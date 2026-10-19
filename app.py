"""Gradio speech recognition and translation with local SeamlessM4T v2 weights."""

from __future__ import annotations

import argparse
from contextlib import contextmanager, nullcontext, ExitStack
import logging
import os
from pathlib import Path
import signal
import tempfile
import threading
import time

PROJECT_DIR = Path(__file__).resolve().parent


def configure_gradio_temp_dir() -> Path:
    """Set and validate the cache BEFORE Gradio freezes its import-time paths."""
    configured = os.environ.get("GRADIO_TEMP_DIR")
    job_id = os.environ.get("SLURM_JOB_ID")
    run_id = f"job-{job_id}" if job_id else f"process-{os.getpid()}"
    cache_dir = (
        Path(configured).expanduser().resolve()
        if configured else PROJECT_DIR / ".cache" / "gradio" / run_id
    )
    try:
        cache_dir.mkdir(mode=0o700, parents=True, exist_ok=True)
        # Checking a real write catches permissions/ACL issues before loading weights.
        with tempfile.TemporaryFile(dir=cache_dir):
            pass
    except OSError as exc:
        raise RuntimeError(
            f"Gradio 缓存目录不可写：{cache_dir}。"
            "请将 GRADIO_TEMP_DIR 设置为当前用户可写的目录。"
        ) from exc
    os.environ["GRADIO_TEMP_DIR"] = str(cache_dir)
    return cache_dir


GRADIO_TEMP_DIR = configure_gradio_temp_dir()

import gradio as gr
import numpy as np
import soundfile as sf

from language_detection import BACKENDS, DEFAULT_BACKEND, DEFAULT_MMS_MODEL_DIR as DEFAULT_LID_MODEL_DIR
from language_detection import LanguageDetectionError, create_language_detector, default_model_dir
import ui
from sample_catalog import load_demo_samples

SAMPLE_DIR = PROJECT_DIR / "data" / "test_samples"
LANGUAGES = [
    ("中文（普通话）", "cmn"), ("中文（繁体输出）", "cmn_Hant"),
    ("英语 · English", "eng"), ("法语 · Français", "fra"),
    ("德语 · Deutsch", "deu"), ("西班牙语 · Español", "spa"),
    ("日语 · 日本語", "jpn"), ("韩语 · 한국어", "kor"),
    ("粤语", "yue"), ("俄语 · Русский", "rus"),
    ("葡萄牙语 · Português", "por"), ("意大利语 · Italiano", "ita"),
    ("阿拉伯语", "arb"), ("印地语", "hin"), ("泰语", "tha"),
    ("越南语", "vie"), ("印尼语", "ind"), ("土耳其语", "tur"),
    ("荷兰语", "nld"), ("波兰语", "pol"), ("乌克兰语", "ukr"),
]
LANGUAGE_CODES = {code for _, code in LANGUAGES}
LANGUAGE_NAMES = {code: name.split(" · ")[0] for name, code in LANGUAGES}
AUTO_LANGUAGE = "auto"
SHOWCASE_SAMPLES = load_demo_samples(SAMPLE_DIR)
if any(item["source_lang"] not in LANGUAGE_CODES or item["target_lang"] not in LANGUAGE_CODES
       for item in SHOWCASE_SAMPLES):
    raise ValueError("展示样本的语言不在文件翻译页面支持范围内。")
EXAMPLES = [[str(SAMPLE_DIR / item["file"]), item["source_lang"], item["target_lang"]]
            for item in SHOWCASE_SAMPLES]
ENGINE_LABEL = "SeamlessM4T v2"
DEFAULT_FILE_TRANSLATION = "speech"
LOGGER = logging.getLogger(__name__)


def read_audio(audio_path: str | None, max_seconds: float) -> tuple[np.ndarray, int]:
    """Decode a bounded clip, validating it before allocating any GPU memory."""
    if not audio_path:
        LOGGER.warning("Audio submission rejected: empty input; recording/upload may be unfinished")
        raise gr.Error("尚未收到音频。录音后请先点击停止，等待“音频已就绪”后再提交；上传失败时请重新录音或上传文件。")
    try:
        with sf.SoundFile(audio_path) as stream:
            if stream.samplerate <= 0 or stream.frames <= 0:
                raise gr.Error("音频为空，请重新上传或录音。")
            if stream.frames / stream.samplerate > max_seconds:
                raise gr.Error(f"音频不能超过 {max_seconds:g} 秒，请截取较短片段。")
            if stream.channels > 8:
                raise gr.Error("音频声道过多，请转换为单声道或双声道。")
            sample_rate = stream.samplerate
            audio = stream.read(
                frames=int(max_seconds * sample_rate) + 1,
                dtype="float32", always_2d=True,
            )
    except gr.Error:
        raise
    except (OSError, RuntimeError, ValueError) as exc:
        LOGGER.warning("Audio decoding failed: %s", exc)
        raise gr.Error("无法读取音频，请使用有效的 MP3、WAV、FLAC 或 OGG 文件。") from exc
    audio = audio.mean(axis=1)
    if audio.size == 0 or not np.isfinite(audio).all():
        raise gr.Error("音频为空或包含无效采样，请更换文件。")
    if not np.any(np.abs(audio) > 1e-5):
        raise gr.Error("音频几乎没有声音，请检查麦克风后重试。")
    return audio, sample_rate


def decode_output(processor, output) -> str:
    """Support ModelOutput (Transformers 5) and older tensor/tuple results."""
    sequences = getattr(output, "sequences", None)
    if sequences is None:
        sequences = output[0] if isinstance(output, tuple) else output
    return processor.decode(sequences[0].tolist(), skip_special_tokens=True).strip()


class SpeechTranslator:
    def __init__(
        self, model_dir: Path, precision: str, max_seconds: float, max_tokens: int,
        lid_model_dir: Path | None = None, lid_max_seconds: float = 10,
        lid_min_confidence: float = 0.5, lid_backend: str = DEFAULT_BACKEND,
        lid_precision: str = "float32",
        text_translator=None, coordinator=None, default_translation_mode=None,
        file_asr=None,
    ):
        self.model_dir = model_dir
        self.precision = precision
        self.max_seconds = max_seconds
        self.max_tokens = max_tokens
        self.processor = None
        self.model = None
        if not np.isfinite(lid_min_confidence) or not 0 <= lid_min_confidence <= 1:
            raise ValueError("语言检测置信度阈值必须在 0–1 之间。")
        self.language_detector = create_language_detector(lid_backend, lid_model_dir, lid_max_seconds, lid_precision)
        self.lid_min_confidence = lid_min_confidence
        self.lock = threading.Lock()
        self.text_translator = text_translator
        self.file_asr = file_asr
        self.file_asr_timeout = float(os.getenv("DEMO_FILE_ASR_TIMEOUT", "120"))
        if not np.isfinite(self.file_asr_timeout) or self.file_asr_timeout <= 0:
            raise ValueError("DEMO_FILE_ASR_TIMEOUT 必须为正数。")
        self.coordinator = coordinator
        self.default_translation_mode = default_translation_mode or DEFAULT_FILE_TRANSLATION

    def load_model(self):
        """Only called at server startup or inference, never when importing the UI."""
        import torch
        from transformers import AutoProcessor, SeamlessM4Tv2Model

        if self.model is not None:
            return
        if not self.model_dir.is_dir():
            raise RuntimeError(f"模型目录不存在：{self.model_dir}")
        if not torch.cuda.is_available():
            raise RuntimeError("未检测到 CUDA GPU，请检查 GPU、驱动及 PyTorch CUDA 环境；在 Slurm 集群上请提交 GPU 作业。")
        dtype = getattr(torch, self.precision)
        if dtype == torch.bfloat16 and not torch.cuda.is_bf16_supported():
            raise RuntimeError("此 GPU 不支持 bfloat16，请选择 float16 或 float32。")
        LOGGER.info("Loading SeamlessM4T v2 from %s (%s)", self.model_dir, self.precision)
        self.processor = AutoProcessor.from_pretrained(self.model_dir, local_files_only=True)
        self.model = SeamlessM4Tv2Model.from_pretrained(
            self.model_dir, dtype=dtype, local_files_only=True,
        ).to("cuda").eval()
        LOGGER.info("Model ready on %s", torch.cuda.get_device_name(0))

    def translate(self, audio_path: str | None, src_lang: str, tgt_lang: str, translation_mode=None):
        # A generator: each yield reports the stage actually running, so the page can
        # show it and fill in the transcript before the translation. The last yield is
        # the result; API clients still receive only that final value.
        if src_lang not in LANGUAGE_CODES | {AUTO_LANGUAGE} or tgt_lang not in LANGUAGE_CODES:
            raise gr.Error("请选择有效的输入语言和输出语言。")
        audio, sample_rate = read_audio(audio_path, self.max_seconds)
        yield from self.translate_audio(audio, sample_rate, src_lang, tgt_lang, translation_mode)

    def translate_audio(self, audio: np.ndarray, sample_rate: int, src_lang: str, tgt_lang: str, translation_mode=None):
        try:
            with self.coordinator.file_inference() if self.coordinator else nullcontext():
                yield from self._translate_audio(audio, sample_rate, src_lang, tgt_lang, translation_mode)
        except ValueError as exc:
            raise gr.Error(str(exc)) from exc

    def _translate_audio(self, audio, sample_rate, src_lang, tgt_lang, translation_mode):
        """The same stage/result generator for in-memory, mono float audio."""
        translation_mode = translation_mode or self.default_translation_mode
        if translation_mode not in {"text", "speech"}:
            raise gr.Error("请选择有效的翻译方式。")
        if translation_mode == "text" and self.text_translator is None:
            raise gr.Error("当前服务没有启用文本翻译模型，请选择 SeamlessM4T 直接语音翻译。")
        if src_lang not in LANGUAGE_CODES | {AUTO_LANGUAGE} or tgt_lang not in LANGUAGE_CODES:
            raise gr.Error("请选择有效的输入语言和输出语言。")
        audio = np.asarray(audio, dtype=np.float32)
        if (
            audio.ndim != 1 or not len(audio) or not np.isfinite(audio).all()
            or not isinstance(sample_rate, (int, np.integer)) or sample_rate <= 0
        ):
            raise gr.Error("内存音频必须是有效的单声道数组和采样率。")
        if len(audio) / sample_rate > self.max_seconds:
            raise gr.Error(f"音频不能超过 {self.max_seconds:g} 秒。")
        import torch
        import torchaudio

        started = time.monotonic()
        # Also protects the model's mutable modality and tokenizer outside Gradio.
        with self.lock:
            try:
                waveform = torch.from_numpy(audio)
                if sample_rate != 16_000:
                    waveform = torchaudio.functional.resample(waveform, sample_rate, 16_000)
                detection_elapsed = None
                if src_lang == AUTO_LANGUAGE:
                    yield "", "", "等待检测", "排队或处理中 · 正在识别语言"
                    prediction = self.language_detector.detect(waveform.numpy())
                    detected_code = self.language_detector.to_seamless_code(prediction.code)
                    if prediction.confidence < self.lid_min_confidence:
                        detected_name = LANGUAGE_NAMES[detected_code] if detected_code else prediction.label
                        raise gr.Error(
                            f"可能是{detected_name}，但检测置信度仅为 "
                            f"{prediction.confidence:.1%}，请手动选择输入语言或提供更清晰的音频。"
                        )
                    if detected_code is None:
                        raise gr.Error(
                            f"检测到 {prediction.label}，当前 Demo 未提供该语言的自动识别映射。"
                            "请手动确认输入语言后重试。"
                        )
                    src_lang = detected_code
                    language_result = f"自动检测（{self.language_detector.display_name}）：{LANGUAGE_NAMES[src_lang]} · 置信度 {prediction.confidence:.1%}"
                    detection_elapsed = prediction.elapsed
                else:
                    language_result = f"手动指定：{LANGUAGE_NAMES[src_lang]}"
                from realtime.backend import ASR_LANGUAGES
                asr_source = src_lang
                use_nemotron = translation_mode == "text" and self.file_asr is not None and asr_source in ASR_LANGUAGES
                asr_name = "Nemotron（非流式）" if use_nemotron else "SeamlessM4T"
                asr_started = time.monotonic()
                LOGGER.info("File ASR started: backend=%s mode=%s source=%s target=%s audio=%.2fs",
                            asr_name, translation_mode, src_lang, tgt_lang, len(audio) / sample_rate)
                yield "", "", language_result, f"排队或处理中 · 正在转写 · {asr_name}"
                if use_nemotron:
                    from realtime.integration import stream_file_transcription
                    transcript = ""
                    drafts = stream_file_transcription(self.file_asr, waveform.numpy(), asr_source,
                                                       timeout=self.file_asr_timeout)
                    try:
                        for draft in drafts:
                            if draft is not None:
                                transcript = draft
                            yield transcript, "", language_result, (
                                f"排队或处理中 · 正在转写 · {asr_name} · {time.monotonic() - asr_started:.1f} 秒")
                    finally:
                        drafts.close()
                else:
                    if translation_mode == "text" and self.file_asr is not None:
                        LOGGER.info("File ASR language %s uses SeamlessM4T compatibility fallback", src_lang)
                    self.load_model()
                    inputs = self.processor(audio=waveform.numpy(), sampling_rate=16_000, return_tensors="pt")
                    # Cast floating features only; masks must remain integer.
                    inputs = {key: value.to(
                        device=self.model.device, dtype=self.model.dtype if value.is_floating_point() else value.dtype,
                    ) for key, value in inputs.items()}
                    with torch.inference_mode():
                        transcript = decode_output(self.processor, self.model.generate(
                            **inputs, tgt_lang=src_lang, generate_speech=False,
                            text_max_new_tokens=self.max_tokens, text_do_sample=False,
                            text_max_time=self.file_asr_timeout,
                        ))
                    if time.monotonic() - asr_started > self.file_asr_timeout:
                        raise gr.Error(f"SeamlessM4T 转写超过 {self.file_asr_timeout:g} 秒，请缩短音频或使用支持的 Nemotron 级联语言。")
                if not transcript.strip():
                    raise gr.Error("未识别到有效原文，请确认输入语言、音量或更换音频。")
                LOGGER.info("File ASR completed: backend=%s elapsed=%.3fs chars=%d",
                            asr_name, time.monotonic() - asr_started, len(transcript))
                if src_lang == tgt_lang:
                    translation = transcript
                else:
                    # Yield outside inference_mode: Gradio may resume the generator on
                    # another worker thread, and grad mode is thread-local.
                    yield transcript, "", language_result, "排队或处理中 · 正在翻译"
                    LOGGER.info("File translation started: mode=%s source=%s target=%s chars=%d",
                                translation_mode, src_lang, tgt_lang, len(transcript))
                    if translation_mode == "text":
                        from realtime.integration import stream_text_translation
                        drafts = stream_text_translation(self.text_translator, transcript, src_lang, tgt_lang)
                        try:
                            for translation in drafts:
                                yield transcript, translation, language_result, "排队或处理中 · 正在翻译"
                        finally:
                            drafts.close()
                    else:
                        with torch.inference_mode():
                            translation = decode_output(self.processor, self.model.generate(
                                **inputs, tgt_lang=tgt_lang, generate_speech=False,
                                text_max_new_tokens=self.max_tokens, text_do_sample=False,
                            ))
                status = f"完成 · 音频 {len(audio) / sample_rate:.1f} 秒 · 用时 {time.monotonic() - started:.1f} 秒"
                if detection_elapsed is not None:
                    status += f"（语言检测 {detection_elapsed:.2f} 秒）"
                yield transcript, translation, language_result, status
                LOGGER.info("File request completed: mode=%s elapsed=%.3fs", translation_mode, time.monotonic() - started)
            except gr.Error:
                raise
            except (LanguageDetectionError, ValueError) as exc:
                LOGGER.warning("File request rejected: %s", exc)
                raise gr.Error(str(exc)) from exc
            except torch.cuda.OutOfMemoryError as exc:
                LOGGER.exception("CUDA out of memory")
                torch.cuda.empty_cache()
                raise gr.Error("GPU 显存不足，请缩短音频后重试。") from exc
            except Exception as exc:
                LOGGER.exception("Speech translation failed")
                raise gr.Error("处理失败，请重试或联系管理员查看服务器日志。") from exc


def example_label(path: str, src: str, tgt: str) -> str:
    """Button text for a bundled sample: language pair plus its real duration."""
    for sample in SHOWCASE_SAMPLES:
        if sample["file"] == Path(path).name:
            return sample["label"]
    short = {"cmn": "中文", "eng": "英语", "fra": "法语"}
    stem = Path(path).stem.split("_")[0]
    spoken = {"zh-CN": "中文", "en": "英语", "fr": "法语"}.get(stem, "自动检测" if src == AUTO_LANGUAGE else short.get(src, LANGUAGE_NAMES[src]))
    try:
        seconds = f" · {sf.info(path).duration:.1f}s"
    except (OSError, RuntimeError):
        seconds = ""
    return f"{spoken} → {short.get(tgt, LANGUAGE_NAMES[tgt])}{seconds}"


def build_demo(translator, demo_mode: bool = False, realtime_enabled: bool = False) -> gr.Blocks:
    engine_label = ENGINE_LABEL + (" · Nemotron" if realtime_enabled else "")
    if getattr(translator, "text_translator", None) is not None:
        engine_label += " · " + translator.text_translator.name.replace(" 文本翻译", "")
    with gr.Blocks(title="语音识别与翻译", analytics_enabled=False, delete_cache=(3600, 3600)) as demo:
        gr.HTML(ui.topbar_html(engine_label, demo_mode), container=False, padding=False, elem_id="st-topbar")
        # Both product modes stay visible, including on narrow screens. Gradio's
        # default "menu" behavior otherwise moves overflowing tabs to a dropdown.
        with gr.Tabs(elem_id="st-tabs", selected="translate", overflow_behavior="wrap"):
            with gr.Tab("语音翻译", id="translate", elem_id="st-tab-translate", render_children=True):
                gr.HTML(ui.hero_html(len(LANGUAGES), demo_mode), container=False, padding=False)
                with gr.Row(elem_id="st-workspace", equal_height=False):
                    with gr.Column(scale=11, elem_id="st-input", elem_classes="st-pane"):
                        with gr.Row(elem_id="st-langbar"):
                            source = gr.Dropdown(
                                [("自动检测", AUTO_LANGUAGE), *LANGUAGES], value=AUTO_LANGUAGE,
                                label="说话语言", elem_id="st-source", scale=5, min_width=120,
                            )
                            swap = gr.Button("⇄", elem_id="st-swap", scale=0, min_width=40)
                            target = gr.Dropdown(LANGUAGES, value="eng", label="翻译为", elem_id="st-target", scale=5, min_width=120)
                        text_enabled = demo_mode or getattr(translator, "text_translator", None) is not None
                        text_name = getattr(getattr(translator, "text_translator", None), "name", "Hy-MT2 文本翻译")
                        mode = gr.Dropdown(
                            [("SeamlessM4T（端到端）", "speech"),
                             (text_name.replace(" 文本翻译", "") + "（级联）", "text")]
                            if text_enabled else [("SeamlessM4T（端到端）", "speech")],
                            value=getattr(translator, "default_translation_mode", DEFAULT_FILE_TRANSLATION),
                            label="翻译模型", info="端到端：音频直接翻译 · 级联：先识别，再翻译",
                            filterable=False, allow_custom_value=False,
                            elem_id="st-translation-mode",
                        )
                        gr.HTML(ui.STAGE_HTML, container=False, padding=False, elem_id="st-stage-visual")
                        audio = gr.Audio(
                            sources=["upload", "microphone"], type="filepath", format="wav",
                            label="上传音频 / 麦克风录音", show_label=False, editable=False, elem_id="st-audio",
                            waveform_options=gr.WaveformOptions(
                                # The stage's voice wave is the live recording visual.
                                waveform_color="#C9C9C6", waveform_progress_color="#8F8F8F",
                                trim_region_color="#8F8F8F", show_recording_waveform=False,
                            ),
                        )
                        with gr.Row(elem_id="st-actions"):
                            submit = gr.Button("识别并翻译", variant="primary", interactive=False, elem_id="st-submit", scale=3)
                            clear = gr.Button("清空", variant="secondary", elem_id="st-clear", scale=1, min_width=88)
                        gr.HTML(ui.PLAYBACK_HTML, container=False, padding=False)
                        gr.HTML(ui.hint_html(translator.max_seconds), container=False, padding=False)
                    with gr.Column(scale=13, elem_id="st-output", elem_classes="st-pane"):
                        with gr.Group(elem_classes="st-card st-card-source"):
                            language_result = gr.Textbox(
                                label="输入语言识别", show_label=False, value="等待检测", interactive=False,
                                elem_id="st-lang-result", container=False,
                            )
                            transcript = gr.Textbox(
                                label="原文", lines=3, max_lines=8, interactive=False, buttons=["copy"],
                                placeholder="说出或上传的内容会显示在这里", elem_id="st-transcript",
                            )
                        with gr.Group(elem_classes="st-card st-card-target"):
                            translation = gr.Textbox(
                                label="译文", lines=4, max_lines=10, interactive=False, buttons=["copy"],
                                placeholder="译文将在这里出现", elem_id="st-translation",
                            )
                        status = gr.Textbox(
                            label="状态", show_label=False, value="等待输入音频", interactive=False,
                            elem_id="st-status", container=False,
                        )
                examples = [example for example in EXAMPLES if Path(example[0]).is_file()]
                example_buttons = []
                if examples:
                    with gr.Column(elem_id="st-examples"):
                        gr.HTML(ui.section_head_html("试试示例", "10–30 秒真人朗读 · 点击填入语言与音频后再提交"), container=False, padding=False)
                        with gr.Row(elem_id="st-example-row"):
                            for path, src, tgt in examples:
                                button = gr.Button(example_label(path, src, tgt), elem_classes="st-example", size="md")
                                button.click(
                                    lambda path=path, src=src, tgt=tgt: (path, src, tgt),
                                    outputs=[audio, source, target], queue=False, show_progress="hidden", api_visibility="private",
                                )
                                example_buttons.append(button)
            with gr.Tab("同声传译", id="interpret", elem_id="st-tab-interpret", render_children=True):
                if realtime_enabled:
                    gr.HTML(ui.realtime_html(), container=False, padding=False)
                else:
                    gr.Markdown("当前启动配置关闭了实时翻译。启用 `--realtime` 后，可在此持续讲话并查看原文与译文。")
        gr.HTML(ui.footer_html("演示模式 · 未加载模型" if demo_mode else engine_label), container=False, padding=False)
        controls = [submit, clear, audio, source, target, mode, swap, *example_buttons]
        # Run readiness/control updates in the browser. A local recording preview
        # can appear before its upload completes; server-side change callbacks also
        # arrive late through a tunnel. Never reuse an earlier recording as fallback.
        audio.change(
            fn=None, inputs=[audio], outputs=[submit, transcript, translation, language_result, status],
            js="""(clip) => {
                window.stFilePlayback?.changed(clip);
                const ready = Boolean(clip && clip.path);
                return [{__type__: "update", interactive: ready}, "", "", "等待检测",
                    ready ? "音频已就绪 · 可以识别并翻译" : "等待音频 · 录音后请先停止并等待上传完成"];
            }""",
            queue=False,
        )
        audio.start_recording(
            fn=None, outputs=[submit, transcript, translation, language_result, status],
            js="""() => {
                window.stFilePlayback?.stop();
                return [{__type__: "update", interactive: false}, "", "", "等待检测",
                    "正在录音 · 请先停止，等待音频上传完成"];
            }""",
            queue=False,
        )
        # This first step has no HTTP round trip. Lock this tab's inputs before
        # capturing the request for the existing serial inference queue.
        submission = submit.click(
            fn=None, inputs=[audio], outputs=[*controls, transcript, translation, language_result, status],
            js=f"""(clip) => {{
                if (!clip || !clip.path) throw new Error("音频尚未就绪，请先停止录音并等待上传完成。");
                window.stFilePlayback?.start(clip);
                return [...Array.from({{length: {len(controls)}}}, () => ({{__type__: "update", interactive: false}})),
                    "", "", "等待检测", "排队或处理中 · 请稍候"];
            }}""",
            queue=False, trigger_mode="once",
        )
        inference = submission.success(
            translator.translate, inputs=[audio, source, target, mode],
            outputs=[transcript, translation, language_result, status],
            concurrency_limit=1, concurrency_id="speech-model", api_name="translate",
            trigger_mode="once", show_progress="hidden",
        )
        inference.failure(
            fn=None, outputs=[status], queue=False,
            js='() => ["处理未完成 · 请查看错误提示，调整输入后重试"]',
        )
        # .then also runs on errors, so a failed detection cannot strand controls.
        inference.then(
            fn=None, inputs=[audio], outputs=controls, queue=False,
            js=f"""(clip) => Array.from({{length: {len(controls)}}}, (_, index) =>
                ({{__type__: "update", interactive: index === 0 ? Boolean(clip && clip.path) : true}}))""",
        )
        for component in (source, target, mode):
            component.change(
                fn=None, inputs=[audio],
                outputs=[transcript, translation, language_result, status], queue=False,
                js="""(clip) => ["", "", "等待检测", clip && clip.path
                    ? "音频已就绪 · 可以识别并翻译" : "等待音频 · 录音后请先停止并等待上传完成"]""",
            )
        swap.click(
            fn=None, inputs=[source, target], outputs=[source, target], queue=False,
            js=f"""(src, tgt) => src === "{AUTO_LANGUAGE}" ? [src, tgt] : [tgt, src]""",
        )
        clear.click(
            fn=None, outputs=[audio, source, transcript, translation, language_result, status, submit],
            js='() => { window.stFilePlayback?.stop(); return [null, "auto", "", "", "等待检测", "等待输入音频", {__type__: "update", interactive: false}]; }',
            queue=False,
        )
    return demo.queue(max_size=16, default_concurrency_limit=1)


def launch_styling(realtime_enabled: bool = False) -> dict:
    """Theme, styles, fonts and front-end script, passed to every launch()."""
    script = ui.load_js()
    if realtime_enabled:
        script = "() => { (" + script + ")(); (" + ui.realtime_js() + ")(); }"
    return {
        "theme": ui.build_theme(), "css": ui.load_css() + (ui.realtime_css() if realtime_enabled else ""), "js": script,
        "head": ui.FONTS_HEAD, "footer_links": [],
    }


def env_bool(name: str, default: bool) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    if value.lower() not in {"1", "0", "true", "false", "yes", "no"}:
        raise ValueError(f"{name} 必须为 1/0、true/false 或 yes/no")
    return value.lower() in {"1", "true", "yes"}


@contextmanager
def startup_deadline(seconds: float):
    """Bound launch, including FRPC's otherwise blocking stdout read, on Linux."""
    def expired(signum, frame):
        LOGGER.error("Gradio 启动/公网隧道创建超过 %g 秒，停止启动。", seconds)
        raise TimeoutError("公网隧道启动超时；请检查计算节点网络或改用反向 SSH 隧道。")

    previous_handler = signal.signal(signal.SIGALRM, expired)
    signal.setitimer(signal.ITIMER_REAL, seconds)
    try:
        yield
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        signal.signal(signal.SIGALRM, previous_handler)


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    from realtime.translation import add_translation_arguments
    add_translation_arguments(parser)
    parser.set_defaults(mt_backend=os.getenv("DEMO_MT_BACKEND", "hy-mt2-vllm"))
    parser.add_argument("--realtime", action=argparse.BooleanOptionalAction, default=env_bool("DEMO_REALTIME", True),
                        help="启用同声传译页，默认启用")
    parser.add_argument("--file-translation", choices=["speech", "text"], default=os.getenv("DEMO_FILE_TRANSLATION", DEFAULT_FILE_TRANSLATION),
                        help="文件页默认 speech：SeamlessM4T 端到端语音翻译；text：先转写，再用文本模型翻译")
    parser.add_argument("--manage-vllm", action=argparse.BooleanOptionalAction, default=env_bool("DEMO_MANAGE_VLLM", True),
                        help="需要 vLLM 时自动启动/清理本机引擎；已有服务用 --no-manage-vllm")
    parser.add_argument("--model-dir", type=Path, default=Path(os.getenv("DEMO_MODEL_DIR", str(PROJECT_DIR / "ckpts/seamless-m4t-v2-large"))))
    parser.add_argument("--host", default=os.getenv("DEMO_HOST", "0.0.0.0"))
    parser.add_argument("--port", type=int, default=int(os.getenv("DEMO_PORT", "7860")))
    parser.add_argument("--share", action=argparse.BooleanOptionalAction, default=env_bool("DEMO_SHARE", True))
    parser.add_argument("--root-path", default=os.getenv("DEMO_ROOT_PATH", ""))
    parser.add_argument("--precision", choices=["float16", "bfloat16", "float32"], default=os.getenv("DEMO_PRECISION", "float16"))
    parser.add_argument("--max-seconds", type=float, default=float(os.getenv("DEMO_MAX_SECONDS", "60")))
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--lid-backend", choices=BACKENDS, default=os.getenv("DEMO_LID_BACKEND", DEFAULT_BACKEND))
    parser.add_argument("--lid-model-dir", type=Path, default=Path(os.environ["DEMO_LID_MODEL_DIR"]) if os.getenv("DEMO_LID_MODEL_DIR") else None)
    parser.add_argument("--lid-precision", choices=["float16", "bfloat16", "float32"], default=os.getenv("DEMO_LID_PRECISION", "float32"))
    parser.add_argument("--lid-max-seconds", type=float, default=float(os.getenv("DEMO_LID_MAX_SECONDS", "10")))
    parser.add_argument("--lid-min-confidence", type=float, default=float(os.getenv("DEMO_LID_MIN_CONFIDENCE", "0.5")))
    parser.add_argument(
        "--demo-mode", action=argparse.BooleanOptionalAction, default=env_bool("DEMO_MODE", False),
        help="不加载模型，仅为内置示例返回数据集参考文本，用于在无 GPU 环境预览界面",
    )
    parser.add_argument("--startup-timeout", type=int, default=int(os.getenv("DEMO_STARTUP_TIMEOUT", "90")))
    args = parser.parse_args(argv)
    if args.mt_backend not in {"seamless", "hy-mt2", "hy-mt2-vllm"} or args.file_translation not in {"text", "speech"}:
        parser.error("无效的文本翻译后端或文件翻译方式。")
    if args.lid_backend not in BACKENDS:
        parser.error("语言检测模块必须为 mms 或 speechbrain。")
    if args.lid_precision not in {"float16", "bfloat16", "float32"}:
        parser.error("MMS 检测精度必须为 float16、bfloat16 或 float32。")
    if args.lid_model_dir is None:
        args.lid_model_dir = default_model_dir(args.lid_backend)
    if args.precision not in {"float16", "bfloat16", "float32"}:
        parser.error("精度必须为 float16、bfloat16 或 float32。")
    if args.startup_timeout <= 0:
        parser.error("启动超时秒数必须大于 0。")
    if not np.isfinite(args.lid_max_seconds) or args.lid_max_seconds < 1:
        parser.error("语言检测片段长度必须至少为 1 秒。")
    if not np.isfinite(args.lid_min_confidence) or not 0 <= args.lid_min_confidence <= 1:
        parser.error("语言检测置信度阈值必须在 0–1 之间。")
    if not 1 <= args.port <= 65535 or not np.isfinite(args.max_seconds) or args.max_seconds <= 0 or args.max_new_tokens <= 0:
        parser.error("端口必须在 1–65535 之间，音频时长和最大 token 数必须大于 0。")
    return args


@contextmanager
def termination_signals():
    """Let Slurm SIGTERM unwind startup/inference and stop owned engine workers."""
    def stop(signum, frame):
        raise SystemExit(128 + signum)

    previous = {sig: signal.signal(sig, stop) for sig in (signal.SIGTERM, signal.SIGINT)}
    try:
        yield
    finally:
        for sig, handler in previous.items():
            signal.signal(sig, handler)


def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    args = parse_args()
    LOGGER.info("GRADIO_TEMP_DIR=%s", GRADIO_TEMP_DIR)
    LOGGER.info("启动设置：share=%s，host=%s，port=%s", args.share, args.host, args.port)
    if not args.share:
        if os.getenv("DEMO_RELAY_TARGET"):
            LOGGER.info("公网链接由登录节点 %s 创建；计算节点仅提供本地服务，等待中转进程打印 PUBLIC_URL。", os.environ["DEMO_RELAY_TARGET"])
        else:
            LOGGER.info("Gradio 公网分享已关闭；外网访问需要已配置的公网 HTTPS 反向代理。")
    username, password = os.getenv("DEMO_AUTH_USER"), os.getenv("DEMO_AUTH_PASSWORD")
    if bool(username) != bool(password):
        raise RuntimeError("DEMO_AUTH_USER 和 DEMO_AUTH_PASSWORD 必须同时设置。")
    from realtime.integration import InferenceCoordinator
    from realtime.translation import create_text_translator, translation_options
    from realtime import create_service
    from realtime.api import create_api, require_gradio_login
    from realtime.engine import ManagedVLLM

    # Start the engine before loading ASR/LID/Seamless, as in the tested vLLM path.
    with termination_signals(), ExitStack() as resources:
        needs_mt = args.realtime or args.file_translation == "text"
        text_translator = create_text_translator(**translation_options(args), require_slurm=False) if needs_mt else None
        engine = None
        if needs_mt and args.mt_backend == "hy-mt2-vllm" and args.manage_vllm and not args.demo_mode:
            engine = resources.enter_context(ManagedVLLM(text_translator, demo_port=args.port))
        service = None
        if args.realtime:
            service = create_service(mock=args.demo_mode, pipeline="cascade", text_translator=text_translator,
                                     require_slurm=False)
            resources.callback(service.close)
            text_translator = service.translator
        elif text_translator is not None and not args.demo_mode:
            text_translator.load()
        file_asr = service.asr if service is not None else None
        if text_translator is not None and file_asr is None and not args.demo_mode:
            from realtime.backend import NemotronASR
            file_asr = NemotronASR(os.getenv("REALTIME_ASR_MODEL_DIR"),
                os.getenv("REALTIME_ASR_PRECISION", "float32"), int(os.getenv("REALTIME_ASR_LOOKAHEAD", "3")))
            file_asr.load()
        coordinator = InferenceCoordinator(service)
        if args.demo_mode:
            from demo_mode import DemoTranslator
            LOGGER.warning("演示模式：文件页仅预览样本/参考文本；实时页使用明确标注的 MOCK 输出。")
            translator = DemoTranslator(SAMPLE_DIR, args.max_seconds, LANGUAGE_NAMES, LANGUAGE_CODES, AUTO_LANGUAGE, read_audio)
            translator.default_translation_mode = args.file_translation
        else:
            translator = SpeechTranslator(
                args.model_dir.resolve(), args.precision, args.max_seconds, args.max_new_tokens,
                args.lid_model_dir, args.lid_max_seconds, args.lid_min_confidence,
                args.lid_backend, args.lid_precision, text_translator=text_translator,
                coordinator=coordinator, default_translation_mode=args.file_translation,
                file_asr=file_asr,
            )
            LOGGER.info("语言检测模块：%s", translator.language_detector.display_name)
            translator.language_detector.load_model()
            translator.load_model()
        demo = build_demo(translator, demo_mode=args.demo_mode, realtime_enabled=args.realtime)
        resources.callback(demo.close)
        app_kwargs = {}
        if service is not None:
            api = create_api(service, LANGUAGES, start_session=coordinator.start_session,
                             authorize=require_gradio_login)
            # Register routes during Gradio's App construction so launch/share/
            # root_path/auth keep their existing behavior and use a single port.
            app_kwargs = {"routes": api.router.routes, "lifespan": api.router.lifespan_context}
        print("正在启动界面，请等待公网入口打印 PUBLIC_URL；本地监听地址不能用于公网访问。", flush=True)
        with startup_deadline(args.startup_timeout):
            demo.launch(
                server_name=args.host, server_port=args.port, share=args.share,
                root_path=args.root_path or None, auth=(username, password) if username else None,
                allowed_paths=[example[0] for example in EXAMPLES if Path(example[0]).is_file()],
                blocked_paths=[str(args.model_dir.resolve()), str(args.lid_model_dir.resolve()),
                    str(PROJECT_DIR / "ckpts"), str(PROJECT_DIR / "outs"), str(PROJECT_DIR / ".env.local"),
                    *([str(text_translator.model_dir.resolve())] if hasattr(text_translator, "model_dir") else [])],
                max_file_size="25mb", show_error=False, inbrowser=False,
                **launch_styling(args.realtime), prevent_thread_lock=True, app_kwargs=app_kwargs,
            )
        if args.share and not demo.share_url:
            raise RuntimeError("公网分享链接创建失败，请检查计算节点的出站网络，或使用 README 中的固定域名方案。")
        if demo.share_url:
            print(f"PUBLIC_URL={demo.share_url}\n请在其他设备的浏览器中打开上述 HTTPS 地址。", flush=True)
            LOGGER.info("PUBLIC_URL=%s", demo.share_url)
        else:
            if os.getenv("DEMO_RELAY_TARGET"):
                print("计算节点服务已就绪，请等待登录节点中转进程打印 PUBLIC_URL。", flush=True)
            else:
                print("当前使用外部反向代理模式，请访问你配置的公网 HTTPS 域名。", flush=True)
        while demo.is_running:
            if engine is not None:
                engine.check()
            time.sleep(.25)


if __name__ == "__main__":
    main()
