"""Gradio speech recognition and translation with local SeamlessM4T v2 weights."""

from __future__ import annotations

import argparse
from contextlib import contextmanager
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
EXAMPLES = [
    [str(SAMPLE_DIR / "zh-CN_en_04.mp3"), "cmn", "eng"],
    [str(SAMPLE_DIR / "en_zh-CN_02.mp3"), "eng", "cmn"],
    [str(SAMPLE_DIR / "fr_en_04.mp3"), "fra", "eng"],
]
CSS = """
.gradio-container {max-width: 1080px !important; margin: auto !important;}
#intro {padding: 12px 4px 20px;}
#intro h1 {font-size: 30px; letter-spacing: -.5px;}
"""
LOGGER = logging.getLogger(__name__)


def read_audio(audio_path: str | None, max_seconds: float) -> tuple[np.ndarray, int]:
    """Decode a bounded clip, validating it before allocating any GPU memory."""
    if not audio_path:
        raise gr.Error("请先上传音频，或使用麦克风录制一段语音。")
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
    def __init__(self, model_dir: Path, precision: str, max_seconds: float, max_tokens: int):
        self.model_dir = model_dir
        self.precision = precision
        self.max_seconds = max_seconds
        self.max_tokens = max_tokens
        self.processor = None
        self.model = None
        self.lock = threading.Lock()

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

    def translate(self, audio_path: str | None, src_lang: str, tgt_lang: str):
        if src_lang not in LANGUAGE_CODES or tgt_lang not in LANGUAGE_CODES:
            raise gr.Error("请选择有效的输入语言和输出语言。")
        audio, sample_rate = read_audio(audio_path, self.max_seconds)
        import torch
        import torchaudio

        started = time.monotonic()
        # Also protects the model's mutable modality and tokenizer outside Gradio.
        with self.lock:
            try:
                self.load_model()
                waveform = torch.from_numpy(audio)
                if sample_rate != 16_000:
                    waveform = torchaudio.functional.resample(waveform, sample_rate, 16_000)
                inputs = self.processor(
                    audio=waveform.numpy(), sampling_rate=16_000, return_tensors="pt",
                )
                # Cast floating features only; attention masks must remain integer.
                inputs = {
                    key: value.to(
                        device=self.model.device,
                        dtype=self.model.dtype if value.is_floating_point() else value.dtype,
                    )
                    for key, value in inputs.items()
                }
                with torch.inference_mode():
                    # The speech encoder does not take a forced source language.
                    # Same-language speech-to-text produces the original transcript.
                    transcript = decode_output(self.processor, self.model.generate(
                        **inputs, tgt_lang=src_lang, generate_speech=False,
                        text_max_new_tokens=self.max_tokens, text_do_sample=False,
                    ))
                    if src_lang == tgt_lang:
                        translation = transcript
                    else:
                        translation = decode_output(self.processor, self.model.generate(
                            **inputs, tgt_lang=tgt_lang, generate_speech=False,
                            text_max_new_tokens=self.max_tokens, text_do_sample=False,
                        ))
                return transcript, translation, f"完成 · 音频 {len(audio) / sample_rate:.1f} 秒 · 用时 {time.monotonic() - started:.1f} 秒"
            except torch.cuda.OutOfMemoryError as exc:
                LOGGER.exception("CUDA out of memory")
                torch.cuda.empty_cache()
                raise gr.Error("GPU 显存不足，请缩短音频后重试。") from exc
            except Exception as exc:
                LOGGER.exception("Speech translation failed")
                raise gr.Error("处理失败，请重试或联系管理员查看服务器日志。") from exc


def build_demo(translator: SpeechTranslator) -> gr.Blocks:
    with gr.Blocks(title="语音识别与翻译", analytics_enabled=False, delete_cache=(3600, 3600)) as demo:
        gr.Markdown(
            "# 语音识别与翻译\n上传音频或录制语音，选择语言，一键查看原文和译文。",
            elem_id="intro",
        )
        with gr.Row():
            with gr.Column(scale=1):
                audio = gr.Audio(
                    sources=["upload", "microphone"], type="filepath", format="wav",
                    label="上传音频 / 麦克风录音", editable=False,
                )
                with gr.Row():
                    source = gr.Dropdown(LANGUAGES, value="cmn", label="输入语言")
                    target = gr.Dropdown(LANGUAGES, value="eng", label="输出语言")
                gr.Markdown(f"请选择音频实际使用的语言；最长 {translator.max_seconds:g} 秒。麦克风录音需要 HTTPS 和浏览器授权。")
                with gr.Row():
                    submit = gr.Button("识别并翻译", variant="primary")
                    clear = gr.Button("清空")
            with gr.Column(scale=1):
                transcript = gr.Textbox(label="原文识别", lines=5, interactive=False)
                translation = gr.Textbox(label="翻译结果", lines=5, interactive=False)
                status = gr.Textbox(label="状态", value="等待输入音频", interactive=False)
        examples = [example for example in EXAMPLES if Path(example[0]).is_file()]
        if examples:
            gr.Examples(
                examples=examples, inputs=[audio, source, target],
                label="试试示例（点击填入后，再点击“识别并翻译”）",
                cache_examples=False,
            )
        submit.click(
            translator.translate, inputs=[audio, source, target],
            outputs=[transcript, translation, status],
            concurrency_limit=1, concurrency_id="speech-model", api_name="translate",
        )
        # Clear previous results when the selected clip or languages change.
        for component in (audio, source, target):
            component.change(
                lambda: ("", "", "等待识别"), outputs=[transcript, translation, status], queue=False,
            )
        clear.click(
            lambda: (None, "", "", "等待输入音频"),
            outputs=[audio, transcript, translation, status], queue=False,
        )
    return demo.queue(max_size=16, default_concurrency_limit=1)


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
    parser.add_argument("--model-dir", type=Path, default=Path(os.getenv("DEMO_MODEL_DIR", str(PROJECT_DIR / "ckpts/seamless-m4t-v2-large"))))
    parser.add_argument("--host", default=os.getenv("DEMO_HOST", "0.0.0.0"))
    parser.add_argument("--port", type=int, default=int(os.getenv("DEMO_PORT", "7860")))
    parser.add_argument("--share", action=argparse.BooleanOptionalAction, default=env_bool("DEMO_SHARE", True))
    parser.add_argument("--root-path", default=os.getenv("DEMO_ROOT_PATH", ""))
    parser.add_argument("--precision", choices=["float16", "bfloat16", "float32"], default=os.getenv("DEMO_PRECISION", "float16"))
    parser.add_argument("--max-seconds", type=float, default=float(os.getenv("DEMO_MAX_SECONDS", "60")))
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--startup-timeout", type=int, default=int(os.getenv("DEMO_STARTUP_TIMEOUT", "90")))
    args = parser.parse_args(argv)
    if args.precision not in {"float16", "bfloat16", "float32"}:
        parser.error("精度必须为 float16、bfloat16 或 float32。")
    if args.startup_timeout <= 0:
        parser.error("启动超时秒数必须大于 0。")
    if not 1 <= args.port <= 65535 or not np.isfinite(args.max_seconds) or args.max_seconds <= 0 or args.max_new_tokens <= 0:
        parser.error("端口必须在 1–65535 之间，音频时长和最大 token 数必须大于 0。")
    return args


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
    translator = SpeechTranslator(args.model_dir.resolve(), args.precision, args.max_seconds, args.max_new_tokens)
    demo = build_demo(translator)
    try:
        translator.load_model()
        print("正在启动界面，请等待公网入口打印 PUBLIC_URL；本地监听地址不能用于公网访问。", flush=True)
        with startup_deadline(args.startup_timeout):
            demo.launch(
                server_name=args.host, server_port=args.port, share=args.share,
                root_path=args.root_path or None, auth=(username, password) if username else None,
                allowed_paths=[example[0] for example in EXAMPLES if Path(example[0]).is_file()],
                blocked_paths=[str(args.model_dir.resolve()), str(PROJECT_DIR / "ckpts"), str(PROJECT_DIR / "outs")],
                max_file_size="25mb", show_error=False, inbrowser=False,
                theme=gr.themes.Soft(primary_hue="blue"), css=CSS,
                prevent_thread_lock=True,
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
        demo.block_thread()
    finally:
        demo.close()


if __name__ == "__main__":
    main()
