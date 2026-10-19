"""Native cache-aware ASR and text-only translation, loaded explicitly on CUDA."""
import logging
import os
from pathlib import Path
import threading
import time
import numpy as np
from .stream_audio import PCMWindows, StreamCancelled, resample_stream
from .cuda_execution import make_inference_stream, inference_stream_context

ROOT = Path(__file__).resolve().parents[1]
LOGGER = logging.getLogger(__name__)
# Only locales with out-of-box transcription, intersected with the existing UI.
ASR_LANGUAGES = {
    "eng": "en-US", "cmn": "zh-CN", "fra": "fr-FR", "deu": "de-DE",
    "spa": "es-ES", "ita": "it-IT", "por": "pt-BR", "rus": "ru-RU",
    "jpn": "ja-JP", "kor": "ko-KR", "arb": "ar-AR", "hin": "hi-IN",
    "vie": "vi-VN", "tur": "tr-TR", "pol": "pl-PL", "nld": "nl-NL",
    "ukr": "uk-UA", "swe": "sv-SE",
}


def require_cuda():
    import torch
    if not torch.cuda.is_available():
        raise RuntimeError("未检测到 CUDA，请通过 Slurm 提交 GPU 作业；CPU 预览使用 --mock。")


def stream_features(processor, pcm, first, last, real_frames, source):
    """The processor drops is_last_audio_chunk; use the extractor for later chunks.

    Tail padding must happen after preemphasis. Padding raw PCM first creates
    an artificial impulse at EOF, and loses short final STFT frames.
    """
    import torch
    required = processor.num_mel_frames_first_audio_chunk if first else processor.num_mel_frames_per_audio_chunk
    if not first and real_frames == 0:
        return torch.zeros((1, required, processor.feature_extractor.feature_size)), None
    if first:
        inputs = processor(pcm, sampling_rate=16000, language=ASR_LANGUAGES[source],
                           is_streaming=True, is_first_audio_chunk=True, return_tensors="pt")
    else:
        inputs = processor.feature_extractor(pcm, sampling_rate=16000, center=False,
                                            is_last_audio_chunk=last, return_tensors="pt")
    if inputs.input_features.shape[1] < real_frames:
        raise RuntimeError("流式 STFT 帧数不足，请检查前处理版本。")
    value = inputs.input_features[:, :real_frames]
    value = torch.nn.functional.pad(value, (0, 0, 0, required - real_frames))
    return value, inputs.get("prompt_ids")


class TokenReceiver:
    """Publish full hypotheses, including Chinese without whitespace boundaries."""
    def __init__(self, processor, on_text, max_tokens):
        self.processor, self.on_text, self.max_tokens = processor, on_text, max_tokens
        self.ids, self.text = [], ""

    def put(self, value):
        ids = value.detach().cpu().reshape(-1).tolist()
        self.ids.extend(ids)
        if len(self.ids) > self.max_tokens + 1:
            raise RuntimeError("ASR 达到输出上限，请缩短测试音频。")
        text = self.processor.decode(self.ids, skip_special_tokens=True).strip()
        # Incomplete UTF-8 byte tokens can transiently decode as U+FFFD.
        if text != self.text and "\ufffd" not in text:
            self.text = text
            self.on_text(text, False)

    def end(self):
        pass  # Finality belongs to a successful generator drain, not streamer.end().


class NemotronASR:
    name = "Nemotron 3.5 流式 ASR"
    def __init__(self, model_dir=None, precision="float32", lookahead=3, max_tokens=16384):
        self.model_dir = Path(model_dir or ROOT / "ckpts/nemotron-3.5-asr-streaming-0.6b")
        self.precision, self.lookahead, self.max_tokens = precision, lookahead, max_tokens
        if precision not in {"float32", "float16", "bfloat16"} or lookahead not in {0, 3, 6, 13}:
            raise ValueError("ASR 精度或 lookahead 无效；可选 0/3/6/13。")
        self.lock = threading.Lock()
        self.cuda_stream = None

    def load(self):
        import torch
        from transformers import AutoModelForRNNT, AutoProcessor
        require_cuda()
        if self.precision == "bfloat16" and not torch.cuda.is_bf16_supported():
            raise ValueError("当前 GPU 不支持 bfloat16")
        self.processor = AutoProcessor.from_pretrained(self.model_dir, local_files_only=True)
        self.processor.set_num_lookahead_tokens(self.lookahead)
        self.model = AutoModelForRNNT.from_pretrained(
            self.model_dir, local_files_only=True, dtype=getattr(torch, self.precision),
        ).to("cuda:0").eval()
        self.cuda_stream = make_inference_stream(self.model.device, priority=-1)
        LOGGER.info("ASR ready: %s, %s, lookahead=%s (%sms)", self.model_dir, self.precision,
                    self.lookahead, self.processor.streaming_latency_ms)

    def transcribe(self, audio, source, *, on_text=None, stop=None, timeout=120):
        """Recognize a complete 16 kHz clip using the official non-streaming API.

        Shares weights/lock with run(), but passes a full feature tensor instead
        of a live iterator. Partial text is only progress, never a completed result.
        """
        import torch
        from transformers import StoppingCriteria, StoppingCriteriaList
        source = "cmn" if source == "cmn_Hant" else source
        if source not in ASR_LANGUAGES:
            raise ValueError(f"Nemotron 不支持此文件输入语言：{source}")
        audio = np.asarray(audio, dtype=np.float32)
        if audio.ndim != 1 or not len(audio) or not np.isfinite(audio).all():
            raise ValueError("离线识别需要有效的单声道 16 kHz 音频。")
        stop = stop if stop is not None else threading.Event()
        started = time.monotonic()

        def check():
            if stop.is_set():
                raise StreamCancelled()
            if time.monotonic() - started > timeout:
                raise ValueError(f"文件转写超过 {timeout:g} 秒，请缩短音频并检查作业日志。")

        class Check(StoppingCriteria):
            def __call__(self, input_ids, scores, **kwargs):
                check()
                return False

        def partial(text, final=False):
            check()
            if on_text is not None:
                on_text(text)

        with self.lock, torch.inference_mode(), inference_stream_context(self.cuda_stream):
            check()
            inputs = self.processor(audio, sampling_rate=16000, language=ASR_LANGUAGES[source],
                                    is_streaming=False, return_tensors="pt")
            check()
            inputs = {key: value.to(device=self.model.device,
                       dtype=self.model.dtype if value.is_floating_point() else value.dtype)
                      if isinstance(value, torch.Tensor) else value for key, value in inputs.items()}
            receiver = TokenReceiver(self.processor, partial, self.max_tokens)
            output = self.model.generate(**inputs, streamer=receiver, max_new_tokens=self.max_tokens,
                stopping_criteria=StoppingCriteriaList([Check()]), do_sample=False,
                return_dict_in_generate=True)
            check()
            if output.sequences.shape[-1] >= self.max_tokens:
                raise ValueError("文件识别达到输出上限，不能把截断原文交给翻译。")
            text = self.processor.decode(output.sequences[0], skip_special_tokens=True).strip()
            if not text:
                raise ValueError("未识别到有效语音，请检查音量、输入语言或更换音频。")
            LOGGER.info("File Nemotron ASR completed: audio=%.2fs elapsed=%.3fs chars=%d language=%s",
                        len(audio) / 16000, time.monotonic() - started, len(text), source)
            return text

    def run(self, chunks, source, on_text, on_progress, stop):
        import torch
        from itertools import chain
        with self.lock, torch.inference_mode(), inference_stream_context(self.cuda_stream):
            processor = self.processor
            windows = iter(PCMWindows(resample_stream(chunks),
                first_frames=processor.num_mel_frames_first_audio_chunk,
                frames=processor.num_mel_frames_per_audio_chunk,
                hop=processor.feature_extractor.hop_length, n_fft=processor.feature_extractor.n_fft))
            try:
                first = next(windows)
            except StopIteration:
                on_text("", True)
                return
            prompt = None
            exhausted = False
            def features():
                nonlocal prompt, exhausted
                for pcm, is_first, real_frames, observed, end, last in chain([first], windows):
                    if stop.is_set():
                        raise StreamCancelled()
                    value, ids = stream_features(processor, pcm, is_first, last, real_frames, source)
                    if ids is not None:
                        prompt = ids.to(self.model.device)
                    yield value.to(self.model.device, dtype=self.model.dtype)
                    on_progress(observed, end / 16000)
                exhausted = True
            stream = features()
            # Prime the prompt without losing the first feature block.
            first_features = next(stream)
            def primed_features():
                yield first_features
                yield from stream
            receiver = TokenReceiver(processor, on_text, self.max_tokens)
            output = self.model.generate(input_features=primed_features(), prompt_ids=prompt,
                num_lookahead_tokens=self.lookahead, streamer=receiver, max_new_tokens=self.max_tokens,
                do_sample=False, return_dict_in_generate=True)
            if stop.is_set():
                raise StreamCancelled()
            # Never pull another blocking audio chunk after generate() has ended.
            if output.sequences.shape[-1] >= self.max_tokens:
                raise RuntimeError("ASR 达到输出上限，不能将截断结果标记为完成。")
            if not exhausted:
                raise RuntimeError("ASR 在音频耗尽前结束，请检查生成上限或模型版本。")
            on_text(processor.decode(output.sequences[0], skip_special_tokens=True).strip(), True)


class SeamlessTextTranslator:
    name = "SeamlessM4T v2 文本翻译"
    def __init__(self, model_dir=None, precision="float16", max_source_tokens=1024, max_new_tokens=1024,
                 existing_translator=None):
        self.model_dir = Path(model_dir or ROOT / "ckpts/seamless-m4t-v2-large")
        if precision not in {"float16", "float32", "bfloat16"}:
            raise ValueError("T2T 精度无效")
        self.precision, self.max_source_tokens, self.max_new_tokens = precision, max_source_tokens, max_new_tokens
        self.existing = existing_translator
        self.lock = existing_translator.lock if existing_translator else threading.Lock()
        self.cuda_stream = None

    def load(self):
        import torch
        from transformers import AutoProcessor, SeamlessM4Tv2ForTextToText
        require_cuda()
        if self.existing:
            self.existing.load_model()
            self.processor, self.model = self.existing.processor, self.existing.model
        else:
            if self.precision == "bfloat16" and not torch.cuda.is_bf16_supported():
                raise ValueError("当前 GPU 不支持 bfloat16")
            self.processor = AutoProcessor.from_pretrained(self.model_dir, local_files_only=True)
            self.model = SeamlessM4Tv2ForTextToText.from_pretrained(self.model_dir,
                local_files_only=True, dtype=getattr(torch, self.precision)).to("cuda:0").eval()
        LOGGER.info("T2T ready: %s (%s)", self.model_dir, self.precision)
        self.cuda_stream = make_inference_stream(self.model.device, priority=0)

    def translate(self, text, source, target):
        if not text or source == target:
            return text
        import torch
        with self.lock, torch.inference_mode(), inference_stream_context(self.cuda_stream):
            inputs = self.processor(text=text, src_lang=source, return_tensors="pt")
            if inputs.input_ids.shape[-1] > self.max_source_tokens:
                raise ValueError("原文超过 T2T 上下文上限，请缩短单次录音；不会静默截断原文。")
            inputs = {k: v.to(self.model.device) for k, v in inputs.items()}
            kwargs = dict(tgt_lang=target, do_sample=False, num_beams=1, max_new_tokens=self.max_new_tokens)
            if self.existing:
                kwargs = dict(tgt_lang=target, generate_speech=False, text_do_sample=False,
                              text_num_beams=1, text_max_new_tokens=self.max_new_tokens)
            output = self.model.generate(**inputs, **kwargs)
            sequences = getattr(output, "sequences", output)
            if isinstance(sequences, tuple):
                sequences = sequences[0]
            eos = self.model.generation_config.eos_token_id
            eos = eos if isinstance(eos, (list, tuple)) else [eos]
            if int(sequences[0, -1]) not in eos:
                raise RuntimeError("译文未正常结束，可能达到生成上限；请缩短录音或增加 T2T 输出上限。")
            return self.processor.decode(sequences[0].tolist(), skip_special_tokens=True).strip()


class MockASR:
    name = "MOCK · 流式 ASR（不运行模型）"
    def run(self, chunks, source, on_text, on_progress, stop):
        total, text = 0., ""
        for audio in resample_stream(chunks):
            if stop.is_set():
                raise StreamCancelled()
            total += len(audio) / 16000
            text = f"[MOCK] 已连续接收 {total:.2f} 秒音频"
            on_text(text, False)
            on_progress(audio, total)
        on_text(text, True)


class MockTextTranslator:
    name = "MOCK · T2T"
    def __init__(self, delay=.15):
        self.delay = delay
    def translate(self, text, source, target):
        time.sleep(self.delay)
        return f"[MOCK {source}→{target}] {text}"


class MockStreamingTextTranslator(MockTextTranslator):
    """CPU-only transport preview; explicit MOCK text is never a model result."""
    def translate_stream(self, text, source, target, *, on_text, stop):
        result = f"[MOCK {source}→{target}] {text}"
        ends = list(range(8, len(result), 8)) + [len(result)]
        for end in ends:
            if stop.wait(self.delay / len(ends)):
                raise StreamCancelled()
            on_text(result[:end])
        return result
