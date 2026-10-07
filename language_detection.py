"""Offline language identification: MMS on CUDA, or SpeechBrain on CPU."""

from dataclasses import dataclass
import logging
import json
from pathlib import Path
import time

import numpy as np

LOGGER = logging.getLogger(__name__)
DEFAULT_MODEL_DIR = Path(__file__).resolve().parent / "ckpts/lang-id-voxlingua107-ecapa"
DEFAULT_MMS_MODEL_DIR = Path(__file__).resolve().parent / "ckpts/mms-lid-256"
DEFAULT_BACKEND = "mms"
BACKENDS = ("mms", "speechbrain")
SPEECHBRAIN_CODES = {
    "zh": "cmn", "en": "eng", "fr": "fra", "de": "deu", "es": "spa",
    "ja": "jpn", "ko": "kor", "ru": "rus", "pt": "por", "it": "ita",
    "ar": "arb", "hi": "hin", "th": "tha", "vi": "vie", "id": "ind",
    "tr": "tur", "nl": "nld", "pl": "pol", "uk": "ukr",
}
MMS_CODES = {code: code for code in SPEECHBRAIN_CODES.values() if code != "arb"}
MMS_CODES.update({"ara": "arb", "yue": "yue"})


class LanguageDetectionError(ValueError):
    """An actionable detection failure that can be shown in the Demo."""


@dataclass(frozen=True)
class LanguagePrediction:
    code: str
    label: str
    confidence: float
    seconds: float
    elapsed: float


class LanguageDetector:
    """Original SpeechBrain detector, retained for the CPU backend."""

    backend = "speechbrain"
    display_name = "SpeechBrain VoxLingua107"
    language_codes = SPEECHBRAIN_CODES

    def __init__(self, model_dir: Path = DEFAULT_MODEL_DIR, max_seconds: float = 10):
        self.model_dir = Path(model_dir).expanduser().resolve()
        if not np.isfinite(max_seconds) or max_seconds < 1:
            raise ValueError("语言检测片段长度必须至少为 1 秒。")
        self.max_seconds = max_seconds
        self.model = None

    def to_seamless_code(self, code: str) -> str | None:
        return self.language_codes.get(code)

    def load_model(self):
        """Load once from local files; never fetch weights or allocate CUDA."""
        if self.model is not None:
            return
        required = ["hyperparams.yaml", "embedding_model.ckpt", "classifier.ckpt", "label_encoder.txt"]
        missing = [name for name in required if not (self.model_dir / name).is_file()]
        if missing:
            raise LanguageDetectionError(
                f"语言检测模型文件缺失：{', '.join(missing)}。请检查 DEMO_LID_MODEL_DIR。"
            )
        try:
            from hyperpyyaml import load_hyperpyyaml
            from speechbrain.inference.classifiers import EncoderClassifier
            from speechbrain.utils.fetching import LocalStrategy
        except ImportError as exc:
            raise LanguageDetectionError("缺少语言检测依赖，请在运行环境安装 requirements.txt。") from exc

        LOGGER.info("Loading SpeechBrain language ID from %s (CPU)", self.model_dir)
        # The downloaded YAML points at a Hub ID by default. Override it before
        # resolving !ref paths so every pretrainer source is an absolute local dir.
        # Avoid from_hparams(), which also tries to fetch optional custom.py.
        with (self.model_dir / "hyperparams.yaml").open(encoding="utf-8") as handle:
            hparams = load_hyperpyyaml(handle, {"pretrained_path": str(self.model_dir)})
        pretrainer = hparams["pretrainer"]
        pretrainer.collect_files(default_source=str(self.model_dir), local_strategy=LocalStrategy.NO_LINK)
        pretrainer.load_collected()
        hparams["label_encoder"].expect_len(hparams["out_n_neurons"])
        self.model = EncoderClassifier(
            modules=hparams["modules"], hparams=hparams, run_opts={"device": "cpu"},
        ).eval()
        LOGGER.info("SpeechBrain language ID ready on CPU")

    def prepare_clip(self, audio_16khz: np.ndarray) -> np.ndarray:
        """Remove low-energy ends and bound classification, without another model."""
        audio = np.asarray(audio_16khz, dtype=np.float32)
        if audio.ndim != 1 or audio.size == 0 or not np.isfinite(audio).all():
            raise LanguageDetectionError("语言检测需要有效的单声道音频。")
        # 20 ms RMS frames. This trims silence; it is not a speech/music detector.
        frame_size = 320
        padded = np.pad(audio, (0, (-audio.size) % frame_size))
        rms = np.sqrt(np.mean(padded.reshape(-1, frame_size) ** 2, axis=1))
        active = np.flatnonzero(rms >= max(float(rms.max()) * 0.1, 1e-4))
        if active.size == 0:
            raise LanguageDetectionError("没有检测到足够的声音，请重新录音或手动选择输入语言。")
        start = int(active[0]) * frame_size
        end = min((int(active[-1]) + 1) * frame_size, audio.size)
        clip = audio[start:min(end, start + int(self.max_seconds * 16_000))]
        if clip.size < 16_000:
            raise LanguageDetectionError("自动检测需要至少 1 秒有效音频，请录制更长片段或手动选择输入语言。")
        return np.ascontiguousarray(clip)

    def detect(self, audio_16khz: np.ndarray) -> LanguagePrediction:
        import torch

        started = time.monotonic()
        clip = self.prepare_clip(audio_16khz)
        self.load_model()
        with torch.inference_mode():
            _, log_score, _, labels = self.model.classify_batch(torch.from_numpy(clip).unsqueeze(0))
        # SpeechBrain returns log posterior probabilities, not probabilities.
        confidence = float(log_score.reshape(-1)[0].exp().item())
        if not np.isfinite(confidence) or not 0 <= confidence <= 1:
            raise LanguageDetectionError("语言检测返回了无效置信度，请手动选择输入语言。")
        label = str(labels[0])
        code = label.split(":", 1)[0].strip()
        return LanguagePrediction(code, label, confidence, clip.size / 16_000, time.monotonic() - started)


class MMSLanguageDetector(LanguageDetector):
    """MMS-LID-256 on the same first visible CUDA GPU as SeamlessM4T."""

    backend = "mms"
    display_name = "MMS-LID-256"
    language_codes = MMS_CODES

    def __init__(self, model_dir: Path = DEFAULT_MMS_MODEL_DIR, max_seconds: float = 10, precision: str = "float32"):
        super().__init__(model_dir, max_seconds)
        if precision not in {"float16", "bfloat16", "float32"}:
            raise ValueError("MMS 检测精度必须为 float16、bfloat16 或 float32。")
        self.precision = precision
        self.feature_extractor = None

    def load_model(self):
        if self.model is not None:
            return
        required = ["config.json", "preprocessor_config.json"]
        missing = [name for name in required if not (self.model_dir / name).is_file()]
        weight_files = ("model.safetensors", "model.safetensors.index.json", "pytorch_model.bin", "pytorch_model.bin.index.json")
        if not any((self.model_dir / name).is_file() for name in weight_files):
            missing.append("model.safetensors / pytorch_model.bin（或分片索引）")
        if missing:
            raise LanguageDetectionError(
                f"MMS-LID-256 模型文件缺失：{', '.join(missing)}。请检查 DEMO_LID_MODEL_DIR。"
            )
        config = json.loads((self.model_dir / "config.json").read_text(encoding="utf-8"))
        if config.get("model_type") != "wav2vec2" or len(config.get("id2label", {})) != 256:
            raise LanguageDetectionError("请使用 MMS-LID-256 的语言分类权重，不能使用 MMS 的语音转写权重。")

        import torch
        from transformers import AutoFeatureExtractor, Wav2Vec2ForSequenceClassification

        if not torch.cuda.is_available():
            raise LanguageDetectionError("MMS 语言检测需要 CUDA GPU，请通过 Slurm 提交 GPU 作业；CPU 检测可选 --lid-backend speechbrain。")
        if self.precision == "bfloat16" and not torch.cuda.is_bf16_supported():
            raise LanguageDetectionError("此 GPU 不支持 MMS bfloat16，请选择 --lid-precision float32 或 float16。")
        LOGGER.info("Loading MMS language ID from %s (%s, cuda:0)", self.model_dir, self.precision)
        feature_extractor = AutoFeatureExtractor.from_pretrained(self.model_dir, local_files_only=True)
        if feature_extractor.sampling_rate != 16_000:
            raise LanguageDetectionError("MMS 语言检测模型的采样率配置必须为 16 kHz。")
        model = Wav2Vec2ForSequenceClassification.from_pretrained(
            self.model_dir, dtype=getattr(torch, self.precision), local_files_only=True,
            use_safetensors=any((self.model_dir / name).is_file() for name in ("model.safetensors", "model.safetensors.index.json")),
        ).to("cuda:0").eval()
        self.feature_extractor = feature_extractor
        self.model = model
        LOGGER.info("MMS language ID ready on cuda:0")

    def detect(self, audio_16khz: np.ndarray) -> LanguagePrediction:
        import torch

        started = time.monotonic()
        clip = self.prepare_clip(audio_16khz)
        self.load_model()
        inputs = self.feature_extractor(clip, sampling_rate=16_000, return_tensors="pt")
        inputs = {
            key: value.to(device=self.model.device, dtype=self.model.dtype if value.is_floating_point() else value.dtype)
            for key, value in inputs.items()
        }
        with torch.inference_mode():
            # Normalize in FP32 even when model weights use a smaller dtype.
            probabilities = self.model(**inputs).logits.float().softmax(dim=-1)[0]
            score, index = probabilities.max(dim=-1)
            confidence = float(score.item())
        if not np.isfinite(confidence) or not 0 <= confidence <= 1:
            raise LanguageDetectionError("MMS 语言检测返回了无效置信度，请手动选择输入语言或调整检测精度。")
        code = str(self.model.config.id2label[index.item()])
        # item() synchronizes the CUDA result; timing includes actual GPU work.
        return LanguagePrediction(code, code, confidence, clip.size / 16_000, time.monotonic() - started)


def default_model_dir(backend: str) -> Path:
    if backend not in BACKENDS:
        raise ValueError(f"未知语言检测模块：{backend}")
    return DEFAULT_MMS_MODEL_DIR if backend == "mms" else DEFAULT_MODEL_DIR


def create_language_detector(
    backend: str = DEFAULT_BACKEND, model_dir: Path | None = None,
    max_seconds: float = 10, precision: str = "float32",
) -> LanguageDetector:
    default_dir = default_model_dir(backend)
    selected_dir = default_dir if model_dir is None else model_dir
    if backend == "mms":
        return MMSLanguageDetector(selected_dir, max_seconds, precision)
    return LanguageDetector(selected_dir, max_seconds)
