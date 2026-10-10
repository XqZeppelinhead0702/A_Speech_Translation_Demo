"""Demo mode: reviews the UI without GPU or weights, using dataset reference text only.

Nothing here runs a model. Results come from data/test_samples/references.csv for the
bundled samples, and the UI labels every result as reference text, not model output.
"""

from __future__ import annotations

import csv
from pathlib import Path
import time

import gradio as gr
import numpy as np

PAIR_CODES = {"zh-CN": "cmn", "en": "eng", "fr": "fra"}
# Short pause so the processing state is visible; it does not imitate real latency.
DEMO_DELAY_SECONDS = 1.6
DEMO_STATUS_PREFIX = "完成 · 演示模式参考文本"


def load_references(sample_dir: Path) -> dict[str, dict[str, str]]:
    path = sample_dir / "references.csv"
    if not path.is_file():
        return {}
    with path.open(encoding="utf-8-sig", newline="") as stream:
        rows = {}
        for row in csv.DictReader(stream):
            source, _, target = row["pair"].partition("_")
            if source in PAIR_CODES and target in PAIR_CODES:
                rows[row["file"]] = {
                    "src": PAIR_CODES[source], "tgt": PAIR_CODES[target],
                    "source_text": row["source_text"], "translation": row["reference_translation"],
                }
        return rows


class DemoTranslator:
    """Same translate() contract as SpeechTranslator, backed by reference text."""

    def __init__(self, sample_dir: Path, max_seconds: float, language_names: dict[str, str],
                 language_codes: set[str], auto_language: str, read_audio):
        self.max_seconds = max_seconds
        self.references = load_references(sample_dir)
        self.language_names = language_names
        self.language_codes = language_codes
        self.auto_language = auto_language
        self.read_audio = read_audio
        # Gradio re-encodes uploads to WAV under a generic name, so samples are matched by
        # their decoded content (duration and loudness), not by file name.
        self.fingerprints = {}
        for name in self.references:
            try:
                audio, sample_rate = read_audio(str(sample_dir / name), max_seconds)
            except gr.Error:
                continue
            self.fingerprints[name] = self.fingerprint(audio, sample_rate)

    @staticmethod
    def fingerprint(audio: np.ndarray, sample_rate: int) -> tuple[float, float]:
        return len(audio) / sample_rate, float(np.sqrt(np.mean(np.square(audio))))

    def match_sample(self, audio: np.ndarray, sample_rate: int) -> dict[str, str] | None:
        duration, rms = self.fingerprint(audio, sample_rate)
        for name, (sample_duration, sample_rms) in self.fingerprints.items():
            if abs(duration - sample_duration) < 0.06 and abs(rms - sample_rms) <= 0.05 * sample_rms:
                return self.references[name]
        return None

    def translate(self, audio_path: str | None, src_lang: str, tgt_lang: str):
        if src_lang not in self.language_codes | {self.auto_language} or tgt_lang not in self.language_codes:
            raise gr.Error("请选择有效的输入语言和输出语言。")
        started = time.monotonic()
        audio, sample_rate = self.read_audio(audio_path, self.max_seconds)
        reference = self.match_sample(audio, sample_rate)
        if reference is None:
            raise gr.Error("演示模式不运行模型，只为内置示例提供数据集参考文本。请点击下方示例，或在 GPU 环境中运行完整模型。")
        if tgt_lang != reference["tgt"]:
            raise gr.Error(
                f"演示模式只有该示例的{self.language_names[reference['tgt']]}参考译文，"
                f"请将输出语言设为{self.language_names[reference['tgt']]}。"
            )
        time.sleep(DEMO_DELAY_SECONDS)
        src = reference["src"]
        if src_lang == self.auto_language:
            language_result = f"演示模式：示例标注语言 · {self.language_names[src]}"
        else:
            language_result = f"手动指定：{self.language_names[src_lang]}"
        status = f"{DEMO_STATUS_PREFIX} · 音频 {len(audio) / sample_rate:.1f} 秒 · 用时 {time.monotonic() - started:.1f} 秒"
        return reference["source_text"], reference["translation"], language_result, status
