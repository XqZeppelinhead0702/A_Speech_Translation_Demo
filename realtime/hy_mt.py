"""Local Hy-MT2 dense (1.8B/7B) text translation; no model load on import."""
import logging
from pathlib import Path
import threading
import time

from .backend import ROOT, require_cuda
from .stream_audio import StreamCancelled
from .cuda_execution import make_inference_stream, inference_stream_context

LOGGER = logging.getLogger(__name__)
# App/Seamless ISO-3 codes -> full names required by Hy-MT prompts.
LANGUAGES = {
    "cmn": ("Chinese", "中文"), "cmn_Hant": ("Traditional Chinese", "繁体中文"),
    "eng": ("English", "英语"), "fra": ("French", "法语"),
    "por": ("Portuguese", "葡萄牙语"), "spa": ("Spanish", "西班牙语"),
    "jpn": ("Japanese", "日语"), "tur": ("Turkish", "土耳其语"),
    "rus": ("Russian", "俄语"), "arb": ("Arabic", "阿拉伯语"),
    "kor": ("Korean", "韩语"), "tha": ("Thai", "泰语"),
    "ita": ("Italian", "意大利语"), "deu": ("German", "德语"),
    "vie": ("Vietnamese", "越南语"), "zsm": ("Malay", "马来语"),
    "ind": ("Indonesian", "印尼语"), "tgl": ("Filipino", "菲律宾语"),
    "hin": ("Hindi", "印地语"), "pol": ("Polish", "波兰语"),
    "ces": ("Czech", "捷克语"), "nld": ("Dutch", "荷兰语"),
    "khm": ("Khmer", "高棉语"), "mya": ("Burmese", "缅甸语"),
    "pes": ("Persian", "波斯语"), "guj": ("Gujarati", "古吉拉特语"),
    "urd": ("Urdu", "乌尔都语"), "tel": ("Telugu", "泰卢固语"),
    "mar": ("Marathi", "马拉地语"), "heb": ("Hebrew", "希伯来语"),
    "ben": ("Bengali", "孟加拉语"), "tam": ("Tamil", "泰米尔语"),
    "ukr": ("Ukrainian", "乌克兰语"), "bod": ("Tibetan", "藏语"),
    "kaz": ("Kazakh", "哈萨克语"), "mon": ("Mongolian", "蒙古语"),
    "uig": ("Uyghur", "维吾尔语"), "yue": ("Cantonese", "粤语"),
}


def translation_prompt(text, source, target):
    if source not in LANGUAGES or target not in LANGUAGES:
        raise ValueError(f"Hy-MT2 不支持此语言组合：{source} → {target}")
    if source in {"cmn", "cmn_Hant", "yue"} or target in {"cmn", "cmn_Hant", "yue"}:
        return f"将以下文本翻译为{LANGUAGES[target][1]}，注意只需要输出翻译后的结果，不要额外解释：\n{text}"
    return (f"Translate the following text into {LANGUAGES[target][0]}. "
            f"Note that you should only output the translated result without any additional explanation:\n{text}")


class FirstTokenTimer:
    """Same-thread HF streamer: skip prompt; time first non-special output token.

    This is generator TTFT, not browser subtitle latency or a streaming MT API.
    """
    def __init__(self, special_ids):
        self.special_ids = set(special_ids)
        self.prompt = True
        self.first_at = None

    def put(self, value):
        if self.prompt:
            self.prompt = False
            return
        if self.first_at is None and any(int(i) not in self.special_ids for i in value.reshape(-1).tolist()):
            self.first_at = time.perf_counter()

    def end(self):
        pass


class TranslationReceiver(FirstTokenTimer):
    """Publish decoded hypotheses, without waiting for whitespace or generator EOF."""
    def __init__(self, tokenizer, on_text, stop=None, interval=.08):
        super().__init__(tokenizer.all_special_ids)
        self.tokenizer, self.on_text, self.stop = tokenizer, on_text, stop
        self.interval, self.last_emit = interval, 0.
        self.ids, self.text = [], ""

    def put(self, value):
        if self.stop is not None and self.stop.is_set():
            raise StreamCancelled()
        prompt = self.prompt
        super().put(value)
        if prompt:
            return
        self.ids.extend(value.reshape(-1).tolist())
        self.publish()

    def publish(self, force=False):
        if self.stop is not None and self.stop.is_set():
            raise StreamCancelled()
        now = time.perf_counter()
        if not force and self.text and now-self.last_emit < self.interval:
            return
        text = self.tokenizer.decode(self.ids, skip_special_tokens=True).strip()
        # A multibyte character may span tokens. Do not display a replacement glyph.
        if text and "\ufffd" not in text and text != self.text:
            self.text, self.last_emit = text, now
            self.on_text(text)

    def end(self):
        self.publish(force=True)  # A draft only: the caller still validates EOS.


class HyMTTextTranslator:
    supported_languages = frozenset(LANGUAGES)

    def __init__(self, model_dir=None, precision="bfloat16", max_source_tokens=4096,
                 max_new_tokens=1024, sampling=False, *, require_slurm=True):
        self.model_dir = Path(model_dir or ROOT / "ckpts/Hy-MT2-1.8B")
        self.name = f"{self.model_dir.name} 文本翻译"
        if precision not in {"bfloat16", "float16", "float32"}:
            raise ValueError("Hy-MT2 精度必须为 bfloat16、float16 或 float32")
        if any(type(v) is not int or v <= 0 for v in [max_source_tokens, max_new_tokens]):
            raise ValueError("Hy-MT2 输入、输出 token 上限必须为正整数")
        self.precision = precision
        self.max_source_tokens, self.max_new_tokens = max_source_tokens, max_new_tokens
        self.sampling = sampling
        self.require_slurm = require_slurm
        self.lock = threading.Lock()
        self.cuda_stream = None
        self.last_metrics = {}

    def load(self):
        import os
        import torch
        from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer
        if self.require_slurm and not os.getenv("SLURM_JOB_ID"):
            raise RuntimeError("Hy-MT2 必须通过 Slurm 提交 GPU 作业。")
        require_cuda()
        if self.precision == "bfloat16" and not torch.cuda.is_bf16_supported():
            raise ValueError("当前 GPU 不支持 bfloat16；可显式选择 float16。")
        config = AutoConfig.from_pretrained(self.model_dir, local_files_only=True, trust_remote_code=False)
        if config.model_type != "hunyuan_v1_dense" or getattr(config, "quantization_config", None):
            raise ValueError("此后端支持原始 Hy-MT2-1.8B/7B dense 权重；FP8、GGUF 和 MoE 未适配。")
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_dir, local_files_only=True, trust_remote_code=False)
        if not self.tokenizer.chat_template:
            raise ValueError("模型目录缺少 chat_template，请补齐官方 tokenizer 文件。")
        self.model = AutoModelForCausalLM.from_pretrained(self.model_dir, config=config,
            local_files_only=True, trust_remote_code=False, dtype=getattr(torch, self.precision),
            attn_implementation="sdpa").to("cuda:0").eval()
        self.cuda_stream = make_inference_stream(self.model.device, priority=0)
        LOGGER.info("Hy-MT2 ready: %s (%s); greedy=%s", self.model_dir, self.precision, not self.sampling)

    def translate(self, text, source, target, *, measure_first_token=False):
        return self._translate(text, source, target, measure_first_token=measure_first_token)

    def translate_stream(self, text, source, target, *, on_text, stop=None):
        """Emit provisional strings, then return the EOS-validated full translation."""
        return self._translate(text, source, target, on_text=on_text, stop=stop)

    def _translate(self, text, source, target, *, measure_first_token=False, on_text=None, stop=None):
        prompt = translation_prompt(text, source, target)
        if stop is not None and stop.is_set():
            raise StreamCancelled()
        if not text or source == target:
            self.last_metrics = {"bypass": True}
            return text
        import torch
        from copy import deepcopy
        with self.lock, torch.inference_mode(), inference_stream_context(self.cuda_stream):
            started = time.perf_counter()
            formatted = self.tokenizer.apply_chat_template(
                [{"role": "user", "content": prompt}], tokenize=False, add_generation_prompt=True)
            inputs = self.tokenizer(formatted, add_special_tokens=False, return_tensors="pt")
            input_tokens = inputs["input_ids"].shape[-1]
            if input_tokens > self.max_source_tokens:
                raise ValueError(f"Hy-MT2 输入含模板共 {input_tokens} tokens，超过 {self.max_source_tokens}；不会静默截断原文。")
            context = self.model.config.max_position_embeddings
            if input_tokens + self.max_new_tokens > context:
                raise ValueError("Hy-MT2 输入加预留输出超过模型上下文容量。")
            inputs = {k: v.to(self.model.device) for k, v in inputs.items()}
            # Preserve this checkpoint's EOS list (7B has two), BOS and PAD IDs.
            generation = deepcopy(self.model.generation_config)
            generation.do_sample, generation.num_beams = self.sampling, 1
            generation.max_new_tokens = self.max_new_tokens
            generation.repetition_penalty = 1.05
            generation.temperature, generation.top_p, generation.top_k = (
                (.7, .6, 20) if self.sampling else (1., 1., 50))
            receiver = (TranslationReceiver(self.tokenizer, on_text, stop) if on_text else
                        FirstTokenTimer(self.tokenizer.all_special_ids) if measure_first_token else None)
            generated_at = time.perf_counter()
            output = self.model.generate(**inputs, generation_config=generation, streamer=receiver)
            if stop is not None and stop.is_set():
                raise StreamCancelled()
            sequences = getattr(output, "sequences", output)
            ids = sequences[0, input_tokens:].tolist()
            eos = generation.eos_token_id
            eos = eos if isinstance(eos, (list, tuple)) else [eos]
            if not ids or ids[-1] not in eos:
                raise RuntimeError("Hy-MT2 译文未正常结束，可能达到输出上限；请增大 --mt-max-new-tokens 或缩短输入。")
            result = self.tokenizer.decode(ids, skip_special_tokens=True).strip()
            if not result:
                raise RuntimeError("Hy-MT2 返回空译文，请检查输入、语言和模型文件。")
            completed = time.perf_counter()
            self.last_metrics = dict(input_tokens=input_tokens, output_tokens=len(ids),
                elapsed_seconds=completed-started, generation_seconds=completed-generated_at,
                first_token_seconds=(receiver.first_at-generated_at if receiver and receiver.first_at else None))
            return result
