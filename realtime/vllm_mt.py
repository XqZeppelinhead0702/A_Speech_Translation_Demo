"""Hy-MT2 via an independently deployed vLLM server; never imports vLLM/CUDA.

Use the local checkpoint's exact prompt token IDs with /completions, avoiding
server chat-template differences. /tokenize verifies the tokenizer at startup.
"""
import asyncio
import json
import logging
import math
import os
from pathlib import Path
import threading
import time
from urllib.parse import urlsplit

from .backend import ROOT
from .hy_mt import LANGUAGES, translation_prompt
from .stream_audio import StreamCancelled

LOGGER = logging.getLogger(__name__)


class VLLMTextTranslator:
    supported_languages = frozenset(LANGUAGES)
    local_gpu = False

    def __init__(self, model_dir=None, precision="bfloat16", max_source_tokens=4096,
                 max_new_tokens=1024, sampling=False, *, api_base="http://127.0.0.1:8000/v1",
                 served_model="hy-mt2"):
        self.model_dir = Path(model_dir or ROOT / "ckpts/Hy-MT2-1.8B")
        self.precision = precision  # server-side dtype; managed launchers pass it to the engine
        if precision not in {"bfloat16", "float16", "float32"}:
            raise ValueError("Hy-MT2 精度必须为 bfloat16、float16 或 float32")
        if any(type(v) is not int or v <= 0 for v in [max_source_tokens, max_new_tokens]):
            raise ValueError("Hy-MT2 输入、输出 token 上限必须为正整数")
        parsed = urlsplit(api_base)
        if (parsed.scheme not in {"http", "https"} or not parsed.hostname or parsed.username
                or parsed.password or parsed.query or parsed.fragment or parsed.path.rstrip("/") != "/v1"):
            raise ValueError("vLLM API 必须为 http(s)://主机:端口/v1；密钥使用 REALTIME_VLLM_API_KEY")
        if not served_model.strip():
            raise ValueError("vLLM served model 不可为空")
        self.api_base, self.served_model = api_base.rstrip("/"), served_model
        self.api_key = os.getenv("REALTIME_VLLM_API_KEY", "")
        self.timeout = float(os.getenv("REALTIME_VLLM_REQUEST_TIMEOUT", "180"))
        if not math.isfinite(self.timeout) or self.timeout <= 0:
            raise ValueError("REALTIME_VLLM_REQUEST_TIMEOUT 必须为正数")
        self.max_source_tokens, self.max_new_tokens = max_source_tokens, max_new_tokens
        self.sampling, self.lock = sampling, threading.Lock()
        self.last_metrics = {}
        self.name = f"{self.model_dir.name} 文本翻译 · vLLM"

    @property
    def headers(self):
        return {"Authorization": f"Bearer {self.api_key}"} if self.api_key else {}

    def load(self):
        # Tokenizer and config only; the translation weights live in the server.
        import httpx
        from transformers import AutoTokenizer
        config = json.loads((self.model_dir / "config.json").read_text())
        if config.get("model_type") != "hunyuan_v1_dense" or config.get("quantization_config"):
            raise ValueError("vLLM 适配器支持原始 Hy-MT2-1.8B/7B dense 权重")
        generation = json.loads((self.model_dir / "generation_config.json").read_text())
        eos = generation.get("eos_token_id", config.get("eos_token_id"))
        self.eos = eos if isinstance(eos, list) else [eos]
        if not self.eos or any(type(v) is not int or v < 0 for v in self.eos):
            raise ValueError("Hy-MT2 缺少有效 EOS 配置")
        self.context = config["max_position_embeddings"]
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_dir, local_files_only=True, trust_remote_code=False)
        if not self.tokenizer.chat_template:
            raise ValueError("Hy-MT2 缺少聊天模板")
        # Explicitly ignore inherited proxy settings: same-node calls stay local.
        # Remote deployment may use a directly reachable URL or an SSH tunnel.
        with httpx.Client(timeout=15, headers=self.headers, trust_env=False) as client:
            result = client.get(self.api_base + "/models")
            self._check_http(result)
            models = result.json().get("data", [])
            model = next((m for m in models if m.get("id") == self.served_model), None)
            if model is None:
                raise RuntimeError("vLLM 未提供指定 served model；检查 --mt-served-model / --served-model-name")
            if type(model.get("max_model_len")) is int:
                self.context = min(self.context, model["max_model_len"])
            formatted, ids = self._prompt("测试。", "cmn", "eng")
            result = client.post(self.api_base.removesuffix("/v1") + "/tokenize",
                json=dict(model=self.served_model, prompt=formatted, add_special_tokens=False))
            self._check_http(result)
            tokenized = result.json()
            if tokenized.get("tokens") != ids:
                raise RuntimeError("vLLM 与客户端 tokenizer 不一致；两端必须使用同一规格、同一版本的 Hy-MT2 文件")
            if type(tokenized.get("max_model_len")) is int:
                self.context = min(self.context, tokenized["max_model_len"])
        LOGGER.info("Hy-MT2 vLLM ready: model=%s; source/output caps=%s/%s; translation weights remain in server",
                    self.served_model, self.max_source_tokens, self.max_new_tokens)

    @staticmethod
    def _check_http(response):
        if response.is_error:
            # Do not include headers/keys or a full source document in exceptions.
            raise RuntimeError(f"vLLM HTTP {response.status_code}；检查模型服务日志、鉴权和上下文上限")

    def _prompt(self, text, source, target):
        formatted = self.tokenizer.apply_chat_template(
            [{"role": "user", "content": translation_prompt(text, source, target)}],
            tokenize=False, add_generation_prompt=True)
        ids = self.tokenizer.encode(formatted, add_special_tokens=False)
        if len(ids) > self.max_source_tokens or len(ids) + self.max_new_tokens > self.context:
            raise ValueError(f"Hy-MT2 输入含模板共 {len(ids)} tokens，超过输入或服务端上下文上限；不会静默截断")
        return formatted, ids

    def translate(self, text, source, target, *, measure_first_token=False):
        return self.translate_stream(text, source, target, on_text=None)

    def translate_stream(self, text, source, target, *, on_text, stop=None):
        translation_prompt(text, source, target)  # validate even on bypass
        if stop is not None and stop.is_set():
            raise StreamCancelled()
        if not text or source == target:
            self.last_metrics = {"bypass": True}
            return text
        with self.lock:
            started = time.perf_counter()
            _, ids = self._prompt(text, source, target)
            self.last_metrics = {}
            return asyncio.run(self._cancellable_request(ids, on_text, stop, started))

    async def _cancellable_request(self, ids, on_text, stop, started):
        async def watch():
            while stop is None or not stop.is_set():
                await asyncio.sleep(.05)
            raise StreamCancelled()
        request = asyncio.create_task(self._stream(ids, on_text, started))
        cancelled = asyncio.create_task(watch())
        try:
            done, _ = await asyncio.wait({request, cancelled}, timeout=self.timeout,
                                         return_when=asyncio.FIRST_COMPLETED)
            if cancelled in done:
                await cancelled
            if request not in done:
                raise RuntimeError("vLLM 请求超时；检查服务端是否积压、编译或推理缓慢")
            if stop is not None and stop.is_set():
                raise StreamCancelled()
            return await request
        finally:
            # Cancelling closes the HTTP stream so the server can abort generation.
            for task in (request, cancelled):
                if not task.done():
                    task.cancel()
            await asyncio.gather(request, cancelled, return_exceptions=True)

    async def _stream(self, ids, on_text, started):
        import httpx
        body = dict(model=self.served_model, prompt=ids, stream=True,
                    stream_options={"include_usage": True}, max_tokens=self.max_new_tokens,
                    temperature=.7 if self.sampling else 0., top_p=.6 if self.sampling else 1.,
                    top_k=20 if self.sampling else -1, repetition_penalty=1.05,
                    stop_token_ids=self.eos, ignore_eos=False)
        text, reason, usage, first_at, emitted_at, emitted, done = "", None, {}, None, 0., "", False
        try:
            async with httpx.AsyncClient(headers=self.headers, timeout=httpx.Timeout(self.timeout, connect=10),
                                          trust_env=False) as client:
                async with client.stream("POST", self.api_base + "/completions", json=body) as response:
                    self._check_http(response)
                    data = []
                    async for line in response.aiter_lines():
                        if line.startswith("data:"):
                            data.append(line[5:].lstrip())
                            if sum(map(len, data)) > 1024*1024:
                                raise RuntimeError("vLLM SSE 事件异常过大")
                            continue
                        if line or not data:  # comments / other SSE fields
                            continue
                        payload, data = "\n".join(data), []
                        if payload == "[DONE]":
                            done = True
                            break
                        event = json.loads(payload)
                        if event.get("error"):
                            raise RuntimeError("vLLM 流式生成出错；请检查模型服务日志")
                        if event.get("usage"):
                            usage = event["usage"]
                        for choice in event.get("choices", []):
                            if choice.get("index", 0) != 0:
                                continue
                            delta = choice.get("text", "")
                            if delta:
                                if reason is not None:
                                    raise RuntimeError("vLLM 在结束事件后继续输出")
                                text += delta
                                first_at = first_at or time.perf_counter()
                                now = time.perf_counter()
                                candidate = text.strip()
                                if (on_text and candidate and "\ufffd" not in candidate
                                        and candidate != emitted and now-emitted_at >= .08):
                                    on_text(candidate)
                                    emitted, emitted_at = candidate, now
                            if choice.get("finish_reason") is not None:
                                reason = choice["finish_reason"]
        except httpx.HTTPError as exc:
            raise RuntimeError(f"vLLM 连接/流式请求失败（{type(exc).__name__}）；检查本机服务日志") from exc
        if not done or reason != "stop":
            raise RuntimeError(f"vLLM 译文未正常结束（finish_reason={reason!r}）；输出上限或连接中断，未确认为完整译文")
        result = text.strip()
        if not result or "\ufffd" in result:
            raise RuntimeError("vLLM 返回空译文或不完整字符")
        if on_text and result != emitted:
            on_text(result)
        self.last_metrics = dict(input_tokens=len(ids), output_tokens=usage.get("completion_tokens"),
            cached_input_tokens=(usage.get("prompt_tokens_details") or {}).get("cached_tokens"),
            elapsed_seconds=time.perf_counter()-started,
            first_token_seconds=first_at-started if first_at else None,
            finish_reason=reason, timing_scope="client request to first nonempty text / complete response")
        return result
