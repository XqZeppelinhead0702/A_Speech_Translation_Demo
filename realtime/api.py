"""Shared realtime HTTP/WebSocket API; importing this module never loads models."""
import base64
from contextlib import asynccontextmanager
import inspect
import json
import logging
import math
from pathlib import Path
import secrets
import threading
import time
import uuid

import numpy as np
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import FileResponse, JSONResponse

from .backend import ASR_LANGUAGES
from .socket import register_socket
from sample_catalog import load_demo_samples

ROOT = Path(__file__).resolve().parents[1]


async def require_gradio_login(connection):
    """Use the same Gradio login cookie as the page, including WebSocket handshakes."""
    application = connection.app
    if application.auth is None and application.auth_dependency is None:
        return
    if application.auth_dependency is not None:
        user = application.auth_dependency(connection)
        if inspect.isawaitable(user):
            user = await user
    else:
        token = (connection.cookies.get(f"access-token-{application.cookie_id}")
                 or connection.cookies.get(f"access-token-unsecure-{application.cookie_id}"))
        user = application.tokens.get(token)
    if not user:
        raise HTTPException(401, "请先登录 Demo，再开始实时会话。")


def replay_samples():
    """Display only the shared showcase catalog, never prepend legacy samples."""
    import soundfile as sf
    directory = ROOT / "data/test_samples"
    samples = []
    for item in load_demo_samples(directory):
        if item["source_lang"] not in ASR_LANGUAGES:
            raise ValueError("展示样本输入语言不在当前流式 ASR 支持范围内。")
        samples.append(dict(path=directory / item["file"], source_lang=item["source_lang"],
                            target_lang=item["target_lang"], label=item["label"],
                            version=item.get("sha256", item["file"])))
    for index, sample in enumerate(samples):
        sample["id"] = index
        sample["duration_seconds"] = None
        if sample["path"].is_file():
            sample["duration_seconds"] = round(sf.info(sample["path"]).duration, 2)
        else:
            logging.warning("Realtime sample is missing: %s", sample["path"])
    return samples


def create_api(service, language_choices, *, prefix="/realtime-api", start_session=None,
               authorize=None, owner_cookie="st_realtime_owner"):
    """Build HTTP/WS routes reusable by Gradio.launch and the private test page."""
    API = prefix.rstrip("/")
    samples = replay_samples()
    replay_threads = []
    tickets = {}
    shutting_down = threading.Event()

    @asynccontextmanager
    async def lifespan(application):
        try:
            yield
        finally:
            shutting_down.set()
            for thread in replay_threads:
                thread.join(timeout=2)
            service.close()

    api = FastAPI(lifespan=lifespan, openapi_url=None, docs_url=None, redoc_url=None)

    async def verify(request):
        if authorize is not None:
            result = authorize(request)
            if inspect.isawaitable(result):
                await result

    async def owner(request):
        await verify(request)
        value = request.cookies.get(owner_cookie)
        if not value or len(value) > 128:
            raise HTTPException(401, "请刷新页面初始化浏览器会话。")
        return value

    async def body(request):
        data = bytearray()
        async for piece in request.stream():
            data.extend(piece)
            if len(data) > 2 * 1024 * 1024:
                raise HTTPException(413, "音频上传块过大。")
        try:
            payload = json.loads(data)
            if not isinstance(payload, dict):
                raise ValueError("需要 JSON 对象")
            return payload
        except (ValueError, UnicodeError) as exc:
            raise HTTPException(400, "无效的请求内容。") from exc

    def call(function, *args, **kwargs):
        try:
            return function(*args, **kwargs)
        except (ValueError, TypeError) as exc:
            raise HTTPException(409, str(exc)) from exc

    @api.get(API + "/meta")
    async def meta(request: Request):
        await verify(request)
        supported = getattr(service.translator, "supported_languages", None)
        languages = [item for item in language_choices if supported is None or item[1] in supported]
        response = JSONResponse({"backend": service.backend.name, "max_seconds": service.max_session_seconds,
                                 "languages": languages, "pipeline": service.pipeline,
                                 "translation_backend": service.translator.name if service.translator else None,
                                 "source_languages": [item for item in languages if item[1] in ASR_LANGUAGES],
                                 "samples": [{key: value for key, value in sample.items() if key != "path"} |
                                             {"available": sample["duration_seconds"] is not None and
                                              sample["duration_seconds"] <= service.max_session_seconds and
                                              (supported is None or (sample["source_lang"] in supported and
                                                                     sample["target_lang"] in supported))}
                                             for sample in samples]})
        if not request.cookies.get(owner_cookie):
            response.set_cookie(owner_cookie, uuid.uuid4().hex, httponly=True, samesite="strict", max_age=86400)
        return response

    def start(payload, owner_id):
        result = call(start_session or service.start_session, source_lang=payload.get("source_lang", "cmn"),
                      target_lang=payload.get("target_lang", "eng"), owner_id=owner_id)
        now = time.monotonic()
        for sid, (_, _, expires) in list(tickets.items()):
            if expires < now:
                del tickets[sid]
        ticket = secrets.token_urlsafe(32)
        tickets[result["session_id"]] = (ticket, owner_id, now + 30)
        return {**result, "stream_ticket": ticket}

    register_socket(api, service, tickets, owner, API)

    @api.post(API + "/diagnostics")
    async def diagnostics(request: Request):
        payload = await body(request)
        call(service.poll_results, payload.get("session_id"), owner_id=await owner(request))
        report = {}
        for key in ["transport", "error", "fallback_reason", "first_asr_display_seconds", "first_translation_display_seconds",
                    "first_complete_translation_display_seconds",
                    "asr_delivery_estimate_ms",
                    "upload_rtt_ms", "result_rtt_ms", "clock_rtt_ms", "delivery_estimate_ms", "max_pending_seconds", "playback_pauses"]:
            value = payload.get(key)
            if isinstance(value, str):
                report[key] = value[:300].replace("\n", " ")
            elif isinstance(value, (float, int)) and math.isfinite(value):
                report[key] = value
        logging.info("Realtime browser diagnostics %s: %s", payload["session_id"], json.dumps(report, ensure_ascii=False))
        return {"ok": True}

    @api.post(API + "/start")
    async def start_route(request: Request):
        return start(await body(request), await owner(request))

    @api.get(API + "/samples/{index}/audio")
    async def sample_audio(index: int, request: Request):
        await owner(request)
        if not 0 <= index < len(samples) or samples[index]["duration_seconds"] is None:
            raise HTTPException(404, "示例文件不存在。")
        # Serve only catalogued sample files; no caller-controlled filesystem paths.
        return FileResponse(samples[index]["path"], media_type="audio/mpeg",
                            headers={"Cache-Control": "private, max-age=3600"})

    @api.post(API + "/audio")
    async def audio_route(request: Request):
        payload = await body(request)
        try:
            data = base64.b64decode(payload["pcm16"], validate=True)
            if len(data) % 2:
                raise ValueError("PCM 字节数不是偶数")
            audio = np.frombuffer(data, dtype="<i2")
            return call(service.push_audio, payload["session_id"], owner_id=await owner(request),
                        chunk_seq=payload["chunk_seq"], sample_rate=payload["sample_rate"], audio=audio)
        except (KeyError, ValueError, TypeError) as exc:
            raise HTTPException(400, "无效的 PCM16 音频块。") from exc

    @api.get(API + "/results")
    async def results_route(request: Request, session_id: str, cursor: int = 0):
        return call(service.poll_results, session_id, cursor=cursor, owner_id=await owner(request))

    @api.post(API + "/finish")
    async def finish_route(request: Request):
        payload = await body(request)
        return call(service.finish_session, payload.get("session_id"),
                    last_chunk_seq=payload.get("last_chunk_seq"), owner_id=await owner(request))

    @api.post(API + "/cancel")
    async def cancel_route(request: Request):
        payload = await body(request)
        return call(service.cancel_session, payload.get("session_id"), owner_id=await owner(request))

    def replay_sample(session_id, owner_id, index):
        import soundfile as sf
        sequence = 0
        try:
            with sf.SoundFile(samples[index]["path"]) as reader:
                size = round(reader.samplerate * 0.16)
                sent = 0
                started = time.monotonic()
                while not shutting_down.is_set():
                    audio = reader.read(size, dtype="float32", always_2d=True)
                    if not len(audio):
                        break
                    sent += len(audio)
                    while time.monotonic() < started + sent / reader.samplerate:
                        if shutting_down.wait(0.05):
                            return
                        if service.poll_results(session_id, owner_id=owner_id)["status"] not in {"listening", "draining"}:
                            return
                    sequence += 1
                    service.push_audio(session_id, chunk_seq=sequence, sample_rate=reader.samplerate,
                                       audio=audio, owner_id=owner_id)
            if not shutting_down.is_set():
                service.finish_session(session_id, last_chunk_seq=sequence, owner_id=owner_id)
        except Exception:
            logging.exception("Realtime sample replay failed")
            try:
                # Preserve the backend's explicit error (buffer, detection, etc.).
                if service.poll_results(session_id, owner_id=owner_id)["status"] in {"listening", "draining"}:
                    service.cancel_session(session_id, owner_id=owner_id)
            except ValueError:
                pass

    @api.post(API + "/replay")
    async def replay_route(request: Request):
        payload, owner_id = await body(request), await owner(request)
        index = payload.get("sample", 0)
        if type(index) is not int or not 0 <= index < len(samples):
            raise HTTPException(400, "无效的示例。")
        sample = samples[index]
        if sample["duration_seconds"] is None:
            raise HTTPException(400, "示例文件不存在，请检查 data/test_samples。")
        if sample["duration_seconds"] > service.max_session_seconds:
            raise HTTPException(400, f"示例超过当前 {service.max_session_seconds:g} 秒输入上限，请选择短样本。")
        session = start({**payload, "source_lang": sample["source_lang"],
                         "target_lang": payload.get("target_lang", sample["target_lang"])}, owner_id)
        replay_threads[:] = [thread for thread in replay_threads if thread.is_alive()]
        thread = threading.Thread(target=replay_sample, args=(session["session_id"], owner_id, index), daemon=True)
        replay_threads.append(thread)
        thread.start()
        return session

    return api
