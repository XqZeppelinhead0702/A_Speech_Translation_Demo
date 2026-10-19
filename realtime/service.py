"""Native streaming ASR with independent, revision-linked text translation."""
from collections import deque
from dataclasses import dataclass, field
import logging
import math
import os
from types import SimpleNamespace
import threading
import time
import uuid

from language_detection import MMS_CODES
from .audio import normalize_audio
from .backend import ASR_LANGUAGES
from .stream_audio import SilenceMeter, StreamCancelled
from .subtitles import SubtitleProjection

LOGGER = logging.getLogger(__name__)
ACTIVE = {"listening", "draining"}
TARGETS = set(MMS_CODES.values()) | {"cmn_Hant"}


@dataclass(frozen=True)
class TranslationJob:
    revision: int
    text: str
    requested: float
    checkpoint: bool = False
    reason: str = "update"


@dataclass
class Session:
    id: str
    owner: str
    source: str
    target: str
    show_transcript: bool
    created: float = field(default_factory=time.monotonic)
    touched: float = field(default_factory=time.monotonic)
    status: str = "listening"
    error: str = ""
    rate: int = 0
    received: int = 0
    acked_seq: int = 0
    buffered_seconds: float = 0.
    processed_seconds: float = 0.
    audio: deque = field(default_factory=deque)
    stop: threading.Event = field(default_factory=threading.Event)
    asr_done: bool = False
    asr_final: bool = False
    asr_version: int = 0
    transcript: str = ""
    translation: str = ""
    translation_version: int = 0
    translation_source: str = ""
    translation_final: bool = False
    translation_draft: str = ""
    translation_draft_version: int = 0
    translation_draft_source: str = ""
    translation_draft_seq: int = 0
    translation_draft_server_time_ms: float | None = None
    first_translation_draft_seconds: float | None = None
    subtitles: SubtitleProjection = field(default_factory=SubtitleProjection)
    pending: TranslationJob | None = None
    checkpoints: deque = field(default_factory=deque)
    checkpoint_versions: set = field(default_factory=set)
    inflight: TranslationJob | None = None
    last_translation_started: float = 0.
    events: list = field(default_factory=list)
    first_asr_seconds: float | None = None
    asr_server_time_ms: float | None = None
    first_translation_seconds: float | None = None
    first_audio_seconds: float | None = None
    coalesced_updates: int = 0
    asr_thread: threading.Thread | None = None


class RealtimeService:
    def __init__(self, asr, translator=None, *, translation_interval=.6, pause_seconds=1.2,
                 threshold=.008, max_buffer_seconds=20, max_session_seconds=120,
                 stream_translation=True):
        if type(stream_translation) is not bool:
            raise ValueError("stream_translation 必须是布尔值。")
        for value in [translation_interval, pause_seconds, threshold, max_buffer_seconds, max_session_seconds]:
            if not isinstance(value, (float, int)) or not math.isfinite(value) or value <= 0:
                raise ValueError("实时服务的间隔、阈值和容量必须为正数。")
        self.asr, self.translator = asr, translator
        self.stream_translation = stream_translation and callable(getattr(translator, "translate_stream", None))
        self.translation_interval, self.pause_seconds, self.threshold = translation_interval, pause_seconds, threshold
        self.max_buffer_seconds, self.max_session_seconds = max_buffer_seconds, max_session_seconds
        self.pipeline = "cascade" if translator else "asr"
        self.backend = SimpleNamespace(name=asr.name + (" + " + translator.name if translator else ""))
        self.sessions = {}
        self.condition = threading.Condition(threading.RLock())
        self.closed = self.asr_busy = False
        self.worker = threading.Thread(target=self._translate_worker, name="stream-t2t", daemon=True)
        self.worker.start()

    def _get(self, sid, owner):
        session = self.sessions.get(sid)
        if session is None or session.owner != owner:
            raise ValueError("会话不存在、已过期或不属于当前用户。")
        return session

    def _discard(self, session):
        session.stop.set()
        session.audio.clear()
        session.buffered_seconds = 0.
        session.pending = None
        session.checkpoints.clear()
        self._clear_draft(session)
        session.subtitles.rollback()
        self.condition.notify_all()

    def _clear_draft(self, s):
        if s.translation_draft_version:
            s.translation_draft_seq += 1
        s.translation_draft, s.translation_draft_source = "", ""
        s.translation_draft_version = 0

    def _on_translation_draft(self, s, job, text):
        with self.condition:
            if s.status not in ACTIVE or s.stop.is_set() or s.inflight is not job:
                return
            if not text or text == s.translation_draft:
                return
            if len(text) > 24000:
                raise RuntimeError("生成中的译文超过容量上限，请缩短会话。")
            s.translation_draft, s.translation_draft_source = text, job.text
            s.translation_draft_version = job.revision
            s.translation_draft_seq += 1
            s.translation_draft_server_time_ms = round(time.time()*1000, 3)
            s.subtitles.partial(text, job.revision, job.text)
            if s.first_translation_draft_seconds is None:
                s.first_translation_draft_seconds = time.monotonic()-s.created
                LOGGER.info("Realtime session %s first MT draft at %.3fs; source v%s", s.id,
                            s.first_translation_draft_seconds, job.revision)
            self.condition.notify_all()

    def _fail(self, session, error):
        session.status, session.error, session.touched = "error", str(error), time.monotonic()
        LOGGER.warning("Realtime session %s failed: %s", session.id, error)
        self._discard(session)

    def _expire(self):
        now = time.monotonic()
        for s in list(self.sessions.values()):
            if s.status == "listening" and now - s.touched > 30:
                s.status, s.error = "cancelled", "长时间没有音频或操作，会话已回收。"
                self._discard(s)
            elif s.status not in ACTIVE and not (s.asr_thread and s.asr_thread.is_alive()) and now - s.touched > 600:
                del self.sessions[s.id]

    def start_session(self, *, source_lang, target_lang, owner_id, show_transcript=True):
        if source_lang not in ASR_LANGUAGES or target_lang not in TARGETS:
            raise ValueError("流式 ASR 暂时需要手动指定支持的输入语言；不支持 auto。")
        supported = getattr(self.translator, "supported_languages", None)
        if supported is not None and (source_lang not in supported or target_lang not in supported):
            raise ValueError(f"当前翻译后端不支持 {source_lang} → {target_lang}。")
        if not isinstance(owner_id, str) or not 1 <= len(owner_id) <= 128:
            raise ValueError("用户会话标识无效。")
        with self.condition:
            self._expire()
            if self.closed:
                raise ValueError("服务已经关闭。")
            if self.asr_busy or any(s.status in ACTIVE for s in self.sessions.values()):
                raise ValueError("实时会话正在使用或结束中，请稍后重试。")
            if len(self.sessions) >= 8:
                terminal = [s for s in self.sessions.values() if s.status not in ACTIVE]
                if terminal:
                    del self.sessions[min(terminal, key=lambda s: s.touched).id]
            s = Session(uuid.uuid4().hex, owner_id, source_lang, target_lang, show_transcript)
            self.sessions[s.id] = s
            LOGGER.info("Realtime session %s started: pipeline=%s source=%s target=%s", s.id, self.pipeline, source_lang, target_lang)
            self.asr_busy = True
            s.asr_thread = threading.Thread(target=self._asr_worker, args=(s,), name="stream-asr", daemon=True)
            s.asr_thread.start()
            return {"session_id": s.id, "status": s.status, "pipeline": self.pipeline}

    def push_audio(self, session_id, *, chunk_seq, sample_rate, audio, owner_id):
        audio = normalize_audio(audio, sample_rate)
        if not len(audio):
            raise ValueError("不能上传空音频块。")
        with self.condition:
            s = self._get(session_id, owner_id)
            if s.status != "listening":
                raise ValueError(s.error or "会话已经停止接收音频。")
            if type(chunk_seq) is not int or chunk_seq < 1:
                raise ValueError("音频块序号需要从 1 开始。")
            if chunk_seq <= s.acked_seq:
                return {"acked_seq": s.acked_seq}
            if chunk_seq != s.acked_seq + 1:
                raise ValueError("音频块序号不连续，请重试缺失的块。")
            if s.rate and s.rate != sample_rate:
                raise ValueError("采样率变化，请重新开始会话。")
            duration = len(audio) / sample_rate
            if s.buffered_seconds + duration > self.max_buffer_seconds or (s.received + len(audio)) / sample_rate > self.max_session_seconds:
                self._fail(s, "达到音频缓冲或会话时长上限；请停止并缩短测试。")
                raise ValueError(s.error)
            s.rate = sample_rate
            if s.first_audio_seconds is None:
                s.first_audio_seconds = time.monotonic() - s.created
                LOGGER.info("Realtime session %s first audio received at %.3fs", s.id, s.first_audio_seconds)
            s.received += len(audio)
            s.audio.append(audio)
            s.buffered_seconds += duration
            s.acked_seq, s.touched = chunk_seq, time.monotonic()
            self.condition.notify_all()
            return {"acked_seq": s.acked_seq}

    def _audio_chunks(self, s):
        while True:
            with self.condition:
                while not s.audio and s.status == "listening" and not s.stop.is_set():
                    self.condition.wait(.5)
                if s.stop.is_set():
                    raise StreamCancelled()
                if not s.audio:
                    return
                audio = s.audio.popleft()
                s.buffered_seconds = max(0., s.buffered_seconds - len(audio) / s.rate)
            yield audio, s.rate

    def _event(self, s, kind, **fields):
        if len(s.events) >= 4096:
            raise RuntimeError("达到更新记录上限，请缩短测试会话。")
        event = dict(session_id=s.id, event_seq=len(s.events) + 1, type=kind,
                     source_lang=s.source, target_lang=s.target,
                     observed_seconds=round(time.monotonic() - s.created, 4),
                     server_time_ms=round(time.time() * 1000, 3), **fields)
        s.events.append(event)
        return event

    def _checkpoint(self, s, reason):
        if not self.translator or not s.asr_version or s.asr_version in s.checkpoint_versions:
            return
        if len(s.checkpoints) >= 64:
            raise RuntimeError("确认译文的等待队列已满，请停止输入并等待处理。")
        s.checkpoint_versions.add(s.asr_version)
        s.checkpoints.append(TranslationJob(s.asr_version, s.transcript, time.monotonic(), True, reason))
        if s.pending and s.pending.revision <= s.asr_version:
            s.pending = None
        self.condition.notify_all()

    def _on_text(self, s, text, final=False):
        with self.condition:
            if s.status not in ACTIVE:
                return
            if len(text) > 12000:
                raise RuntimeError("原文超过会话文本上限，请缩短录音。")
            changed = text != s.transcript
            if changed:
                s.asr_version += 1
                s.transcript = text
                if s.pending:
                    s.coalesced_updates += 1
                s.pending = TranslationJob(s.asr_version, text, time.monotonic()) if self.translator else None
                if s.first_asr_seconds is None and text:
                    s.first_asr_seconds = time.monotonic() - s.created
                    LOGGER.info("Realtime session %s first ASR at %.3fs (%.3fs after first audio)",
                                s.id, s.first_asr_seconds, s.first_asr_seconds - (s.first_audio_seconds or 0.))
            if changed or final:
                event = self._event(s, "asr", asr_version=s.asr_version,
                                    transcript=text if s.show_transcript else "", is_final=final)
                s.asr_server_time_ms = event["server_time_ms"]
            s.asr_final = final
            if text.rstrip().endswith(("。", "！", "？", ".", "!", "?")):
                self._checkpoint(s, "punctuation")
            if final:
                self._checkpoint(s, "stop")
            self.condition.notify_all()

    def _asr_worker(self, s):
        meter = SilenceMeter(threshold=self.threshold)
        def progress(audio, seconds):
            silent = meter.feed(audio)
            with self.condition:
                if s.status in ACTIVE:
                    s.processed_seconds = seconds
                    if silent >= self.pause_seconds:
                        self._checkpoint(s, "pause")
        try:
            self.asr.run(self._audio_chunks(s), s.source,
                         lambda text, final=False: self._on_text(s, text, final), progress, s.stop)
            with self.condition:
                if s.status in ACTIVE and s.status != "draining":
                    raise RuntimeError("ASR 在输入停止前结束，请检查生成上限或模型版本。")
                if s.status in ACTIVE:
                    if not s.asr_final:
                        self._on_text(s, s.transcript, True)
                    s.asr_done = True
                    self._finish_if_ready(s)
        except StreamCancelled:
            pass
        except Exception as exc:
            LOGGER.exception("Streaming ASR failed")
            with self.condition:
                if s.status in ACTIVE:
                    self._fail(s, exc)
        finally:
            with self.condition:
                self.asr_busy = False
                self.condition.notify_all()

    def _finish_if_ready(self, s):
        if s.status != "draining" or not s.asr_done or s.inflight or s.pending or s.checkpoints:
            return
        if self.translator and s.transcript:
            if s.translation_version != s.asr_version:
                self._checkpoint(s, "stop")
                return
            s.translation_final = True
        s.status, s.touched = "finished", time.monotonic()
        self.condition.notify_all()

    def _translate_worker(self):
        while True:
            with self.condition:
                self._expire()
                if self.closed:
                    return
                selected = None
                now = time.monotonic()
                for s in self.sessions.values():
                    if s.status not in ACTIVE:
                        continue
                    if s.checkpoints:
                        job = s.checkpoints.popleft()
                    elif s.pending and (s.status == "draining" or now - s.last_translation_started >= self.translation_interval):
                        job, s.pending = s.pending, None
                    else:
                        continue
                    s.inflight, s.last_translation_started = job, now
                    selected = s, job
                    break
                if selected is None:
                    self.condition.wait(.05)
                    continue
            s, job = selected
            started = time.monotonic()
            try:
                # Reuse a completed partial for an identical mandatory checkpoint.
                with self.condition:
                    cached = s.translation if s.translation_version == job.revision else None
                if cached is not None:
                    translated = cached
                elif self.stream_translation:
                    translated = self.translator.translate_stream(job.text, s.source, s.target,
                        on_text=lambda text, session=s, selected_job=job:
                            self._on_translation_draft(session, selected_job, text), stop=s.stop)
                else:
                    translated = self.translator.translate(job.text, s.source, s.target)
                completed = time.monotonic()
                with self.condition:
                    if s.status in ACTIVE:
                        # Publish even if ASR moved on; bind the output to its source.
                        # Do not discard completed MT merely because newer ASR exists.
                        if job.revision >= s.translation_version:
                            s.translation, s.translation_source = translated, job.text
                            s.translation_version = job.revision
                            s.translation_final = job.checkpoint and s.asr_final and job.revision == s.asr_version
                            s.subtitles.complete(translated, job.revision, job.text)
                        if s.first_translation_seconds is None:
                            s.first_translation_seconds = completed - s.created
                            LOGGER.info("Realtime session %s first translation at %.3fs; queue=%.3fs inference=%.3fs",
                                        s.id, s.first_translation_seconds, started - job.requested, completed - started)
                        self._event(s, "translation", asr_version=s.asr_version,
                            translation_version=job.revision, translation_source=job.text, translation=translated,
                            is_final=job.checkpoint, reason=job.reason,
                            queue_seconds=round(started - job.requested, 4),
                            inference_seconds=round(completed - started, 4))
                    s.inflight = None
                    self._clear_draft(s)
                    self._finish_if_ready(s)
            except StreamCancelled:
                with self.condition:
                    s.inflight = None
                    self._clear_draft(s)
                    s.subtitles.rollback()
                    if s.status in ACTIVE:
                        self._fail(s, "翻译提前中止，请重新开始会话。")
            except Exception as exc:
                LOGGER.exception("Streaming T2T failed")
                with self.condition:
                    s.inflight = None
                    if s.status in ACTIVE:
                        self._fail(s, exc)

    def poll_results(self, session_id, *, cursor=0, owner_id):
        with self.condition:
            s = self._get(session_id, owner_id)
            if type(cursor) is not int or not 0 <= cursor <= len(s.events):
                raise ValueError("结果游标无效。")
            snapshot = dict(transcript=s.transcript if s.show_transcript else "", asr_version=s.asr_version,
                asr_final=s.asr_final, translation=s.translation, translation_version=s.translation_version,
                translation_source=s.translation_source, translation_final=s.translation_final,
                lag_versions=max(0, s.asr_version - s.translation_version),
                translation_pending=bool(s.pending or s.checkpoints or s.inflight),
                translation_draft=s.translation_draft, translation_draft_version=s.translation_draft_version,
                translation_draft_source=s.translation_draft_source, translation_draft_seq=s.translation_draft_seq,
                translation_draft_server_time_ms=s.translation_draft_server_time_ms,
                first_translation_draft_seconds=s.first_translation_draft_seconds,
                first_asr_seconds=s.first_asr_seconds, first_translation_seconds=s.first_translation_seconds,
                asr_server_time_ms=s.asr_server_time_ms,
                first_audio_seconds=s.first_audio_seconds,
                coalesced_updates=s.coalesced_updates, **s.subtitles.snapshot())
            return dict(events=[dict(e) for e in s.events[cursor:]], cursor=len(s.events), snapshot=snapshot,
                status=s.status, error=s.error, pipeline=self.pipeline, backend=self.backend.name,
                language_info=f"手动指定：{s.source}", acked_seq=s.acked_seq,
                received_seconds=s.received / s.rate if s.rate else 0.,
                processed_seconds=s.processed_seconds, queued_seconds=s.buffered_seconds,
                pending_checkpoints=len(s.checkpoints), server_time_ms=round(time.time() * 1000, 3))

    def finish_session(self, session_id, *, last_chunk_seq, owner_id):
        with self.condition:
            s = self._get(session_id, owner_id)
            if type(last_chunk_seq) is not int or last_chunk_seq != s.acked_seq:
                raise ValueError("最后音频块尚未确认，不能结束会话。")
            if s.status == "listening":
                s.status, s.touched = "draining", time.monotonic()
                self.condition.notify_all()
            return {"status": s.status}

    def cancel_session(self, session_id, *, owner_id):
        with self.condition:
            s = self._get(session_id, owner_id)
            if s.status in ACTIVE:
                LOGGER.info("Realtime session %s cancelled: received=%.3fs processed=%.3fs", s.id,
                            s.received / s.rate if s.rate else 0., s.processed_seconds)
                s.status, s.touched = "cancelled", time.monotonic()
                self._discard(s)
            return {"status": s.status}

    def close(self):
        with self.condition:
            self.closed = True
            for s in self.sessions.values():
                if s.status in ACTIVE:
                    s.status = "cancelled"
                    self._discard(s)
            self.condition.notify_all()
            threads = [self.worker] + [s.asr_thread for s in self.sessions.values() if s.asr_thread]
        deadline = time.monotonic() + 10
        for thread in threads:
            thread.join(timeout=max(0., deadline - time.monotonic()))


def create_service(*, mock=None, pipeline=None, translator=None, mt_options=None,
                   stream_translation=None, require_slurm=True, text_translator=None, **options):
    from .backend import MockASR, MockTextTranslator, MockStreamingTextTranslator, NemotronASR
    from .translation import create_text_translator
    mock = os.getenv("REALTIME_MOCK", "0") == "1" if mock is None else mock
    pipeline = pipeline or os.getenv("REALTIME_PIPELINE", "cascade")
    if stream_translation is None:
        configured = os.getenv("REALTIME_MT_STREAMING", "1")
        if configured not in {"0", "1"}:
            raise ValueError("REALTIME_MT_STREAMING 必须为 0 或 1")
        stream_translation = configured == "1"
    if pipeline not in {"asr", "cascade"}:
        raise ValueError("REALTIME_PIPELINE 必须为 asr 或 cascade")
    # Construct/validate the adapter before any GPU load. ASR-only never loads MT.
    adapter = (text_translator if text_translator is not None else
               create_text_translator(**(mt_options or {}), existing_translator=translator,
                                      require_slurm=require_slurm)) if pipeline == "cascade" else None
    options.setdefault("translation_interval", float(os.getenv("REALTIME_TRANSLATION_INTERVAL", ".6")))
    options.setdefault("max_session_seconds", float(os.getenv("REALTIME_MAX_SECONDS", "120")))
    if mock:
        asr = MockASR()
        mt = (MockStreamingTextTranslator() if callable(getattr(adapter, "translate_stream", None))
              else MockTextTranslator()) if pipeline == "cascade" else None
        if mt is not None:
            mt.name += " · " + adapter.name
            mt.supported_languages = getattr(adapter, "supported_languages", None)
    else:
        if require_slurm and not os.getenv("SLURM_JOB_ID"):
            raise RuntimeError("真实流式模型必须通过 Slurm 提交；CPU 检查使用 mock=True。")
        asr = NemotronASR(os.getenv("REALTIME_ASR_MODEL_DIR"),
            os.getenv("REALTIME_ASR_PRECISION", "float32"), int(os.getenv("REALTIME_ASR_LOOKAHEAD", "3")))
        asr.load()
        mt = adapter
        if mt:
            mt.load()
    return RealtimeService(asr, mt, stream_translation=stream_translation, **options)
