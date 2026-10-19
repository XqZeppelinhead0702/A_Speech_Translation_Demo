"""Share a text translator between file and realtime requests without contention."""
from contextlib import contextmanager
import threading


class InferenceCoordinator:
    def __init__(self, service=None):
        self.service = service
        self.lock = threading.RLock()
        self.file_busy = False

    @contextmanager
    def file_inference(self):
        with self.lock:
            busy = self.file_busy
            if self.service is not None:
                with self.service.condition:
                    busy |= self.service.asr_busy or any(
                        s.status in {"listening", "draining"} or s.inflight is not None
                        for s in self.service.sessions.values())
            if busy:
                raise ValueError("模型正在处理其他任务，请先结束实时会话或等待当前翻译完成。")
            self.file_busy = True
        try:
            yield
        finally:
            with self.lock:
                self.file_busy = False

    def start_session(self, **kwargs):
        with self.lock:
            if self.file_busy:
                raise ValueError("正在处理文件/录音翻译，请完成后再开始实时会话。")
            return self.service.start_session(**kwargs)


def stream_text_translation(translator, text, source, target):
    """Bridge callback generation to Gradio, keeping only the latest pending draft.

    The adapter still validates EOS before returning its final result. Closing the
    generator cancels the request and waits for its worker to release the model.
    """
    if not callable(getattr(translator, "translate_stream", None)):
        yield translator.translate(text, source, target)
        return
    yield from _stream_operation(lambda on_text, stop: translator.translate_stream(
        text, source, target, on_text=on_text, stop=stop), "file-t2t")


def stream_file_transcription(asr, audio, source, *, timeout):
    """Yield cumulative ASR text, or None as a heartbeat while the encoder runs."""
    yield from _stream_operation(lambda on_text, stop: asr.transcribe(
        audio, source, on_text=on_text, stop=stop, timeout=timeout), "file-asr", heartbeat=1)


def _stream_operation(operation, name, heartbeat=None):
    condition = threading.Condition()
    stop = threading.Event()
    state = {"draft": None, "done": False, "result": None, "error": None}

    def on_text(draft):
        with condition:
            state["draft"] = draft
            condition.notify_all()

    def run():
        try:
            state["result"] = operation(on_text, stop)
        except Exception as exc:
            state["error"] = exc
        finally:
            with condition:
                state["done"] = True
                condition.notify_all()

    worker = threading.Thread(target=run, name=name, daemon=True)
    worker.start()
    try:
        while True:
            with condition:
                condition.wait_for(lambda: state["draft"] is not None or state["done"], timeout=heartbeat)
                draft, state["draft"] = state["draft"], None
                done = state["done"]
            if draft is not None:
                yield draft
            elif not done and heartbeat is not None:
                yield None
            if done:
                if state["error"] is not None:
                    raise state["error"]
                yield state["result"]
                return
    finally:
        stop.set()
        # No subsequent request may overlap a still-cancelling native translator.
        worker.join()
