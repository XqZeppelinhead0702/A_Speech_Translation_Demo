"""Lifecycle of a private vLLM child process shared by both Gradio tabs."""
import contextlib
import logging
import math
import os
from pathlib import Path
import signal
import socket
import subprocess
import sys
import time
import urllib.error
import urllib.request
from urllib.parse import urlsplit

ROOT = Path(__file__).resolve().parents[1]
LOGGER = logging.getLogger(__name__)


class ManagedVLLM:
    def __init__(self, adapter, *, demo_port=7860):
        self.adapter = adapter
        self.process = self.log = None
        parsed = urlsplit(adapter.api_base)
        if parsed.scheme != "http" or parsed.hostname != "127.0.0.1":
            raise ValueError("自动启动 vLLM 需要 http://127.0.0.1:端口/v1；已有服务请使用 --no-manage-vllm。")
        self.port = parsed.port or 80
        if self.port == demo_port:
            raise ValueError("vLLM 端口不能与 Demo 端口相同。")
        self.timeout = float(os.getenv("REALTIME_VLLM_STARTUP_TIMEOUT", "1800"))
        if not math.isfinite(self.timeout) or self.timeout <= 0:
            raise ValueError("REALTIME_VLLM_STARTUP_TIMEOUT 必须为正数。")
        run_id = os.getenv("SLURM_JOB_ID") or f"process-{os.getpid()}"
        self.log_path = ROOT / "outs" / f"vllm-engine-{run_id}.log"

    def __enter__(self):
        # Do not adopt or stop an unrelated process already occupying this port.
        with socket.socket() as probe:
            try:
                probe.bind(("127.0.0.1", self.port))
            except OSError as exc:
                raise RuntimeError("vLLM 端口已占用；修改 REALTIME_VLLM_API_BASE，或用 --no-manage-vllm 连接已有服务。") from exc
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        self.log = self.log_path.open("a", encoding="utf-8")
        command = ["bash", "scripts/run/serve_vllm.sh", "--model-dir", str(self.adapter.model_dir.resolve()),
                   "--precision", self.adapter.precision, "--port", str(self.port),
                   "--served-model", self.adapter.served_model]
        if not os.getenv("SLURM_JOB_ID"):
            command.append("--allow-local")
        environment = dict(os.environ)
        environment.setdefault("DEMO_PYTHON", sys.executable)
        try:
            self.process = subprocess.Popen(command, cwd=ROOT, env=environment,
                stdout=self.log, stderr=subprocess.STDOUT, start_new_session=True)
            LOGGER.info("启动共用 vLLM 服务；日志：%s", self.log_path)
            opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
            request = urllib.request.Request(f"http://127.0.0.1:{self.port}/health", headers=self.adapter.headers)
            deadline, next_report = time.monotonic() + self.timeout, 0.
            while True:
                self.check()
                if time.monotonic() >= deadline:
                    raise RuntimeError(f"vLLM 启动超时；查看 {self.log_path}")
                try:
                    with opener.open(request, timeout=1) as response:
                        if response.status == 200:
                            return self
                except (OSError, urllib.error.URLError):
                    pass
                if time.monotonic() >= next_report:
                    LOGGER.info("等待 vLLM 就绪，首次启动可能编译/捕获计算图…")
                    next_report = time.monotonic() + 30
                time.sleep(.5)
        except BaseException:
            self.close()
            raise

    def check(self):
        if self.process is not None and self.process.poll() is not None:
            raise RuntimeError(f"vLLM 已退出（退出码 {self.process.returncode}）；查看 {self.log_path}")

    def close(self):
        if self.process is not None:
            with contextlib.suppress(ProcessLookupError):
                os.killpg(self.process.pid, signal.SIGTERM)
            with contextlib.suppress(subprocess.TimeoutExpired):
                self.process.wait(timeout=8)
            with contextlib.suppress(ProcessLookupError):
                os.killpg(self.process.pid, signal.SIGKILL)
            self.process.wait()
            self.process = None
        if self.log is not None:
            self.log.close()
            self.log = None

    def __exit__(self, *_):
        self.close()
