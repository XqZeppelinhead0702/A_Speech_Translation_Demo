"""Share a compute-node Gradio service through a networked login node, without GPU use."""

import argparse
import os
from pathlib import Path
import secrets
import select
import shlex
import signal
import socket
import subprocess
import sys
import threading
import time
from urllib.parse import urlsplit, urlunsplit

PROJECT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_DIR))


def stop(signum, frame):
    raise SystemExit(0)


def connect(args):
    remote_program = args.project_dir / "scripts" / "login_relay.py"
    remote_environment = [f"GRADIO_TEMP_DIR={args.project_dir / '.cache/gradio' / ('relay-' + args.job_id)}"]
    if args.proxy_url:
        remote_environment.extend(f"{name}={args.proxy_url}" for name in ("HTTP_PROXY", "HTTPS_PROXY", "http_proxy", "https_proxy"))
    remote_environment.extend(["NO_PROXY=localhost,127.0.0.1,::1", "no_proxy=localhost,127.0.0.1,::1"])
    remote_command = shlex.join([
        "env", *remote_environment,
        str(args.python), "-u", str(remote_program), "check" if args.check_only else "serve",
        "--relay-port", str(args.relay_port),
        *(["--test-tunnel"] if args.check_only and args.test_tunnel else []),
    ])
    command = [
        "ssh", "-T", "-o", "BatchMode=yes", "-o", "StrictHostKeyChecking=yes",
        "-o", "ConnectTimeout=10", "-o", "ExitOnForwardFailure=yes",
        "-o", "ServerAliveInterval=15", "-o", "ServerAliveCountMax=3",
        "-R", f"127.0.0.1:{args.relay_port}:127.0.0.1:{args.app_port}",
    ]
    if args.identity_file:
        command.extend(["-i", str(args.identity_file), "-o", "IdentitiesOnly=yes"])
    command.extend([args.target, remote_command])
    print(f"[登录节点中转] 连接 {args.target}，回环转发端口 {args.relay_port}…", flush=True)
    # Keeping stdin open ties the remote relay's lifetime to this Slurm process.
    process = subprocess.Popen(command, stdin=subprocess.PIPE)
    try:
        result = process.wait()
        if result:
            print(
                "[登录节点中转失败] 请检查内网 SSH 可达性、免交互密钥、known_hosts、"
                "端口转发权限，以及登录节点能否访问相同的项目/Conda 路径。",
                file=sys.stderr, flush=True,
            )
        return result
    finally:
        if process.stdin:
            process.stdin.close()
        if process.poll() is None:
            process.terminate()
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()


def watch_connection(stopping=None):
    # SSH closes stdin when its client exits, including scancel on the compute node.
    stopping = stopping if stopping is not None else threading.Event()
    descriptor = sys.stdin.fileno()
    while not stopping.is_set():
        readable, _, _ = select.select([descriptor], [], [], 0.2)
        if readable and not os.read(descriptor, 1):
            if not stopping.is_set():
                os.kill(os.getpid(), signal.SIGTERM)
            return


def serve(args):
    stopping = threading.Event()
    watcher = threading.Thread(target=watch_connection, args=(stopping,), daemon=False)
    watcher.start()
    tunnel_module = None
    try:
        os.chdir(PROJECT_DIR)
        import app  # Only UI/utilities; no model is instantiated or loaded.
        import httpx
        from gradio import networking, tunneling
        from scripts.check_public_access import check_public_access
        tunnel_module = tunneling
        print(f"[登录节点中转] 实际运行节点：{socket.gethostname()}", flush=True)
        check_public_access(prepare=True)
        backend = f"http://127.0.0.1:{args.relay_port}/"
        print("[登录节点中转] 网络检查通过，等待计算节点 Demo 就绪…", flush=True)
        deadline = time.monotonic() + args.backend_timeout
        while True:
            try:
                response = httpx.get(backend, timeout=3, trust_env=False)
                if response.status_code in {200, 301, 302, 303, 307, 308, 401}:
                    break
            except httpx.HTTPError:
                pass
            if time.monotonic() >= deadline:
                raise RuntimeError("等待计算节点服务超时，请查看模型加载日志及内网转发端口。")
            time.sleep(1)
        with app.startup_deadline(args.startup_timeout):
            raw_url = networking.setup_tunnel(
                local_host="127.0.0.1", local_port=args.relay_port,
                share_token=secrets.token_hex(16), share_server_address=None,
                share_server_tls_certificate=None,
            )
        parsed = urlsplit(raw_url)
        public_url = urlunsplit(("https", parsed.netloc, parsed.path, parsed.query, parsed.fragment))
        print(f"PUBLIC_URL={public_url}\n在其他设备的浏览器中打开上述 HTTPS 地址。", flush=True)
        while True:
            if any(t.proc is None or t.proc.poll() is not None for t in tunneling.CURRENT_TUNNELS):
                raise RuntimeError("登录节点公网分享客户端已退出，结束作业。")
            time.sleep(1)
    finally:
        stopping.set()
        watcher.join()
        if tunnel_module is not None:
            for tunnel in tunnel_module.CURRENT_TUNNELS:
                tunnel.kill()


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="mode", required=True)
    client = commands.add_parser("connect")
    client.add_argument("--target", required=True)
    client.add_argument("--project-dir", type=Path, default=PROJECT_DIR)
    client.add_argument("--python", type=Path, default=Path(sys.executable))
    client.add_argument("--app-port", type=int, default=7860)
    client.add_argument("--relay-port", type=int, default=17860)
    client.add_argument("--job-id", default=os.getenv("SLURM_JOB_ID", str(os.getpid())))
    client.add_argument("--identity-file", type=Path, default=Path(os.environ["DEMO_RELAY_SSH_KEY"]).expanduser() if os.getenv("DEMO_RELAY_SSH_KEY") else None)
    client.add_argument("--proxy-url", default=os.getenv("DEMO_RELAY_PROXY_URL", ""))
    client.add_argument("--check-only", action="store_true", help="Only check networking in the real SSH session, without a server or GPU")
    client.add_argument("--test-tunnel", action="store_true", help="With --check-only, create and immediately close a diagnostic share tunnel")
    server = commands.add_parser("serve")
    server.add_argument("--relay-port", type=int, required=True)
    server.add_argument("--backend-timeout", type=int, default=300)
    server.add_argument("--startup-timeout", type=int, default=90)
    checker = commands.add_parser("check")
    checker.add_argument("--relay-port", type=int, default=17860)
    checker.add_argument("--test-tunnel", action="store_true")
    args = parser.parse_args(argv)
    if not 1 <= args.relay_port <= 65535 or not 1 <= getattr(args, "app_port", 7860) <= 65535:
        parser.error("端口必须在 1–65535 之间。")
    return args


def main():
    args = parse_args()
    signal.signal(signal.SIGTERM, stop)
    signal.signal(signal.SIGINT, stop)
    try:
        if args.mode == "connect":
            return connect(args)
        if args.mode == "check":
            from scripts.check_public_access import check_public_access
            print(f"[登录节点网络检查] 实际运行节点：{socket.gethostname()}", flush=True)
            check_public_access(prepare=True)
            if args.test_tunnel:
                import app
                from gradio import networking, tunneling
                try:
                    with app.startup_deadline(60):
                        networking.setup_tunnel(
                            local_host="127.0.0.1", local_port=args.relay_port,
                            share_token=secrets.token_hex(16), share_server_address=None,
                            share_server_tls_certificate=None,
                        )
                    print("[登录节点网络检查] 公网隧道握手通过；诊断隧道立即关闭。", flush=True)
                finally:
                    for tunnel in tunneling.CURRENT_TUNNELS:
                        tunnel.kill()
            return 0
        serve(args)
        return 0
    except Exception as exc:
        print(f"[登录节点中转失败] {exc}", file=sys.stderr, flush=True)
        return 1


if __name__ == "__main__":
    sys.exit(main())
