"""Check Gradio share networking / prepare its client, without starting a server or GPU."""

import argparse
import hashlib
import os
from pathlib import Path
import socket
import ssl
import base64
import sys
import time
from urllib.request import proxy_bypass
from urllib.parse import urlsplit

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import app  # Sets a writable Gradio directory before importing Gradio.
import httpx
from gradio import networking, tunneling


def check_share_connection(host, port, timeout):
    """Probe the same HTTP CONNECT proxy route FRPC uses, if configured."""
    proxy = os.environ.get("HTTP_PROXY") or os.environ.get("http_proxy")
    if not proxy or proxy_bypass(host):
        with socket.create_connection((host, port), timeout=timeout):
            return
    parsed = urlsplit(proxy)
    if parsed.scheme not in {"http", "https"} or not parsed.hostname:
        raise OSError("分享客户端的 HTTP_PROXY 需要 http/https 代理 URL")
    connection = socket.create_connection((parsed.hostname, parsed.port or (443 if parsed.scheme == "https" else 80)), timeout=timeout)
    try:
        if parsed.scheme == "https":
            connection = ssl.create_default_context().wrap_socket(connection, server_hostname=parsed.hostname)
        authority = f"{host}:{port}"
        headers = [f"CONNECT {authority} HTTP/1.1", f"Host: {authority}"]
        if parsed.username is not None:
            from urllib.parse import unquote
            credentials = f"{unquote(parsed.username)}:{unquote(parsed.password or '')}"
            headers.append("Proxy-Authorization: Basic " + base64.b64encode(credentials.encode()).decode())
        connection.sendall(("\r\n".join(headers) + "\r\n\r\n").encode())
        with connection.makefile("rb") as response:
            status = response.readline(4096).split()
            if len(status) < 2 or status[1] != b"200":
                raise OSError("代理拒绝分享服务器的 CONNECT 请求")
    finally:
        connection.close()


def check_public_access(prepare=False, timeout=10, attempts=3):
    proxy = os.environ.get("https_proxy") or os.environ.get("HTTPS_PROXY") or os.environ.get("all_proxy") or os.environ.get("ALL_PROXY")
    proxy_hint = ""
    if proxy:
        parsed = urlsplit(proxy)
        if parsed.hostname in {"127.0.0.1", "localhost", "::1"}:
            proxy_hint = (
                f" 当前 HTTPS 代理指向本机 {parsed.hostname}:{parsed.port}。"
                "请确认代理监听在当前节点；登录节点和计算节点的回环地址彼此独立。"
            )
            print(f"[代理提示]{proxy_hint}", flush=True)
    print("[公网检查] 获取 Gradio 分享服务地址…", flush=True)
    custom_server = networking.GRADIO_SHARE_SERVER_ADDRESS
    if custom_server:
        host, port = custom_server.rsplit(":", 1)
        port = int(port)
    else:
        last_error = None
        for attempt in range(attempts):
            try:
                response = httpx.get(networking.GRADIO_API_SERVER, timeout=timeout)
                response.raise_for_status()
                payload = response.json()[0]
                host, port = payload["host"], int(payload["port"])
                last_error = None
                break
            except Exception as exc:
                last_error = exc
                print(f"[公网检查] API 请求第 {attempt + 1}/{attempts} 次失败（{type(exc).__name__}）。", flush=True)
                if attempt + 1 < attempts:
                    time.sleep(1)
        if last_error is not None:
            raise RuntimeError(
                f"节点 {socket.gethostname()} 无法连接 {networking.GRADIO_API_SERVER}（{type(last_error).__name__}）。"
                "请检查当前节点的出站 HTTPS、DNS 或代理。"
                + proxy_hint
            ) from last_error

    print(f"[公网检查] 测试分享服务器 TCP：{host}:{port}…", flush=True)
    try:
        check_share_connection(host, port, timeout)
    except OSError as exc:
        raise RuntimeError(
            f"无法连接分享服务器 {host}:{port}（{type(exc).__name__}）。"
            "仅能访问 HTTPS 不代表能建立 Gradio 隧道，请确认当前节点或所用代理允许该出站 TCP 端口。"
        ) from exc

    binary = Path(tunneling.BINARY_PATH)
    if not binary.is_file():
        if not prepare:
            raise RuntimeError(f"缺少分享客户端：{binary}。请运行本脚本 --prepare 下载。")
        print(f"[公网检查] 下载官方分享客户端至 {binary}…", flush=True)
        try:
            tunneling.Tunnel.download_binary()
        except Exception as exc:
            raise RuntimeError(
                f"无法准备分享客户端（{type(exc).__name__}）。"
                f"请检查 {tunneling.BINARY_URL} 是否可访问，以及 {binary.parent} 是否可写。"
            ) from exc
    expected = tunneling.CHECKSUMS.get(tunneling.BINARY_URL)
    if expected and hashlib.sha256(binary.read_bytes()).hexdigest() != expected:
        raise RuntimeError(f"分享客户端校验失败：{binary}。请移走此文件后重新执行 --prepare。")
    if not binary.stat().st_mode & 0o111:
        raise RuntimeError(f"分享客户端没有执行权限：{binary}。请为当前用户添加执行权限。")
    print("[公网检查] 通过。此检查不加载模型、不启动服务器；公网 URL 将在 Demo 建立隧道后生成。", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prepare", action="store_true", help="Download and checksum-verify the official FRPC client if missing")
    args = parser.parse_args()
    try:
        check_public_access(prepare=args.prepare)
    except Exception as exc:
        print(f"[公网检查失败] {exc}", file=sys.stderr, flush=True)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
