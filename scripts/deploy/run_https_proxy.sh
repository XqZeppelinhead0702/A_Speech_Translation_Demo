#!/usr/bin/env bash
# Run on the machine exposing the public HTTPS entry, with Caddy installed.
set -euo pipefail
if (($# < 1 || $# > 2)); then
    echo "用法：run_https_proxy.sh 实际域名 [后端端口，默认 7860]" >&2
    exit 2
fi
domain="$1"
port="${2:-7860}"
if [[ ! "$domain" =~ ^[A-Za-z0-9][A-Za-z0-9.-]*\.[A-Za-z0-9-]+$ ]]; then
    echo "请提供实际 DNS 域名，不要包含协议、路径或端口。" >&2
    exit 2
fi
if [[ ! "$port" =~ ^[0-9]{1,5}$ ]] || ((10#$port < 1 || 10#$port > 65535)); then
    echo "后端端口必须在 1–65535 之间。" >&2
    exit 2
fi
command -v caddy >/dev/null || { echo "请先安装 Caddy，详见 docs/deploy_local.md。" >&2; exit 1; }
proxy_config="$(mktemp)"
proxy_pid=""
cleanup() {
    trap - EXIT TERM INT
    [[ -z "$proxy_pid" ]] || kill "$proxy_pid" 2>/dev/null || true
    wait 2>/dev/null || true
    rm -f -- "$proxy_config"
}
trap cleanup EXIT
trap 'exit 143' TERM
trap 'exit 130' INT
printf '%s {\n    reverse_proxy 127.0.0.1:%s\n}\n' "$domain" "$port" > "$proxy_config"
caddy validate --config "$proxy_config" --adapter caddyfile
caddy run --config "$proxy_config" --adapter caddyfile &
proxy_pid=$!
wait "$proxy_pid"
