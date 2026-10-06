#!/usr/bin/env bash
# Internal dispatcher. Prefer the named entry scripts documented in README.
set -eo pipefail
source "$(dirname -- "${BASH_SOURCE[0]}")/common.sh"
mode="${1:-}"
if (($#)); then shift; fi
case "$mode" in
    share|local|login-relay|public-tunnel) ;;
    *) echo "用法：serve.sh share|local|login-relay|public-tunnel [app.py 参数]" >&2; exit 2 ;;
esac
command -v ffmpeg >/dev/null || { echo "缺少 FFmpeg，请安装或设置 DEMO_FFMPEG_DIR。" >&2; exit 1; }
export DEMO_PORT="${DEMO_PORT:-7860}"
app_args=("$@")
# Keep the application's port and the SSH forwarding destination identical.
for ((i=0; i<${#app_args[@]}; i++)); do
    case "${app_args[i]}" in
        --port)
            ((i + 1 < ${#app_args[@]})) || { echo "--port 缺少端口号。" >&2; exit 2; }
            export DEMO_PORT="${app_args[i+1]}" ;;
        --port=*) export DEMO_PORT="${app_args[i]#--port=}" ;;
    esac
done
demo_port_valid "$DEMO_PORT" || { echo "应用端口必须在 1–65535 之间。" >&2; exit 2; }
if [[ "$mode" == share || "$mode" == local ]]; then
    unset DEMO_RELAY_TARGET DEMO_TUNNEL_TARGET
    if [[ "$mode" == share ]]; then
        timeout 90s "$DEMO_PYTHON" -u scripts/check_public_access.py --prepare
        exec "$DEMO_PYTHON" -u app.py "${app_args[@]}" --host 127.0.0.1 --port "$DEMO_PORT" --share
    fi
    exec "$DEMO_PYTHON" -u app.py "${app_args[@]}" --host "${DEMO_HOST:-127.0.0.1}" --port "$DEMO_PORT" --no-share
fi

command -v ssh >/dev/null || { echo "缺少 SSH 客户端。" >&2; exit 1; }
if [[ "$mode" == login-relay ]]; then
    : "${DEMO_RELAY_TARGET:?请在 .env.local 中设置 DEMO_RELAY_TARGET 为登录节点 SSH 目标}"
    unset DEMO_TUNNEL_TARGET
    demo_port_valid "${DEMO_RELAY_PORT:-17860}" || { echo "登录节点中转端口无效。" >&2; exit 2; }
    relay_args=(connect --target "$DEMO_RELAY_TARGET"
        --relay-port "${DEMO_RELAY_PORT:-17860}" --app-port "$DEMO_PORT"
        --python "${DEMO_RELAY_PYTHON:-$(command -v "$DEMO_PYTHON")}" 
        --project-dir "${DEMO_RELAY_PROJECT_DIR:-$DEMO_PROJECT_DIR}")
    # Check real remote SSH networking/tunnel setup before starting the GPU model.
    timeout 90s "$DEMO_PYTHON" -u scripts/login_relay.py "${relay_args[@]}" --check-only --test-tunnel
else
    : "${DEMO_TUNNEL_TARGET:?请在 .env.local 中设置 DEMO_TUNNEL_TARGET 为公网服务器 SSH 目标}"
    unset DEMO_RELAY_TARGET
    demo_port_valid "${DEMO_TUNNEL_PORT:-17860}" || { echo "公网服务器转发端口无效。" >&2; exit 2; }
    ssh_options=(-T -o BatchMode=yes -o StrictHostKeyChecking=yes -o ConnectTimeout=10
        -o ExitOnForwardFailure=yes -o ServerAliveInterval=15 -o ServerAliveCountMax=3)
    ssh "${ssh_options[@]}" "$DEMO_TUNNEL_TARGET" true
fi

tunnel_pid=""
app_pid=""
cleanup() {
    trap - EXIT TERM INT
    [[ -z "$app_pid" ]] || kill "$app_pid" 2>/dev/null || true
    [[ -z "$tunnel_pid" ]] || kill "$tunnel_pid" 2>/dev/null || true
    wait 2>/dev/null || true
}
trap cleanup EXIT
trap 'exit 143' TERM
trap 'exit 130' INT
if [[ "$mode" == login-relay ]]; then
    "$DEMO_PYTHON" -u scripts/login_relay.py "${relay_args[@]}" &
else
    ssh -N "${ssh_options[@]}" \
        -R "127.0.0.1:${DEMO_TUNNEL_PORT:-17860}:127.0.0.1:${DEMO_PORT}" "$DEMO_TUNNEL_TARGET" &
fi
tunnel_pid=$!
"$DEMO_PYTHON" -u app.py "${app_args[@]}" --host 127.0.0.1 --port "$DEMO_PORT" --no-share &
app_pid=$!
set +e
wait -n "$app_pid" "$tunnel_pid"
result=$?
set -e
echo "Demo 或中转进程已退出，结束服务。" >&2
# An unexpected clean child exit also means the composite service is unavailable.
if [[ "$result" -eq 0 ]]; then result=1; fi
exit "$result"
