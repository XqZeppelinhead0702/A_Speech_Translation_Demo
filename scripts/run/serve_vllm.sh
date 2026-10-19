#!/usr/bin/env bash
# Separate Hy-MT2 process; shares the Demo Python unless explicitly overridden.
set -eo pipefail
source "$(dirname -- "${BASH_SOURCE[0]}")/common.sh"
vllm_python="${REALTIME_VLLM_PYTHON:-$DEMO_PYTHON}"
command -v "$vllm_python" >/dev/null || { echo "找不到 vLLM Python；默认使用 DEMO_PYTHON，可选 REALTIME_VLLM_PYTHON 覆盖" >&2; exit 1; }
model_dir="${REALTIME_VLLM_MODEL_DIR:-${REALTIME_HY_MT_MODEL_DIR:-ckpts/Hy-MT2-1.8B}}"
precision="${REALTIME_HY_MT_PRECISION:-bfloat16}"
port="${REALTIME_VLLM_PORT:-8000}"
served_model="${REALTIME_VLLM_SERVED_MODEL:-hy-mt2}"
check_only=0
allow_local=0
extra=()
while (($#)); do
    case "$1" in
        --model-dir|--precision|--port|--served-model)
            (($# >= 2)) || { echo "$1 缺少参数" >&2; exit 2; }
            case "$1" in
                --model-dir) model_dir="$2" ;;
                --precision) precision="$2" ;;
                --port) port="$2" ;;
                --served-model) served_model="$2" ;;
            esac
            shift 2 ;;
        --check-only) check_only=1; shift ;;
        --allow-local) allow_local=1; shift ;;
        --) shift; extra=("$@"); break ;;
        *) echo "未知参数：$1（额外 vLLM 参数放在 -- 后）" >&2; exit 2 ;;
    esac
done
demo_port_valid "$port" || { echo "vLLM 端口无效" >&2; exit 2; }
case "$precision" in bfloat16|float16|float32) ;; *) echo "精度无效" >&2; exit 2 ;; esac
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
# Scope the compatibility hook to this engine/check and its spawn workers.
# The Demo/ASR process keeps its original import behavior, even in the same env.
case "${REALTIME_VLLM_TOKENIZER_COMPAT:-1}" in
    1)
        export SPEECH_DEMO_VLLM_COMPAT=1
        export PYTHONPATH="$DEMO_PROJECT_DIR/scripts/vllm_compat${PYTHONPATH:+:$PYTHONPATH}"
        ;;
    0) unset SPEECH_DEMO_VLLM_COMPAT ;;
    *) echo "REALTIME_VLLM_TOKENIZER_COMPAT 必须为 0 或 1" >&2; exit 2 ;;
esac
"$vllm_python" -u scripts/check_vllm.py --model-dir "$model_dir" --gpu-info
((check_only == 0)) || exit 0
[[ -n "${SLURM_JOB_ID:-}" || "$allow_local" == 1 ]] || { echo "集群请通过 Slurm 提交；普通 GPU 机器显式加 --allow-local；CPU 检查可用 --check-only" >&2; exit 1; }
engine_args=(--model "$model_dir" --served-model-name "$served_model" --host 127.0.0.1 --port "$port"
    --dtype "$precision" --tensor-parallel-size 1
    --gpu-memory-utilization "${REALTIME_VLLM_GPU_MEMORY_UTILIZATION:-0.4}"
    --max-model-len "${REALTIME_VLLM_MAX_MODEL_LEN:-8192}"
    --max-num-seqs "${REALTIME_VLLM_MAX_NUM_SEQS:-1}"
    --max-num-batched-tokens "${REALTIME_VLLM_MAX_BATCHED_TOKENS:-512}"
    --enable-prefix-caching --enable-chunked-prefill)
if [[ -n "${REALTIME_VLLM_MODEL_IMPL:-}" ]]; then engine_args+=(--model-impl "$REALTIME_VLLM_MODEL_IMPL"); fi
case "${REALTIME_VLLM_ENFORCE_EAGER:-0}" in
    1) engine_args+=(--enforce-eager) ;;
    0) ;;
    *) echo "REALTIME_VLLM_ENFORCE_EAGER 必须为 0 或 1" >&2; exit 2 ;;
esac
# vLLM reads its own VLLM_API_KEY. Do not place keys on the process command line.
if [[ -n "${REALTIME_VLLM_API_KEY:-}" ]]; then export VLLM_API_KEY="$REALTIME_VLLM_API_KEY"; fi
echo "启动独立 vLLM 服务：127.0.0.1:$port；显存预算比例 ${REALTIME_VLLM_GPU_MEMORY_UTILIZATION:-0.4}"
exec "$vllm_python" -u -m vllm.entrypoints.openai.api_server "${engine_args[@]}" "${extra[@]}"
