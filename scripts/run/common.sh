#!/usr/bin/env bash
# Shared setup for portable launchers. Source this file; do not run it alone.
set -eo pipefail
DEMO_PROJECT_DIR="${DEMO_PROJECT_DIR:-$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)}"
cd -- "$DEMO_PROJECT_DIR"
config_file="${DEMO_CONFIG_FILE:-$DEMO_PROJECT_DIR/.env.local}"
if [[ -f "$config_file" ]]; then
    set -a
    source "$config_file"
    set +a
elif [[ -n "${DEMO_CONFIG_FILE:-}" ]]; then
    echo "找不到 DEMO_CONFIG_FILE 指定的配置文件。" >&2
    exit 1
fi
# Module/Conda shell hooks may refer to unset shell variables.
if [[ -n "${DEMO_SETUP_SH:-}" ]]; then source "$DEMO_SETUP_SH"; fi
if [[ -n "${DEMO_MODULES:-}" ]]; then
    command -v module >/dev/null || { echo "当前 shell 无 module 命令，请使用 DEMO_SETUP_SH 初始化。" >&2; exit 1; }
    read -r -a demo_modules <<< "$DEMO_MODULES"
    module load "${demo_modules[@]}"
fi
if [[ -n "${DEMO_CONDA_SH:-}" ]]; then source "$DEMO_CONDA_SH"; fi
if [[ -n "${DEMO_CONDA_ENV:-}" ]]; then
    command -v conda >/dev/null || { echo "请设置 DEMO_CONDA_SH，或预先激活 Python 环境。" >&2; exit 1; }
    conda activate "$DEMO_CONDA_ENV"
fi
set -u
export DEMO_PROJECT_DIR
export DEMO_PYTHON="${DEMO_PYTHON:-python}"
command -v "$DEMO_PYTHON" >/dev/null || { echo "找不到配置的 Python。" >&2; exit 1; }
if [[ -n "${DEMO_FFMPEG_DIR:-}" ]]; then export PATH="$DEMO_FFMPEG_DIR:$PATH"; fi
export PYTHONUNBUFFERED=1 GRADIO_ANALYTICS_ENABLED=False
# Bound default CPU worker pools to the Slurm allocation before importing Torch.
if [[ -n "${SLURM_JOB_ID:-}" ]]; then
    export OMP_NUM_THREADS="${OMP_NUM_THREADS:-${SLURM_CPUS_PER_TASK:-1}}"
    export MKL_NUM_THREADS="${MKL_NUM_THREADS:-$OMP_NUM_THREADS}"
fi
export NO_PROXY="localhost,127.0.0.1,0.0.0.0,::1${NO_PROXY:+,$NO_PROXY}${no_proxy:+,$no_proxy}"
export no_proxy="$NO_PROXY"
case "${DEMO_PROXY_MODE:-inherit}" in
    direct) unset HTTP_PROXY HTTPS_PROXY ALL_PROXY http_proxy https_proxy all_proxy ;;
    inherit) ;;
    *) echo "DEMO_PROXY_MODE 必须为 direct 或 inherit。" >&2; exit 1 ;;
esac

demo_port_valid() {
    [[ "$1" =~ ^[0-9]{1,5}$ ]] && ((10#$1 >= 1 && 10#$1 <= 65535))
}
