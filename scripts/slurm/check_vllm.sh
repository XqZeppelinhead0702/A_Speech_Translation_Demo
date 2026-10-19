#!/usr/bin/env bash
#SBATCH --job-name=vllm_compat_check
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --gres=gpu:1
#SBATCH --output=outs/vllm-check-%j.out
#SBATCH --error=outs/vllm-check-%j.err
# Metadata/tokenizer + read-only nvidia-smi only; no model weights or GPU kernels.
set -eo pipefail
[[ -n "${SLURM_JOB_ID:-}" ]] || { echo "请从项目根目录使用 sbatch 提交" >&2; exit 1; }
cd -- "${DEMO_PROJECT_DIR:-${SLURM_SUBMIT_DIR:?缺少提交目录}}"
export DEMO_PROJECT_DIR="$PWD"
source scripts/run/common.sh
diagnostic_python="${REALTIME_VLLM_PYTHON:-$DEMO_PYTHON}"
exec "$diagnostic_python" -u scripts/check_vllm.py --gpu-info \
  --model-dir "${REALTIME_VLLM_MODEL_DIR:-${REALTIME_HY_MT_MODEL_DIR:-ckpts/Hy-MT2-1.8B}}" "$@"
