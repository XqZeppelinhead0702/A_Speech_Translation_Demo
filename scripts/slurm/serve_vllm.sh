#!/usr/bin/env bash
#SBATCH --job-name=hy_mt2_vllm
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --output=outs/vllm-%j.out
#SBATCH --error=outs/vllm-%j.err
set -eo pipefail
[[ -n "${SLURM_JOB_ID:-}" ]] || { echo "请从项目根目录使用 sbatch 提交" >&2; exit 1; }
cd -- "${DEMO_PROJECT_DIR:-${SLURM_SUBMIT_DIR:?缺少提交目录}}"
export DEMO_PROJECT_DIR="$PWD"
exec bash scripts/run/serve_vllm.sh "$@"
