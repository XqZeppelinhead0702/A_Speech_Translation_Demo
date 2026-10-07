#!/usr/bin/env bash
#SBATCH --job-name=speech_demo
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH --output=outs/demo-%j.out
#SBATCH --error=outs/demo-%j.err

set -eo pipefail
if [[ -z "${SLURM_JOB_ID:-}" ]]; then
    echo "请在项目根目录使用 sbatch 提交此 GPU 作业。" >&2
    exit 1
fi
# sbatch copies this file to a spool directory: use the submit directory instead.
project_dir="${DEMO_PROJECT_DIR:-${SLURM_SUBMIT_DIR:?缺少提交目录}}"
cd -- "$project_dir"
export DEMO_PROJECT_DIR="$PWD"
exec bash "$DEMO_PROJECT_DIR/scripts/run/serve.sh" share "$@"
