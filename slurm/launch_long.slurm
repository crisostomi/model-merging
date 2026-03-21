#!/bin/bash
#
#SBATCH -D /leonardo_work/IscrC_TVU/dcrisost/model-merging
#SBATCH --job-name=tv-analysis-long
#SBATCH --output=./slurm/tv-analysis-long-%j.out
#SBATCH --error=./slurm/tv-analysis-long-%j.err
#SBATCH --time=06:00:00
#SBATCH --ntasks=1
#SBATCH --mem=64G
#SBATCH --partition=boost_usr_prod
#SBATCH --gres=gpu:1
#SBATCH --account=IscrC_TVU
#
# Long-running analysis jobs — up to 6h, 64GB RAM
# Usage: sbatch slurm/launch_long.slurm scripts/analyze_layer_importance.py
#

set -euo pipefail

export PATH="$HOME/.local/bin:$PATH"
export http_proxy='http://login01:3133'
export https_proxy='http://login01:3133'
export SSL_CERT_FILE=/etc/ssl/certs/ca-bundle.crt

SCRATCH_BASE=/leonardo_scratch/large/userexternal/dcrisost/model-merging
export MODELS_PATH="${SCRATCH_BASE}/models"
export HF_HOME="${SCRATCH_BASE}/hf_cache"
export HF_DATASETS_CACHE="${SCRATCH_BASE}/hf_cache/datasets"

mkdir -p "$MODELS_PATH" "$HF_HOME" "$HF_DATASETS_CACHE"

echo "== Job ${SLURM_JOB_ID} started at $(date) on $(hostname) =="

SCRIPT="${1:-scripts/analyze_layer_importance.py}"
shift || true

srun uv run python "$SCRIPT" "$@"

echo ""
echo "== Job ${SLURM_JOB_ID} finished at $(date) =="
