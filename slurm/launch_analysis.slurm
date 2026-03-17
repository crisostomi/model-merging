#!/bin/bash
#
#SBATCH -D /leonardo_work/IscrC_TVU/dcrisost/model-merging
#SBATCH --job-name=tv-analysis
#SBATCH --output=./slurm/tv-analysis-%j.out
#SBATCH --error=./slurm/tv-analysis-%j.err
#SBATCH --time=02:00:00
#SBATCH --ntasks=1
#SBATCH --mem=32G
#SBATCH --partition=boost_usr_prod
#SBATCH --gres=gpu:1
#SBATCH --account=IscrC_TVU
#
# Task vector analysis — runs on CPU but needs 32GB RAM for 8 models
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

# Run the analysis script specified as first argument, or default to analyze_task_vectors.py
SCRIPT="${1:-scripts/analyze_task_vectors.py}"
shift || true

srun uv run python "$SCRIPT" "$@"

echo ""
echo "== Job ${SLURM_JOB_ID} finished at $(date) =="
