#!/bin/bash
#
#SBATCH -D /leonardo_work/IscrC_TVU/dcrisost/model-merging
#SBATCH --job-name=merging-eval
#SBATCH --output=./slurm/merging-eval-%j.out
#SBATCH --error=./slurm/merging-eval-%j.err
#SBATCH --time=24:00:00
#SBATCH --ntasks=1
#SBATCH --mem=32G
#SBATCH --partition=boost_usr_prod
#SBATCH --gres=gpu:1
#SBATCH --account=IscrC_TVU
#
# ── Model Merging: multi-task evaluation via SLURM ───────────────────────
#
# Usage:
#   sbatch slurm/launch_merging_eval.slurm                              # defaults
#   sbatch slurm/launch_merging_eval.slurm merger=isotropic benchmark=N8
#   sbatch slurm/launch_merging_eval.slurm merger=tsv benchmark=N20
#
# All arguments are forwarded as Hydra overrides to evaluate_multitask_merging.py.
# ────────────────────────────────────────────────────────────────────────

set -euo pipefail

# ── environment ─────────────────────────────────────────────────────────
export PATH="$HOME/.local/bin:$PATH"

# ── proxy for internet access on compute nodes ──
export http_proxy='http://login01:3133'
export https_proxy='http://login01:3133'
export SSL_CERT_FILE=/etc/ssl/certs/ca-bundle.crt

# ── caches ──
SCRATCH_BASE=/leonardo_scratch/large/userexternal/dcrisost/model-merging
export MODELS_PATH="${SCRATCH_BASE}/models"
# Use existing HF cache from home dir (models already downloaded there)
export HF_HOME="$HOME/.cache/huggingface"
export HF_DATASETS_CACHE="${SCRATCH_BASE}/hf_cache/datasets"
# Offline mode: all models/datasets must be pre-cached
export HF_HUB_OFFLINE=1
export WANDB_DIR="${SCRATCH_BASE}/wandb"
export WANDB_DATA_DIR="${SCRATCH_BASE}/wandb"
export WANDB_CACHE_DIR="${SCRATCH_BASE}/wandb/cache"
export WANDB_ARTIFACTS_DIR="${SCRATCH_BASE}/wandb/artifacts"

# ── create dirs if needed ───────────────────────────────────────────────
mkdir -p slurm
mkdir -p "$MODELS_PATH" "$HF_HOME" "$HF_DATASETS_CACHE" "$WANDB_DIR"

# ── auto-tag from Hydra overrides ─────────────────────────────────────
TAGS="autoresearch"
for arg in "$@"; do
    case "$arg" in
        merger=*)   TAGS="${TAGS},${arg#merger=}" ;;
        benchmark=*) TAGS="${TAGS},${arg#benchmark=}" ;;
    esac
done
TAG_OVERRIDE="core.tags=[${TAGS}]"

# ── run ─────────────────────────────────────────────────────────────────
echo "== Job ${SLURM_JOB_ID} started at $(date) on $(hostname) =="
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
echo ""

if [ $# -eq 0 ]; then
    echo "No Hydra overrides supplied — running with config defaults"
    echo ""
    srun uv run python scripts/evaluate_multitask_merging.py "${TAG_OVERRIDE}"
else
    echo "Hydra overrides: $@ ${TAG_OVERRIDE}"
    echo ""
    srun uv run python scripts/evaluate_multitask_merging.py "$@" "${TAG_OVERRIDE}"
fi

echo ""
echo "== Job ${SLURM_JOB_ID} finished at $(date) =="
