#!/bin/bash
#SBATCH --job-name=evaluate_launcher
#SBATCH --time=0:30:00
#SBATCH --output=logs/launcher_%j.out
#SBATCH --error=logs/launcher_%j.err
#SBATCH --account=IscrC_LENS
#SBATCH --partition=boost_usr_prod
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#
# Launcher script for evaluate_multitask_merging with Hydra submitit scheduler
# Submit with: sbatch cluster/evaluate_submitit.slurm merger.alpha=0.1,0.2,0.3
#
# This job runs the Hydra launcher which spawns separate SLURM jobs for each 
# parameter combination via the submitit_slurm launcher.

set -e

# Use absolute paths (BASH_SOURCE doesn't work in SLURM copied scripts)
PROJECT_DIR="/leonardo_work/IscrC_LENS/dcrisost/model-merging"

# Change to project directory
cd "$PROJECT_DIR" || exit 1

# Source environment variables
source "$PROJECT_DIR/cluster/cluster.env"

# Run with Hydra multirun mode and submitit launcher
# All arguments are passed directly to Hydra
HYDRA_FULL_ERROR=1 uv run python scripts/evaluate_multitask_merging.py -m \
    'hydra/launcher=submitit_slurm' \
    'benchmark=N20' \
    'nn/encoder=b32' \
    "$@"
