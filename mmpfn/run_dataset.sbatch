#!/bin/bash
# One dataset = one scheduler job. Usage (from mmpfn/):
#   sbatch run_dataset.sbatch <dataset> [task]        e.g.  sbatch run_dataset.sbatch cbis_ddsm mass
# Adjust -p / -q / -A / --gres below to your cluster; check with `sinfo` and your cluster docs.
#SBATCH -J mmpfn
#SBATCH -N 1
#SBATCH -c 8
#SBATCH --mem=64G
#SBATCH -G a100:1
#SBATCH -t 48:00:00
#SBATCH -p general
#SBATCH -o logs/slurm-%j.out
set -euo pipefail
DATASET="$1"; TASK="${2:-}"
BRANCH=$(git rev-parse --abbrev-ref HEAD)
LOG="logs/${DATASET}${TASK:+_$TASK}_${BRANCH}.log"
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate mmpfn
cd "$SLURM_SUBMIT_DIR"
echo "[$(date)] branch=$BRANCH dataset=$DATASET task=$TASK node=$(hostname) gpu=$(nvidia-smi -L)" | tee -a "$LOG"
CUDA_VISIBLE_DEVICES=0 python -u run.py "$DATASET" $TASK >> "$LOG" 2>&1
echo "[$(date)] done" >> "$LOG"
