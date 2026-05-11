#!/usr/bin/env bash
#SBATCH --job-name=verse-splits
#SBATCH -q primary
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --time=00:15:00
#SBATCH --output=logs/verse-splits-%j.out
#SBATCH --error=logs/verse-splits-%j.err
#SBATCH --mail-type=END,FAIL

set -euo pipefail

cd "${SLURM_SUBMIT_DIR:-$(pwd)}"
. slurm/_common.sh

singularity exec \
    --bind "${REPO_ROOT}:${REPO_ROOT}" \
    --bind "${DATA_DIR}:${DATA_DIR}" \
    "${CONTAINER_SIF}" \
    python -m verse_pipeline.splits \
        --manifest "${REORIENTED_DIR}/placed_manifest.json" \
        --out_dir  "${REORIENTED_DIR}/splits" \
        --n_folds  "${N_FOLDS:-5}" \
        --seed     "${SPLIT_SEED:-20260511}" \
        --holdout  "${HOLDOUT_MODE:-verse20_test}"
