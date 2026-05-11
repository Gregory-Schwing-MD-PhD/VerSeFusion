#!/usr/bin/env bash
#SBATCH --job-name=verse-splits
#SBATCH --output=logs/verse-splits-%j.out
#SBATCH --error=logs/verse-splits-%j.err
#SBATCH --partition=compute
#SBATCH --qos=normal
#SBATCH --time=00:15:00
#SBATCH --mem=4G
#SBATCH --cpus-per-task=1

. "$(dirname "$0")/_common.sh"

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
