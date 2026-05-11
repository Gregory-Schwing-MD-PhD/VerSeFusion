#!/usr/bin/env bash
#SBATCH --job-name=verse-unify
#SBATCH --output=logs/verse-unify-%j.out
#SBATCH --error=logs/verse-unify-%j.err
#SBATCH --partition=compute
#SBATCH --qos=normal
#SBATCH --time=01:00:00
#SBATCH --mem=8G
#SBATCH --cpus-per-task=2

. "$(dirname "$0")/_common.sh"

singularity exec \
    --bind "${REPO_ROOT}:${REPO_ROOT}" \
    --bind "${DATA_DIR}:${DATA_DIR}" \
    "${CONTAINER_SIF}" \
    python -m verse_pipeline.unify \
        --raw_dir "${RAW_DIR}" \
        --out_dir "${UNIFIED_DIR}" \
        --prefer "${DEDUP_PREFER:-verse20}" \
        --mode symlink
