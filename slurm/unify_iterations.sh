#!/usr/bin/env bash
#SBATCH --job-name=verse-unify
#SBATCH -q primary
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=01:00:00
#SBATCH --output=logs/verse-unify-%j.out
#SBATCH --error=logs/verse-unify-%j.err
#SBATCH --mail-type=END,FAIL

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
