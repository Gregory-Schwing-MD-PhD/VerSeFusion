#!/usr/bin/env bash
#SBATCH --job-name=verse-reorient
#SBATCH --output=logs/verse-reorient-%j.out
#SBATCH --error=logs/verse-reorient-%j.err
#SBATCH --partition=compute
#SBATCH --qos=normal
#SBATCH --time=02:00:00
#SBATCH --mem=16G
#SBATCH --cpus-per-task=4

. "$(dirname "$0")/_common.sh"

singularity exec \
    --bind "${REPO_ROOT}:${REPO_ROOT}" \
    --bind "${DATA_DIR}:${DATA_DIR}" \
    "${CONTAINER_SIF}" \
    python -m verse_pipeline.reorient \
        --in_dir  "${UNIFIED_DIR}" \
        --out_dir "${REORIENTED_DIR}"
