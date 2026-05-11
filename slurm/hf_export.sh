#!/usr/bin/env bash
#SBATCH --job-name=verse-hf-export
#SBATCH --output=logs/verse-hf-export-%j.out
#SBATCH --error=logs/verse-hf-export-%j.err
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
    python -m verse_pipeline.hf_export \
        --in_dir   "${REORIENTED_DIR}" \
        --out_dir  "${HF_DIR}" \
        --mode     copy
