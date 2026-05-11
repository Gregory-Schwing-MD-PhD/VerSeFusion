#!/usr/bin/env bash
#SBATCH --job-name=verse-lstv-audit
#SBATCH --output=logs/verse-lstv-audit-%j.out
#SBATCH --error=logs/verse-lstv-audit-%j.err
#SBATCH --partition=compute
#SBATCH --qos=normal
#SBATCH --time=00:10:00
#SBATCH --mem=4G
#SBATCH --cpus-per-task=1

. "$(dirname "$0")/_common.sh"

singularity exec \
    --bind "${REPO_ROOT}:${REPO_ROOT}" \
    --bind "${DATA_DIR}:${DATA_DIR}" \
    "${CONTAINER_SIF}" \
    python -m verse_pipeline.lstv --audit \
        --manifest "${REORIENTED_DIR}/placed_manifest.json"
