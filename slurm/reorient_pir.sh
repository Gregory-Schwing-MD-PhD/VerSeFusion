#!/usr/bin/env bash
#SBATCH --job-name=verse-reorient
#SBATCH -q primary
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=02:00:00
#SBATCH --output=logs/verse-reorient-%j.out
#SBATCH --error=logs/verse-reorient-%j.err
#SBATCH --mail-type=END,FAIL

. "$(dirname "$0")/_common.sh"

singularity exec \
    --bind "${REPO_ROOT}:${REPO_ROOT}" \
    --bind "${DATA_DIR}:${DATA_DIR}" \
    "${CONTAINER_SIF}" \
    python -m verse_pipeline.reorient \
        --in_dir  "${UNIFIED_DIR}" \
        --out_dir "${REORIENTED_DIR}"
