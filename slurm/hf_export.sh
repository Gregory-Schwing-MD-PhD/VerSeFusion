#!/usr/bin/env bash
#SBATCH --job-name=verse-hf-export
#SBATCH -q primary
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=01:00:00
#SBATCH --output=logs/verse-hf-export-%j.out
#SBATCH --error=logs/verse-hf-export-%j.err
#SBATCH --mail-type=END,FAIL

set -euo pipefail

cd "${SLURM_SUBMIT_DIR:-$(pwd)}"
. slurm/_common.sh

singularity exec \
    --bind "${REPO_ROOT}:${REPO_ROOT}" \
    --bind "${DATA_DIR}:${DATA_DIR}" \
    "${CONTAINER_SIF}" \
    python -m verse_pipeline.hf_export \
        --in_dir   "${REORIENTED_DIR}" \
        --out_dir  "${HF_DIR}" \
        --mode     copy
