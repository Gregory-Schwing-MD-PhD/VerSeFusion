#!/usr/bin/env bash
#SBATCH --job-name=verse-unify
#SBATCH -q primary
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:30:00
#SBATCH --output=logs/verse-unify-%j.out
#SBATCH --error=logs/verse-unify-%j.err
#SBATCH --mail-type=END,FAIL

# Group raw MICCAI files by series_id, materialise scan-<series_id>/ dirs,
# resolve cross-release ties via demographics spreadsheet (TUM-published).

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$(pwd)}"
. slurm/_common.sh

singularity exec \
    --bind "${REPO_ROOT}:${REPO_ROOT}" \
    --bind "${DATA_DIR}:${DATA_DIR}" \
    "${CONTAINER_SIF}" \
    python -m verse_pipeline.unify \
        --raw_dir       "${RAW_DIR}" \
        --out_dir       "${UNIFIED_DIR}" \
        --demographics  "${REPO_ROOT}/configs/verse_demographics.xlsx"
