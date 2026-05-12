#!/usr/bin/env bash
#SBATCH --job-name=verse-download
#SBATCH -q primary
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --time=02:00:00
#SBATCH --output=logs/verse-download-%j.out
#SBATCH --error=logs/verse-download-%j.err
#SBATCH --mail-type=END,FAIL

# Download VerSe in MICCAI-challenge format from OSF nodes:
#   VerSe19: https://osf.io/923ap/
#   VerSe20: https://osf.io/b2wxj/
#
# Auto-recovers subjects missing from MICCAI by falling back to the
# BIDS-format mirrors (jtfa5, 4skx2).  The gap is discovered by diffing
# MICCAI listings against the demographics CSV — no hardcoded subject list,
# so the fallback automatically narrows or empties as TUM patches MICCAI.
#
# Two phases:
#   1) Serial listing of MICCAI v19+v20, gap diff, BIDS fallback listing.
#   2) Parallel download with 8 worker threads.
# Resumable; cached files are skipped on re-runs.
#
# Disable the BIDS fallback with: DOWNLOAD_FLAGS="--no_bids_fallback" sbatch ...

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$(pwd)}"
. slurm/_common.sh

DOWNLOAD_WORKERS="${DOWNLOAD_WORKERS:-8}"
DOWNLOAD_FLAGS="${DOWNLOAD_FLAGS:-}"

singularity exec \
    --bind "${REPO_ROOT}:${REPO_ROOT}" \
    --bind "${DATA_DIR}:${DATA_DIR}" \
    "${CONTAINER_SIF}" \
    python -m verse_pipeline.download \
        --out_dir       "${RAW_DIR}" \
        --workers       "${DOWNLOAD_WORKERS}" \
        --demographics  "${REPO_ROOT}/configs/verse_demographics.csv" \
        ${DOWNLOAD_FLAGS}
