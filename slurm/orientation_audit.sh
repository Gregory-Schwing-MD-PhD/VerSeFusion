#!/usr/bin/env bash
#SBATCH --job-name=verse-orient-audit
#SBATCH -q primary
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=8G
#SBATCH --time=00:15:00
#SBATCH --output=logs/verse-orient-audit-%j.out
#SBATCH --error=logs/verse-orient-audit-%j.err
#SBATCH --mail-type=END,FAIL

# Verify every canonical (or corrected) scan is PIR-oriented.
# Only reads headers via nibabel, no full volume loads, so this is fast and
# memory-cheap.

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$(pwd)}"
. slurm/_common.sh

ORIENT_INPUT="${ORIENT_INPUT:-${DATA_DIR}/canonical}"
ORIENT_OUT="${ORIENT_OUT:-${DATA_DIR}/orientation}"
ORIENT_WORKERS="${ORIENT_WORKERS:-8}"
ORIENT_FLAGS="${ORIENT_FLAGS:-}"

mkdir -p "${ORIENT_OUT}"

singularity exec \
    --bind "${REPO_ROOT}:${REPO_ROOT}" \
    --bind "${DATA_DIR}:${DATA_DIR}" \
    "${CONTAINER_SIF}" \
    python -m verse_pipeline.orientation_audit \
        --input_dir "${ORIENT_INPUT}" \
        --out_dir   "${ORIENT_OUT}" \
        --workers   "${ORIENT_WORKERS}" \
        ${ORIENT_FLAGS}
