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

set -euo pipefail

cd "${SLURM_SUBMIT_DIR:-$(pwd)}"
. slurm/_common.sh

# Prefer the post-VERIDAH corrected tree if it exists and is non-empty,
# otherwise fall back to the unified tree.  This means:
#   - `make correct-slurm` + `make reorient-slurm` => reorients corrected/.
#   - `make reorient-slurm` alone                  => reorients unified/.
CORRECTED_DIR="${CORRECTED_DIR:-${DATA_DIR}/corrected}"
if [[ -d "${CORRECTED_DIR}" ]] && [[ -n "$(ls -A "${CORRECTED_DIR}" 2>/dev/null)" ]]; then
    REORIENT_IN_DIR="${CORRECTED_DIR}"
    echo "Reorient input: ${REORIENT_IN_DIR}  (post-VERIDAH corrected)"
else
    REORIENT_IN_DIR="${UNIFIED_DIR}"
    echo "Reorient input: ${REORIENT_IN_DIR}  (NO VERIDAH corrections applied)"
fi

singularity exec \
    --bind "${REPO_ROOT}:${REPO_ROOT}" \
    --bind "${DATA_DIR}:${DATA_DIR}" \
    "${CONTAINER_SIF}" \
    python -m verse_pipeline.reorient \
        --in_dir  "${REORIENT_IN_DIR}" \
        --out_dir "${REORIENTED_DIR}"
