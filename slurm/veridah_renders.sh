#!/usr/bin/env bash
#SBATCH --job-name=verse-veridah-renders
#SBATCH -q primary
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=00:30:00
#SBATCH --output=logs/verse-veridah-renders-%j.out
#SBATCH --error=logs/verse-veridah-renders-%j.err
#SBATCH --mail-type=END,FAIL

# Generate before/after side-by-side renders for VERIDAH-corrected subjects.
#
# Reads:  data/canonical/, data/corrected/, data/corrected/veridah_manifest.json
# Writes: data/corrected/renders/<series_id>_before_after.png + index.html
#
# Only ~25 subjects, so this is a quick job — under 5 minutes.

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$(pwd)}"
. slurm/_common.sh

RENDER_WORKERS="${RENDER_WORKERS:-4}"
RENDER_DPI="${RENDER_DPI:-90}"
CANONICAL_DIR="${DATA_DIR}/canonical"
CORRECTED_DIR="${DATA_DIR}/corrected"
RENDER_OUT="${CORRECTED_DIR}/renders"
VERIDAH_RENDER_FLAGS="${VERIDAH_RENDER_FLAGS:-}"

mkdir -p "${RENDER_OUT}"

singularity exec \
    --bind "${REPO_ROOT}:${REPO_ROOT}" \
    --bind "${DATA_DIR}:${DATA_DIR}" \
    "${CONTAINER_SIF}" \
    python -m verse_pipeline.visualize_corrections \
        --canonical_dir "${CANONICAL_DIR}" \
        --corrected_dir "${CORRECTED_DIR}" \
        --out_dir       "${RENDER_OUT}" \
        --workers       "${RENDER_WORKERS}" \
        --dpi           "${RENDER_DPI}" \
        ${VERIDAH_RENDER_FLAGS}
