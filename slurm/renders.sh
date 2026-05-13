#!/usr/bin/env bash
#SBATCH --job-name=verse-renders
#SBATCH -q primary
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --output=logs/verse-renders-%j.out
#SBATCH --error=logs/verse-renders-%j.err
#SBATCH --mail-type=END,FAIL

# Generate per-scan QC PNGs from canonically reoriented scans.
#
# Reads from data/canonical/ (output of reorient.sh).
# Output: data/qc/renders/<series_id>.png
#         data/qc/renders/renders_manifest.json
#         data/qc/renders/index.html       (gallery)
#
# Each render is fast (~1-3s) since the data is already in PIR — no
# orientation handling needed inside visualize.py.

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$(pwd)}"
. slurm/_common.sh

RENDER_WORKERS="${RENDER_WORKERS:-8}"
RENDER_DPI="${RENDER_DPI:-80}"
RENDER_DIR="${DATA_DIR}/qc/renders"
RENDER_INPUT="${RENDER_INPUT:-${DATA_DIR}/canonical}"
RENDER_FLAGS="${RENDER_FLAGS:-}"

mkdir -p "${RENDER_DIR}"

singularity exec \
    --bind "${REPO_ROOT}:${REPO_ROOT}" \
    --bind "${DATA_DIR}:${DATA_DIR}" \
    "${CONTAINER_SIF}" \
    python -m verse_pipeline.visualize \
        --input_dir "${RENDER_INPUT}" \
        --out_dir   "${RENDER_DIR}" \
        --workers   "${RENDER_WORKERS}" \
        --dpi       "${RENDER_DPI}" \
        ${RENDER_FLAGS}

singularity exec \
    --bind "${REPO_ROOT}:${REPO_ROOT}" \
    --bind "${DATA_DIR}:${DATA_DIR}" \
    "${CONTAINER_SIF}" \
    python -m verse_pipeline.render_gallery \
        --renders_dir    "${RENDER_DIR}" \
        --qc_manifest    "${DATA_DIR}/qc/qc_manifest.json" \
        --unify_manifest "${UNIFIED_DIR}/unify_manifest.json" \
        --out_path       "${RENDER_DIR}/index.html"
