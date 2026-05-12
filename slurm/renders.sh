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

# Generate per-scan QC PNGs showing CT + mask overlay + centroid markers.
#
# Output: data/qc/renders/<series_id>.png
#         data/qc/renders/renders_manifest.json
#         data/qc/renders/index.html       (gallery)
#
# Resource sizing: each render loads one CT volume (~30-300MB) and one mask
# (~10-100MB), reorients to PIR, and rasterizes 3 panels via matplotlib.
# ~3-8 seconds per scan; 374 scans on 8 workers ≈ 5-10 minutes wall time.

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$(pwd)}"
. slurm/_common.sh

RENDER_WORKERS="${RENDER_WORKERS:-8}"
RENDER_DPI="${RENDER_DPI:-80}"
RENDER_DIR="${DATA_DIR}/qc/renders"
RENDER_FLAGS="${RENDER_FLAGS:-}"

mkdir -p "${RENDER_DIR}"

# Generate PNGs (default: all scans; override with RENDER_FLAGS="--flagged_from data/qc/qc_manifest.json")
singularity exec \
    --bind "${REPO_ROOT}:${REPO_ROOT}" \
    --bind "${DATA_DIR}:${DATA_DIR}" \
    "${CONTAINER_SIF}" \
    python -m verse_pipeline.visualize \
        --unified_dir "${UNIFIED_DIR}" \
        --out_dir     "${RENDER_DIR}" \
        --workers     "${RENDER_WORKERS}" \
        --dpi         "${RENDER_DPI}" \
        ${RENDER_FLAGS}

# Build the HTML gallery
singularity exec \
    --bind "${REPO_ROOT}:${REPO_ROOT}" \
    --bind "${DATA_DIR}:${DATA_DIR}" \
    "${CONTAINER_SIF}" \
    python -m verse_pipeline.render_gallery \
        --renders_dir    "${RENDER_DIR}" \
        --qc_manifest    "${DATA_DIR}/qc/qc_manifest.json" \
        --unify_manifest "${UNIFIED_DIR}/unify_manifest.json" \
        --out_path       "${RENDER_DIR}/index.html"
