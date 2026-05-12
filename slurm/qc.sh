#!/usr/bin/env bash
#SBATCH --job-name=verse-qc
#SBATCH -q primary
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=00:45:00
#SBATCH --output=logs/verse-qc-%j.out
#SBATCH --error=logs/verse-qc-%j.err
#SBATCH --mail-type=END,FAIL

# Per-scan alignment audit for the unified VerSe corpus.
#
# For each scan-<series_id>/ dir, checks:
#   1. files_present       — required CT/mask/centroid on disk
#   2. headers_readable    — nibabel can load both NIfTI headers
#   3. shape_match         — CT and mask same voxel grid
#   4. affine_match        — CT and mask same world coordinate frame
#   5. label_inventory     — VerSe labels 1-28 only, voxel counts sane
#   6. centroid_alignment  — each labeled centroid lands on the right mask voxel
#
# Output: data/qc/qc_manifest.json
#
# Resource sizing: ProcessPoolExecutor over scans, each worker loads one mask
# volume (~10-100MB) plus reads CT/mask headers.  8 workers × ~250MB peak each
# fits comfortably in 32G.  Most scans complete in 1-3 seconds; 374 scans on
# 8 workers ≈ 2-5 minutes wall time.

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$(pwd)}"
. slurm/_common.sh

QC_WORKERS="${QC_WORKERS:-8}"
QC_DIR="${DATA_DIR}/qc"

singularity exec \
    --bind "${REPO_ROOT}:${REPO_ROOT}" \
    --bind "${DATA_DIR}:${DATA_DIR}" \
    "${CONTAINER_SIF}" \
    python -m verse_pipeline.qc \
        --unified_dir "${UNIFIED_DIR}" \
        --out_dir     "${QC_DIR}" \
        --workers     "${QC_WORKERS}"
