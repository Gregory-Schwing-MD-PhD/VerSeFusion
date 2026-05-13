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

# Per-scan QC audit for canonically reoriented VerSe scans.
#
# Reads from data/canonical/ (output of reorient.sh).
# Writes data/qc/qc_manifest.json.
#
# Resource sizing: 374 scans × header read + mask data load (~50MB peak) on
# 8 workers ≈ 2-5 minutes wall time.

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$(pwd)}"
. slurm/_common.sh

QC_WORKERS="${QC_WORKERS:-8}"
QC_DIR="${DATA_DIR}/qc"
QC_INPUT="${QC_INPUT:-${DATA_DIR}/canonical}"

mkdir -p "${QC_DIR}"

singularity exec \
    --bind "${REPO_ROOT}:${REPO_ROOT}" \
    --bind "${DATA_DIR}:${DATA_DIR}" \
    "${CONTAINER_SIF}" \
    python -m verse_pipeline.qc \
        --input_dir "${QC_INPUT}" \
        --out_dir   "${QC_DIR}" \
        --workers   "${QC_WORKERS}"
