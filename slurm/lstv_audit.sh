#!/usr/bin/env bash
#SBATCH --job-name=verse-lstv-audit
#SBATCH -q primary
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=00:30:00
#SBATCH --output=logs/verse-lstv-audit-%j.out
#SBATCH --error=logs/verse-lstv-audit-%j.err
#SBATCH --mail-type=END,FAIL

# Label-based LSTV / TLTV classification of the corrected dataset.
#
# Reads:  data/corrected/scan-*/scan-*_msk.nii.gz
# Writes: data/lstv/lstv_audit_manifest.json
#         data/lstv/lstv_audit_summary.csv
#
# The audit streams each mask slice-by-slice (chunked along largest axis)
# rather than loading the full volume, so peak memory per worker is small
# even for full-body VerSe scans (512×512×1500+).  32G is generous headroom.

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$(pwd)}"
. slurm/_common.sh

AUDIT_WORKERS="${AUDIT_WORKERS:-8}"
AUDIT_INPUT="${AUDIT_INPUT:-${DATA_DIR}/corrected}"
AUDIT_OUT="${AUDIT_OUT:-${DATA_DIR}/lstv}"

mkdir -p "${AUDIT_OUT}"

singularity exec \
    --bind "${REPO_ROOT}:${REPO_ROOT}" \
    --bind "${DATA_DIR}:${DATA_DIR}" \
    "${CONTAINER_SIF}" \
    python -m verse_pipeline.lstv_audit \
        --input_dir "${AUDIT_INPUT}" \
        --out_dir   "${AUDIT_OUT}" \
        --workers   "${AUDIT_WORKERS}"
