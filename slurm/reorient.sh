#!/usr/bin/env bash
#SBATCH --job-name=verse-reorient
#SBATCH -q primary
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=01:00:00
#SBATCH --output=logs/verse-reorient-%j.out
#SBATCH --error=logs/verse-reorient-%j.err
#SBATCH --mail-type=END,FAIL

# Reorient unified VerSe scans to canonical PIR orientation.
#
# Output: data/canonical/scan-<series_id>/{ct,msk}.nii.gz   (REAL files, not symlinks)
#         data/canonical/scan-<series_id>/snp.png            (symlink to original)
#         data/canonical/scan-<series_id>/meta.json
#         data/canonical/canonical_manifest.json
#
# Resource sizing: each worker loads CT+mask (~100MB peak), applies an axis
# permutation/flip (no resampling), saves the reoriented NIfTIs.  About 3-8s
# per scan; 374 scans on 8 workers ≈ 5-10 minutes wall time.  Memory peaks
# briefly at ~2GB during save; 64G is generous headroom.
#
# Disk usage: ~18GB total for 374 scans (data preserved bit-for-bit, just
# permuted).  Equivalent to a full copy of the original unified tree.

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$(pwd)}"
. slurm/_common.sh

REORIENT_WORKERS="${REORIENT_WORKERS:-8}"
CANONICAL_DIR="${DATA_DIR}/canonical"

mkdir -p "${CANONICAL_DIR}"

singularity exec \
    --bind "${REPO_ROOT}:${REPO_ROOT}" \
    --bind "${DATA_DIR}:${DATA_DIR}" \
    "${CONTAINER_SIF}" \
    python -m verse_pipeline.reorient \
        --unified_dir "${UNIFIED_DIR}" \
        --out_dir     "${CANONICAL_DIR}" \
        --workers     "${REORIENT_WORKERS}"
