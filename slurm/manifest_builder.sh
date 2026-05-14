#!/usr/bin/env bash
#SBATCH --job-name=verse-manifest
#SBATCH -q primary
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=4G
#SBATCH --time=00:15:00
#SBATCH --output=logs/verse-manifest-%j.out
#SBATCH --error=logs/verse-manifest-%j.err
#SBATCH --mail-type=END,FAIL

# Build the master manifest for VerSeFusion.  Aggregates per-stage info
# (canonical metadata + veridah corrections + LSTV audit + unify splits)
# into a single CSV/JSON.  CV folds are produced in a separate stage
# (splits-slurm) so they can be regenerated with different seeds /
# strata without rebuilding the manifest.

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$(pwd)}"
. slurm/_common.sh

OUT_DIR="${MANIFEST_OUT_DIR:-${DATA_DIR}/manifest}"
mkdir -p "${OUT_DIR}"

echo "============================================================"
echo "Manifest build job"
echo "  canonical_dir:   ${DATA_DIR}/canonical"
echo "  corrected_dir:   ${DATA_DIR}/corrected"
echo "  unify_manifest:  ${DATA_DIR}/unified/unify_manifest.json"
echo "  lstv_audit:      ${DATA_DIR}/lstv/lstv_audit_manifest.json"
echo "  output_dir:      ${OUT_DIR}"
echo "============================================================"

singularity exec \
    --bind "${REPO_ROOT}:${REPO_ROOT}" \
    --bind "${DATA_DIR}:${DATA_DIR}" \
    "${CONTAINER_SIF}" \
    python -m verse_pipeline.manifest_builder \
        --canonical_dir  "${DATA_DIR}/canonical" \
        --corrected_dir  "${DATA_DIR}/corrected" \
        --unify_manifest "${DATA_DIR}/unified/unify_manifest.json" \
        --lstv_audit     "${DATA_DIR}/lstv/lstv_audit_manifest.json" \
        --output_dir     "${OUT_DIR}"

echo ""
echo "Manifest done.  Next: make splits-slurm"
