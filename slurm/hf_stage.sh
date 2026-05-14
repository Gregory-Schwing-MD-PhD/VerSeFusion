#!/usr/bin/env bash
#SBATCH --job-name=verse-hf-stage
#SBATCH -q primary
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=01:30:00
#SBATCH --output=logs/verse-hf-stage-%j.out
#SBATCH --error=logs/verse-hf-stage-%j.err
#SBATCH --mail-type=END,FAIL

# Stage the VerSeFusion dataset (and 10-case sample) WITHOUT uploading.
#
# Same orientation gate, same staging logic as hf_export.sh, just stops
# before the HuggingFace push so you can inspect data/hf_staging/ and
# data/hf_staging_sample/ before deciding to upload.
#
# No HF_TOKEN required.  No network calls.
#
# Hardcoded repos (these still get baked into the staged README):
#   Full:    gregoryschwingmdphd/VerseFusion
#   Sample:  gregoryschwingmdphd/VerseFusion-Sample
#
# Optional env:
#   HF_REPO_ID                 Override full-dataset repo
#   HF_SAMPLE_REPO_ID          Override sample repo
#   HF_SAMPLE_N                Sample size (default 10; set 0 to skip sample)
#   HF_DATASET_PRETTY_NAME     Pretty name for README ("VerSeFusion" default)
#   HF_SYMLINK=1               Symlink NIfTIs instead of copying
#   HF_STAGING_DIR             Override staging dir (default: $DATA_DIR/hf_staging)
#   HF_SAMPLE_STAGING_DIR      Override sample staging dir
#                              (default: $DATA_DIR/hf_staging_sample)
#   HF_PREVIEW_DIR             Optional dir of preview PNGs to include

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$(pwd)}"
. slurm/_common.sh

HF_REPO_ID="${HF_REPO_ID:-gregoryschwingmdphd/VerseFusion}"
HF_SAMPLE_REPO_ID="${HF_SAMPLE_REPO_ID:-gregoryschwingmdphd/VerseFusion-Sample}"
HF_LSTV_REPO_ID="${HF_LSTV_REPO_ID:-gregoryschwingmdphd/VerseFusion-LSTV}"
HF_SAMPLE_N="${HF_SAMPLE_N:-10}"
HF_DATASET_PRETTY_NAME="${HF_DATASET_PRETTY_NAME:-VerSeFusion}"
HF_STAGING_DIR="${HF_STAGING_DIR:-${DATA_DIR}/hf_staging}"
HF_SAMPLE_STAGING_DIR="${HF_SAMPLE_STAGING_DIR:-${DATA_DIR}/hf_staging_sample}"
HF_LSTV_STAGING_DIR="${HF_LSTV_STAGING_DIR:-${DATA_DIR}/hf_staging_lstv}"
HF_MANIFEST_CSV="${HF_MANIFEST_CSV:-${DATA_DIR}/manifest/manifest.csv}"

EXPORT_FLAGS=(
    --no_upload
    --sample_n             "${HF_SAMPLE_N}"
    --sample_repo_id       "${HF_SAMPLE_REPO_ID}"
    --sample_staging_dir   "${HF_SAMPLE_STAGING_DIR}"
    --lstv_repo_id         "${HF_LSTV_REPO_ID}"
    --lstv_staging_dir     "${HF_LSTV_STAGING_DIR}"
)
# Manifest from stage 10 (gives us LSTV + cv_fold columns in the staged dataset)
if [[ -f "${HF_MANIFEST_CSV}" ]]; then
    EXPORT_FLAGS+=( --manifest_csv "${HF_MANIFEST_CSV}" )
else
    echo "WARNING: ${HF_MANIFEST_CSV} not found — staged dataset will lack manifest.csv."
    echo "         Run 'make manifest-slurm' first."
fi
# Stage mode: default 'hardlink' (instant on same-FS).  Override with
# HF_STAGE_MODE=copy or HF_STAGE_MODE=symlink, or the legacy HF_SYMLINK=1.
if [[ -n "${HF_STAGE_MODE:-}" ]]; then
    EXPORT_FLAGS+=( --stage_mode "${HF_STAGE_MODE}" )
fi
[[ "${HF_SYMLINK:-0}" = "1" ]] && EXPORT_FLAGS+=( --symlink )
[[ -n "${HF_PREVIEW_DIR:-}" ]] && EXPORT_FLAGS+=( --preview_dir "${HF_PREVIEW_DIR}" )

mkdir -p "${HF_STAGING_DIR}" "${HF_SAMPLE_STAGING_DIR}" "${HF_LSTV_STAGING_DIR}"

echo "============================================================"
echo "HF stage job (NO UPLOAD)"
echo "  full repo:        ${HF_REPO_ID}"
echo "  sample repo:      ${HF_SAMPLE_REPO_ID}"
echo "  sample_n:         ${HF_SAMPLE_N}"
echo "  pretty_name:      ${HF_DATASET_PRETTY_NAME}"
echo "  full staging:     ${HF_STAGING_DIR}"
echo "  sample staging:   ${HF_SAMPLE_STAGING_DIR}"
echo "  lstv repo:        ${HF_LSTV_REPO_ID}"
echo "  lstv staging:     ${HF_LSTV_STAGING_DIR}"
echo "============================================================"

singularity exec \
    --bind "${REPO_ROOT}:${REPO_ROOT}" \
    --bind "${DATA_DIR}:${DATA_DIR}" \
    "${CONTAINER_SIF}" \
    python -m verse_pipeline.hf_export \
        --canonical_dir   "${DATA_DIR}/canonical" \
        --corrected_dir   "${DATA_DIR}/corrected" \
        --unify_manifest  "${DATA_DIR}/unified/unify_manifest.json" \
        --staging_dir     "${HF_STAGING_DIR}" \
        --repo_id         "${HF_REPO_ID}" \
        --dataset_pretty_name "${HF_DATASET_PRETTY_NAME}" \
        --workers         8 \
        "${EXPORT_FLAGS[@]}"

echo ""
echo "============================================================"
echo "Staging complete.  Inspect:"
echo "  du -sh ${HF_STAGING_DIR}"
echo "  cat ${HF_STAGING_DIR}/README.md"
echo "  cat ${HF_SAMPLE_STAGING_DIR}/sample_selection.json | jq .selected_ids"
echo ""
echo "When ready to upload, run:"
echo "  HF_TOKEN=hf_xxx make hf-export-slurm"
echo "============================================================"
