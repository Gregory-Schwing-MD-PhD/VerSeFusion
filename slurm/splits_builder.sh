#!/usr/bin/env bash
#SBATCH --job-name=verse-splits
#SBATCH -q primary
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=2G
#SBATCH --time=00:05:00
#SBATCH --output=logs/verse-splits-%j.out
#SBATCH --error=logs/verse-splits-%j.err
#SBATCH --mail-type=END,FAIL

# Generate 5-fold stratified CV splits from the master manifest.
# Patient-level, stratified by lstv_class.  Test patients are held out
# (no fold assignment), trainval patients are split across 5 folds.
#
# Reads:   data/manifest/manifest.csv
# Writes:  data/manifest/splits_5fold.json
#
# Optional env:
#   SPLITS_N_FOLDS    Number of CV folds (default 5)
#   SPLITS_SEED       Seed for fold assignment (default 42)
#   SPLITS_MANIFEST   Override manifest path
#   SPLITS_OUT        Override output path

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$(pwd)}"
. slurm/_common.sh

N_FOLDS="${SPLITS_N_FOLDS:-5}"
SEED="${SPLITS_SEED:-42}"
MANIFEST_CSV="${SPLITS_MANIFEST:-${DATA_DIR}/manifest/manifest.csv}"
OUT_PATH="${SPLITS_OUT:-${DATA_DIR}/manifest/splits_5fold.json}"

if [[ ! -f "${MANIFEST_CSV}" ]]; then
    echo "ERROR: manifest.csv not found at ${MANIFEST_CSV}"
    echo "       Run 'make manifest-slurm' first."
    exit 1
fi

mkdir -p "$(dirname "${OUT_PATH}")"

echo "============================================================"
echo "Splits build job"
echo "  manifest_csv:    ${MANIFEST_CSV}"
echo "  out:             ${OUT_PATH}"
echo "  n_folds:         ${N_FOLDS}"
echo "  seed:            ${SEED}"
echo "============================================================"

singularity exec \
    --bind "${REPO_ROOT}:${REPO_ROOT}" \
    --bind "${DATA_DIR}:${DATA_DIR}" \
    "${CONTAINER_SIF}" \
    python -m verse_pipeline.splits_builder \
        --manifest_csv "${MANIFEST_CSV}" \
        --out          "${OUT_PATH}" \
        --n_folds      "${N_FOLDS}" \
        --seed         "${SEED}"

echo ""
echo "============================================================"
echo "Splits done.  Inspect:"
echo "  jq '.subtype_counts' ${OUT_PATH}"
echo "  jq '.folds[0] | {fold, n_train: (.train_patients|length), n_val: (.val_patients|length)}' ${OUT_PATH}"
echo "============================================================"
