#!/usr/bin/env bash
#SBATCH --job-name=verse-veridah
#SBATCH -q primary
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=00:30:00
#SBATCH --output=logs/verse-veridah-%j.out
#SBATCH --error=logs/verse-veridah-%j.err
#SBATCH --mail-type=END,FAIL

# Apply Möller 2026 VERIDAH manual label corrections to canonical PIR scans.
#
# Reads:  data/canonical/  (output of reorient.sh)
# Writes: data/corrected/  (PIR + Möller corrections to ~25 subjects)
#
# About 25 subjects get real corrected masks; the other ~349 are symlinks
# straight to canonical/.  Total wall time ~1-3 minutes.

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$(pwd)}"
. slurm/_common.sh

VERIDAH_WORKERS="${VERIDAH_WORKERS:-8}"
CANONICAL_DIR="${DATA_DIR}/canonical"
CORRECTED_DIR="${DATA_DIR}/corrected"
CORRECTIONS_CSV="${REPO_ROOT}/configs/veridah_corrections.csv"
VERIDAH_FLAGS="${VERIDAH_FLAGS:-}"

mkdir -p "${CORRECTED_DIR}"

singularity exec \
    --bind "${REPO_ROOT}:${REPO_ROOT}" \
    --bind "${DATA_DIR}:${DATA_DIR}" \
    "${CONTAINER_SIF}" \
    python -m verse_pipeline.veridah \
        --in_dir          "${CANONICAL_DIR}" \
        --out_dir         "${CORRECTED_DIR}" \
        --corrections_csv "${CORRECTIONS_CSV}" \
        --workers         "${VERIDAH_WORKERS}" \
        ${VERIDAH_FLAGS}
