#!/usr/bin/env bash
#SBATCH --job-name=verse-correct
#SBATCH -q primary
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:30:00
#SBATCH --output=logs/verse-correct-%j.out
#SBATCH --error=logs/verse-correct-%j.err
#SBATCH --mail-type=END,FAIL

# Apply Moeller 2026 VERIDAH manual label corrections to ~25 subjects.
# Pass-through symlinks the other ~330 subjects unchanged.

set -euo pipefail

cd "${SLURM_SUBMIT_DIR:-$(pwd)}"
. slurm/_common.sh

CORRECTED_DIR="${CORRECTED_DIR:-${DATA_DIR}/corrected}"
mkdir -p "${CORRECTED_DIR}"

singularity exec \
    --bind "${REPO_ROOT}:${REPO_ROOT}" \
    --bind "${DATA_DIR}:${DATA_DIR}" \
    "${CONTAINER_SIF}" \
    python -m verse_pipeline.veridah \
        --in_dir          "${UNIFIED_DIR}" \
        --out_dir         "${CORRECTED_DIR}" \
        --corrections_csv "${REPO_ROOT}/configs/veridah_corrections.csv"
