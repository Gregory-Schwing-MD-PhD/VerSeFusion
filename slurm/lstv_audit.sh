#!/usr/bin/env bash
#SBATCH --job-name=verse-lstv-audit
#SBATCH -q primary
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --time=00:10:00
#SBATCH --output=logs/verse-lstv-audit-%j.out
#SBATCH --error=logs/verse-lstv-audit-%j.err
#SBATCH --mail-type=END,FAIL

set -euo pipefail

cd "${SLURM_SUBMIT_DIR:-$(pwd)}"
. slurm/_common.sh

singularity exec \
    --bind "${REPO_ROOT}:${REPO_ROOT}" \
    --bind "${DATA_DIR}:${DATA_DIR}" \
    "${CONTAINER_SIF}" \
    python -m verse_pipeline.lstv --audit \
        --manifest "${REORIENTED_DIR}/placed_manifest.json"
