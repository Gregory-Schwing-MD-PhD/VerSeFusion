#!/usr/bin/env bash
#SBATCH --job-name=verse-download
#SBATCH -q primary
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=08:00:00
#SBATCH --output=logs/verse-download-%j.out
#SBATCH --error=logs/verse-download-%j.err
#SBATCH --mail-type=END,FAIL

# Download VerSe in MICCAI-challenge format from OSF nodes:
#   VerSe19: https://osf.io/923ap/
#   VerSe20: https://osf.io/b2wxj/
# Throttled and resumable.  Expect ~1500-1600 files total across both releases.

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$(pwd)}"
. slurm/_common.sh

singularity exec \
    --bind "${REPO_ROOT}:${REPO_ROOT}" \
    --bind "${DATA_DIR}:${DATA_DIR}" \
    "${CONTAINER_SIF}" \
    python -m verse_pipeline.download --out_dir "${RAW_DIR}"
