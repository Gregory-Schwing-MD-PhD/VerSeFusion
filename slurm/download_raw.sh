#!/usr/bin/env bash
#SBATCH --job-name=verse-download
#SBATCH -q primary
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --time=08:00:00
#SBATCH --output=logs/verse-download-%j.out
#SBATCH --error=logs/verse-download-%j.err
#SBATCH --mail-type=END,FAIL

# Pull the six VerSe S3 zips on Warrior HPC.
# 30 GB total; 8h is generous and accounts for spotty S3 throughput.

set -euo pipefail

# SLURM copies the script body to a temp dir, so $0 / $(dirname $0) point
# nowhere useful.  Anchor on SLURM_SUBMIT_DIR (always set by sbatch) so
# _common.sh resolves to the real repo path.
cd "${SLURM_SUBMIT_DIR:-$(pwd)}"
. slurm/_common.sh

singularity exec \
    --bind "${REPO_ROOT}:${REPO_ROOT}" \
    --bind "${DATA_DIR}:${DATA_DIR}" \
    "${CONTAINER_SIF}" \
    python -m verse_pipeline.download \
        --out_dir "${RAW_DIR}" \
        --log_level INFO
