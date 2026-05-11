#!/usr/bin/env bash
#SBATCH --job-name=verse-download
#SBATCH --output=logs/verse-download-%j.out
#SBATCH --error=logs/verse-download-%j.err
#SBATCH --partition=compute
#SBATCH --qos=normal
#SBATCH --time=08:00:00
#SBATCH --mem=8G
#SBATCH --cpus-per-task=4

# Pull the six VerSe S3 zips on Warrior HPC.
# 30 GB total; 8h is generous and accounts for spotty S3 throughput.

. "$(dirname "$0")/_common.sh"

singularity exec \
    --bind "${REPO_ROOT}:${REPO_ROOT}" \
    --bind "${DATA_DIR}:${DATA_DIR}" \
    "${CONTAINER_SIF}" \
    python -m verse_pipeline.download \
        --out_dir "${RAW_DIR}" \
        --log_level INFO
