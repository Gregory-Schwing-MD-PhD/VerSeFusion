#!/usr/bin/env bash
#SBATCH --job-name=verse-manifest
#SBATCH -q primary
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:30:00
#SBATCH --output=logs/verse-manifest-%j.out
#SBATCH --error=logs/verse-manifest-%j.err
#SBATCH --mail-type=END,FAIL

. "$(dirname "$0")/_common.sh"

singularity exec \
    --bind "${REPO_ROOT}:${REPO_ROOT}" \
    --bind "${DATA_DIR}:${DATA_DIR}" \
    "${CONTAINER_SIF}" \
    python -m verse_pipeline.manifest \
        --in_dir   "${REORIENTED_DIR}" \
        --out_path "${REORIENTED_DIR}/placed_manifest.json" \
        --unify_manifest "${UNIFIED_DIR}/unify_manifest.json"
