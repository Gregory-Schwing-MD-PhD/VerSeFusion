#!/usr/bin/env bash
#SBATCH --job-name=versefusion_hpc_pull
#SBATCH -q primary
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=02:00:00
#SBATCH --output=logs/hpc_pull_%j.out
#SBATCH --error=logs/hpc_pull_%j.err
#SBATCH --mail-type=END,FAIL

# =============================================================================
# Pulls one Docker Hub image and writes containers/versefusion.sif.
#
# By default, pulls ${DOCKERHUB_USER}/ctspinopelvic1k:latest — since
# VerSeFusion's deps are a strict subset of CTSpinoPelvic1K's, the same
# container runs both repos.  Override with SOURCE_REPO=versefusion when
# a dedicated image is available.
#
# Usage (from the project root):
#     sbatch slurm/hpc_pull.sh
#     DOCKERHUB_USER=myuser sbatch slurm/hpc_pull.sh
#     SOURCE_REPO=versefusion sbatch slurm/hpc_pull.sh
#
# Or via make:
#     make hpc-pull
# =============================================================================

set -euo pipefail

PROJECT_ROOT="${SLURM_SUBMIT_DIR:-$(pwd)}"
cd "${PROJECT_ROOT}"
mkdir -p logs containers

# ── Scrub host env that can leak into the pull ───────────────────────────────
unset JAVA_HOME LD_LIBRARY_PATH PYTHONPATH R_LIBS R_LIBS_USER R_LIBS_SITE
# ── Nextflow/conda env (same layout used elsewhere in the project) ───────────
export CONDA_PREFIX="${CONDA_PREFIX:-${HOME}/mambaforge/envs/nextflow}"
export PATH="${CONDA_PREFIX}/bin:${PATH}"
# ── Singularity runtime dirs ─────────────────────────────────────────────────
export SINGULARITY_TMPDIR="/tmp/${USER}_job_${SLURM_JOB_ID}"
export XDG_RUNTIME_DIR="${SINGULARITY_TMPDIR}/runtime"
export NXF_SINGULARITY_CACHEDIR="${HOME}/singularity_cache"
mkdir -p "${SINGULARITY_TMPDIR}" "${XDG_RUNTIME_DIR}" "${NXF_SINGULARITY_CACHEDIR}"
trap 'rm -rf "${SINGULARITY_TMPDIR}"' EXIT
export CONDA_PREFIX="${HOME}/mambaforge/envs/nextflow"
export PATH="${CONDA_PREFIX}/bin:${PATH}"
unset JAVA_HOME; which singularity
export NXF_SINGULARITY_HOME_MOUNT=true
unset LD_LIBRARY_PATH PYTHONPATH R_LIBS R_LIBS_USER R_LIBS_SITE

echo "======================================================================"
echo " VerSeFusion hpc_pull"
echo "   Job ID         : ${SLURM_JOB_ID:-local}"
echo "   Node           : $(hostname)"
echo "   DOCKERHUB_USER : ${DOCKERHUB_USER:-gregoryschwingmdphd}"
echo "   SOURCE_REPO    : ${SOURCE_REPO:-ctspinopelvic1k}"
echo "   TAG            : ${TAG:-latest}"
echo "   SIF out dir    : ${PROJECT_ROOT}/containers"
echo "   Cache dir      : ${NXF_SINGULARITY_CACHEDIR}"
echo "   Started        : $(date)"
echo "======================================================================"

# Call the pull script directly (NOT `make hpc-pull` — that would re-submit).
SIF_DIR="${PROJECT_ROOT}/containers" \
    bash "${PROJECT_ROOT}/scripts/hpc_pull.sh"

echo ""
echo " Completed at $(date)"
echo "======================================================================"
