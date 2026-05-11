#!/usr/bin/env bash
# =============================================================================
# VerSeFusion — HPC Singularity Pull
# scripts/hpc_pull.sh
#
# Pulls a Docker Hub image and writes containers/versefusion.sif so every
# slurm/*.sh wrapper finds it via the default CONTAINER_SIF path.
#
# Because VerSeFusion's runtime deps are a strict subset of CTSpinoPelvic1K's,
# the *default* source image is `${DOCKERHUB_USER}/ctspinopelvic1k:${TAG}` —
# i.e. by default this script pulls your existing CTSpinoPelvic1K container
# and lands it at containers/versefusion.sif so VerSeFusion's pipeline runs
# against it unchanged.  Once you publish a dedicated versefusion image,
# set SOURCE_REPO=versefusion to switch.
#
# Prereqs on HPC:
#   - Singularity >= 3.9 (module load singularity, or in PATH)
#   - Internet access from login / compute node
#   - Disk: ~3 GB
#
# Usage:
#   bash scripts/hpc_pull.sh
#   DOCKERHUB_USER=myuser bash scripts/hpc_pull.sh
#   SOURCE_REPO=versefusion bash scripts/hpc_pull.sh   # dedicated image
#   TAG=v0.1.0 bash scripts/hpc_pull.sh
# =============================================================================

set -euo pipefail

# --- knobs ------------------------------------------------------------------
DOCKERHUB_USER="${DOCKERHUB_USER:-gregoryschwingmdphd}"
SOURCE_REPO="${SOURCE_REPO:-ctspinopelvic1k}"   # `ctspinopelvic1k` (default, reuse) | `versefusion`
TAG="${TAG:-latest}"
SIF_DIR="${SIF_DIR:-$(pwd)/containers}"
OUT_SIF="${SIF_DIR}/versefusion.sif"

# --- helpers ----------------------------------------------------------------
log() { echo "[$(date '+%H:%M:%S')] $*"; }
die() { echo "[ERROR] $*" >&2; exit 1; }

command -v singularity >/dev/null 2>&1 \
    || { module load singularity 2>/dev/null \
            || die "Singularity not found.  Try: module load singularity"; }

mkdir -p "${SIF_DIR}"

case "${SOURCE_REPO}" in
    ctspinopelvic1k|versefusion) ;;
    *) die "SOURCE_REPO must be 'ctspinopelvic1k' or 'versefusion' (got: ${SOURCE_REPO})" ;;
esac

SOURCE_IMAGE="docker://${DOCKERHUB_USER}/${SOURCE_REPO}:${TAG}"

log "=== VerSeFusion HPC Singularity Pull ==="
log "User          : ${DOCKERHUB_USER}"
log "Source image  : ${SOURCE_IMAGE}"
log "Output SIF    : ${OUT_SIF}"

# --- pull -------------------------------------------------------------------
singularity pull --force "${OUT_SIF}" "${SOURCE_IMAGE}"
log "  ✓ ${OUT_SIF}  ($(du -sh "${OUT_SIF}" | cut -f1))"

# --- self-check -------------------------------------------------------------
log "Verifying runtime imports inside the SIF..."
if singularity exec "${OUT_SIF}" python -c "
import nibabel, numpy, yaml, sklearn, tqdm, requests
print(f'  nibabel  {nibabel.__version__}')
print(f'  numpy    {numpy.__version__}')
print(f'  sklearn  {sklearn.__version__}')
" ; then
    log "  ✓ all required modules importable"
else
    die "Required modules missing from ${SOURCE_IMAGE}.  Pick a different image or rebuild."
fi

# --- next steps -------------------------------------------------------------
cat <<EOF

  Next steps:
    # 1.  Pull the six VerSe S3 zips
    sbatch slurm/download_raw.sh

    # 2.  Run the rest of the pipeline
    sbatch slurm/unify_iterations.sh
    sbatch slurm/reorient_pir.sh
    sbatch slurm/build_manifest.sh
    sbatch slurm/make_splits.sh
    sbatch slurm/hf_export.sh

EOF

log "Pull complete."
