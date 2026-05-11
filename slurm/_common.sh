#!/usr/bin/env bash
# =============================================================================
# Shared SLURM preamble.
#
# Every slurm/*.sh script in this repo sources this file via
# `. slurm/_common.sh` (the caller must `cd "${SLURM_SUBMIT_DIR:-$(pwd)}"`
# first, so this path resolves to the real repo location rather than the
# SLURM temp dir).  It is responsible for:
#
#   1. Scrubbing leaky host env vars that can corrupt the job
#      (JAVA_HOME, LD_LIBRARY_PATH, PYTHONPATH, R_LIBS*).
#   2. Activating the conda env used elsewhere in the project
#      (${HOME}/mambaforge/envs/nextflow) so `singularity` and `python`
#      resolve to the right binaries.
#   3. Setting up per-job Singularity runtime dirs (SINGULARITY_TMPDIR,
#      XDG_RUNTIME_DIR, NXF_SINGULARITY_CACHEDIR).
#   4. Loading non-secret defaults from configs/default.env and the
#      gitignored secrets layer from env.local (if either exists).
#   5. Exporting PYTHONPATH (+ SINGULARITYENV_PYTHONPATH) so
#      `verse_pipeline` is importable without `pip install -e .`.
#
# After this file is sourced, the calling script has:
#   REPO_ROOT, DATA_DIR, RAW_DIR, UNIFIED_DIR, REORIENTED_DIR,
#   HF_DIR, LOG_DIR, CONTAINER_SIF, PYTHONPATH — all guaranteed non-empty.
# =============================================================================

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${REPO_ROOT}"

# ── Scrub host env that can leak into the pull ───────────────────────────────
unset JAVA_HOME LD_LIBRARY_PATH PYTHONPATH R_LIBS R_LIBS_USER R_LIBS_SITE
# ── Nextflow/conda env (same layout used elsewhere in the project) ───────────
export CONDA_PREFIX="${CONDA_PREFIX:-${HOME}/mambaforge/envs/nextflow}"
export PATH="${CONDA_PREFIX}/bin:${PATH}"
# ── Singularity runtime dirs ─────────────────────────────────────────────────
export SINGULARITY_TMPDIR="/tmp/${USER}_job_${SLURM_JOB_ID:-local}"
export XDG_RUNTIME_DIR="${SINGULARITY_TMPDIR}/runtime"
export NXF_SINGULARITY_CACHEDIR="${HOME}/singularity_cache"
mkdir -p "${SINGULARITY_TMPDIR}" "${XDG_RUNTIME_DIR}" "${NXF_SINGULARITY_CACHEDIR}"
trap 'rm -rf "${SINGULARITY_TMPDIR}"' EXIT
export CONDA_PREFIX="${HOME}/mambaforge/envs/nextflow"
export PATH="${CONDA_PREFIX}/bin:${PATH}"
unset JAVA_HOME; which singularity
export NXF_SINGULARITY_HOME_MOUNT=true
unset LD_LIBRARY_PATH PYTHONPATH R_LIBS R_LIBS_USER R_LIBS_SITE

# --- load non-secret defaults -------------------------------------------------
# Regex must include digits so VERSE19_ZIPS etc. aren't filtered out.
# default.env holds only single-word scalars; multi-word lists and path
# expressions are NOT supported here — see configs/default.env header.
if [[ -f configs/default.env ]]; then
    # shellcheck disable=SC1091
    set -a; source <(grep -E '^[A-Z0-9_]+=' configs/default.env); set +a
fi

# --- optional secrets layer ---------------------------------------------------
if [[ -f env.local ]]; then
    # shellcheck disable=SC1091
    set -a; source env.local; set +a
fi

# --- defaults the env files do not set (paths live here, not in default.env) -
DATA_DIR="${DATA_DIR:-${REPO_ROOT}/data}"
LOG_DIR="${LOG_DIR:-${REPO_ROOT}/logs}"
RAW_DIR="${RAW_DIR:-${DATA_DIR}/raw}"
UNIFIED_DIR="${UNIFIED_DIR:-${DATA_DIR}/unified}"
REORIENTED_DIR="${REORIENTED_DIR:-${DATA_DIR}/reoriented}"
HF_DIR="${HF_DIR:-${DATA_DIR}/hf_export}"
CONTAINER_SIF="${CONTAINER_SIF:-${REPO_ROOT}/containers/versefusion.sif}"

mkdir -p "${LOG_DIR}"

# --- make src/ importable as `verse_pipeline` without pip install -e . -------
# Both the host conda env and the reused CTSpinoPelvic1K container lack
# `verse_pipeline`; pointing PYTHONPATH at src/ keeps things working without
# either an install step or a dedicated VerSeFusion image.  REPO_ROOT is
# bind-mounted into every singularity exec (--bind "${REPO_ROOT}:${REPO_ROOT}"),
# so this path resolves to the same location inside and outside the container.
# SINGULARITYENV_PYTHONPATH guarantees the value crosses the
# `singularity exec` boundary regardless of host-env strip policy.
export PYTHONPATH="${REPO_ROOT}/src${PYTHONPATH:+:${PYTHONPATH}}"
export SINGULARITYENV_PYTHONPATH="${PYTHONPATH}"

echo "============================================================"
echo "VerSeFusion job"
echo "  host:           $(hostname)"
echo "  date:           $(date -Iseconds)"
echo "  repo:           ${REPO_ROOT}"
echo "  DATA_DIR:       ${DATA_DIR}"
echo "  CONTAINER_SIF:  ${CONTAINER_SIF}"
echo "  PYTHONPATH:     ${PYTHONPATH}"
echo "  SLURM_JOB_ID:   ${SLURM_JOB_ID:-N/A}"
echo "============================================================"
