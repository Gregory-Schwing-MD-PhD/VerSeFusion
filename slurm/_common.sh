#!/usr/bin/env bash
# Shared SLURM preamble.  Source from each job script via `. slurm/_common.sh`.
# Loads configs/default.env and configs/env.local (if present).

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${REPO_ROOT}"

# --- load non-secret defaults -------------------------------------------------
if [[ -f configs/default.env ]]; then
    # shellcheck disable=SC1091
    set -a; source <(grep -E '^[A-Z_]+=' configs/default.env); set +a
fi

# --- optional secrets layer ---------------------------------------------------
if [[ -f env.local ]]; then
    # shellcheck disable=SC1091
    set -a; source env.local; set +a
fi

# --- defaults the env files may not set --------------------------------------
DATA_DIR="${DATA_DIR:-${REPO_ROOT}/data}"
LOG_DIR="${LOG_DIR:-${REPO_ROOT}/logs}"
RAW_DIR="${RAW_DIR:-${DATA_DIR}/raw}"
UNIFIED_DIR="${UNIFIED_DIR:-${DATA_DIR}/unified}"
REORIENTED_DIR="${REORIENTED_DIR:-${DATA_DIR}/reoriented}"
HF_DIR="${HF_DIR:-${DATA_DIR}/hf_export}"
CONTAINER_SIF="${CONTAINER_SIF:-${REPO_ROOT}/containers/versefusion.sif}"

mkdir -p "${LOG_DIR}"

echo "============================================================"
echo "VerSeFusion job"
echo "  host:           $(hostname)"
echo "  date:           $(date -Iseconds)"
echo "  repo:           ${REPO_ROOT}"
echo "  DATA_DIR:       ${DATA_DIR}"
echo "  CONTAINER_SIF:  ${CONTAINER_SIF}"
echo "  SLURM_JOB_ID:   ${SLURM_JOB_ID:-N/A}"
echo "============================================================"
