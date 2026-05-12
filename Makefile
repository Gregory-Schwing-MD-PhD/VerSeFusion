# =============================================================================
# VerSeFusion — Makefile
#
# Drives the full pipeline:
#   download → unify → qc → (reorient + veridah, see chunk 2)
#
# Most stages run via SLURM (the *-slurm targets) because the cluster has
# Singularity + the canonical environment.  Each stage also has a *-local
# fallback for interactive use on a login node.
#
# Run `make help` for a summary of every target.
# =============================================================================

REPO_ROOT     := $(shell pwd)
DATA_DIR      := $(REPO_ROOT)/data
RAW_DIR       := $(DATA_DIR)/raw
UNIFIED_DIR   := $(DATA_DIR)/unified
QC_DIR        := $(DATA_DIR)/qc
LOGS_DIR      := $(REPO_ROOT)/logs

CONFIG_DIR    := $(REPO_ROOT)/configs
DEMOGRAPHICS  := $(CONFIG_DIR)/verse_demographics.csv

CONTAINERS_DIR := $(REPO_ROOT)/containers
CONTAINER_SIF  := $(CONTAINERS_DIR)/versefusion.sif
CONTAINER_DEF  := $(CONTAINERS_DIR)/versefusion.def

SRC_DIR       := $(REPO_ROOT)/src

# Tunables — override on the command line: `make download-slurm DOWNLOAD_WORKERS=16`
DOWNLOAD_WORKERS ?= 8
QC_WORKERS       ?= 8

.DEFAULT_GOAL := help

# Use bash, fail on first error
SHELL := /bin/bash
.SHELLFLAGS := -eu -o pipefail -c

# -----------------------------------------------------------------------------
# Help
# -----------------------------------------------------------------------------

.PHONY: help
help:
	@echo "VerSeFusion pipeline targets"
	@echo ""
	@echo "  Setup:"
	@echo "    setup              Create dirs, fetch demographics if missing"
	@echo "    container          Build Singularity .sif from .def"
	@echo ""
	@echo "  Stage 1 — Download (OSF → data/raw/):"
	@echo "    download-slurm     Submit SLURM job (recommended on cluster)"
	@echo "    download-local     Run interactively on the current node"
	@echo "    download-clean     Wipe data/raw/ (irreversible)"
	@echo ""
	@echo "  Stage 2 — Unify (data/raw/ → data/unified/):"
	@echo "    unify-slurm        Submit SLURM job"
	@echo "    unify-local        Run interactively"
	@echo "    unify-clean        Wipe data/unified/"
	@echo ""
	@echo "  Stage 3 — QC alignment audit (data/unified/ → data/qc/):"
	@echo "    qc-slurm           Submit SLURM job"
	@echo "    qc-local           Run interactively"
	@echo "    qc-clean           Wipe data/qc/"
	@echo ""
	@echo "  Convenience:"
	@echo "    all-slurm          download → unify → qc, sequentially (SLURM)"
	@echo "    status             Show what's been produced so far"
	@echo "    clean-all          Wipe data/ entirely (DESTRUCTIVE)"
	@echo ""
	@echo "  Tunables (override on command line):"
	@echo "    DOWNLOAD_WORKERS=$(DOWNLOAD_WORKERS)  parallel download threads"
	@echo "    QC_WORKERS=$(QC_WORKERS)        parallel QC workers"
	@echo ""
	@echo "  Example:"
	@echo "    make download-slurm DOWNLOAD_WORKERS=16"
	@echo "    DOWNLOAD_FLAGS=\"--no_bids_fallback\" sbatch slurm/download_raw.sh"

# -----------------------------------------------------------------------------
# Setup
# -----------------------------------------------------------------------------

.PHONY: setup
setup:
	@mkdir -p $(RAW_DIR) $(UNIFIED_DIR) $(QC_DIR) $(LOGS_DIR)
	@if [ ! -f $(DEMOGRAPHICS) ]; then \
	    echo "ERROR: $(DEMOGRAPHICS) not found."; \
	    echo "Place TUM's verse_datasets_age_sex_19.xlsx, exported as CSV, at this path."; \
	    exit 1; \
	fi
	@echo "Setup OK. Dirs created, demographics found."

.PHONY: container
container:
	@mkdir -p $(CONTAINERS_DIR)
	@if [ ! -f $(CONTAINER_DEF) ]; then \
	    echo "ERROR: $(CONTAINER_DEF) not found; cannot build container."; \
	    exit 1; \
	fi
	cd $(CONTAINERS_DIR) && singularity build --fakeroot \
	    $(CONTAINER_SIF) $(CONTAINER_DEF)

# -----------------------------------------------------------------------------
# Stage 1 — Download
# -----------------------------------------------------------------------------

.PHONY: download-slurm download-local download-clean

download-slurm: setup
	@mkdir -p $(LOGS_DIR)
	sbatch $(REPO_ROOT)/slurm/download_raw.sh

download-local: setup
	cd $(REPO_ROOT) && \
	PYTHONPATH=$(SRC_DIR) python -m verse_pipeline.download \
	    --out_dir      $(RAW_DIR) \
	    --workers      $(DOWNLOAD_WORKERS) \
	    --demographics $(DEMOGRAPHICS)

download-clean:
	@echo "Removing $(RAW_DIR) ..."
	rm -rf $(RAW_DIR)
	@mkdir -p $(RAW_DIR)

# -----------------------------------------------------------------------------
# Stage 2 — Unify
# -----------------------------------------------------------------------------

.PHONY: unify-slurm unify-local unify-clean

unify-slurm: setup
	@mkdir -p $(LOGS_DIR)
	sbatch $(REPO_ROOT)/slurm/unify_iterations.sh

unify-local: setup
	cd $(REPO_ROOT) && \
	PYTHONPATH=$(SRC_DIR) python -m verse_pipeline.unify \
	    --raw_dir      $(RAW_DIR) \
	    --out_dir      $(UNIFIED_DIR) \
	    --demographics $(DEMOGRAPHICS)

unify-clean:
	@echo "Removing $(UNIFIED_DIR) ..."
	rm -rf $(UNIFIED_DIR)
	@mkdir -p $(UNIFIED_DIR)

# -----------------------------------------------------------------------------
# Stage 3 — QC alignment audit
# -----------------------------------------------------------------------------

.PHONY: qc-slurm qc-local qc-clean

qc-slurm: setup
	@mkdir -p $(LOGS_DIR)
	sbatch $(REPO_ROOT)/slurm/qc.sh

qc-local: setup
	cd $(REPO_ROOT) && \
	PYTHONPATH=$(SRC_DIR) python -m verse_pipeline.qc \
	    --unified_dir $(UNIFIED_DIR) \
	    --out_dir     $(QC_DIR) \
	    --workers     $(QC_WORKERS)

qc-clean:
	@echo "Removing $(QC_DIR) ..."
	rm -rf $(QC_DIR)
	@mkdir -p $(QC_DIR)

# -----------------------------------------------------------------------------
# Convenience: run everything
# -----------------------------------------------------------------------------

# `all-slurm` submits each stage in sequence with --dependency=afterok so
# they chain automatically through the SLURM queue.  Each step waits for
# the previous one to complete successfully before starting.
.PHONY: all-slurm
all-slurm: setup
	@mkdir -p $(LOGS_DIR)
	@DOWNLOAD_JOB=$$(sbatch --parsable $(REPO_ROOT)/slurm/download_raw.sh); \
	echo "submitted download as job $$DOWNLOAD_JOB"; \
	UNIFY_JOB=$$(sbatch --parsable --dependency=afterok:$$DOWNLOAD_JOB \
	    $(REPO_ROOT)/slurm/unify_iterations.sh); \
	echo "submitted unify as job $$UNIFY_JOB (after $$DOWNLOAD_JOB)"; \
	QC_JOB=$$(sbatch --parsable --dependency=afterok:$$UNIFY_JOB \
	    $(REPO_ROOT)/slurm/qc.sh); \
	echo "submitted qc as job $$QC_JOB (after $$UNIFY_JOB)"; \
	echo ""; \
	echo "Watch with:  squeue -u $$USER"

# -----------------------------------------------------------------------------
# Status / inspection
# -----------------------------------------------------------------------------

.PHONY: status
status:
	@echo "===== VerSeFusion pipeline status ====="
	@echo ""
	@echo "Demographics:"
	@if [ -f $(DEMOGRAPHICS) ]; then \
	    echo "  $(DEMOGRAPHICS)  ($$(wc -l < $(DEMOGRAPHICS)) lines)"; \
	else \
	    echo "  MISSING: $(DEMOGRAPHICS)"; \
	fi
	@echo ""
	@echo "Container:"
	@if [ -f $(CONTAINER_SIF) ]; then \
	    echo "  $(CONTAINER_SIF)  ($$(du -h $(CONTAINER_SIF) | cut -f1))"; \
	else \
	    echo "  MISSING: $(CONTAINER_SIF)  (run 'make container')"; \
	fi
	@echo ""
	@echo "Stage 1 (download):"
	@if [ -f $(RAW_DIR)/download_manifest.json ]; then \
	    echo "  manifest: $(RAW_DIR)/download_manifest.json"; \
	    jq -r '"  n_files=\(.n_files)  downloaded=\(.n_downloaded)  cached=\(.n_cached)  failed=\(.n_failed)"' \
	        $(RAW_DIR)/download_manifest.json 2>/dev/null \
	        || echo "  (jq not available — check manifest manually)"; \
	else \
	    echo "  not run yet"; \
	fi
	@echo ""
	@echo "Stage 2 (unify):"
	@if [ -f $(UNIFIED_DIR)/unify_manifest.json ]; then \
	    echo "  manifest: $(UNIFIED_DIR)/unify_manifest.json"; \
	    jq -r '"  n_scans=\(.n_scans)  n_patients=\(.n_patients)  complete=\(.completeness.n_complete)"' \
	        $(UNIFIED_DIR)/unify_manifest.json 2>/dev/null \
	        || echo "  (jq not available)"; \
	else \
	    echo "  not run yet"; \
	fi
	@echo ""
	@echo "Stage 3 (qc):"
	@if [ -f $(QC_DIR)/qc_manifest.json ]; then \
	    echo "  manifest: $(QC_DIR)/qc_manifest.json"; \
	    jq -r '.by_status | "  PASS=\(.PASS)  WARN=\(.WARN)  FAIL=\(.FAIL)  SKIP=\(.SKIP)"' \
	        $(QC_DIR)/qc_manifest.json 2>/dev/null \
	        || echo "  (jq not available)"; \
	else \
	    echo "  not run yet"; \
	fi
	@echo ""
	@echo "Recent SLURM logs:"
	@ls -lt $(LOGS_DIR)/*.err 2>/dev/null | head -5 | awk '{print "  " $$0}' \
	    || echo "  (no logs yet)"

# -----------------------------------------------------------------------------
# Destructive cleanup
# -----------------------------------------------------------------------------

.PHONY: clean-all
clean-all:
	@echo "WARNING: this will delete $(DATA_DIR) entirely."
	@echo "Press Ctrl-C in the next 5 seconds to abort..."
	@sleep 5
	rm -rf $(DATA_DIR)
	@echo "$(DATA_DIR) removed."
