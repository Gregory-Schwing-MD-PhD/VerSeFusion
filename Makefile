# =============================================================================
# VerSeFusion — Makefile
#
# Pipeline order:
#   download → unify → reorient → qc → renders → (veridah, chunk 2)
#
# data/raw/        OSF-fetched MICCAI/BIDS files (gitignored, ~60GB)
# data/unified/    Per-scan dirs with symlinks to raw, demographics-keyed
# data/canonical/  Reoriented to PIR; real NIfTI files (~18GB)
# data/qc/         Per-scan QC manifest from canonical
# data/qc/renders/ Per-scan PNGs + HTML gallery
#
# Run `make help` for a summary of every target.
# =============================================================================

REPO_ROOT     := $(shell pwd)
DATA_DIR      := $(REPO_ROOT)/data
RAW_DIR       := $(DATA_DIR)/raw
UNIFIED_DIR   := $(DATA_DIR)/unified
CANONICAL_DIR := $(DATA_DIR)/canonical
QC_DIR        := $(DATA_DIR)/qc
RENDER_DIR    := $(DATA_DIR)/qc/renders
LOGS_DIR      := $(REPO_ROOT)/logs

CONFIG_DIR    := $(REPO_ROOT)/configs
DEMOGRAPHICS  := $(CONFIG_DIR)/verse_demographics.csv

CONTAINERS_DIR := $(REPO_ROOT)/containers
CONTAINER_SIF  := $(CONTAINERS_DIR)/versefusion.sif
CONTAINER_DEF  := $(CONTAINERS_DIR)/versefusion.def

SRC_DIR       := $(REPO_ROOT)/src

DOWNLOAD_WORKERS  ?= 8
REORIENT_WORKERS  ?= 8
QC_WORKERS        ?= 8
RENDER_WORKERS    ?= 8
RENDER_DPI        ?= 80

.DEFAULT_GOAL := help

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
	@echo "    setup              Create dirs, check demographics file"
	@echo "    container          Build Singularity .sif from .def"
	@echo ""
	@echo "  Stage 1 — Download (OSF → data/raw/):"
	@echo "    download-slurm     Submit SLURM job"
	@echo "    download-local     Run interactively"
	@echo "    download-clean     Wipe data/raw/"
	@echo ""
	@echo "  Stage 2 — Unify (data/raw/ → data/unified/):"
	@echo "    unify-slurm        Submit SLURM job"
	@echo "    unify-local        Run interactively"
	@echo "    unify-clean        Wipe data/unified/"
	@echo ""
	@echo "  Stage 3 — Reorient to PIR (data/unified/ → data/canonical/):"
	@echo "    reorient-slurm     Submit SLURM job"
	@echo "    reorient-local     Run interactively"
	@echo "    reorient-clean     Wipe data/canonical/"
	@echo ""
	@echo "  Stage 4 — QC audit (data/canonical/ → data/qc/):"
	@echo "    qc-slurm           Submit SLURM job"
	@echo "    qc-local           Run interactively"
	@echo "    qc-clean           Wipe data/qc/ (keeps renders subdir)"
	@echo ""
	@echo "  Stage 5 — Visual renders (data/canonical/ → data/qc/renders/):"
	@echo "    renders-slurm      Generate per-scan PNGs + HTML gallery"
	@echo "    renders-flagged    Render only WARN/FAIL scans from qc_manifest"
	@echo "    renders-local      Render 10 scans interactively (smoke test)"
	@echo "    renders-gallery    Rebuild HTML index from existing PNGs"
	@echo "    renders-clean      Wipe data/qc/renders/"
	@echo ""
	@echo "  Convenience:"
	@echo "    all-slurm          download → unify → reorient → qc → renders chain"
	@echo "    status             Show what's been produced so far"
	@echo "    clean-all          Wipe data/ entirely (DESTRUCTIVE)"
	@echo ""
	@echo "  Tunables (override on command line):"
	@echo "    DOWNLOAD_WORKERS=$(DOWNLOAD_WORKERS)  parallel download threads"
	@echo "    REORIENT_WORKERS=$(REORIENT_WORKERS)  parallel reorient workers"
	@echo "    QC_WORKERS=$(QC_WORKERS)        parallel QC workers"
	@echo "    RENDER_WORKERS=$(RENDER_WORKERS)    parallel render workers"
	@echo "    RENDER_DPI=$(RENDER_DPI)       PNG resolution (150 for paper)"

# -----------------------------------------------------------------------------
# Setup
# -----------------------------------------------------------------------------

.PHONY: setup
setup:
	@mkdir -p $(RAW_DIR) $(UNIFIED_DIR) $(CANONICAL_DIR) $(QC_DIR) $(RENDER_DIR) $(LOGS_DIR)
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
# Stage 3 — Reorient to PIR
# -----------------------------------------------------------------------------

.PHONY: reorient-slurm reorient-local reorient-clean

reorient-slurm: setup
	@mkdir -p $(LOGS_DIR)
	sbatch $(REPO_ROOT)/slurm/reorient.sh

reorient-local: setup
	cd $(REPO_ROOT) && \
	PYTHONPATH=$(SRC_DIR) python -m verse_pipeline.reorient \
	    --unified_dir $(UNIFIED_DIR) \
	    --out_dir     $(CANONICAL_DIR) \
	    --workers     $(REORIENT_WORKERS)

reorient-clean:
	@echo "Removing $(CANONICAL_DIR) ..."
	rm -rf $(CANONICAL_DIR)
	@mkdir -p $(CANONICAL_DIR)

# -----------------------------------------------------------------------------
# Stage 4 — QC audit
# -----------------------------------------------------------------------------

.PHONY: qc-slurm qc-local qc-clean

qc-slurm: setup
	@mkdir -p $(LOGS_DIR)
	sbatch $(REPO_ROOT)/slurm/qc.sh

qc-local: setup
	cd $(REPO_ROOT) && \
	PYTHONPATH=$(SRC_DIR) python -m verse_pipeline.qc \
	    --input_dir $(CANONICAL_DIR) \
	    --out_dir   $(QC_DIR) \
	    --workers   $(QC_WORKERS)

qc-clean:
	@echo "Removing $(QC_DIR) (preserving $(RENDER_DIR)) ..."
	@find $(QC_DIR) -maxdepth 1 -type f -delete 2>/dev/null || true
	@mkdir -p $(QC_DIR)

# -----------------------------------------------------------------------------
# Stage 5 — Visual renders
# -----------------------------------------------------------------------------

.PHONY: renders-slurm renders-flagged renders-local renders-gallery renders-clean

renders-slurm: setup
	@mkdir -p $(LOGS_DIR)
	sbatch $(REPO_ROOT)/slurm/renders.sh

renders-flagged: setup
	@mkdir -p $(LOGS_DIR)
	@if [ ! -f $(QC_DIR)/qc_manifest.json ]; then \
	    echo "ERROR: $(QC_DIR)/qc_manifest.json not found.  Run 'make qc-slurm' first."; \
	    exit 1; \
	fi
	RENDER_FLAGS="--flagged_from $(QC_DIR)/qc_manifest.json" \
	    sbatch $(REPO_ROOT)/slurm/renders.sh

renders-local: setup
	cd $(REPO_ROOT) && \
	PYTHONPATH=$(SRC_DIR) python -m verse_pipeline.visualize \
	    --input_dir $(CANONICAL_DIR) \
	    --out_dir   $(RENDER_DIR) \
	    --workers   2 \
	    --limit     10
	$(MAKE) renders-gallery

renders-gallery:
	cd $(REPO_ROOT) && \
	PYTHONPATH=$(SRC_DIR) python -m verse_pipeline.render_gallery \
	    --renders_dir    $(RENDER_DIR) \
	    --qc_manifest    $(QC_DIR)/qc_manifest.json \
	    --unify_manifest $(UNIFIED_DIR)/unify_manifest.json \
	    --out_path       $(RENDER_DIR)/index.html

renders-clean:
	@echo "Removing $(RENDER_DIR) ..."
	rm -rf $(RENDER_DIR)
	@mkdir -p $(RENDER_DIR)

# -----------------------------------------------------------------------------
# Convenience: chain everything via SLURM dependencies
# -----------------------------------------------------------------------------

.PHONY: all-slurm
all-slurm: setup
	@mkdir -p $(LOGS_DIR)
	@DOWNLOAD_JOB=$$(sbatch --parsable $(REPO_ROOT)/slurm/download_raw.sh); \
	echo "submitted download as job $$DOWNLOAD_JOB"; \
	UNIFY_JOB=$$(sbatch --parsable --dependency=afterok:$$DOWNLOAD_JOB \
	    $(REPO_ROOT)/slurm/unify_iterations.sh); \
	echo "submitted unify as job $$UNIFY_JOB (after $$DOWNLOAD_JOB)"; \
	REORIENT_JOB=$$(sbatch --parsable --dependency=afterok:$$UNIFY_JOB \
	    $(REPO_ROOT)/slurm/reorient.sh); \
	echo "submitted reorient as job $$REORIENT_JOB (after $$UNIFY_JOB)"; \
	QC_JOB=$$(sbatch --parsable --dependency=afterok:$$REORIENT_JOB \
	    $(REPO_ROOT)/slurm/qc.sh); \
	echo "submitted qc as job $$QC_JOB (after $$REORIENT_JOB)"; \
	RENDER_JOB=$$(sbatch --parsable --dependency=afterok:$$QC_JOB \
	    $(REPO_ROOT)/slurm/renders.sh); \
	echo "submitted renders as job $$RENDER_JOB (after $$QC_JOB)"; \
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
	    jq -r '"  n_files=\(.n_files)  downloaded=\(.n_downloaded)  cached=\(.n_cached)  failed=\(.n_failed)"' \
	        $(RAW_DIR)/download_manifest.json 2>/dev/null \
	        || echo "  manifest exists, jq unavailable"; \
	else \
	    echo "  not run yet"; \
	fi
	@echo ""
	@echo "Stage 2 (unify):"
	@if [ -f $(UNIFIED_DIR)/unify_manifest.json ]; then \
	    jq -r '"  n_scans=\(.n_scans)  n_patients=\(.n_patients)  complete=\(.completeness.n_complete)"' \
	        $(UNIFIED_DIR)/unify_manifest.json 2>/dev/null \
	        || echo "  manifest exists, jq unavailable"; \
	else \
	    echo "  not run yet"; \
	fi
	@echo ""
	@echo "Stage 3 (reorient):"
	@if [ -f $(CANONICAL_DIR)/canonical_manifest.json ]; then \
	    jq -r '"  target_orientation=\(.target_orientation)  n_ok=\(.n_ok)  n_failed=\(.n_failed)"' \
	        $(CANONICAL_DIR)/canonical_manifest.json 2>/dev/null \
	        || echo "  manifest exists, jq unavailable"; \
	else \
	    echo "  not run yet"; \
	fi
	@echo ""
	@echo "Stage 4 (qc):"
	@if [ -f $(QC_DIR)/qc_manifest.json ]; then \
	    jq -r '.by_status | "  PASS=\(.PASS)  WARN=\(.WARN)  FAIL=\(.FAIL)  SKIP=\(.SKIP)"' \
	        $(QC_DIR)/qc_manifest.json 2>/dev/null \
	        || echo "  manifest exists, jq unavailable"; \
	else \
	    echo "  not run yet"; \
	fi
	@echo ""
	@echo "Stage 5 (renders):"
	@if [ -f $(RENDER_DIR)/renders_manifest.json ]; then \
	    jq -r '"  n_rendered=\(.n_rendered)  n_failed=\(.n_failed)"' \
	        $(RENDER_DIR)/renders_manifest.json 2>/dev/null \
	        || echo "  manifest exists, jq unavailable"; \
	    if [ -f $(RENDER_DIR)/index.html ]; then \
	        echo "  gallery: $(RENDER_DIR)/index.html"; \
	    fi; \
	else \
	    echo "  not run yet"; \
	fi
	@echo ""
	@echo "Recent SLURM logs:"
	@ls -lt $(LOGS_DIR)/*.err 2>/dev/null | head -5 | awk '{print "  " $$0}' \
	    || echo "  (no logs yet)"

.PHONY: clean-all
clean-all:
	@echo "WARNING: this will delete $(DATA_DIR) entirely."
	@echo "Press Ctrl-C in the next 5 seconds to abort..."
	@sleep 5
	rm -rf $(DATA_DIR)
	@echo "$(DATA_DIR) removed."
