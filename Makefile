# =============================================================================
# VerSeFusion — Makefile
#
# Pipeline order:
#   download → unify → reorient → qc → renders → veridah → veridah-renders
#            → audit (LSTV/TLTV) → orient (verify PIR) → manifest
#            → splits (5-fold CV) → hf-stage / hf-upload
#
# data/raw/                OSF-fetched MICCAI/BIDS files (gitignored, ~60GB)
# data/unified/            Per-scan dirs with symlinks to raw
# data/canonical/          Reoriented to PIR; real NIfTI files (~18GB)
# data/qc/                 QC manifest from canonical
# data/qc/renders/         Per-scan PNGs + HTML gallery
# data/corrected/          PIR + VERIDAH label corrections (~25 subjects)
# data/corrected/renders/  Before/after PNGs + gallery for corrected subjects
# data/lstv/               LSTV / TLTV audit (manifest + CSV)
# data/orientation/        Per-scan PIR verification
# data/hf_staging/         Staged for HuggingFace push
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
CORRECTED_DIR := $(DATA_DIR)/corrected
CORRECTED_RENDER_DIR := $(CORRECTED_DIR)/renders
LSTV_DIR      := $(DATA_DIR)/lstv
ORIENT_DIR    := $(DATA_DIR)/orientation
MANIFEST_DIR  := $(DATA_DIR)/manifest
HF_STAGING    := $(DATA_DIR)/hf_staging
LOGS_DIR      := $(REPO_ROOT)/logs

# Splits tunables (stage 10b)
SPLITS_N_FOLDS ?= 5
SPLITS_SEED    ?= 42

CONFIG_DIR    := $(REPO_ROOT)/configs
DEMOGRAPHICS  := $(CONFIG_DIR)/verse_demographics.csv
VERIDAH_CSV   := $(CONFIG_DIR)/veridah_corrections.csv

CONTAINERS_DIR := $(REPO_ROOT)/containers
CONTAINER_SIF  := $(CONTAINERS_DIR)/versefusion.sif
CONTAINER_DEF  := $(CONTAINERS_DIR)/versefusion.def

SRC_DIR       := $(REPO_ROOT)/src

DOWNLOAD_WORKERS  ?= 8
REORIENT_WORKERS  ?= 8
QC_WORKERS        ?= 8
RENDER_WORKERS    ?= 8
RENDER_DPI        ?= 80
VERIDAH_WORKERS   ?= 8
AUDIT_WORKERS     ?= 8

# Hardcoded HuggingFace repos. Override at the command line if needed:
#   make hf-export-slurm HF_REPO_ID=other-account/repo
HF_REPO_ID            ?= gregoryschwingmdphd/VerseFusion
HF_SAMPLE_REPO_ID     ?= gregoryschwingmdphd/VerseFusion-Sample
HF_SAMPLE_N           ?= 10
HF_DATASET_PRETTY_NAME ?= VerSeFusion

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
	@echo "    renders-local      Smoke test: render 10 scans interactively"
	@echo "    renders-gallery    Rebuild HTML index from existing PNGs"
	@echo "    renders-clean      Wipe data/qc/renders/"
	@echo ""
	@echo "  Stage 6 — VERIDAH label corrections (data/canonical/ → data/corrected/):"
	@echo "    veridah-slurm      Apply Möller 2026 corrections"
	@echo "    veridah-local      Run interactively"
	@echo "    veridah-clean      Wipe data/corrected/ (keeps renders subdir)"
	@echo ""
	@echo "  Stage 7 — VERIDAH before/after renders:"
	@echo "    veridah-renders-slurm        Submit SLURM job"
	@echo "    veridah-renders-local        Run interactively"
	@echo "    veridah-renders-advisories   Also render advisory-only cases"
	@echo "    veridah-renders-clean        Wipe data/corrected/renders/"
	@echo ""
	@echo "  Stage 8 — LSTV / TLTV audit (data/corrected/ → data/lstv/):"
	@echo "    audit-slurm        Submit SLURM job"
	@echo "    audit-local        Run interactively"
	@echo "    audit-canonical    Audit data/canonical/ for pre-VERIDAH stats"
	@echo "    audit-clean        Wipe data/lstv/"
	@echo ""
	@echo "  Stage 9 — Orientation audit (verify all scans are PIR):"
	@echo "    orient-slurm       Submit SLURM job"
	@echo "    orient-local       Run interactively"
	@echo "    orient-strict      Run with --strict (non-zero exit on failure)"
	@echo "    orient-clean       Wipe data/orientation/"
	@echo ""
	@echo "  Stage 10a — Master manifest:"
	@echo "    manifest-slurm     Submit SLURM job (~1 min)"
	@echo "    manifest-local     Run interactively"
	@echo "    manifest-clean     Wipe data/manifest/"
	@echo ""
	@echo "  Stage 10b — 5-fold stratified CV splits (patient-level, by lstv_class):"
	@echo "    splits-slurm       Submit SLURM job (~30 sec)"
	@echo "    splits-local       Run interactively"
	@echo "    Override: SPLITS_N_FOLDS=10, SPLITS_SEED=123"
	@echo ""
	@echo "  Stage 11 — HuggingFace export:"
	@echo "    Repos (hardcoded):"
	@echo "      full:    $(HF_REPO_ID)"
	@echo "      sample:  $(HF_SAMPLE_REPO_ID)   (top $(HF_SAMPLE_N) by label count)"
	@echo "    hf-stage-slurm     Stage both repos via SLURM, NO upload"
	@echo "    hf-stage-local     Stage both repos interactively, NO upload (quick smoke test)"
	@echo "    hf-export-slurm    Stage + upload BOTH via SLURM. Usage:"
	@echo "                         HF_TOKEN=hf_xxx make hf-export-slurm"
	@echo "    hf-clean           Wipe data/hf_staging/ and data/hf_staging_sample/"
	@echo "    Override: HF_REPO_ID, HF_SAMPLE_REPO_ID, HF_SAMPLE_N (0=skip sample),"
	@echo "              HF_PUBLIC=1, HF_DATASET_PRETTY_NAME"
	@echo ""
	@echo "  Convenience:"
	@echo "    all-slurm          Chain stages 1-8 via --dependency=afterok"
	@echo "    status             Show what's been produced so far"
	@echo "    clean-all          Wipe data/ entirely (DESTRUCTIVE)"
	@echo ""
	@echo "  Tunables (override on command line):"
	@echo "    DOWNLOAD_WORKERS=$(DOWNLOAD_WORKERS)  REORIENT_WORKERS=$(REORIENT_WORKERS)"
	@echo "    QC_WORKERS=$(QC_WORKERS)        RENDER_WORKERS=$(RENDER_WORKERS)"
	@echo "    VERIDAH_WORKERS=$(VERIDAH_WORKERS)    AUDIT_WORKERS=$(AUDIT_WORKERS)"
	@echo "    RENDER_DPI=$(RENDER_DPI)"

# -----------------------------------------------------------------------------
# Setup
# -----------------------------------------------------------------------------

.PHONY: setup
setup:
	@mkdir -p $(RAW_DIR) $(UNIFIED_DIR) $(CANONICAL_DIR) $(QC_DIR) $(RENDER_DIR) \
	          $(CORRECTED_DIR) $(CORRECTED_RENDER_DIR) $(LSTV_DIR) \
	          $(ORIENT_DIR) $(MANIFEST_DIR) $(HF_STAGING) $(LOGS_DIR)
	@if [ ! -f $(DEMOGRAPHICS) ]; then \
	    echo "ERROR: $(DEMOGRAPHICS) not found."; \
	    exit 1; \
	fi
	@echo "Setup OK. Dirs created, demographics found."

.PHONY: container
container:
	@mkdir -p $(CONTAINERS_DIR)
	@if [ ! -f $(CONTAINER_DEF) ]; then \
	    echo "ERROR: $(CONTAINER_DEF) not found."; \
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
# Stage 3 — Reorient
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
# Stage 4 — QC
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
# Stage 5 — Renders
# -----------------------------------------------------------------------------

.PHONY: renders-slurm renders-flagged renders-local renders-gallery renders-clean

renders-slurm: setup
	@mkdir -p $(LOGS_DIR)
	sbatch $(REPO_ROOT)/slurm/renders.sh

renders-flagged: setup
	@mkdir -p $(LOGS_DIR)
	@if [ ! -f $(QC_DIR)/qc_manifest.json ]; then \
	    echo "ERROR: $(QC_DIR)/qc_manifest.json not found. Run 'make qc-slurm' first."; \
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
# Stage 6 — VERIDAH label corrections
# -----------------------------------------------------------------------------

.PHONY: veridah-slurm veridah-local veridah-clean

veridah-slurm: setup
	@mkdir -p $(LOGS_DIR)
	@if [ ! -f $(VERIDAH_CSV) ]; then \
	    echo "ERROR: $(VERIDAH_CSV) not found."; \
	    exit 1; \
	fi
	sbatch $(REPO_ROOT)/slurm/veridah.sh

veridah-local: setup
	cd $(REPO_ROOT) && \
	PYTHONPATH=$(SRC_DIR) python -m verse_pipeline.veridah \
	    --in_dir          $(CANONICAL_DIR) \
	    --out_dir         $(CORRECTED_DIR) \
	    --corrections_csv $(VERIDAH_CSV) \
	    --workers         $(VERIDAH_WORKERS)

veridah-clean:
	@echo "Removing $(CORRECTED_DIR) (preserving renders) ..."
	@find $(CORRECTED_DIR) -maxdepth 1 -type f -delete 2>/dev/null || true
	@find $(CORRECTED_DIR) -maxdepth 1 -type d -name "scan-*" -exec rm -rf {} + 2>/dev/null || true
	@mkdir -p $(CORRECTED_DIR)

# -----------------------------------------------------------------------------
# Stage 7 — VERIDAH before/after renders
# -----------------------------------------------------------------------------

.PHONY: veridah-renders-slurm veridah-renders-local veridah-renders-advisories veridah-renders-clean

veridah-renders-slurm: setup
	@mkdir -p $(LOGS_DIR)
	@if [ ! -f $(CORRECTED_DIR)/veridah_manifest.json ]; then \
	    echo "ERROR: $(CORRECTED_DIR)/veridah_manifest.json not found. Run 'make veridah-slurm' first."; \
	    exit 1; \
	fi
	sbatch $(REPO_ROOT)/slurm/veridah_renders.sh

veridah-renders-local: setup
	cd $(REPO_ROOT) && \
	PYTHONPATH=$(SRC_DIR) python -m verse_pipeline.visualize_corrections \
	    --canonical_dir $(CANONICAL_DIR) \
	    --corrected_dir $(CORRECTED_DIR) \
	    --out_dir       $(CORRECTED_RENDER_DIR) \
	    --workers       2 \
	    --dpi           $(RENDER_DPI)

veridah-renders-advisories: setup
	@mkdir -p $(LOGS_DIR)
	@if [ ! -f $(CORRECTED_DIR)/veridah_manifest.json ]; then \
	    echo "ERROR: $(CORRECTED_DIR)/veridah_manifest.json not found. Run 'make veridah-slurm' first."; \
	    exit 1; \
	fi
	VERIDAH_RENDER_FLAGS="--include_advisories" \
	    sbatch $(REPO_ROOT)/slurm/veridah_renders.sh

veridah-renders-clean:
	@echo "Removing $(CORRECTED_RENDER_DIR) ..."
	rm -rf $(CORRECTED_RENDER_DIR)
	@mkdir -p $(CORRECTED_RENDER_DIR)

# -----------------------------------------------------------------------------
# Stage 8 — LSTV / TLTV label-based audit
# -----------------------------------------------------------------------------

.PHONY: audit-slurm audit-local audit-canonical audit-clean

audit-slurm: setup
	@mkdir -p $(LOGS_DIR)
	sbatch $(REPO_ROOT)/slurm/lstv_audit.sh

audit-local: setup
	cd $(REPO_ROOT) && \
	PYTHONPATH=$(SRC_DIR) python -m verse_pipeline.lstv_audit \
	    --input_dir $(CORRECTED_DIR) \
	    --out_dir   $(LSTV_DIR) \
	    --workers   $(AUDIT_WORKERS)

audit-canonical: setup
	cd $(REPO_ROOT) && \
	PYTHONPATH=$(SRC_DIR) python -m verse_pipeline.lstv_audit \
	    --input_dir $(CANONICAL_DIR) \
	    --out_dir   $(LSTV_DIR)/canonical \
	    --workers   $(AUDIT_WORKERS)

audit-clean:
	@echo "Removing $(LSTV_DIR) ..."
	rm -rf $(LSTV_DIR)
	@mkdir -p $(LSTV_DIR)

# -----------------------------------------------------------------------------
# Stage 9 — Orientation audit (PIR verification)
# -----------------------------------------------------------------------------

.PHONY: orient-slurm orient-local orient-strict orient-clean

orient-slurm: setup
	@mkdir -p $(LOGS_DIR)
	sbatch $(REPO_ROOT)/slurm/orientation_audit.sh

orient-local: setup
	cd $(REPO_ROOT) && \
	PYTHONPATH=$(SRC_DIR) python -m verse_pipeline.orientation_audit \
	    --input_dir $(CANONICAL_DIR) \
	    --out_dir   $(ORIENT_DIR) \
	    --workers   $(AUDIT_WORKERS)

# Fail the SLURM job if any scan isn't PIR (non-zero exit code)
orient-strict: setup
	@mkdir -p $(LOGS_DIR)
	ORIENT_FLAGS="--strict" sbatch $(REPO_ROOT)/slurm/orientation_audit.sh

orient-clean:
	@echo "Removing $(ORIENT_DIR) ..."
	rm -rf $(ORIENT_DIR)
	@mkdir -p $(ORIENT_DIR)

# -----------------------------------------------------------------------------
# Stage 10a — Master manifest (per-scan metadata aggregation)
# -----------------------------------------------------------------------------

.PHONY: manifest-slurm manifest-local manifest-clean

manifest-slurm: setup
	@mkdir -p $(LOGS_DIR) $(MANIFEST_DIR)
	sbatch $(REPO_ROOT)/slurm/manifest_builder.sh

manifest-local: setup
	@mkdir -p $(MANIFEST_DIR)
	cd $(REPO_ROOT) && \
	PYTHONPATH=$(SRC_DIR) python -m verse_pipeline.manifest_builder \
	    --canonical_dir  $(CANONICAL_DIR) \
	    --corrected_dir  $(CORRECTED_DIR) \
	    --unify_manifest $(UNIFIED_DIR)/unify_manifest.json \
	    --lstv_audit     $(LSTV_DIR)/lstv_audit_manifest.json \
	    --output_dir     $(MANIFEST_DIR)

manifest-clean:
	@echo "Removing $(MANIFEST_DIR) ..."
	rm -rf $(MANIFEST_DIR)
	@mkdir -p $(MANIFEST_DIR)

# -----------------------------------------------------------------------------
# Stage 10b — 5-fold stratified CV splits (patient-level, by lstv_class)
# -----------------------------------------------------------------------------

.PHONY: splits-slurm splits-local

splits-slurm: setup
	@mkdir -p $(LOGS_DIR) $(MANIFEST_DIR)
	@if [[ ! -f "$(MANIFEST_DIR)/manifest.csv" ]]; then \
	    echo "ERROR: $(MANIFEST_DIR)/manifest.csv not found.  Run 'make manifest-slurm' first."; \
	    exit 1; \
	fi
	sbatch \
	    --export=ALL,SPLITS_N_FOLDS="$(SPLITS_N_FOLDS)",SPLITS_SEED="$(SPLITS_SEED)" \
	    $(REPO_ROOT)/slurm/splits_builder.sh

splits-local: setup
	@mkdir -p $(MANIFEST_DIR)
	cd $(REPO_ROOT) && \
	PYTHONPATH=$(SRC_DIR) python -m verse_pipeline.splits_builder \
	    --manifest_csv $(MANIFEST_DIR)/manifest.csv \
	    --out          $(MANIFEST_DIR)/splits_5fold.json \
	    --n_folds      $(SPLITS_N_FOLDS) \
	    --seed         $(SPLITS_SEED)

# -----------------------------------------------------------------------------
# Stage 11 — HuggingFace dataset export
# -----------------------------------------------------------------------------

.PHONY: hf-stage-slurm hf-stage-local hf-export-slurm hf-clean

# Stage both repos via SLURM, NO upload.  No HF_TOKEN required.
# Pass overrides via env: HF_SAMPLE_N=20 make hf-stage-slurm
hf-stage-slurm: setup
	@mkdir -p $(LOGS_DIR)
	sbatch \
	    --export=ALL,HF_REPO_ID="$(HF_REPO_ID)",HF_SAMPLE_REPO_ID="$(HF_SAMPLE_REPO_ID)",HF_SAMPLE_N="$(HF_SAMPLE_N)",HF_DATASET_PRETTY_NAME="$(HF_DATASET_PRETTY_NAME)" \
	    $(REPO_ROOT)/slurm/hf_stage.sh

# Stage interactively (no SLURM, no upload).  Useful for quick smoke tests
# on small datasets or when you want immediate feedback.
hf-stage-local: setup
	cd $(REPO_ROOT) && \
	PYTHONPATH=$(SRC_DIR) python -m verse_pipeline.hf_export \
	    --canonical_dir         $(CANONICAL_DIR) \
	    --corrected_dir         $(CORRECTED_DIR) \
	    --unify_manifest        $(UNIFIED_DIR)/unify_manifest.json \
	    --staging_dir           $(HF_STAGING) \
	    --repo_id               $(HF_REPO_ID) \
	    --dataset_pretty_name   $(HF_DATASET_PRETTY_NAME) \
	    --sample_n              $(HF_SAMPLE_N) \
	    --sample_repo_id        $(HF_SAMPLE_REPO_ID) \
	    --sample_staging_dir    $(HF_STAGING)_sample \
	    --no_upload \
	    --workers               8

# Stage + upload BOTH full dataset and sample via SLURM. Single command, just needs the token:
#   HF_TOKEN=hf_xxx make hf-export-slurm
hf-export-slurm: setup
	@mkdir -p $(LOGS_DIR)
	@if [[ -z "$$HF_TOKEN" ]] && [[ ! -f "$$HOME/.cache/huggingface/token" ]]; then \
	    echo "ERROR: HF_TOKEN env var not set, and no cached token at ~/.cache/huggingface/token"; \
	    echo "       Usage:  HF_TOKEN=hf_xxx make hf-export-slurm"; \
	    echo "       Or run: huggingface-cli login"; \
	    exit 1; \
	fi
	sbatch \
	    --export=ALL,HF_REPO_ID="$(HF_REPO_ID)",HF_SAMPLE_REPO_ID="$(HF_SAMPLE_REPO_ID)",HF_SAMPLE_N="$(HF_SAMPLE_N)",HF_TOKEN="$$HF_TOKEN",HF_PUBLIC="$${HF_PUBLIC:-0}",HF_DATASET_PRETTY_NAME="$(HF_DATASET_PRETTY_NAME)" \
	    $(REPO_ROOT)/slurm/hf_export.sh

hf-clean:
	@echo "Removing $(HF_STAGING) and $(HF_STAGING)_sample ..."
	rm -rf $(HF_STAGING) $(HF_STAGING)_sample
	@mkdir -p $(HF_STAGING) $(HF_STAGING)_sample

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
	VERIDAH_JOB=$$(sbatch --parsable --dependency=afterok:$$QC_JOB \
	    $(REPO_ROOT)/slurm/veridah.sh); \
	echo "submitted veridah as job $$VERIDAH_JOB (after $$QC_JOB)"; \
	VR_JOB=$$(sbatch --parsable --dependency=afterok:$$VERIDAH_JOB \
	    $(REPO_ROOT)/slurm/veridah_renders.sh); \
	echo "submitted veridah-renders as job $$VR_JOB (after $$VERIDAH_JOB)"; \
	AUDIT_JOB=$$(sbatch --parsable --dependency=afterok:$$VERIDAH_JOB \
	    $(REPO_ROOT)/slurm/lstv_audit.sh); \
	echo "submitted audit as job $$AUDIT_JOB (after $$VERIDAH_JOB)"; \
	ORIENT_JOB=$$(sbatch --parsable --dependency=afterok:$$REORIENT_JOB \
	    $(REPO_ROOT)/slurm/orientation_audit.sh); \
	echo "submitted orient-audit as job $$ORIENT_JOB (after $$REORIENT_JOB)"; \
	echo ""; \
	echo "HF upload (Stage 10) is deliberately NOT chained; run 'make hf-upload' when ready."; \
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
	@echo "Stage 6 (veridah):"
	@if [ -f $(CORRECTED_DIR)/veridah_manifest.json ]; then \
	    jq -r '"  n_corrected=\(.n_corrected)  n_passthrough=\(.n_passthrough)  by_type=\(.by_correction_type)"' \
	        $(CORRECTED_DIR)/veridah_manifest.json 2>/dev/null \
	        || echo "  manifest exists, jq unavailable"; \
	else \
	    echo "  not run yet"; \
	fi
	@echo ""
	@echo "Stage 7 (veridah-renders):"
	@if [ -f $(CORRECTED_RENDER_DIR)/renders_manifest.json ]; then \
	    jq -r '"  n_rendered=\(.n_rendered)  n_failed=\(.n_failed)"' \
	        $(CORRECTED_RENDER_DIR)/renders_manifest.json 2>/dev/null \
	        || echo "  manifest exists, jq unavailable"; \
	    if [ -f $(CORRECTED_RENDER_DIR)/index.html ]; then \
	        echo "  gallery: $(CORRECTED_RENDER_DIR)/index.html"; \
	    fi; \
	else \
	    echo "  not run yet"; \
	fi
	@echo ""
	@echo "Stage 8 (lstv-audit):"
	@if [ -f $(LSTV_DIR)/lstv_audit_manifest.json ]; then \
	    jq -r '.summary.headline_counts | "  has_L6=\(.has_L6_label)  lacks_L5_LSJ_in_FOV=\(.lacks_L5_label_with_LSJ_in_FOV)  has_T13=\(.has_T13_label)  lacks_T12_TLJ_in_FOV=\(.lacks_T12_label_with_TLJ_in_FOV)"' \
	        $(LSTV_DIR)/lstv_audit_manifest.json 2>/dev/null \
	        || echo "  manifest exists, jq unavailable"; \
	    jq -r '.summary.lstv_class_counts | "  LSTV: \(.)"' \
	        $(LSTV_DIR)/lstv_audit_manifest.json 2>/dev/null || true; \
	    jq -r '.summary.tltv_class_counts | "  TLTV: \(.)"' \
	        $(LSTV_DIR)/lstv_audit_manifest.json 2>/dev/null || true; \
	else \
	    echo "  not run yet"; \
	fi
	@echo ""
	@echo "Stage 9 (orientation-audit):"
	@if [ -f $(ORIENT_DIR)/orientation_audit.json ]; then \
	    jq -r '.summary | "  total=\(.n_scans)  pass=\(.n_passes_PIR)  ct_not_pir=\(.n_ct_not_PIR)  msk_not_pir=\(.n_msk_not_PIR)  shape_mismatch=\(.n_shape_mismatch)"' \
	        $(ORIENT_DIR)/orientation_audit.json 2>/dev/null \
	        || echo "  manifest exists, jq unavailable"; \
	else \
	    echo "  not run yet"; \
	fi
	@echo ""
	@echo "Stage 10 (hf-staging):"
	@if [ -d $(HF_STAGING)/scans ]; then \
	    echo "  staging_dir: $(HF_STAGING)"; \
	    echo "  size:        $$(du -sh $(HF_STAGING) 2>/dev/null | cut -f1)"; \
	    echo "  n_subjects:  $$(find $(HF_STAGING)/scans -maxdepth 1 -mindepth 1 -type d | wc -l)"; \
	else \
	    echo "  not staged yet"; \
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
