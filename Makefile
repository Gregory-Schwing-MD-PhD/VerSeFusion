# =============================================================================
# VerSeFusion — Makefile
#
# Each target has a local variant (runs in current shell) and a -slurm variant
# (submits the equivalent script under slurm/).  All paths are configurable
# via env or configs/default.env (sourced automatically).
# =============================================================================

SHELL := /bin/bash
.SHELLFLAGS := -eu -o pipefail -c
.ONESHELL:

# --- load non-secret defaults -------------------------------------------------
ifneq (,$(wildcard configs/default.env))
    include configs/default.env
    export
endif

# --- optional secrets layer (gitignored) --------------------------------------
ifneq (,$(wildcard env.local))
    include env.local
    export
endif

# --- defaults (may be overridden via env or configs/default.env) --------------
DATA_DIR       ?= $(CURDIR)/data
RAW_DIR        ?= $(DATA_DIR)/raw
UNIFIED_DIR    ?= $(DATA_DIR)/unified
REORIENTED_DIR ?= $(DATA_DIR)/reoriented
HF_DIR         ?= $(DATA_DIR)/hf_export
LOG_DIR        ?= $(CURDIR)/logs

PYTHON         ?= python3
PIP            ?= pip

.DEFAULT_GOAL := help

# =============================================================================
# meta
# =============================================================================
.PHONY: help
help:  ## Show this help.
	@echo ""
	@echo "VerSeFusion — make targets"
	@echo "=========================="
	@awk 'BEGIN{FS=":.*?## "} /^[a-zA-Z_-]+:.*?## / {printf "  \033[36m%-22s\033[0m %s\n", $$1, $$2}' $(MAKEFILE_LIST)
	@echo ""
	@echo "Configurable paths:"
	@echo "  DATA_DIR       = $(DATA_DIR)"
	@echo "  RAW_DIR        = $(RAW_DIR)"
	@echo "  UNIFIED_DIR    = $(UNIFIED_DIR)"
	@echo "  REORIENTED_DIR = $(REORIENTED_DIR)"
	@echo "  HF_DIR         = $(HF_DIR)"
	@echo ""

.PHONY: install
install:  ## pip install -e .[dev]
	$(PIP) install -e ".[dev]"

.PHONY: clean
clean:  ## Remove __pycache__ and build artefacts (preserves data/).
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	rm -rf build dist *.egg-info .pytest_cache .ruff_cache .mypy_cache
	@echo "Clean done.  data/ preserved."

.PHONY: deep-clean
deep-clean: clean  ## Remove pipeline outputs too (data/, logs/, work/).
	rm -rf $(DATA_DIR) $(LOG_DIR) work/ .nextflow*

# =============================================================================
# stage 0 — container pull (HPC)
# =============================================================================
.PHONY: hpc-pull
hpc-pull:  ## Pull a Docker Hub image -> containers/versefusion.sif on Warrior HPC.
	mkdir -p containers logs
	bash scripts/hpc_pull.sh

.PHONY: hpc-pull-slurm
hpc-pull-slurm:  ## Submit hpc-pull as a SLURM job.
	mkdir -p $(LOG_DIR)
	sbatch slurm/hpc_pull.sh

# =============================================================================
# stage 1 — download
# =============================================================================
.PHONY: download
download:  ## Pull the six VerSe S3 zips (resumable, sha256 verified).
	mkdir -p $(RAW_DIR) $(LOG_DIR)
	$(PYTHON) -m verse_pipeline.download --out_dir $(RAW_DIR)

.PHONY: download-slurm
download-slurm:  ## Submit download as a SLURM job.
	mkdir -p $(LOG_DIR)
	sbatch slurm/download_raw.sh

# =============================================================================
# stage 2 — unify
# =============================================================================
.PHONY: unify
unify:  ## Merge VerSe19+VerSe20, dedup 105-image overlap (VerSe20 wins).
	mkdir -p $(UNIFIED_DIR) $(LOG_DIR)
	$(PYTHON) -m verse_pipeline.unify \
	    --raw_dir $(RAW_DIR) \
	    --out_dir $(UNIFIED_DIR)

.PHONY: unify-slurm
unify-slurm:
	sbatch slurm/unify_iterations.sh

# =============================================================================
# stage 3 — reorient to PIR
# =============================================================================
.PHONY: reorient
reorient:  ## Reorient every CT + mask to PIR (matches CTSpinoPelvic1K).
	mkdir -p $(REORIENTED_DIR) $(LOG_DIR)
	$(PYTHON) -m verse_pipeline.reorient \
	    --in_dir $(UNIFIED_DIR) \
	    --out_dir $(REORIENTED_DIR)

.PHONY: reorient-slurm
reorient-slurm:
	sbatch slurm/reorient_pir.sh

# =============================================================================
# stage 4 — manifest + LSTV audit
# =============================================================================
.PHONY: manifest
manifest:  ## Build placed_manifest.json with vertebra inventory + LSTV/T13 flags.
	$(PYTHON) -m verse_pipeline.manifest \
	    --in_dir $(REORIENTED_DIR) \
	    --out_path $(REORIENTED_DIR)/placed_manifest.json

.PHONY: manifest-slurm
manifest-slurm:
	sbatch slurm/build_manifest.sh

.PHONY: lstv-audit
lstv-audit:  ## Print LSTV / T13 / normal counts to stdout.
	$(PYTHON) -m verse_pipeline.lstv --audit \
	    --manifest $(REORIENTED_DIR)/placed_manifest.json

# =============================================================================
# stage 5 — splits
# =============================================================================
.PHONY: splits
splits:  ## 5-fold CV stratified by LSTV+T13.
	$(PYTHON) -m verse_pipeline.splits \
	    --manifest $(REORIENTED_DIR)/placed_manifest.json \
	    --out_dir $(REORIENTED_DIR)/splits

# =============================================================================
# stage 6 — HuggingFace export
# =============================================================================
.PHONY: hf-export
hf-export:  ## Build HuggingFace DatasetDict under data/hf_export/.
	mkdir -p $(HF_DIR)
	$(PYTHON) -m verse_pipeline.hf_export \
	    --in_dir $(REORIENTED_DIR) \
	    --out_dir $(HF_DIR)

.PHONY: hf-export-slurm
hf-export-slurm:
	sbatch slurm/hf_export.sh

# =============================================================================
# QA / utility
# =============================================================================
.PHONY: inventory
inventory:  ## Print per-source / per-split subject counts.
	$(PYTHON) -m verse_pipeline.cli_inventory \
	    --manifest $(REORIENTED_DIR)/placed_manifest.json

.PHONY: test
test:  ## Run pytest suite.
	$(PYTHON) -m pytest tests/ -v

.PHONY: lint
lint:  ## Run ruff + mypy.
	$(PYTHON) -m ruff check src/ tests/
	$(PYTHON) -m mypy src/

.PHONY: full-pipeline
full-pipeline: download unify reorient manifest splits hf-export  ## Run every stage end-to-end (local).
	@echo "Full pipeline complete.  Output at $(HF_DIR)"
