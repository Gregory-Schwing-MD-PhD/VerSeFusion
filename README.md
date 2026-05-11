# VerSeFusion

A reproducible pipeline to download, unify, dedup, and benchmark-export the
VerSe 2019 and VerSe 2020 CT vertebrae segmentation datasets as a single
curated corpus with explicit handling of enumeration anomalies (LSTV, T13)
and a crosswalk to the CTSpinoPelvic1K label scheme.

---

## Why this repo exists

VerSe19 and VerSe20 are published as two overlapping releases of the same
underlying cohort:

| Release  | Subjects | Annotated vertebrae | Notes                                              |
|----------|---------:|--------------------:|----------------------------------------------------|
| VerSe19  |      141 |               1 725 | Initial release.                                    |
| VerSe20  |      300 |               4 142 | Enriched for anatomical variants + fracture grades. |
| Overlap  |  105 imgs|                   - | series shared between VerSe19 and VerSe20           |
| Fused    |      355 |              ~4 505 | dedup'd, single canonical source per subject        |

The VerSe maintainers publish a BIDS-restructured form on S3 that is the
recommended download (the OSF mirrors carry the older MICCAI-challenge
schema).  This repo:

1. Pulls the six S3 zips reproducibly with checksum verification.
2. Unifies VerSe19 + VerSe20 into one subject-keyed tree, deduplicating
   the 105-image overlap by preferring the VerSe20 release (newer
   annotations, fracture grading included).
3. Re-orients every CT and mask to PIR to match the CTSpinoPelvic1K
   convention.
4. Builds a placed_manifest.json with per-subject metadata: vertebra
   inventory, LSTV/T13 flags, voxel spacing, FOV.
5. Stratifies a 5-fold CV split on the LSTV/T13/normal axis.
6. Crosswalks the VerSe label scheme (1-28) to the CTSpinoPelvic1K
   10-class scheme for direct external validation.
7. Exports a HuggingFace-compatible flat directory for downstream training.

## Primary downstream uses

- External validation of nnU-Net v2 / VERIDAH / SPINEPS / TotalSegmentator
  models trained on CTSpinoPelvic1K.
- Pretraining corpus for the nnU-Net v2 trainer.
- Standalone benchmark for vertebrae labelling and segmentation with
  LSTV-stratified evaluation.

## Repository layout

```
VerSeFusion/
|-- Makefile                  one-line entry points (see `make help`)
|-- README.md                 this file
|-- LICENSE                   MIT (code) / CC-BY-SA-4.0 (data, upstream)
|-- pyproject.toml
|-- requirements.txt
|-- configs/
|   |-- default.env           non-secret defaults (paths, version pins)
|   `-- label_scheme.yaml     VerSe -> CTSpinoPelvic1K crosswalk
|-- containers/
|   `-- README.md             SIF pull / reuse instructions
|-- slurm/                    Warrior-HPC SLURM wrappers
|   |-- _common.sh            shared env scrub + conda + singularity setup
|   |-- hpc_pull.sh           pull SIF into containers/
|   |-- download_raw.sh
|   |-- unify_iterations.sh
|   |-- reorient_pir.sh
|   |-- build_manifest.sh
|   |-- lstv_audit.sh
|   |-- make_splits.sh
|   `-- hf_export.sh
|-- nextflow/
|   |-- main.nf
|   `-- nextflow.config
|-- src/verse_pipeline/
|   |-- download.py           wget + sha256, resumable
|   |-- unify.py              VerSe19+20 dedup
|   |-- reorient.py           PIR reorientation
|   |-- manifest.py           placed_manifest.json builder
|   |-- lstv.py               LSTV / T13 detection
|   |-- splits.py             5-fold CV
|   |-- label_crosswalk.py    VerSe <-> CTSpinoPelvic1K
|   |-- hf_export.py          HuggingFace export
|   |-- cli_inventory.py      per-source/split subject counts
|   `-- utils/{bids,centroid_json,nifti}.py
|-- scripts/
|   |-- hpc_pull.sh           SIF pull worker
|   |-- inventory.py
|   `-- qc_overview.py
|-- tests/                    pytest suite (41 tests)
|-- data/                     gitignored - staging dir
`-- docs/
    |-- design.md
    |-- label_scheme.md
    `-- crosswalk.md
```

## Quick start

```bash
# 0.  one-time: pull a Singularity image into containers/versefusion.sif
make hpc-pull-slurm       # defaults to reusing the ctspinopelvic1k image

# 1.  pull the six S3 zips (~30 GB total, resumable)
make download-slurm

# 2.  unify VerSe19 + VerSe20, dedup the 105-image overlap
make unify-slurm

# 3.  reorient everything to PIR
make reorient-slurm

# 4.  build the manifest with LSTV/T13 flags
make manifest-slurm

# 5.  5-fold stratified splits
make splits

# 6.  HuggingFace flat-directory export
make hf-export-slurm

# audit at any time
make lstv-audit            # prints normal/lstv/t13/both counts
make inventory             # subjects per source/split/anomaly category
```

Every `make X-slurm` target submits through `slurm/<target>.sh` with the
right SBATCH directives.  The corresponding `make X` runs the same logic
locally on the calling host (useful for development; not recommended for
the 30 GB download).

PYTHONPATH is set automatically in both paths -- the Makefile exports
`$(CURDIR)/src` for local targets, and `slurm/_common.sh` exports
`PYTHONPATH` + `SINGULARITYENV_PYTHONPATH` for jobs running inside the
container.  No `pip install -e .` is required.

## Label scheme

VerSe ships a 28-class vertebra index (see `docs/label_scheme.md`).  Sacrum
and coccyx labels (26, 27) exist in the *scheme* but are not annotated in
the dataset -- VerSe is a vertebrae-only corpus.  Hips and pelvis are
absent entirely.

The crosswalk to CTSpinoPelvic1K's 10-class scheme is therefore lossy in
one direction (VerSe -> CTSpinoPelvic1K drops thoracic and cervical) and
incomplete in the other (CTSpinoPelvic1K -> VerSe has no hip labels to
map to).  See `docs/crosswalk.md`.

| VerSe label | Region                       | CTSpinoPelvic1K equivalent |
|------------:|------------------------------|---------------------------:|
| 1-7         | C1-C7                        |     - (not in CTSPP1K)     |
| 8-19        | T1-T12                       |     -                      |
| 20-24       | L1-L5                        |          1-5               |
| 25          | L6 (lumbarized LSTV)         |           6                |
| 26          | sacrum (unannotated in VerSe)|           7                |
| 27          | coccyx (unannotated)         |     -                      |
| 28          | T13 (extra thoracic)         |     -                      |

## Citation

If you use this pipeline, please cite the original VerSe papers (Sekuboyina
2021, Loffler 2020, Liebl 2021).  The data license is CC-BY-SA 4.0 and
attaches to any derivative exports this repo produces.

## License

- Code: MIT (see `LICENSE`).
- Data exports (`data/unified/`, `data/reoriented/`, `data/hf_export/`):
  inherit CC-BY-SA 4.0 from upstream VerSe.

## Acknowledgements

This repo would not exist without the VerSe team at TUM (Sekuboyina, Liebl,
Loffler, Kirschke et al.) who built and curated the original dataset, and
without Hendrik Moller's VERIDAH work which clarified the role of sequence
prediction in resolving LSTV labelling ambiguity.
