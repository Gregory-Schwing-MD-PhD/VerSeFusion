# VerSeFusion

A reproducible pipeline to download, unify, deduplicate, **label-correct**, and
benchmark-export the **VerSe 2019** and **VerSe 2020** CT vertebrae
segmentation datasets as a single curated corpus with explicit handling of
enumeration anomalies (LSTV, T13) and a crosswalk to the
[CTSpinoPelvic1K](https://github.com/gschwing/CTSpinoPelvic1K) label scheme.

---

## What this repo gives you

Running the full pipeline produces **355 subjects** (matches the canonical
TUM count of 374 scans from 355 patients), drawn from two upstream releases
that ship with disjoint subject-ID ranges:

| Source           | Subjects | Notes                                              |
|------------------|---------:|----------------------------------------------------|
| VerSe19          |      141 | `sub-verse{001..400}` range                        |
| VerSe20          |      214 | `sub-verse{500..800}` + `sub-gl{001..500}` ranges  |
| **Total unique** |  **355** | dedup'd, single canonical scan per subject         |

Each subject carries: a CT volume, a vertebra segmentation mask (28-class
VerSe label scheme), and centroid coordinates in voxel space.  All three are
reoriented to **PIR** (Posterior / Inferior / Right) to match the
CTSpinoPelvic1K convention.

### What's different about VerSeFusion vs. just downloading VerSe directly

1. **OSF as the canonical source.**  The S3 endpoint cited in upstream
   documentation (`s3.bonescreen.de`) has been offline since May 2024;
   VerSeFusion downloads instead from OSF nodes `jtfa5` (VerSe19) and `4skx2`
   (VerSe20), which host the same BIDS-restructured form.  See
   [`docs/design.md`](docs/design.md).

2. **VERIDAH manual label corrections applied.**  Moeller et al. (2026)
   manually reviewed VerSe and published 25 label-correction entries (mostly
   T13 cases where the supernumerary thoracic vertebra was annotated as L1
   with subsequent lumbar labels shifted down).  VerSeFusion incorporates
   these via the `correct` stage (`configs/veridah_corrections.csv`).
   To our knowledge, this is the first publicly available distribution of
   VerSe with VERIDAH-validated ground truth.

3. **PIR reorientation throughout.**  CT, mask, and centroid voxel
   coordinates are transformed in lockstep so models trained on
   CTSpinoPelvic1K can be evaluated on VerSeFusion without an
   inference-time reorientation step.

4. **Stratified 5-fold CV splits** on the LSTV/T13/normal axis so reported
   DSC can be sliced by anomaly category at evaluation time.

5. **HuggingFace-compatible flat export** for downstream training and
   benchmarking.

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
|   |-- default.env           non-secret defaults (single-word scalars only)
|   |-- label_scheme.yaml     VerSe -> CTSpinoPelvic1K crosswalk
|   `-- veridah_corrections.csv   Moeller 2026 manual corrections (25 rows)
|-- containers/
|   `-- README.md             SIF pull / reuse instructions
|-- slurm/                    Warrior-HPC SLURM wrappers
|   |-- _common.sh            shared env scrub + conda + singularity setup
|   |-- hpc_pull.sh           pull SIF into containers/
|   |-- download_raw.sh       stage 1
|   |-- unify_iterations.sh   stage 2
|   |-- apply_corrections.sh  stage 2.5 (VERIDAH)
|   |-- reorient_pir.sh       stage 3 (auto-prefers corrected/)
|   |-- build_manifest.sh     stage 4
|   |-- make_splits.sh        stage 5
|   `-- hf_export.sh          stage 6
|-- nextflow/
|   |-- main.nf
|   `-- nextflow.config
|-- src/verse_pipeline/
|   |-- download.py           OSF REST API + throttling + retry-on-429
|   |-- unify.py              VerSe19+20 merge with dedup
|   |-- veridah.py            apply Moeller 2026 manual corrections
|   |-- reorient.py           PIR reorientation (CT + mask + centroids)
|   |-- manifest.py           placed_manifest.json builder
|   |-- lstv.py               LSTV / T13 detection from centroid JSON
|   |-- splits.py             5-fold CV
|   |-- label_crosswalk.py    VerSe <-> CTSpinoPelvic1K
|   |-- hf_export.py          HuggingFace flat-directory export
|   |-- cli_inventory.py      per-source / per-anomaly counts
|   `-- utils/{bids,centroid_json,nifti}.py
|-- scripts/
|   |-- hpc_pull.sh           SIF pull worker
|   |-- inventory.py
|   `-- qc_overview.py
|-- tests/                    pytest suite (41 tests)
|-- data/                     gitignored - staging dir
|   |-- raw/{verse19,verse20}/    OSF-mirrored BIDS layout
|   |-- unified/sub-verseNNN/     post-dedup
|   |-- corrected/sub-verseNNN/   post-VERIDAH (mostly symlinks; ~15 materialized)
|   |-- reoriented/sub-verseNNN/  PIR throughout
|   `-- hf_export/                ct/, labels/, centroids/, splits/
`-- docs/
    |-- design.md
    |-- label_scheme.md
    `-- crosswalk.md
```

## Quick start

```bash
# 0.  one-time: pull a Singularity image into containers/versefusion.sif
make hpc-pull-slurm       # defaults to reusing the ctspinopelvic1k image

# 1.  pull all VerSe files from OSF (~1465 files, ~25 GB, resumable)
make download-slurm

# 2.  unify VerSe19 + VerSe20 into one subject-keyed tree
make unify-slurm

# 2.5  apply Moeller 2026 manual label corrections
make correct-slurm

# 3.  reorient everything to PIR (auto-prefers corrected/ from step 2.5)
make reorient-slurm

# 4.  build the manifest with LSTV/T13 flags
make manifest-slurm

# 5.  5-fold stratified splits
make splits

# 6.  HuggingFace flat-directory export
make hf-export-slurm

# audit at any time
make lstv-audit            # prints normal/lstv/t13/both counts
make inventory             # subjects per source / split / anomaly category
```

Every `make X-slurm` target submits through `slurm/<target>.sh` with the
right SBATCH directives.  The corresponding `make X` runs the same logic
locally on the calling host (useful for development; not recommended for
the 25 GB download).

`PYTHONPATH` is set automatically in both paths -- the Makefile exports
`$(CURDIR)/src` for local targets, and `slurm/_common.sh` exports
`PYTHONPATH` + `SINGULARITYENV_PYTHONPATH` for jobs running inside the
container.  **No `pip install -e .` is required.**

## Data sources and provenance

### Why OSF, not S3

The upstream README at https://github.com/anjany/verse documents six S3
URLs at `s3.bonescreen.de/public/VerSe-complete/` as the canonical download.
**Those URLs have been offline since May 2024** (see
[verse#17](https://github.com/anjany/verse/issues/17)).  The BIDS-
restructured ("subject-based") form of the data is mirrored on OSF under
two child storage nodes the upstream README never explicitly linked to:

- VerSe 2019 subject-based: https://osf.io/jtfa5/
- VerSe 2020 subject-based: https://osf.io/4skx2/

`src/verse_pipeline/download.py` walks these via OSF's public REST API
(`https://api.osf.io/v2/nodes/<id>/files/osfstorage/`) with:

- **Pre-request throttling** (0.8 s per call) to stay under OSF's
  ~100 req/min unauthenticated rate limit.
- **Retry-on-429** with exponential backoff (2 s -> 4 s -> 8 s, up to 6
  attempts), honouring `Retry-After` when OSF sends it.
- **Per-file resumability**: rerunning skips files already on disk at the
  expected size; only missing or corrupted files re-download.

### VERIDAH manual corrections

Moeller H. et al. (2026) — *VERIDAH: Solving Enumeration Anomaly Aware
Vertebra Labeling across Imaging Sequences*, arXiv:2601.14066 — manually
reviewed VerSe and identified 25 subjects with mislabeled enumeration
anomalies (predominantly T13 cases where the extra thoracic vertebra was
annotated as L1 with subsequent lumbar labels shifted down).

VerSeFusion incorporates these via the `correct` stage:

```
configs/veridah_corrections.csv  ->  src/verse_pipeline/veridah.py
        |
        v
data/unified/  ->  data/corrected/
    (335 subjects symlinked through unchanged)
    (~15 subjects with materialized corrected mask + centroid)
```

Two correction types are supported:

- **T13 shift** (12 cases).  Remap: 20 -> 28, 21 -> 20, 22 -> 21, ...
- **LabelOverride** (3 cases: `verse559`, `verse606`, `verse642_dir-sag`).
  Full sequence replacement -- the entire vertebra label sequence was
  re-seeded after VERIDAH review.

Per-subject provenance (including the explicit `{old_label: new_label}`
remap actually applied) is recorded in `data/corrected/veridah_manifest.json`
for citation in the paper.

The remaining 10 rows in the CSV are advisory-only: they flag TLTV
(thoracolumbar transitional vertebra) and stump-rib morphology that
Moeller's team noted but did not modify the label sequence for.  These
flags surface in the manifest but do not change voxel labels.

## Label scheme

VerSe ships a 28-class vertebra index (see `docs/label_scheme.md`).
Sacrum and coccyx labels (26, 27) exist in the scheme but are **not
annotated** in the dataset -- VerSe is a vertebrae-only corpus.  Hips and
pelvis are absent entirely.

The crosswalk to CTSpinoPelvic1K's 10-class scheme is lossy in one
direction (VerSe -> CTSpinoPelvic1K drops thoracic and cervical) and
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

If you use this pipeline, please cite **both** the original VerSe papers
**and** the VERIDAH corrections paper:

```bibtex
@article{sekuboyina2021verse,
  title   = {VerSe: A Vertebrae Labelling and Segmentation Benchmark
             for Multi-detector CT Images},
  author  = {Sekuboyina, Anjany and others},
  journal = {Medical Image Analysis},
  year    = {2021}
}

@article{loffler2020verse,
  title   = {A Vertebral Segmentation Dataset with Fracture Grading},
  author  = {Loffler, Maximilian T. and others},
  journal = {Radiology: Artificial Intelligence},
  year    = {2020}
}

@article{liebl2021verse,
  title   = {A Computed Tomography Vertebral Segmentation Dataset with
             Anatomical Variations and Multi-Vendor Scanner Data},
  author  = {Liebl, Hans and Schinz, David and others},
  year    = {2021}
}

@article{moller2026veridah,
  title   = {VERIDAH: Solving Enumeration Anomaly Aware Vertebra Labeling
             across Imaging Sequences},
  author  = {Moeller, Hendrik and others},
  journal = {arXiv preprint arXiv:2601.14066},
  year    = {2026}
}
```

The data license is **CC-BY-SA 4.0** and attaches to any derivative
exports this repo produces.  The VERIDAH corrections themselves are
redistributed with the same license per agreement with the authors.

## License

- **Code**: MIT (see `LICENSE`).
- **Data exports** (`data/unified/`, `data/corrected/`,
  `data/reoriented/`, `data/hf_export/`): inherit **CC-BY-SA 4.0** from
  upstream VerSe.

## Acknowledgements

This repo would not exist without:

- The VerSe team at TUM (Sekuboyina, Liebl, Loffler, Kirschke et al.) who
  built and curated the original dataset.
- Hendrik Moeller (TUM) for sharing the VERIDAH manual corrections and
  for the underlying labeling methodology that motivated this fusion.
- The community of users in `verse#17` who surfaced the OSF subject-based
  node IDs that replaced the dead S3 endpoint.
