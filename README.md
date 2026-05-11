# VerSeFusion

> A reproducible pipeline to download, unify, dedup, and benchmark-export the
> **VerSe 2019** and **VerSe 2020** CT vertebrae segmentation datasets as a
> single curated corpus with explicit handling of enumeration anomalies
> (LSTV, T13) and a crosswalk to the [CTSpinoPelvic1K] label scheme.

[CTSpinoPelvic1K]: https://github.com/gschwing/CTSpinoPelvic1K  <!-- update when public -->

---

## Why this repo exists

VerSe19 and VerSe20 are published as **two overlapping releases** of the same
underlying cohort:

| Release  | Subjects | Annotated vertebrae | Anatomical variants                                |
|----------|---------:|--------------------:|-----------------------------------------------------|
| VerSe19  |      141 |               1 725 | mixed pathology, some LSTV                          |
| VerSe20  |      300 |               4 142 | enriched: **77 enum. anomalies, 161 transitional**  |
| Overlap  |  105 imgs|                   — | series shared between VerSe19 & VerSe20             |
| **Fused**|  **355** |          **~4 505** | dedup'd, single canonical source per subject        |

The VerSe maintainers publish a BIDS-restructured form on S3 that is the
recommended download (the OSF mirrors carry the older MICCAI-challenge
schema).  This repo:

1.  Pulls the six S3 zips reproducibly with checksum verification.
2.  Unifies VerSe19 + VerSe20 into one subject-keyed tree, deduplicating
    the 105-image overlap by preferring the VerSe20 release (newer
    annotations, fracture grading included).
3.  Re-orients every CT and mask to **PIR** to match the CTSpinoPelvic1K
    convention (so models trained on one can be evaluated on the other
    without re-orienting at inference time).
4.  Builds a `placed_manifest.json` with per-subject metadata: vertebra
    inventory, LSTV/T13 flags, scanner manufacturer, voxel spacing, FOV.
5.  Stratifies a 5-fold CV split on the LSTV/T13/normal axis.
6.  Crosswalks the VerSe label scheme (1–28) to the CTSpinoPelvic1K 10-class
    scheme for direct external validation of models trained on
    CTSpinoPelvic1K.
7.  Exports a HuggingFace-compatible `DatasetDict` for downstream training
    and benchmarking.

## Primary downstream uses

- **External validation** of nnU-Net v2 / VERIDAH / SPINEPS / TotalSegmentator
  models trained on CTSpinoPelvic1K.
- **Pretraining corpus** for the nnU-Net v2 trainer (`tools/nnunet_wandb_variant.py`
  from CTSpinoPelvic1K) — VerSe's varied FOV and pathology mix is
  complementary to CTSpinoPelvic1K's COLONOG-derived abdominal CTs.
- **Standalone benchmark** for vertebrae labelling and segmentation with
  LSTV stratified evaluation — something the original VerSe challenge did
  not report.

## Repository layout

```
VerSeFusion/
├── Makefile                  one-line entry points (see `make help`)
├── README.md                 this file
├── LICENSE                   MIT (code) / CC-BY-SA-4.0 (data, upstream)
├── pyproject.toml
├── requirements.txt
├── configs/
│   ├── default.env           non-secret defaults (paths, version pins)
│   └── label_scheme.yaml     VerSe → CTSpinoPelvic1K crosswalk
├── containers/
│   └── README.md             Docker Hub pull instructions
├── slurm/                    Warrior-HPC SLURM wrappers
│   ├── download_raw.sh
│   ├── unify_iterations.sh
│   ├── reorient_pir.sh
│   ├── build_manifest.sh
│   ├── lstv_audit.sh
│   ├── make_splits.sh
│   └── hf_export.sh
├── nextflow/
│   ├── main.nf               full pipeline DAG
│   └── nextflow.config
├── src/verse_pipeline/
│   ├── download.py           wget + sha256, resumable
│   ├── unify.py              VerSe19+20 dedup
│   ├── reorient.py           PIR reorientation (matches CTSpinoPelvic1K)
│   ├── manifest.py           placed_manifest.json builder
│   ├── lstv.py               LSTV / T13 detection from centroid JSON
│   ├── splits.py             5-fold CV, stratified by LSTV+T13
│   ├── label_crosswalk.py    VerSe ↔ CTSpinoPelvic1K 10-class
│   ├── hf_export.py          HuggingFace DatasetDict export
│   └── utils/
│       ├── bids.py           parse BIDS-style filenames
│       ├── centroid_json.py  read/write VerSe ctd.json
│       └── nifti.py          orientation, affine helpers
├── scripts/
│   ├── inventory.py          subjects per source/split
│   └── qc_overview.py        regenerate snp.png overviews
├── tests/                    pytest smoke tests
├── data/                     gitignored — staging dir
│   ├── raw/{verse19,verse20}
│   ├── unified/
│   ├── reoriented/
│   └── hf_export/
└── docs/
    ├── design.md
    ├── label_scheme.md
    └── crosswalk.md
```

## Quick start

```bash
# 1. clone + install
git clone https://github.com/<you>/VerSeFusion.git
cd VerSeFusion
make install                       # pip install -e .[dev]

# 2. pull the six S3 zips (≈ 30 GB total, resumable)
make download

# 3. unify VerSe19 + VerSe20, dedup the 105-image overlap
make unify

# 4. reorient everything to PIR
make reorient

# 5. build the manifest with LSTV/T13 flags
make manifest

# 6. 5-fold stratified splits
make splits

# 7. HuggingFace export
make hf-export
```

On the Warrior HPC, every `make` target has a matching `make <target>-slurm`
that submits the job through `slurm/<target>.sh` with the right partition,
GRES, and Singularity bindings.

## Label scheme

VerSe ships a 28-class vertebra index (see `docs/label_scheme.md`).  Sacrum
and coccyx labels (26, 27) exist in the **scheme** but are **not annotated**
in the dataset — VerSe is a vertebrae-only corpus.  Hips and pelvis are
absent entirely.

The crosswalk to CTSpinoPelvic1K's 10-class scheme is therefore lossy in
one direction (VerSe → CTSpinoPelvic1K drops thoracic and cervical) and
incomplete in the other (CTSpinoPelvic1K → VerSe has no hip/sacrum labels
to map to).  See `docs/crosswalk.md`.

| VerSe label | Region                       | CTSpinoPelvic1K equivalent |
|------------:|------------------------------|---------------------------:|
| 1–7         | C1–C7                        |     — (not in CTSPP1K)     |
| 8–19        | T1–T12                       |     —                      |
| 20–24       | L1–L5                        |          1–5               |
| 25          | L6 (lumbarized LSTV)         |           6                |
| 26          | sacrum (unannotated in VerSe)|           7                |
| 27          | coccyx (unannotated)         |     —                      |
| 28          | T13 (extra thoracic)         |     —                      |

## Citation

If you use this pipeline, please cite the original VerSe papers (Sekuboyina
2021, Löffler 2020, Liebl 2021) — see `docs/citation.bib`.  The data
license is **CC-BY-SA 4.0** and that license attaches to any derivative
exports this repo produces.

## License

- **Code**: MIT (see `LICENSE`).
- **Data exports** (`data/unified/`, `data/reoriented/`, `data/hf_export/`):
  inherit **CC-BY-SA 4.0** from upstream VerSe.

## Acknowledgements

This repo would not exist without the VerSe team at TUM (Sekuboyina, Liebl,
Löffler, Kirschke et al.) who built and curated the original dataset, and
without Hendrik Möller's VERIDAH work which clarified the role of sequence
prediction in resolving LSTV labelling ambiguity.
