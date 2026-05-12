# VerSeFusion-LSTV

A reproducible pipeline that constructs a Castellvi-graded vertebrae segmentation
corpus from the public VerSe distributions, fusing both releases (VerSe 2019 +
VerSe 2020) into a single patient-level dataset with auditable provenance.

This repository accompanies the NeurIPS 2026 Datasets & Benchmarks Track
submission. **All identifiers in this repository are anonymized for the
double-blind review process.**

## Why this dataset

VerSe is the canonical public CT-spine segmentation benchmark, but the
distribution is a moving target: there are two parallel releases (2019 and
2020), two parallel file formats (MICCAI-challenge and BIDS), and a known
limitation that **lumbosacral transitional vertebrae (LSTV) of Castellvi
grades 3 and 4 are not segmented by design** — TUM's annotation protocol
excludes fused transitional vertebrae from the segmentation labels.

VerSe-trained segmentation models therefore systematically fail on the most
pathologically important LSTV cases, the very ones a surgeon needs to plan
around. This pipeline:

1. Re-fuses the two VerSe releases into a single 374-scan / 355-patient corpus
2. Resolves multi-series patients via TUM's published demographic table
3. Recovers subjects partially or fully missing from MICCAI by falling back
   to the BIDS distribution
4. Audits every scan for CT/mask/centroid geometric agreement
5. Adds Castellvi-grade LSTV annotations to enable training of LSTV-aware
   segmentation and grading models

## Pretrained weights

Pretrained nnU-Net (5-fold) checkpoints are hosted on HuggingFace because
they exceed GitHub's storage limits. See `WEIGHTS.md` for download
instructions.

## Quick start

```bash
# 1. Set up the conda environment + Singularity container
make setup

# 2. Run the full pipeline (each stage logs to logs/)
make download-slurm     # ~25 min, ~60 GB → data/raw/
make unify-slurm        # ~30 s, → data/unified/
make qc-slurm           # ~5 min, → data/qc/
```

Each stage is independently re-runnable; outputs are deterministic.

## Pipeline overview

```
                  configs/verse_demographics.csv
                          │
                          ▼
   OSF MICCAI nodes ──┐
   (923ap, b2wxj)     ├──► download ──► data/raw/{verse19,verse20}/
   OSF BIDS nodes  ───┤        (auto-discover gaps, fallback to BIDS
   (jtfa5, 4skx2)     │         for any missing CT / mask / centroid)
                      │
                      └──► unify ──► data/unified/scan-<series_id>/
                                       (one canonical dir per scan,
                                        symlinks + meta.json)
                                          │
                                          ▼
                                       qc ──► data/qc/qc_manifest.json
                                          │
                                          ▼
                              (Castellvi annotation +
                               nnU-Net training — separate stages,
                               see paper for details)
```

### Stage 1 — Download

```bash
make download-slurm
```

Auto-discovers the gap between TUM's published demographics (374 expected
series) and what OSF's MICCAI-format nodes actually publish. For any subject
missing one or more required kinds (CT, mask, centroid), the script falls
back to OSF's BIDS-format mirrors and fetches only the missing files.

The fallback is per-kind, not just per-subject: if MICCAI has a CT but no
centroid for verse051, only the centroid is pulled from BIDS, MICCAI's CT
takes precedence.

Downloads are streamed in parallel (`ThreadPoolExecutor`, default 8 workers)
against OSF's S3-backed CDN. Listing happens serially with throttling to
respect OSF's API rate limit; downloads from the CDN aren't rate-limited.
Resumable — files already present at the expected size are counted as
cached on re-runs.

Per-file manifest at `data/raw/download_manifest.json` records source
(`miccai` or `bids_fallback`), remote path, local path, and size.

```bash
# Disable the BIDS fallback if you want a pure MICCAI subset:
DOWNLOAD_FLAGS="--no_bids_fallback" sbatch slurm/download_raw.sh

# Crank workers up to 16:
DOWNLOAD_WORKERS=16 sbatch slurm/download_raw.sh
```

### Stage 2 — Unify

```bash
make unify-slurm
```

Walks `data/raw/`, parses MICCAI- and BIDS-style filenames through a single
permissive parser (handles `verseNNN.nii.gz`, `verseNNN_CT-iso.nii.gz`,
`sub-verseNNN_dir-iso_ct.nii.gz`, `GL003.nii.gz`, plus multi-series
patient variants like `verse400_verse090_CT-iso.nii.gz`), groups files by
`(release, series_id)`, and materializes one canonical directory per scan
at `data/unified/scan-<series_id>/`:

```
data/unified/scan-verse014/
├── scan-verse014_ct.nii.gz         (symlink to raw CT)
├── scan-verse014_msk.nii.gz        (symlink to raw mask)
├── scan-verse014_ctd.json          (symlink to raw centroid)
├── scan-verse014_snp.png           (symlink to raw snapshot)
└── scan-verse014_meta.json         (generated, see schema below)
```

When a series appears in both releases (105 cross-release subjects per TUM's
demographics), the v20 copy is preferred but both source paths are recorded.

Each scan's `meta.json` carries:

```json
{
  "series_id":             "verse014",
  "patient_id":            "verse014",        // sibling-scan grouping key
  "chosen_release":        "verse19",
  "other_releases":        ["verse20"],
  "split":                 "training",
  "position":              "1 of 1",
  "in_v19":                true,
  "in_v20":                false,
  "sex":                   "F",
  "age":                   72,
  "source_paths":          {"ct": "...", "msk": "...", "ctd": "...", "snp": "..."},
  "missing_kinds":         [],
  "source_format":         "miccai",           // miccai | bids
  "centroid_coord_system": "asl_iso_1mm",      // asl_iso_1mm | voxel
  "version":               "0.2.0"
}
```

The `source_format` and `centroid_coord_system` fields are critical for
downstream stages: MICCAI centroids are in 1 mm isotropic ASL space, BIDS
centroids are in per-image voxel space. The reorient stage dispatches on
this flag.

Manifest at `data/unified/unify_manifest.json` aggregates counts:

```bash
jq '{n_scans, n_patients, n_multi_series, by_release, by_split,
     by_source_format, completeness}' data/unified/unify_manifest.json
```

### Stage 3 — QC (alignment audit)

```bash
make qc-slurm
```

Per-scan QC audits the unified corpus for geometric consistency. For each
scan, six checks run independently:

| Check                 | What it verifies                                          | Why it matters |
|-----------------------|------------------------------------------------------------|----------------|
| `files_present`       | CT, mask, centroid present on disk                         | catches stale symlinks |
| `headers_readable`    | nibabel can load both NIfTI headers                        | catches truncated downloads |
| `shape_match`         | CT and mask have identical voxel grids                     | nnU-Net assumes this |
| `affine_match`        | CT and mask agree on direction matrix + spacing            | catches LPS/RAS mismatches |
| `label_inventory`     | mask labels in VerSe range [1, 28], no tiny artifacts      | catches label noise |
| `centroid_alignment`  | each centroid's `(label, x, y, z)` lands on matching mask voxel | end-to-end ground-truth integrity |

Each check returns `PASS / WARN / FAIL / SKIP` with reason strings; the
per-scan overall status is the worst-of.

The centroid-alignment check is the gold standard: it converts MICCAI's
1mm-iso-ASL centroids to voxel space using the CT's affine (or uses voxel
coords directly for BIDS-recovered subjects), rounds to the nearest integer,
and verifies `mask[voxel] == centroid_label` for every labeled vertebra.

#### Querying QC results

```bash
# Headline numbers
jq '.by_status, .by_check' data/qc/qc_manifest.json

# Every flagged scan
jq '.flagged_scans' data/qc/qc_manifest.json

# Drill into one specific scan
jq '.scans[] | select(.series_id == "verse051")' data/qc/qc_manifest.json

# Distribution of centroid match rates
jq '[.scans[] | .checks.centroid_alignment.match_rate // empty] | sort' \
   data/qc/qc_manifest.json
```

## Repository layout

```
VerSeFusion/
├── configs/
│   ├── default.env                  # Project paths, container settings
│   ├── verse_demographics.csv       # TUM-published 374-row demographic table
│   ├── veridah_corrections.csv      # Möller LSTV label corrections (chunk 2)
│   └── label_scheme.yaml            # VerSe vertebra label scheme
├── slurm/
│   ├── _common.sh                   # Shared environment for all SLURM jobs
│   ├── download_raw.sh              # Stage 1
│   ├── unify_iterations.sh          # Stage 2
│   └── qc.sh                        # Stage 3
├── src/
│   └── verse_pipeline/
│       ├── __init__.py
│       ├── download.py              # OSF fetcher + BIDS-fallback discovery
│       ├── unify.py                 # raw → unified scan-dirs
│       ├── qc.py                    # per-scan alignment audit
│       └── utils/
│           ├── __init__.py
│           ├── demographics.py      # CSV loader, patient/series indexing
│           └── miccai.py            # MICCAI+BIDS filename parser
├── containers/                      # gitignored; built locally
│   └── versefusion.sif
├── data/                            # gitignored
│   ├── raw/                         # Stage 1 output
│   ├── unified/                     # Stage 2 output
│   └── qc/                          # Stage 3 output
├── logs/                            # gitignored
├── Makefile
├── README.md
├── WEIGHTS.md                       # HuggingFace checkpoint download
└── .gitignore
```

## Dataset accounting

After running through unify on the full 374-row demographic table:

| Metric                     | Count |
|----------------------------|------:|
| Total scans                | 374   |
| Unique patients            | 355   |
| Multi-series patients      | 18    |
| Cross-release (v19 + v20)  | 105   |
| Glocker cohort (gl* prefix)| 30    |
| MICCAI-sourced             | 359   |
| BIDS-sourced (partial recovery) | 15 |
| Complete (CT + msk + ctd)  | 373   |
| Missing one component      | 1     |

One subject (`verse072`) lacks a published centroid file in both MICCAI and
BIDS distributions. The CT and segmentation are present; users requiring
centroids should either exclude this subject or derive centroids from the
segmentation labels.

## Reproducibility

All sources of randomness in the pipeline are controllable:

- Demographic ordering: deterministic from `configs/verse_demographics.csv`
- File ordering: deterministic (sorted before parallel dispatch)
- Cross-release tiebreaking: deterministic preference for v20
- QC parallelism: process pool with deterministic future ordering

Re-running any stage with the same inputs produces identical outputs.

## Dependencies

The pipeline runs inside a Singularity container (`containers/versefusion.sif`)
built from `containers/versefusion.def`. The container provides:

- Python 3.11
- nibabel, numpy
- requests
- tqdm

Build with:

```bash
make container
```

This requires root or the `--fakeroot` flag. On a cluster without root,
build on a local workstation and `scp` the resulting `.sif`.

## Limitations

1. **One subject lacks a published centroid** (`verse072`). CT and mask
   are intact; downstream users must decide whether to exclude or derive.

2. **BIDS-recovered centroids are in voxel space, not 1mm-iso-ASL**. The
   reorient stage handles this by dispatching on the
   `centroid_coord_system` field in each scan's meta.json. Users
   bypassing reorient must do the conversion themselves.

3. **Castellvi grade-3/4 LSTV are not segmented in the upstream VerSe
   labels.** The transitional vertebra is excluded by TUM's annotation
   protocol. Our supplementary Castellvi-grade annotations fill this gap
   (see paper).

4. **Multi-series patients** are flattened to per-scan rows, not
   per-patient. For patient-level splits, use the `patient_id` field in
   each scan's meta.json.

## Citation

If you use this pipeline or the derived corpus, please cite:

```bibtex
@inproceedings{anonymous2026versefusion,
    title  = {[anonymous title for double-blind review]},
    author = {Anonymous},
    booktitle = {NeurIPS Datasets and Benchmarks Track},
    year   = {2026},
    note   = {Under review}
}
```

The upstream VerSe distributions should also be cited:

```bibtex
@article{sekuboyina2021verse,
    title   = {{VerSe}: A vertebrae labelling and segmentation benchmark
               for multi-detector {CT} images},
    author  = {Sekuboyina, Anjany and others},
    journal = {Medical Image Analysis},
    volume  = {73},
    year    = {2021},
}
```

## License

The pipeline code in this repository is released under the MIT License.

The underlying VerSe dataset is distributed by TUM under CC BY 4.0; users
of the derived corpus must comply with VerSe's terms. Pretrained weights
follow the same CC BY 4.0 license.

See `LICENSE` for the full text.
