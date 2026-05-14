# VerSeFusion

A reproducible pipeline that constructs an LSTV/TLTV-stratified vertebrae
segmentation corpus from the public VerSe distributions, fusing both releases
(VerSe 2019 + VerSe 2020) into a single patient-level dataset with auditable
provenance, model-derived enumeration-anomaly corrections, and patient-level
5-fold cross-validation splits.

This repository accompanies the NeurIPS 2026 Datasets & Benchmarks Track
submission. **All identifiers in this repository will be anonymized for the
double-blind review process.**

## Companion repositories

| Repo | Contents |
|---|---|
| **VerSeFusion** | Full 374-scan corpus, ~22 GB. |
| **VerSeFusion-LSTV** | 68-scan anomaly subset (lumbarization + T13-supernumerary + T12-absent). |
| **VerSeFusion-Sample** | 10 scans by label completeness, ~500 MB — for smoke tests and tutorials. |

All three are produced by the same pipeline, shipped with the same `dataset_interface.py`
loader, and reference the same patient-level cross-validation splits.

## Why this dataset

VerSe is the canonical public CT-spine segmentation benchmark, but its
distribution and annotation scheme have known limitations that systematically
disadvantage anomaly-aware segmentation research:

- **Two parallel releases (2019 and 2020) with substantial overlap** — 105 of
  ~370 patients appear in both, and the published comparisons disagree on which
  release should be treated as canonical for those overlapping cases.
- **Two file formats** — MICCAI-style and BIDS-style, with different filename
  conventions, centroid coordinate spaces, and orientation conventions.
- **Castellvi grade III and IV LSTV are not segmented by design** — TUM's
  annotation protocol excludes fully-fused transitional vertebrae from the
  segmentation labels, removing precisely the cases that surgeons need to plan
  around.
- **Enumeration anomalies (extra/missing vertebrae) are labeled inconsistently** —
  a T13 may be labeled as T12, an L6 may be labeled as L5, depending on the
  annotator's interpretation of the sequence.

VerSe-trained segmentation models therefore systematically fail on the most
pathologically important LSTV cases. Wrong-level spine surgery — preventable
when transitional vertebrae are correctly identified preoperatively — remains a
significant patient-safety problem (Konin & Walz, 2010; Hanhivaara et al., 2024).

This pipeline addresses each limitation:

1. Fuses both VerSe releases into a single 374-scan / 355-patient corpus with
   deterministic cross-release tie-breaking.
2. Falls back from MICCAI to BIDS distributions when MICCAI publishes
   incomplete subjects.
3. Reorients every scan to PIR canonical space at 1 mm isotropic resolution.
4. Runs VERIDAH (Möller et al., 2026) as a referee over every scan, correcting
   14 enumeration-anomaly label assignments where VerSe's labels disagree with
   VERIDAH's sequence verdict.
5. Audits every scan with a 4-way LSTV/TLTV taxonomy
   (`normal`, `lumbarization`, `t13_supernumerary`, `truncated`) at both the
   structural and field-of-view levels.
6. Provides patient-level 5-fold stratified CV splits with a fixed 15% test
   holdout.

## Pipeline overview

```
                  configs/verse_demographics.csv
                          │
                          ▼
  OSF MICCAI nodes ──┐
  (923ap, b2wxj)     │      ┌──────────────────────────────────────────┐
                     ├──►   │  Stage  1   download    → data/raw/      │
  OSF BIDS nodes  ───┤      │  Stage  2   unify       → data/unified/  │
  (jtfa5, 4skx2)     │      │  Stage  3   reorient    → data/canonical/│
                     │      │  Stage  4   qc          → data/qc/       │
                     │      │  Stage  5   renders     → data/qc/renders│
                     │      │  Stage  6   veridah     → data/corrected/│
                     │      │  Stage  7   veridah-renders               │
                     │      │  Stage  8   lstv-audit  → data/lstv/     │
                     │      │  Stage  9   orient-audit                   │
                     │      │  Stage 10a  manifest    → data/manifest/ │
                     │      │  Stage 10b  splits                         │
                     │      │  Stage 11   hf-stage + hf-export           │
                     └──►   └──────────────────────────────────────────┘
```

Each stage is independently re-runnable, idempotent, and produces a structured
JSON manifest. Outputs are deterministic given the same inputs and seed.

## Dataset statistics

### Overall corpus

| Metric | Count |
|---|---:|
| Total scans | 374 |
| Unique patients | 355 |
| Multi-series patients | 19 |
| Cross-release (v19 + v20) | 105 |
| MICCAI-sourced | 361 |
| BIDS-sourced (partial recovery) | 13 |
| Complete (CT + msk + ctd) | 373 |
| Missing one component | 1 |
| VERIDAH corrections applied | 14 |

### LSTV/TLTV anomaly stratification

| Class | Scans | Definition | Audit-categorical source |
|---|---:|---|---|
| `normal` | 306 | No transitional vertebra in FOV | otherwise |
| `lumbarization` | 44 | L6 present (sacral lumbarization, or true sixth lumbar) | `lstv_class_audit == "lumbarization"` |
| `t13_supernumerary` | 18 | T13 present (thirteenth thoracic vertebra) | `tltv_class_audit == "t13_supernumerary"` |
| `truncated` | 6 | T12 absent in-FOV (not FOV-truncated) | `tltv_class_audit == "t12_absent"` |

Anomaly prevalence in this corpus is 68/374 ≈ 18.2%, roughly 4× the lowest
estimates and within the upper range of population studies (LSTV prevalence
ranges 4–37% depending on cohort and grading threshold; cf. Konin & Walz, 2010).

### 5-fold cross-validation splits

15% patient-level stratified test holdout, then 5-fold stratified CV on the
remaining 85%. Stratification uses patient-level worst-case LSTV class
collapse (a patient with one t13 scan + one normal scan counts as t13).

|  | Patients | Scans |
|---|---:|---:|
| Test (held out) | 54 | 57 |
| CV pool | 301 | 317 |
| Per fold: train | 240–242 | ~255 |
| Per fold: val | 59–61 | ~62 |

Per-class breakdown in the test holdout: 3 t13, 7 lumbarization, 1 truncated,
43 normal. Each CV fold's val side gets 3 t13, 7–8 lumbarization, 1 truncated,
48–49 normal — distribution flat across folds within ±1.

## Recommended use: training LSTV/TLTV-aware segmenters

The 18% anomaly prevalence in this corpus is intentionally inflated relative to
the population base rate of clinically significant LSTV (the surgically relevant
Castellvi III/IV grades comprise roughly 5–10% of populations). This makes the
corpus a practical training target for anomaly-aware segmentation models in a
way that natural-distribution sampling does not.

### Strategy 1 — Pretrain on the full corpus, fine-tune on the LSTV subset

```bash
# Pretrain (nnU-Net 5-fold, fold 0 shown)
nnUNetv2_train Dataset501_VerSeFusion 3d_fullres 0 -tr nnUNetTrainer_VerseFusion

# Fine-tune on the LSTV subset (lr=1e-5, 50 epochs)
nnUNetv2_train Dataset502_VerSeFusionLSTV 3d_fullres 0 \
    -tr nnUNetTrainerNoMirroring \
    -pretrained_weights checkpoints/Dataset501_VerSeFusion/fold_0/checkpoint_final.pth
```

This is the simplest baseline. Track held-out validation Dice **per LSTV class**,
not population-mean — overall Dice can be 0.95 while LSTV-cohort Dice is 0.6,
because the normal-anatomy scans dominate the average.

### Strategy 2 — Anomaly oversampling within a single training run

Use the canonical `splits_5fold.json` as the patient-level fold structure, but
oversample LSTV cases by 5–10× within training batches via a
`WeightedRandomSampler`. nnU-Net supports class-level weighting through the
`oversample_foreground_percent` hook (Isensee et al., 2021).

### Strategy 3 — Multi-task segmentation + classification

Attach a global classification head to the encoder predicting `lstv_class` —
one of `{normal, lumbarization, t13_supernumerary, truncated}`. The auxiliary
classification loss regularizes the encoder to attend to anomaly cues that pure
voxel-wise Dice doesn't reward. The architecture mirrors VERIDAH's
sequence-aware vertebra-level voting (Möller et al., 2026), and the training
signal is balanced 1:1 between segmentation and classification.

### Strategy 4 — VERIDAH-in-the-loop label refinement

For research applications where ground-truth labels themselves are suspect,
run VERIDAH at inference time as a referee on every prediction. Disagreements
between VERIDAH's sequence verdict and the predicted segmentation's label
assignment are an early warning for enumeration-anomaly failure modes — this
is the mechanism by which the 14 VERIDAH-corrected scans in this corpus were
identified.

### Evaluation: per-cohort metrics, not population averages

When evaluating any LSTV-aware segmenter, report Dice / Hausdorff /
surface-distance **separately** for each of the four LSTV classes. Population
averages will systematically obscure cohort-level failure. Hanhivaara et al.
(2024) document the same masking effect in classification accuracy: CT has 84%
balanced accuracy for LSTV detection but only 76% sensitivity, and
per-grade-stratified metrics are the only honest way to characterize a model.

## Quick start

```bash
# 1. Set up the conda environment + Singularity container
make setup

# 2. Run the pipeline (each stage queues an independent SLURM job)
make download-slurm          # ~25 min, ~60 GB → data/raw/
make unify-slurm             # ~30 s, → data/unified/
make reorient-slurm          # ~10 min, → data/canonical/
make qc-slurm                # ~5 min, → data/qc/
make renders-slurm           # ~3 min, → data/qc/renders/
make veridah-slurm           # ~15 min, → data/corrected/
make veridah-renders-slurm   # ~30 s, → data/corrected/renders/
make lstv-audit-slurm        # ~20 s, → data/lstv/
make orient-slurm            # ~20 s, → data/orientation/
make manifest-slurm          # ~10 s, → data/manifest/manifest.{csv,json}
make splits-slurm            # ~10 s, → data/manifest/splits_5fold.json
HF_TOKEN=hf_xxx make hf-export-slurm   # stage + push all three HF repos
```

Each stage is independently re-runnable; outputs are deterministic. `make status`
prints a summary of completed stages and their key statistics.

## Loading the data

The `dataset_interface.py` shim shipped in every HF repo provides a uniform
loader. To use:

```python
from huggingface_hub import snapshot_download
from verse_pipeline.dataset_interface import VerSeFusion

# Metadata only — NIfTIs lazy-load on access (~5 MB transfer)
path = snapshot_download(
    repo_id="gregoryschwingmdphd/VerseFusion",
    repo_type="dataset",
    allow_patterns=[
        "manifest.csv", "manifest.json", "splits_5fold.json",
        "veridah_manifest.json", "orientation_audit.json",
        "dataset_interface.py", "*.md",
    ],
)

ds = VerSeFusion(path)
print(ds.stats())                          # all 374 scans

# Patient-level splits — same identity across folds
train, val = ds.fold(0)                    # ~240 train + ~61 val patients
test       = ds.test_set()                 # 54 patients, untouched across folds

# Iterate
for case in ds:
    ct  = case.load_ct()                   # (Z, Y, X) numpy array, HU
    msk = case.load_mask()                 # (Z, Y, X) int8, labels 1-28
    print(case.series_id, case.lstv_class, case.veridah_applied)
```

The PyTorch wrapper `VerSeFusionDataset` is included for direct integration with
DataLoader-based training.

## Stage details

### Stage 1 — Download

Auto-discovers the gap between TUM's published demographics (374 expected
series) and what OSF's MICCAI-format nodes actually publish. For any subject
missing one or more required kinds (CT, mask, centroid), the script falls back
to OSF's BIDS-format mirrors and fetches only the missing files. Per-kind, not
per-subject: if MICCAI has a CT but no centroid for `verse051`, only the
centroid is pulled from BIDS.

### Stage 2 — Unify

Parses MICCAI- and BIDS-style filenames through a permissive parser, groups
files by `(release, series_id)`, and materializes one canonical directory per
scan at `data/unified/scan-<series_id>/`. When a series appears in both
releases, the v20 copy is preferred. Each scan's `meta.json` records source
paths, demographics, and `source_format` / `centroid_coord_system` flags
critical for downstream coordinate handling.

### Stage 3 — Reorient

Reorients every CT and mask to PIR (Posterior-Inferior-Right) canonical space
at 1 mm isotropic resolution. MICCAI centroids (1 mm ASL space) and BIDS
centroids (voxel space) are dispatched to the appropriate transform via the
`centroid_coord_system` flag from stage 2.

### Stage 4 — QC

Per-scan alignment audit. Six independent checks per scan:
`files_present`, `headers_readable`, `shape_match`, `affine_match`,
`label_inventory`, `centroid_alignment`. The centroid-alignment check is the
gold standard: for each labeled centroid, converts to voxel space and verifies
`mask[voxel] == centroid_label`.

### Stage 5 — Renders

Generates 9-panel QC visualization gallery for every scan: orthogonal
mid-volume views, label-colored mask overlays, and an mm-scaled max-intensity
projection. Output as a static HTML gallery at `data/qc/renders/index.html`
for rapid human review of all 374 scans.

### Stage 6 — VERIDAH

Runs Möller et al.'s (2026) VERIDAH model — a vertebra-level sequence
predictor trained to resolve enumeration anomalies — as a referee over every
scan. Where VERIDAH disagrees with VerSe's label assignment and meets the
confidence threshold, a correction is recorded in
`data/corrected/veridah_manifest.json`. Two correction types are emitted:

- **`t13_shift`** (12 scans): VerSe labeled a vertebra as L1 that VERIDAH
  identifies as T13. The mask's label is shifted up by one, recovering the
  missing T13 annotation. This is the most common VerSe annotation gap.
- **`label_override`** (2 scans): VerSe's label was internally inconsistent
  (e.g., two adjacent vertebrae with the same label); VERIDAH provided the
  resolved sequence.

VERIDAH reports 96.3% accuracy for thoracic enumeration anomalies and 97.2%
for lumbar on CT (Möller et al., 2026); these 14 corrections all met its
confidence threshold and were also confirmed by independent visual inspection.

### Stage 7 — VERIDAH renders

Side-by-side render of pre- and post-correction segmentation for the 14
corrected scans, plus 11 representative passthrough scans, for review.

### Stage 8 — LSTV audit

Per-scan structural audit emitting a 4-way `lstv_class` plus underlying
categoricals (`lstv_class_audit`, `tltv_class_audit`). Distinguishes genuine
T12 absence (anomaly) from T12-outside-FOV (artifact), and distinguishes
L6-present (anomaly) from L5-fov-truncated (artifact). The audit's categorical
output is the source of truth for downstream stratification — boolean flags
(`has_t13`, `has_l6`) are kept for backward compatibility but not used for
class assignment.

### Stage 9 — Orientation audit

Defense-in-depth verification that every CT and mask in `data/canonical/` is
in PIR orientation at the expected shape. Independent of stage 3 to catch any
intermediate hand-editing.

### Stage 10 — Manifest + splits

Stage 10a emits a single 27-column manifest (`manifest.csv` / `manifest.json`)
joining unify, VERIDAH, LSTV audit, and canonical metadata.

Stage 10b generates patient-level 5-fold stratified CV splits with a 15% test
holdout (configurable via `SPLITS_TEST_FRACTION`). Stratification is over the
4-way `lstv_class` taxonomy using a per-stratum round-robin (sklearn's
StratifiedKFold cannot handle strata smaller than `n_folds`, which is the case
for `truncated` with 6 patients). Seeded; outputs are deterministic.

### Stage 11 — HF export

Three-phase HuggingFace dataset export:

- **Phase 1**: Full corpus to `VerSeFusion` (374 scans, ~22 GB).
- **Phase 2**: Top-10 by label completeness to `VerSeFusion-Sample`.
- **Phase 3**: All 68 LSTV-class scans to `VerSeFusion-LSTV`.

Each phase ships the same set of top-level files: `manifest.csv`,
`manifest.json`, `splits_5fold.json`, `veridah_manifest.json`,
`orientation_audit.json`, `dataset_interface.py`, README, LICENSE. Uses
hardlink staging (instant on same-FS) and HuggingFace's `upload_large_folder`
(parallel, resumable).

## Repository layout

```
VerSeFusion/
├── configs/
│   ├── default.env                  # Project paths, container settings
│   ├── verse_demographics.csv       # TUM-published 374-row demographic table
│   ├── veridah_corrections.csv      # Möller LSTV label corrections
│   └── label_scheme.yaml            # VerSe vertebra label scheme + T13
├── slurm/                           # One SLURM script per stage
├── src/verse_pipeline/
│   ├── download.py                  # Stage 1
│   ├── unify.py                     # Stage 2
│   ├── reorient.py                  # Stage 3
│   ├── qc.py                        # Stage 4
│   ├── render_qc.py                 # Stage 5
│   ├── veridah_runner.py            # Stage 6
│   ├── veridah_render.py            # Stage 7
│   ├── lstv_audit.py                # Stage 8
│   ├── orient_audit.py              # Stage 9
│   ├── manifest_builder.py          # Stage 10a
│   ├── splits_builder.py            # Stage 10b
│   ├── hf_export.py                 # Stage 11
│   ├── dataset_interface.py         # Python loader (shipped in all HF repos)
│   └── utils/
├── containers/                      # gitignored
├── data/                            # gitignored
├── logs/                            # gitignored
├── Makefile
├── README.md
└── LICENSE
```

## Reproducibility

All sources of randomness in the pipeline are seeded and controllable:

- Demographic ordering: deterministic from `configs/verse_demographics.csv`.
- File ordering: deterministic (sorted before parallel dispatch).
- Cross-release tiebreaking: deterministic preference for v20.
- QC parallelism: process pool with deterministic future ordering.
- Splits: seeded (default `42`), patient-level stratified.

Re-running any stage with the same inputs and seed produces identical outputs.

## Dependencies

The pipeline runs inside a Singularity container (`containers/versefusion.sif`)
built from `containers/versefusion.def`:

- Python 3.11, nibabel, numpy, pandas, scipy
- PyTorch (for VERIDAH inference)
- `spineps` (VERIDAH ships as part of this package)
- `huggingface_hub`
- `requests`, `tqdm`

```bash
make container
```

This requires root or `--fakeroot`. On a cluster without root, build on a
local workstation and `scp` the resulting `.sif`.

## Limitations

1. **Castellvi grade is not directly annotated.** This corpus captures
   *structural* LSTV/TLTV (presence of L6, T13, or absence of T12) but does
   not include the Castellvi I–IV grade. Castellvi grading requires assessment
   of transverse-process morphology and is partially derivable from the
   segmentation mask via post-hoc analysis; a supplementary Castellvi
   annotation effort is described in the accompanying paper.
2. **`truncated` cohort is small (n=6).** Genuine T12-absent cases are
   underrepresented because TLJ-truncated CTs (T12 simply outside the FOV)
   outnumber true T12-absent scans by ~13:1. Confidence intervals on
   truncated-class metrics will be wide.
3. **One subject lacks a centroid file** (`verse072`) in both MICCAI and BIDS.
   CT and segmentation are intact; users requiring centroids should derive
   from the segmentation.
4. **BIDS-recovered centroids are in voxel space, not 1 mm ASL.** Stage 3
   handles this via `centroid_coord_system` dispatch; users bypassing stage 3
   must convert themselves.
5. **VERIDAH corrections are model-derived.** The 14 corrected labels follow
   VERIDAH's CT-mode sequence prediction (97.2% accuracy on its own evaluation
   set, Möller et al. 2026); 2 of the 14 (`label_override` kind) were further
   confirmed by manual review.

## Citation

If you use this corpus or pipeline, please cite the accompanying paper:

```bibtex
@inproceedings{anonymous2026versefusion,
  title  = {[anonymous title for double-blind review]},
  author = {Anonymous},
  booktitle = {Advances in Neural Information Processing Systems (NeurIPS),
               Datasets and Benchmarks Track},
  year   = {2026},
  note   = {Under review}
}
```

The upstream VerSe and VERIDAH publications must also be cited:

```bibtex
@article{sekuboyina2021verse,
  title   = {{VerSe}: A vertebrae labelling and segmentation benchmark for
             multi-detector {CT} images},
  author  = {Sekuboyina, Anjany and Husseini, Malek E. and L{\"o}ffler,
             Maximilian and others},
  journal = {Medical Image Analysis},
  volume  = {73},
  pages   = {102166},
  year    = {2021},
  doi     = {10.1016/j.media.2021.102166}
}

@article{moller2026veridah,
  title   = {{VERIDAH}: Solving Enumeration Anomaly Aware Vertebra Labeling
             across Imaging Sequences},
  author  = {M{\"o}ller, Hendrik and Schoen, Hanna and Graf, Robert and
             Atad, Matan and Molinier, Nathan and Sekuboyina, Anjany and
             others},
  journal = {arXiv preprint arXiv:2601.14066},
  year    = {2026}
}

@article{moller2024spineps,
  title   = {{SPINEPS} — automatic whole spine segmentation of T2-weighted
             {MR} images using a two-phase approach to multi-class semantic
             and instance segmentation},
  author  = {M{\"o}ller, Hendrik and Graf, Robert and Schmitt, Joachim and
             others},
  journal = {European Radiology},
  year    = {2024},
  doi     = {10.1007/s00330-024-11155-y}
}

@article{warszawer2025totalspineseg,
  title   = {{TotalSpineSeg}: Robust Spine Segmentation with Landmark-Based
             Labeling in {MRI}},
  author  = {Warszawer, Yehuda and Molinier, Nathan and Valo{\v{s}}ek, Jan
             and Benveniste, Pierre-Louis and others},
  year    = {2025},
  url     = {https://github.com/neuropoly/totalspineseg}
}

@article{isensee2021nnunet,
  title   = {{nnU-Net}: a self-configuring method for deep learning-based
             biomedical image segmentation},
  author  = {Isensee, Fabian and Jaeger, Paul F. and Kohl, Simon A. A. and
             Petersen, Jens and Maier-Hein, Klaus H.},
  journal = {Nature Methods},
  volume  = {18},
  number  = {2},
  pages   = {203--211},
  year    = {2021}
}

@article{castellvi1984,
  title   = {Lumbosacral transitional vertebrae and their relationship with
             lumbar extradural defects},
  author  = {Castellvi, A. E. and Goldstein, L. A. and Chan, D. P.},
  journal = {Spine},
  volume  = {9},
  number  = {5},
  pages   = {493--495},
  year    = {1984}
}

@article{konin2010lstv,
  title   = {Lumbosacral transitional vertebrae: classification, imaging
             findings, and clinical relevance},
  author  = {Konin, George P. and Walz, Daniel M.},
  journal = {AJNR Am J Neuroradiol},
  volume  = {31},
  number  = {10},
  pages   = {1778--1786},
  year    = {2010}
}

@article{hanhivaara2024castellvi,
  title   = {Castellvi classification of lumbosacral transitional vertebrae:
             comparison between conventional radiography, {CT}, and {MRI}},
  author  = {Hanhivaara, Jaakko and M{\"a}{\"a}tt{\"a}, Juhani H. and
             Kinnunen, Pietari and Niinim{\"a}ki, Jaakko and Nevalainen,
             Mika T.},
  journal = {Acta Radiologica},
  year    = {2024},
  doi     = {10.1177/02841851241289355}
}

@article{seilanian2025angles,
  title   = {Lumbosacral vertebral angles can predict lumbosacral transitional
             vertebrae on routine sagittal {MRI}},
  author  = {Seilanian Toosi, Farrokh and Mahdianfar, Bahare and Zarifian,
             Ahmadreza and others},
  journal = {Arch Bone Jt Surg},
  volume  = {13},
  number  = {5},
  pages   = {271--280},
  year    = {2025}
}
```

## License

The pipeline code in this repository is released under the **MIT License**.

The underlying VerSe-derived imagery and segmentation labels are redistributed
under **CC BY 4.0**, the license of the parent VerSe distributions
(Sekuboyina et al., 2021). Users must comply with VerSe's attribution terms.

VERIDAH-derived label corrections are redistributed under the same CC BY 4.0
terms, consistent with the upstream model's release (Möller et al., 2026).

See `LICENSE` for full text.
