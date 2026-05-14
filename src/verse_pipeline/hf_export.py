"""
hf_export.py — package the VerSeFusion dataset for HuggingFace upload.

Two phases:

  STAGE   Build a HF-compliant directory tree at staging_dir, pulling files
          from data/corrected/ where VERIDAH applied a correction and from
          data/canonical/ otherwise.  Writes README.md (dataset card with
          YAML frontmatter), splits.csv, and per-subject directories with
          ct.nii.gz, mask.nii.gz, and meta.json.

  UPLOAD  Push staging_dir to a HuggingFace dataset repo via huggingface_hub.
          (Skipped if --no_upload; useful for inspecting staging output.)

ORIENTATION GATE
----------------
Before staging, the script runs the same checks as orientation_audit.py and
REFUSES TO PROCEED if any scan's CT or mask isn't PIR-oriented.  This is the
guard against "scans positioned inconsistently" — if it passes, every render
and every NIfTI on HF will be in the same canonical frame.

The staging step ALSO re-verifies the staged files (in case anything went
wrong during copy/symlink), so the orientation guarantee is end-to-end.

Output structure
----------------
  staging_dir/
    README.md                      ← dataset card (YAML frontmatter + body)
    LICENSE                        ← CC-BY-4.0 by default
    splits.csv                     ← series_id, split (training/validation/test)
    orientation_audit.json         ← proof of PIR consistency
    scans/
      <series_id>/
        ct.nii.gz
        mask.nii.gz
        meta.json                  ← per-scan provenance
    corrections/
      veridah_manifest.json        ← which subjects had labels corrected
    previews/                      (optional, if --include_previews)
      <series_id>.png

Authentication: requires HF_TOKEN in env (or `huggingface-cli login` cache).
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import os
import shutil
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import numpy as np
import nibabel as nib

log = logging.getLogger("verse.hf_export")

EXPECTED_AXCODES: tuple[str, str, str] = ("P", "I", "R")


# =============================================================================
# orientation gate (lightweight, no per-scan I/O of full volumes)
# =============================================================================

def _verify_one(scan_dir_str: str) -> dict[str, Any]:
    scan_dir = Path(scan_dir_str)
    series_id = scan_dir.name.replace("scan-", "")
    meta_path = scan_dir / f"scan-{series_id}_meta.json"
    if not meta_path.exists():
        return {"series_id": series_id, "passes": False,
                "error": f"no meta.json at {meta_path}"}
    meta = json.loads(meta_path.read_text())
    try:
        ct_img  = nib.load(str(meta["source_paths"]["ct"]))
        msk_img = nib.load(str(meta["source_paths"]["msk"]))
        ct_ax   = nib.aff2axcodes(ct_img.affine)
        msk_ax  = nib.aff2axcodes(msk_img.affine)
    except Exception as e:
        return {"series_id": series_id, "passes": False,
                "error": f"{type(e).__name__}: {e}"}
    passes = (ct_ax == EXPECTED_AXCODES and msk_ax == EXPECTED_AXCODES and
              tuple(ct_img.shape[:3]) == tuple(msk_img.shape[:3]))
    return {
        "series_id":   series_id,
        "passes":      passes,
        "ct_axcodes":  list(ct_ax),
        "msk_axcodes": list(msk_ax),
        "ct_shape":    list(ct_img.shape[:3]),
        "msk_shape":   list(msk_img.shape[:3]),
    }


def orientation_gate(input_dir: Path, workers: int = 8) -> dict[str, Any]:
    """Return a summary; raises RuntimeError if any scan fails PIR check."""
    scan_dirs = sorted(d for d in input_dir.iterdir()
                       if d.is_dir() and d.name.startswith("scan-"))
    log.info("Orientation gate: checking %d scans in %s", len(scan_dirs), input_dir)

    results: list[dict[str, Any]] = []
    with ProcessPoolExecutor(max_workers=workers) as ex:
        futures = {ex.submit(_verify_one, str(d)): d for d in scan_dirs}
        for fut in as_completed(futures):
            results.append(fut.result())

    results.sort(key=lambda r: r.get("series_id", ""))
    failed = [r for r in results if not r.get("passes")]

    if failed:
        log.error("ORIENTATION GATE FAILED: %d / %d scans not PIR",
                  len(failed), len(results))
        for r in failed[:20]:
            log.error("  %s  ct=%s  msk=%s",
                      r["series_id"], r.get("ct_axcodes"), r.get("msk_axcodes"))
        if len(failed) > 20:
            log.error("  ... and %d more", len(failed) - 20)
        raise RuntimeError(
            f"Orientation gate failed: {len(failed)} / {len(results)} scans "
            f"are not PIR-oriented.  Run reorient stage and fix before upload."
        )

    log.info("Orientation gate PASSED: all %d scans are PIR-oriented.",
             len(results))
    return {
        "input_dir":      str(input_dir.resolve()),
        "n_scans":        len(results),
        "n_passes":       len(results) - len(failed),
        "expected_axcodes": list(EXPECTED_AXCODES),
        "subjects":       results,
    }


# =============================================================================
# resolve sources (corrected overrides canonical)
# =============================================================================

def _resolve_source(series_id: str,
                    canonical_dir: Path,
                    corrected_dir: Path | None) -> tuple[Path, str]:
    """Return (scan_dir_to_use, source_tag).  Prefer corrected/ if VERIDAH applied."""
    if corrected_dir is not None:
        corr_scan = corrected_dir / f"scan-{series_id}"
        corr_meta = corr_scan / f"scan-{series_id}_meta.json"
        if corr_meta.exists():
            meta = json.loads(corr_meta.read_text())
            if meta.get("veridah_applied"):
                return corr_scan, "corrected"
    return canonical_dir / f"scan-{series_id}", "canonical"


# =============================================================================
# sample selection — score by label completeness, take top N
# =============================================================================

def _score_one(args_tuple: tuple[str, str, str | None]) -> dict[str, Any]:
    """Count unique vertebra labels in one mask (chunked streaming to keep RAM low)."""
    series_id, canonical_dir_str, corrected_dir_str = args_tuple
    canonical_dir = Path(canonical_dir_str)
    corrected_dir = Path(corrected_dir_str) if corrected_dir_str else None

    src_scan_dir, source_tag = _resolve_source(series_id, canonical_dir, corrected_dir)
    meta_path = src_scan_dir / f"scan-{series_id}_meta.json"
    if not meta_path.exists():
        return {"series_id": series_id, "error": f"no meta.json at {meta_path}"}
    meta = json.loads(meta_path.read_text())

    msk_path = Path(meta["source_paths"]["msk"])
    if not msk_path.exists():
        return {"series_id": series_id, "error": f"mask not on disk: {msk_path}"}

    try:
        img = nib.load(str(msk_path))
        shape = tuple(int(s) for s in img.shape[:3])
        biggest_axis = int(np.argmax(shape))
        chunk_size = 32
        labels: set[int] = set()
        for start in range(0, shape[biggest_axis], chunk_size):
            end = min(start + chunk_size, shape[biggest_axis])
            slicer = [slice(None)] * 3
            slicer[biggest_axis] = slice(start, end)
            chunk = np.asarray(img.dataobj[tuple(slicer)])
            for v in np.unique(chunk):
                if v != 0:
                    labels.add(int(v))
    except Exception as e:
        return {"series_id": series_id, "error": f"{type(e).__name__}: {e}"}

    return {
        "series_id":       series_id,
        "n_labels":        len(labels),
        "labels":          sorted(labels),
        "veridah_applied": bool(meta.get("veridah_applied", False)),
        "source_tag":      source_tag,
    }


def select_anomaly_ids_from_manifest(
    manifest_csv: Path,
) -> tuple[list[str], list[dict[str, Any]]]:
    """Pick every scan with an LSTV/TLTV/anomaly class label.

    Includes scans where lstv_class is one of:
      t13_supernumerary, lumbarization, truncated.
    Excludes 'normal' (and any other unrecognized class).

    Returns (selected_series_ids, scoring_records).
    """
    if not manifest_csv.exists():
        raise FileNotFoundError(
            f"manifest.csv required for LSTV selection but not found at "
            f"{manifest_csv}.  Run `make manifest-slurm` first."
        )
    import pandas as pd
    df = pd.read_csv(manifest_csv)
    if "lstv_class" not in df.columns:
        raise ValueError(
            f"{manifest_csv} is missing the 'lstv_class' column.  Re-run "
            f"`make manifest-slurm` to regenerate."
        )

    anomaly_classes = ("t13_supernumerary", "lumbarization", "truncated")
    sub = df[df["lstv_class"].isin(anomaly_classes)].copy()
    sub = sub.sort_values(
        by=["lstv_class", "series_id"],
        ascending=[True, True],
    )

    log.info("Selected %d scans for LSTV/anomaly export:", len(sub))
    for cls in anomaly_classes:
        n = int((sub["lstv_class"] == cls).sum())
        log.info("  %-22s  %d", cls, n)
    log.info("  (excluded normal: %d)",
             int((df["lstv_class"] == "normal").sum()))

    scoring_records: list[dict[str, Any]] = []
    for _, row in sub.iterrows():
        scoring_records.append({
            "series_id":       row["series_id"],
            "lstv_class":      row["lstv_class"],
            "n_labels":        int(row["n_labels"]) if pd.notna(row["n_labels"]) else 0,
            "veridah_applied": bool(row.get("veridah_applied", False)),
        })
    return sub["series_id"].tolist(), scoring_records


def select_sample_ids_from_manifest(
    manifest_csv: Path,
    n:            int,
) -> tuple[list[str], list[dict[str, Any]]]:
    """Pick top-N scans by label completeness from the master manifest.

    Sort order (deterministic):
      1. n_labels DESC       — most-annotated scans first
      2. veridah_applied first — showcase the corrections
      3. series_id ASC       — stable tiebreak

    Reads the per-scan `n_labels` from data/manifest/manifest.csv (already
    computed by the LSTV audit in stage 8 and surfaced in stage 10a), so
    this is a single CSV read — no mask I/O.
    """
    if not manifest_csv.exists():
        raise FileNotFoundError(
            f"manifest.csv required for sample selection but not found at "
            f"{manifest_csv}.  Run `make manifest-slurm` first."
        )
    import pandas as pd
    df = pd.read_csv(manifest_csv)
    if "n_labels" not in df.columns:
        raise ValueError(
            f"{manifest_csv} is missing the 'n_labels' column.  Re-run "
            f"`make manifest-slurm` to regenerate."
        )

    log.info("Selecting top %d sample scans by label completeness from %s",
             n, manifest_csv.name)

    df_sorted = df.sort_values(
        by=["n_labels", "veridah_applied", "series_id"],
        ascending=[False, False, True],
        na_position="last",
    ).head(n)

    log.info("Selected %d scans for sample:", len(df_sorted))
    log.info("  %-12s  %-10s  %-8s  %s", "series_id", "n_labels", "veridah", "lstv_class")
    for _, row in df_sorted.iterrows():
        log.info("  %-12s  %-10d  %-8s  %s",
                 row["series_id"],
                 int(row["n_labels"]) if pd.notna(row["n_labels"]) else 0,
                 str(bool(row.get("veridah_applied", False))).lower(),
                 row.get("lstv_class", "?"))

    scoring_records: list[dict[str, Any]] = []
    for _, row in df_sorted.iterrows():
        scoring_records.append({
            "series_id":       row["series_id"],
            "n_labels":        int(row["n_labels"]) if pd.notna(row["n_labels"]) else 0,
            "veridah_applied": bool(row.get("veridah_applied", False)),
            "lstv_class":      row.get("lstv_class"),
            "source_tag":      "manifest",
        })
    return df_sorted["series_id"].tolist(), scoring_records


def select_sample_ids(canonical_dir: Path,
                       corrected_dir: Path | None,
                       n: int,
                       workers: int = 8) -> tuple[list[str], list[dict[str, Any]]]:
    """Return (top_n_series_ids, full_scoring_results).

    DEPRECATED legacy path that reads mask files via chunked streaming.
    Slow (~minutes for 374 scans on networked storage).  Kept as a
    fallback only.  Prefer `select_sample_ids_from_manifest` whenever a
    manifest.csv is available.

    Scoring criteria, in order:
      1. n_labels descending (most-annotated scans first)
      2. veridah_applied first (showcase the corrections)
      3. series_id ascending (deterministic tiebreak)
    """
    scan_dirs = sorted(d for d in canonical_dir.iterdir()
                       if d.is_dir() and d.name.startswith("scan-"))
    log.warning("Falling back to slow mask-scanning sample selection.  "
                "Pass --manifest_csv for a fast path (~1000× faster).")
    log.info("Scoring %d scans for sample selection (target top %d by label completeness)",
             len(scan_dirs), n)

    args_list = [
        (d.name.replace("scan-", ""), str(canonical_dir),
         str(corrected_dir) if corrected_dir else None)
        for d in scan_dirs
    ]

    results: list[dict[str, Any]] = []
    n_done = 0
    last_log = time.monotonic()
    start = last_log

    with ProcessPoolExecutor(max_workers=workers) as ex:
        futures = {ex.submit(_score_one, a): a[0] for a in args_list}
        for fut in as_completed(futures):
            r = fut.result()
            results.append(r)
            n_done += 1
            if time.monotonic() - last_log >= 5.0:
                elapsed = time.monotonic() - start
                rate = n_done / elapsed if elapsed > 0 else 0.0
                log.info("  scored %d/%d  (%.1f scans/s)", n_done, len(scan_dirs), rate)
                last_log = time.monotonic()

    valid = [r for r in results if "error" not in r]
    invalid = [r for r in results if "error" in r]
    if invalid:
        log.warning("  %d scans skipped due to errors:", len(invalid))
        for r in invalid[:5]:
            log.warning("    %s: %s", r["series_id"], r["error"])

    valid.sort(key=lambda r: (-r["n_labels"], not r["veridah_applied"], r["series_id"]))
    selected = valid[:n]

    log.info("Selected %d scans for sample:", len(selected))
    log.info("  %-12s  %-10s  %-8s  %s", "series_id", "n_labels", "veridah", "source")
    for r in selected:
        log.info("  %-12s  %-10d  %-8s  %s",
                 r["series_id"], r["n_labels"],
                 str(r["veridah_applied"]).lower(), r["source_tag"])

    return [r["series_id"] for r in selected], valid


# =============================================================================
# per-subject stage
# =============================================================================

def _stage_one(args: tuple[str, str, str | None, str, str, dict[str, Any] | None, str]) -> dict[str, Any]:
    (series_id, canonical_dir_str, corrected_dir_str, staging_scans_dir_str,
     preview_src_dir_str, splits_lookup, stage_mode) = args

    canonical_dir = Path(canonical_dir_str)
    corrected_dir = Path(corrected_dir_str) if corrected_dir_str else None
    src_scan_dir, source_tag = _resolve_source(series_id, canonical_dir, corrected_dir)

    src_meta_path = src_scan_dir / f"scan-{series_id}_meta.json"
    if not src_meta_path.exists():
        return {"series_id": series_id, "error": f"meta.json missing at {src_meta_path}"}
    src_meta = json.loads(src_meta_path.read_text())

    src_ct  = Path(src_meta["source_paths"]["ct"])
    src_msk = Path(src_meta["source_paths"]["msk"])
    if not src_ct.exists() or not src_msk.exists():
        return {"series_id": series_id,
                "error": f"source NIfTI missing (ct={src_ct.exists()} msk={src_msk.exists()})"}

    out_subject_dir = Path(staging_scans_dir_str) / series_id
    out_subject_dir.mkdir(parents=True, exist_ok=True)

    out_ct   = out_subject_dir / "ct.nii.gz"
    out_msk  = out_subject_dir / "mask.nii.gz"
    out_meta = out_subject_dir / "meta.json"

    # Materialize.  Default: hardlink (instant; same-inode pointer on the same
    # filesystem).  Falls back to copy automatically if hardlink fails (e.g.,
    # canonical/ and staging/ are on different filesystems).  Hardlinks are
    # transparent to HF upload — `upload_folder` reads file bytes the same way
    # regardless of how they got there.  Explicit `--stage_mode copy` does a
    # real byte copy (slow, useful only if you need a self-contained tarball
    # off the source FS); `--stage_mode symlink` makes pointers (also fine
    # for upload but doesn't survive cross-FS rsync without --copy-links).
    pairs = [(src_ct, out_ct), (src_msk, out_msk)]
    for src, dst in pairs:
        if dst.exists() or dst.is_symlink():
            dst.unlink()
    if stage_mode == "symlink":
        for src, dst in pairs:
            dst.symlink_to(src.resolve())
        materialize_method = "symlink"
    elif stage_mode == "copy":
        for src, dst in pairs:
            shutil.copy2(str(src), str(dst))
        materialize_method = "copy"
    else:   # hardlink (default), with copy fallback
        materialize_method = "hardlink"
        for src, dst in pairs:
            try:
                os.link(str(src), str(dst))
            except OSError:
                shutil.copy2(str(src), str(dst))
                materialize_method = "copy_fallback"

    # Re-verify the staged files are PIR (defense in depth: ensure the upload
    # bundle itself is consistent, even if someone hand-edited canonical/).
    ct_img  = nib.load(str(out_ct))
    msk_img = nib.load(str(out_msk))
    ct_ax   = nib.aff2axcodes(ct_img.affine)
    msk_ax  = nib.aff2axcodes(msk_img.affine)
    if ct_ax != EXPECTED_AXCODES or msk_ax != EXPECTED_AXCODES:
        return {"series_id": series_id,
                "error": f"staged NIfTI not PIR  ct={ct_ax} msk={msk_ax}"}

    spacing = [float(np.linalg.norm(ct_img.affine[:3, k])) for k in range(3)]
    shape   = list(int(s) for s in ct_img.shape[:3])

    # Clean meta.json: strip absolute source_paths (which are private),
    # keep provenance fields.
    out_meta_dict = {
        "series_id":     series_id,
        "source_tag":    source_tag,                 # "canonical" or "corrected"
        "files": {
            "ct":   "ct.nii.gz",
            "mask": "mask.nii.gz",
        },
        "orientation_axcodes": list(EXPECTED_AXCODES),
        "spacing_mm":    spacing,
        "shape":         shape,
        "split":         (splits_lookup or {}).get(series_id),
    }
    # Carry through provenance fields if present
    for key in ("patient_id", "source_format", "verse_subset", "verse_split",
                "veridah_applied", "veridah_correction_type", "veridah_remap",
                "tltv", "sr_left", "sr_right"):
        if key in src_meta:
            out_meta_dict[key] = src_meta[key]

    out_meta.write_text(json.dumps(out_meta_dict, indent=2))

    # Optionally copy a preview PNG if available
    preview_status = "skipped"
    preview_src_dir = Path(preview_src_dir_str) if preview_src_dir_str else None
    if preview_src_dir is not None and preview_src_dir.is_dir():
        for cand in [preview_src_dir / f"{series_id}.png",
                     preview_src_dir / f"{series_id}_before_after.png"]:
            if cand.exists():
                dst = out_subject_dir.parent.parent / "previews" / f"{series_id}.png"
                dst.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(cand, dst)
                preview_status = "copied"
                break

    return {
        "series_id":          series_id,
        "source_tag":         source_tag,
        "shape":              shape,
        "spacing_mm":         spacing,
        "preview_status":     preview_status,
        "materialize_method": materialize_method,
    }


# =============================================================================
# README / splits / LICENSE
# =============================================================================

LICENSE_TEXT = """Creative Commons Attribution 4.0 International (CC BY 4.0)

You are free to:
  - Share — copy and redistribute the material in any medium or format
  - Adapt — remix, transform, and build upon the material for any purpose,
    even commercially.

Under the following terms:
  - Attribution — You must give appropriate credit, provide a link to the
    license, and indicate if changes were made.

Full text: https://creativecommons.org/licenses/by/4.0/legalcode
"""

README_TEMPLATE = """---
license: cc-by-4.0
task_categories:
- image-segmentation
language:
- en
size_categories:
- n<1K
tags:
- medical-imaging
- spine
- ct
- segmentation
- vertebra
- lstv
- tltv
- verse
pretty_name: "VerSeFusion: Re-fused VerSe 2019+2020 with VERIDAH corrections"
---

# {dataset_pretty_name}

A re-fused, PIR-canonical version of the VerSe 2019 and VerSe 2020 vertebra
segmentation challenges, with VERIDAH (Möller 2026) label corrections applied
for thoracolumbar transitional vertebrae.

## Dataset stats

- **Total scans:** {n_scans}
- **Total patients:** {n_patients}
- **Splits:** training={n_train}, validation={n_val}, test={n_test}
- **Source:** VerSe 2019 + VerSe 2020 (combined) with VERIDAH corrections
- **Canonical orientation:** PIR (axis 0 = P, axis 1 = I, axis 2 = R)
- **VERIDAH-corrected subjects:** {n_corrected}

## Orientation

Every scan in this dataset has been reoriented to a single canonical frame:

- **axis 0** increases toward **P** (posterior — i.e., anterior → posterior)
- **axis 1** increases toward **I** (inferior — i.e., superior → inferior; this is the spine axis)
- **axis 2** increases toward **R** (right — i.e., left → right)

This is verified end-to-end: see `orientation_audit.json` for the
per-subject report.  Rendering conventions in `previews/`:

- **Coronal:** head at top, patient's right at viewer's right
- **Axial:** anterior at top, patient's right at viewer's right
- **Sagittal:** head at top, anterior at left

## Structure

```
{repo_id}/
├── README.md
├── LICENSE
├── splits.csv                  # series_id → split (training/validation/test)
├── orientation_audit.json      # per-subject orientation verification
├── scans/
│   └── <series_id>/
│       ├── ct.nii.gz           # CT volume, HU values, PIR-oriented
│       ├── mask.nii.gz         # vertebra labels (uint8), PIR-oriented
│       └── meta.json           # per-scan provenance
├── corrections/
│   └── veridah_manifest.json   # which subjects had labels corrected
└── previews/                   # optional QC renders
    └── <series_id>.png
```

## Label schema

| Label | Anatomy | | Label | Anatomy |
|-------|---------|-|-------|---------|
| 1–7   | C1–C7   | | 20    | L1 |
| 8     | T1      | | 21    | L2 |
| 9     | T2      | | 22    | L3 |
| 10    | T3      | | 23    | L4 |
| 11    | T4      | | 24    | L5 |
| 12    | T5      | | 25    | L6 (supernumerary lumbar) |
| 13    | T6      | | 26    | sacrum (variably annotated) |
| 14    | T7      | | 27    | coccyx |
| 15    | T8      | | 28    | T13 (supernumerary thoracic) |
| 16    | T9      | | | |
| 17    | T10     | | | |
| 18    | T11     | | | |
| 19    | T12     | | | |

## Loading example

```python
import nibabel as nib

ct  = nib.load("scans/verse001/ct.nii.gz")
msk = nib.load("scans/verse001/mask.nii.gz")

# Both are guaranteed to be PIR-oriented:
assert nib.aff2axcodes(ct.affine)  == ('P', 'I', 'R')
assert nib.aff2axcodes(msk.affine) == ('P', 'I', 'R')
```

## Citation

If you use this dataset, please cite the original VerSe challenges and the
VERIDAH corrections paper:

```bibtex
@article{{sekuboyina2021verse,
  title={{VerSe: A vertebrae labelling and segmentation benchmark for multi-detector CT images}},
  author={{Sekuboyina, A. and others}},
  journal={{Medical Image Analysis}},
  year={{2021}}
}}

@article{{loffler2020verse2020,
  title={{A vertebral segmentation dataset with fracture grading}},
  author={{Löffler, M.T. and others}},
  journal={{Radiology: Artificial Intelligence}},
  year={{2020}}
}}

@article{{moller2026veridah,
  title={{VERIDAH: Vertebral identification and transitional anomaly detection}},
  author={{Möller, H. and others}},
  year={{2026}}
}}
```

## Acknowledgments

VerSe challenge data: Technical University Munich.  VERIDAH corrections:
H. Möller et al. (2026).
"""


def write_license(staging_dir: Path) -> Path:
    p = staging_dir / "LICENSE"
    p.write_text(LICENSE_TEXT)
    return p


def write_splits_csv(staging_dir: Path,
                     splits_lookup: dict[str, str],
                     ordered_ids: list[str]) -> Path:
    p = staging_dir / "splits.csv"
    with p.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["series_id", "split"])
        for sid in ordered_ids:
            w.writerow([sid, splits_lookup.get(sid, "")])
    return p


def write_readme(staging_dir: Path,
                 dataset_pretty_name: str,
                 repo_id: str,
                 stats: dict[str, Any],
                 is_sample: bool = False,
                 parent_repo: str | None = None) -> Path:
    body = README_TEMPLATE.format(
        dataset_pretty_name=dataset_pretty_name,
        repo_id=repo_id,
        n_scans=stats.get("n_scans", "?"),
        n_patients=stats.get("n_patients", "?"),
        n_train=stats.get("n_train", "?"),
        n_val=stats.get("n_val", "?"),
        n_test=stats.get("n_test", "?"),
        n_corrected=stats.get("n_corrected", "?"),
    )
    if is_sample:
        sample_note = (
            "\n\n## Note: this is a sample\n\n"
            f"This is a {stats.get('n_scans')}-scan sample from the full dataset, "
            f"chosen as the most-completely-labeled scans (highest unique-vertebra-label "
            f"count, with VERIDAH-corrected subjects prioritized to showcase the "
            f"thoracolumbar transitional-vertebra corrections).\n\n"
            f"For the full {dataset_pretty_name.replace('-Sample', '')} dataset, see: "
            f"https://huggingface.co/datasets/{parent_repo or '...'}\n"
        )
        body = body + sample_note
    p = staging_dir / "README.md"
    p.write_text(body)
    return p


def copy_veridah_manifest(corrected_dir: Path | None,
                          staging_dir: Path,
                          subset_ids: set[str] | None = None) -> Path | None:
    if corrected_dir is None:
        return None
    src = corrected_dir / "veridah_manifest.json"
    if not src.exists():
        return None
    manifest = json.loads(src.read_text())
    if subset_ids is not None:
        # Filter corrections list to only subjects in the subset
        manifest["corrections"] = [c for c in manifest.get("corrections", [])
                                   if c.get("series_id") in subset_ids]
        manifest["subset_filtered"] = True
        manifest["n_corrected"] = sum(1 for c in manifest["corrections"]
                                       if c.get("veridah_applied"))
    dst = staging_dir / "corrections" / "veridah_manifest.json"
    dst.parent.mkdir(parents=True, exist_ok=True)
    dst.write_text(json.dumps(manifest, indent=2))
    return dst


def copy_manifest(manifest_csv: Path | None,
                   staging_dir: Path,
                   subset_ids: set[str] | None = None) -> Path | None:
    """Copy manifest.csv, manifest.json, splits_5fold.json, and the
    dataset_interface.py shim into the staging dir.

    If `subset_ids` is given, manifest rows are filtered to that subset
    AND splits_5fold.json is filtered too (test/train/val sets restricted
    to subjects in the subset).  This is how the sample export's manifest
    stays consistent with the 10 scans it actually ships.
    """
    if manifest_csv is None:
        return None
    if not manifest_csv.exists():
        log.warning("manifest.csv not found at %s; staging dir will lack manifest",
                    manifest_csv)
        return None

    import pandas as pd
    manifest_dir = manifest_csv.parent

    # --- manifest.csv ---------------------------------------------------------
    df = pd.read_csv(manifest_csv)
    if subset_ids is not None:
        df = df[df["series_id"].isin(subset_ids)].reset_index(drop=True)
    dst_csv = staging_dir / "manifest.csv"
    df.to_csv(dst_csv, index=False)
    log.info("Wrote %s (%d rows)", dst_csv, len(df))

    # --- manifest.json --------------------------------------------------------
    src_json = manifest_dir / "manifest.json"
    if src_json.exists():
        man = json.loads(src_json.read_text())
        if subset_ids is not None:
            man["subjects"] = [s for s in man.get("subjects", [])
                                if s.get("series_id") in subset_ids]
            man["n_subjects"] = len(man["subjects"])
            man["subset_filtered"] = True
        (staging_dir / "manifest.json").write_text(
            json.dumps(man, indent=2, default=str)
        )
        log.info("Wrote %s (%d subjects)", staging_dir / "manifest.json",
                 man.get("n_subjects", "?"))

    # --- manifest_summary.json (only for full export) -------------------------
    if subset_ids is None:
        src_summary = manifest_dir / "manifest_summary.json"
        if src_summary.exists():
            shutil.copy2(src_summary, staging_dir / "manifest_summary.json")

    # --- splits_5fold.json ----------------------------------------------------
    src_splits = manifest_dir / "splits_5fold.json"
    if src_splits.exists():
        doc = json.loads(src_splits.read_text())
        if subset_ids is not None:
            keep = subset_ids
            doc["test_series_ids"]   = [s for s in doc.get("test_series_ids", []) if s in keep]
            doc["test_patients"]     = sorted({p for p, pa in doc.get("patient_attrs", {}).items()
                                                if any(s in keep for s in pa.get("series_ids", []))
                                                and p in doc.get("test_patients", [])})
            for f in doc.get("folds", []):
                f["train_series_ids"] = [s for s in f.get("train_series_ids", []) if s in keep]
                f["val_series_ids"]   = [s for s in f.get("val_series_ids",   []) if s in keep]
            doc["subset_filtered"] = True
        (staging_dir / "splits_5fold.json").write_text(
            json.dumps(doc, indent=2, default=str)
        )
        log.info("Wrote %s", staging_dir / "splits_5fold.json")
    else:
        log.warning("splits_5fold.json not found at %s — staged dataset will lack CV folds",
                    src_splits)

    # --- dataset_interface.py shim --------------------------------------------
    # Ship the Python loader inside the dataset itself, matching the
    # CTSpinoPelvic1K convention.  Try the repo first, then the src/ dir
    # adjacent to wherever the manifest builder ran.
    import verse_pipeline
    pkg_dir = Path(verse_pipeline.__file__).parent
    src_interface = pkg_dir / "dataset_interface.py"
    if src_interface.exists():
        shutil.copy2(src_interface, staging_dir / "dataset_interface.py")
        log.info("Wrote %s", staging_dir / "dataset_interface.py")

    return dst_csv


def write_radiologist_review_csv(
    staging_dir:       Path,
    manifest_csv_path: Path | None,
    lstv_audit_path:   Path | None = None,
    subset_ids:        set[str] | None = None,
) -> Path | None:
    """Write a radiologist-friendly CSV summarizing each scan's anomaly status.

    Columns are projected from manifest.csv and (when available) joined with
    the LSTV audit's per-scan evidence strings — the auditor's plain-English
    reasoning for why a scan was classified as e.g. "lumbarization" vs
    "lsj_fov_truncated".  These evidence strings are typically what a
    radiologist wants when deciding which cases to spot-check.

    Sorted with anomalies on top (t13 > lumb > trunc > normal), then by
    series_id.
    """
    if manifest_csv_path is None or not manifest_csv_path.exists():
        log.warning("Skipping radiologist_review.csv — no manifest.csv at %s",
                    manifest_csv_path)
        return None

    import pandas as pd
    df = pd.read_csv(manifest_csv_path)
    if subset_ids is not None:
        df = df[df["series_id"].astype(str).isin(subset_ids)].copy()

    # Join in evidence strings from the LSTV audit if we can find it
    evidence: dict[str, dict[str, str]] = {}
    if lstv_audit_path and lstv_audit_path.exists():
        try:
            audit = json.loads(lstv_audit_path.read_text())
            for sub in audit.get("subjects", []) or []:
                sid = sub.get("series_id")
                if sid:
                    evidence[str(sid)] = {
                        "lstv_evidence": sub.get("lstv_evidence", "") or "",
                        "tltv_evidence": sub.get("tltv_evidence", "") or "",
                    }
        except (OSError, ValueError) as e:
            log.warning("Could not read LSTV audit at %s for evidence join: %s",
                        lstv_audit_path, e)

    df["lstv_evidence"] = df["series_id"].astype(str).map(
        lambda s: evidence.get(s, {}).get("lstv_evidence", ""))
    df["tltv_evidence"] = df["series_id"].astype(str).map(
        lambda s: evidence.get(s, {}).get("tltv_evidence", ""))

    # Project radiologist-facing columns (in this order); skip any that don't exist
    radiologist_cols = [
        "series_id",
        "lstv_class",            # 4-way summary cohort
        "lstv_class_audit",      # raw LSJ verdict from auditor
        "tltv_class_audit",      # raw TLJ verdict from auditor
        "lstv_evidence",         # plain-English reasoning
        "tltv_evidence",
        "n_labels",
        "labels_present",
        "has_T13",
        "has_L6",
        "lacks_T12_TLJ_in_FOV",
        "veridah_applied",
        "veridah_kind",
        "veridah_action",
        "source_dataset",
        "source_format",
        "sex",
        "age",
        "patient_id",
        "split",                 # VerSe's original (not ML splits)
        "ct_relative_path",
        "mask_relative_path",
    ]
    cols = [c for c in radiologist_cols if c in df.columns]
    out_df = df[cols].copy()

    # Sort: anomalies first by clinical priority, then by series_id
    class_order = {
        "t13_supernumerary": 0,
        "lumbarization":     1,
        "truncated":         2,
        "normal":            3,
    }
    out_df["_sort_key"] = out_df["lstv_class"].map(class_order).fillna(99)
    out_df = (out_df.sort_values(["_sort_key", "series_id"])
                    .drop(columns=["_sort_key"])
                    .reset_index(drop=True))

    out_path = staging_dir / "radiologist_review.csv"
    out_df.to_csv(out_path, index=False)
    log.info("Wrote radiologist_review.csv (%d rows, %d columns) — %s",
             len(out_df), len(cols), out_path)
    return out_path


def write_browse_by_class_md(
    staging_dir:       Path,
    manifest_csv_path: Path | None,
    subset_ids:        set[str] | None = None,
) -> Path | None:
    """Write a 'browse by anomaly class' markdown index for the dataset.

    Groups every scan under its `lstv_class`, with internal links to its
    scans/<series_id>/ directory.  Renders nicely on HuggingFace's dataset
    file viewer.  Drops the 'normal' section if there are more than 50
    normals (too long to be useful as a browse-by-class index).
    """
    if manifest_csv_path is None or not manifest_csv_path.exists():
        return None

    import pandas as pd
    df = pd.read_csv(manifest_csv_path)
    if subset_ids is not None:
        df = df[df["series_id"].astype(str).isin(subset_ids)].copy()
    if len(df) == 0:
        return None

    class_titles = {
        "t13_supernumerary": "T13 supernumerary  (extra thoracic vertebra)",
        "lumbarization":     "Lumbarization  (L6 present — sixth lumbar / S1 lumbarized)",
        "truncated":         "T12 absent  (genuine absence, not FOV truncation)",
        "normal":            "Normal lumbosacral / thoracolumbar anatomy",
    }
    class_order = ["t13_supernumerary", "lumbarization", "truncated", "normal"]

    total = len(df)
    n_anom = int((df["lstv_class"] != "normal").sum()) if "lstv_class" in df.columns else 0

    lines = [
        "# Browse by anomaly class",
        "",
        f"Total scans: **{total}** — of which **{n_anom}** carry an LSTV/TLTV "
        "anomaly (`lstv_class != normal`).",
        "",
        "Click any series_id to navigate to its scan directory; CT and mask "
        "files are at `scans/{series_id}/ct.nii.gz` and `mask.nii.gz`.  See "
        "`radiologist_review.csv` for the auditor's plain-English evidence "
        "strings per scan.",
        "",
        "---",
        "",
    ]

    for cls in class_order:
        sub = df[df["lstv_class"] == cls] if "lstv_class" in df.columns else df.head(0)
        n = len(sub)
        if n == 0:
            continue
        # For the full corpus, the normal section would be 300+ entries — skip it
        if cls == "normal" and n > 50:
            lines.append(f"## normal  ({n} scans)")
            lines.append("")
            lines.append(f"_{n} normal-anatomy scans omitted from this index.  "
                         f"See `radiologist_review.csv` for the complete list._")
            lines.append("")
            continue
        lines.append(f"## {cls}  —  {class_titles.get(cls, cls)}  ({n} scans)")
        lines.append("")
        for _, row in sub.sort_values("series_id").iterrows():
            sid     = row["series_id"]
            n_lbl   = int(row["n_labels"]) if pd.notna(row.get("n_labels")) else 0
            tags    = []
            if bool(row.get("veridah_applied", False)):
                kind = row.get("veridah_kind") or "veridah-corrected"
                tags.append(f"_{kind}_")
            tag_str = "  " + " • ".join(tags) if tags else ""
            lines.append(f"- [`{sid}`](scans/{sid}/)  —  {n_lbl} labels{tag_str}")
        lines.append("")

    out = staging_dir / "BROWSE_BY_CLASS.md"
    out.write_text("\n".join(lines))
    log.info("Wrote browse-by-class index: %s (%d sections)", out,
             sum(1 for cls in class_order if len(df[df.get("lstv_class") == cls]) > 0))
    return out


def write_orientation_audit_proof(staging_dir: Path, gate_result: dict[str, Any]) -> Path:
    p = staging_dir / "orientation_audit.json"
    p.write_text(json.dumps(gate_result, indent=2))
    return p


def copy_orientation_audit(
    source_path: Path | None,
    staging_dir: Path,
    subset_ids:  set[str] | None = None,
) -> Path | None:
    """Copy stage 9's orientation audit manifest into the staging dir.

    This is the fast path: instead of re-running the orientation gate
    (which loads 374×2 NIfTI headers), we reuse stage 9's already-computed
    audit.  If `subset_ids` is given, per-subject results are filtered to
    just those subjects (for the sample export).

    Returns the destination Path on success, or None if no audit was
    available to copy.
    """
    if source_path is None or not source_path.exists():
        log.warning("Orientation audit not found at %s; staged dataset will "
                    "lack orientation_audit.json.  Run `make orient-slurm` "
                    "first or pass --gate to re-run the check during staging.",
                    source_path)
        return None
    try:
        audit = json.loads(source_path.read_text())
    except (OSError, ValueError) as e:
        log.warning("Could not read %s: %s — skipping orientation_audit.json",
                    source_path, e)
        return None

    if subset_ids is not None:
        # Filter any list-of-subjects field down to the subset
        for key in ("subjects", "results", "entries"):
            if isinstance(audit.get(key), list):
                audit[key] = [
                    e for e in audit[key]
                    if (e.get("series_id") or e.get("scan_id")) in subset_ids
                ]
        audit["subset_filtered"] = True
        audit["n_subset"] = len(subset_ids)

    dst = staging_dir / "orientation_audit.json"
    dst.write_text(json.dumps(audit, indent=2, default=str))
    log.info("Copied orientation audit (%d subjects) -> %s",
             sum(len(audit.get(k, [])) for k in ("subjects", "results", "entries"))
             or audit.get("total", 0) or audit.get("n_scans", 0),
             dst)
    return dst


# =============================================================================
# upload
# =============================================================================

def upload_to_hf(staging_dir: Path, repo_id: str,
                 private: bool = True, token: str | None = None) -> str:
    try:
        from huggingface_hub import HfApi, create_repo
    except ImportError:
        raise RuntimeError(
            "huggingface_hub not installed.  Run: pip install huggingface_hub"
        )

    if token is None:
        token = os.environ.get("HF_TOKEN")
    if token is None:
        log.warning("No HF_TOKEN in env — relying on `huggingface-cli login` cache.")

    log.info("Creating repo %s (private=%s) ...", repo_id, private)
    create_repo(repo_id, repo_type="dataset", private=private,
                exist_ok=True, token=token)

    api = HfApi(token=token)
    ignore_patterns = [".DS_Store", "*.pyc", "__pycache__", ".ipynb_checkpoints"]

    # Optional override — leave unset to let upload_large_folder pick its own
    # default (one worker per core).  Only set HF_UPLOAD_WORKERS if you're on
    # a low-memory node and need to bound the hashing footprint.
    upload_num_workers_env = os.environ.get("HF_UPLOAD_WORKERS")
    upload_num_workers = (int(upload_num_workers_env)
                          if upload_num_workers_env else None)

    # Use upload_large_folder when available — it parallelizes, chunks, and
    # is resumable on failure, which matters for a 22GB dataset.  Falls back
    # to plain upload_folder on older huggingface_hub installs.
    use_large = hasattr(api, "upload_large_folder")
    if use_large:
        log.info("Uploading %s -> %s via upload_large_folder "
                 "(parallel, resumable, num_workers=%s) ...",
                 staging_dir, repo_id,
                 upload_num_workers if upload_num_workers else "default")
        kwargs: dict[str, Any] = {
            "folder_path": str(staging_dir),
            "repo_id":     repo_id,
            "repo_type":   "dataset",
        }
        if upload_num_workers is not None:
            kwargs["num_workers"] = upload_num_workers
        try:
            api.upload_large_folder(**kwargs, ignore_patterns=ignore_patterns)
        except TypeError:
            log.warning("upload_large_folder rejected ignore_patterns; retrying without")
            api.upload_large_folder(**kwargs)
    else:
        log.info("Uploading %s -> %s via upload_folder ...", staging_dir, repo_id)
        api.upload_folder(
            folder_path=str(staging_dir),
            repo_id=repo_id,
            repo_type="dataset",
            commit_message="Initial dataset upload",
            ignore_patterns=ignore_patterns,
        )

    url = f"https://huggingface.co/datasets/{repo_id}"
    log.info("Upload complete: %s", url)
    return url


# =============================================================================
# orchestration
# =============================================================================

def stage_dataset(
    canonical_dir:  Path,
    corrected_dir:  Path | None,
    staging_dir:    Path,
    unify_manifest_path: Path,
    preview_dir:    Path | None = None,
    workers:        int = 8,
    stage_mode:     str = "hardlink",
    subset_series_ids: list[str] | None = None,
    run_gate:       bool = False,
    orient_audit_path: Path | None = None,
    lstv_audit_path:   Path | None = None,
    manifest_csv:   Path | None = None,
) -> dict[str, Any]:
    """Stage subjects + write top-level files.

    By default, the orientation gate is NOT re-run during staging (stage 9
    already validated all scans).  Stage 9's audit at `orient_audit_path`
    is copied into the staging dir as orientation_audit.json.  Pass
    `run_gate=True` to re-validate from scratch (slow but defensive).

    If `subset_series_ids` is given, only those subjects are staged and
    the orientation audit is filtered to that subset before writing.

    stage_mode:
        "hardlink" (default): same-FS hardlinks, instant, falls back to copy
        "copy":               full byte copy via shutil.copy2
        "symlink":            symbolic links
    """

    # 1. Orientation: by default reuse stage 9's audit (fast path).
    if run_gate:
        log.info("Running fresh orientation gate (run_gate=True)")
        gate_result = orientation_gate(canonical_dir, workers=workers)
    else:
        gate_result = None  # nothing to write from a re-run; we'll copy stage 9's

    # 2. Resolve splits + patient lookup.  Prefer manifest.csv (from stage 10a)
    # which already aggregated everything correctly; fall back to a permissive
    # reader against unify_manifest.json that handles both 'subjects' and
    # 'scans' top-level keys.
    splits_lookup: dict[str, str] = {}
    patient_lookup: dict[str, str] = {}    # series_id -> patient_id

    if manifest_csv is not None and manifest_csv.exists():
        import pandas as pd
        _df = pd.read_csv(manifest_csv)
        for _, _row in _df.iterrows():
            _sid = str(_row["series_id"])
            _sp  = _row.get("split")
            if _sp and not pd.isna(_sp):
                splits_lookup[_sid] = str(_sp)
            _pid = _row.get("patient_id")
            if _pid and not pd.isna(_pid):
                patient_lookup[_sid] = str(_pid)
        log.info("Loaded splits + patients from manifest.csv "
                 "(%d entries with splits, %d with patient_id)",
                 len(splits_lookup), len(patient_lookup))
    else:
        if not unify_manifest_path.exists():
            raise FileNotFoundError(
                f"Neither manifest.csv nor unify manifest found.  "
                f"Looked for manifest_csv={manifest_csv} and "
                f"unify_manifest_path={unify_manifest_path}."
            )
        _raw = json.loads(unify_manifest_path.read_text())
        # Permissive top-level key resolution
        _records: list[dict[str, Any]] = []
        if isinstance(_raw, list):
            _records = [r for r in _raw if isinstance(r, dict)]
        elif isinstance(_raw, dict):
            for _k in ("subjects", "scans", "records", "entries"):
                if isinstance(_raw.get(_k), list):
                    _records = [r for r in _raw[_k] if isinstance(r, dict)]
                    break
        for _sub in _records:
            _sid = _sub.get("series_id") or _sub.get("scan_id")
            if not _sid:
                continue
            _sid = str(_sid)
            _sp = _sub.get("split") or _sub.get("verse_split")
            if _sp:
                splits_lookup[_sid] = str(_sp)
            _pid = _sub.get("patient_id") or _sub.get("patient")
            if _pid:
                patient_lookup[_sid] = str(_pid)
        log.info("Loaded splits + patients from unify manifest "
                 "(%d entries with splits, %d with patient_id)",
                 len(splits_lookup), len(patient_lookup))

    # 3. Stage each subject in parallel
    scan_dirs = sorted(d for d in canonical_dir.iterdir()
                        if d.is_dir() and d.name.startswith("scan-"))
    series_ids = [d.name.replace("scan-", "") for d in scan_dirs]
    if subset_series_ids is not None:
        wanted = set(subset_series_ids)
        series_ids = [s for s in series_ids if s in wanted]
        log.info("Subset mode: %d / %d subjects selected",
                 len(series_ids), len(scan_dirs))

    staging_scans = staging_dir / "scans"
    staging_scans.mkdir(parents=True, exist_ok=True)
    if preview_dir is not None:
        (staging_dir / "previews").mkdir(parents=True, exist_ok=True)

    log.info("Staging %d subjects into %s (stage_mode=%s)",
             len(series_ids), staging_dir, stage_mode)

    args_list = [
        (sid, str(canonical_dir),
         str(corrected_dir) if corrected_dir else None,
         str(staging_scans),
         str(preview_dir) if preview_dir else None,
         splits_lookup, stage_mode)
        for sid in series_ids
    ]

    stage_results: list[dict[str, Any]] = []
    n_done = 0
    last_log = time.monotonic()
    start = last_log

    with ProcessPoolExecutor(max_workers=workers) as ex:
        futures = {ex.submit(_stage_one, a): a[0] for a in args_list}
        for fut in as_completed(futures):
            r = fut.result()
            stage_results.append(r)
            n_done += 1
            if "error" in r:
                log.warning("  %s: %s", r["series_id"], r["error"])
            if time.monotonic() - last_log >= 10.0:
                elapsed = time.monotonic() - start
                rate = n_done / elapsed if elapsed > 0 else 0.0
                log.info("  staged %d / %d  (%.1f scans/s)",
                         n_done, len(series_ids), rate)
                last_log = time.monotonic()

    stage_results.sort(key=lambda r: r.get("series_id", ""))
    n_staged = sum(1 for r in stage_results if "error" not in r)
    log.info("Stage done: %d / %d ok", n_staged, len(stage_results))

    # Summarize how files were materialized (catches hardlink-fell-back-to-copy)
    method_counts: dict[str, int] = {}
    for r in stage_results:
        m = r.get("materialize_method", "?")
        method_counts[m] = method_counts.get(m, 0) + 1
    log.info("  materialize method counts: %s", method_counts)

    # 4. Top-level files
    subset_set = set(series_ids) if subset_series_ids is not None else None
    if gate_result is not None:
        # Fresh gate was run — write its result as the proof file
        write_orientation_audit_proof(staging_dir, gate_result)
    else:
        # Copy stage 9's audit instead (filtered to subset if applicable)
        copy_orientation_audit(orient_audit_path, staging_dir, subset_set)
    write_license(staging_dir)
    write_splits_csv(staging_dir, splits_lookup, sorted(series_ids))

    # Filter veridah manifest to subset if subsetting
    copy_veridah_manifest(corrected_dir, staging_dir, subset_set)
    copy_manifest(manifest_csv, staging_dir, subset_set)

    # Radiologist-facing review CSV + browse-by-class markdown index.
    # Both are cheap (~1ms each), useful for clinician review of any subset.
    write_radiologist_review_csv(
        staging_dir, manifest_csv, lstv_audit_path, subset_set,
    )
    write_browse_by_class_md(staging_dir, manifest_csv, subset_set)

    # 5. README stats — compute per-subset
    split_counts: dict[str, int] = {}
    for sid in series_ids:
        sp = splits_lookup.get(sid, "unknown")
        split_counts[sp] = split_counts.get(sp, 0) + 1

    n_corrected = 0
    if corrected_dir is not None and (corrected_dir / "veridah_manifest.json").exists():
        vm = json.loads((corrected_dir / "veridah_manifest.json").read_text())
        if subset_set is not None:
            n_corrected = sum(1 for c in vm.get("corrections", [])
                              if c.get("series_id") in subset_set
                              and c.get("veridah_applied"))
        else:
            n_corrected = vm.get("n_corrected", 0)

    # Patient counts: use the patient_lookup we built earlier (from either
    # manifest.csv or the permissive unify reader).
    relevant_sids = (set(series_ids) if subset_series_ids is None
                     else set(series_ids))
    n_patients = len({patient_lookup[s] for s in relevant_sids
                      if s in patient_lookup})

    return {
        "n_scans":       n_staged,
        "n_patients":    n_patients,
        "n_train":       split_counts.get("training", 0),
        "n_val":         split_counts.get("validation", 0),
        "n_test":        split_counts.get("test", 0),
        "n_corrected":   n_corrected,
        "stage_results": stage_results,
        "splits_lookup": splits_lookup,
        "subset_ids":    sorted(series_ids) if subset_series_ids is not None else None,
    }


# =============================================================================
# CLI
# =============================================================================

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="verse-hf-export",
        description="Stage and (optionally) upload the VerSeFusion dataset to HuggingFace.",
    )
    p.add_argument("--canonical_dir", type=Path, required=True)
    p.add_argument("--corrected_dir", type=Path,
                   help="If given, VERIDAH-corrected masks override canonical for those subjects.")
    p.add_argument("--unify_manifest", type=Path, required=True)
    p.add_argument("--staging_dir",   type=Path, required=True)
    p.add_argument("--preview_dir",   type=Path,
                   help="Optional dir of preview PNGs (e.g., data/qc/renders/).")

    p.add_argument("--repo_id",       type=str,
                   default="gregoryschwingmdphd/VerseFusion",
                   help="HuggingFace dataset repo. Default: gregoryschwingmdphd/VerseFusion.")
    p.add_argument("--dataset_pretty_name", type=str,
                   default="VerSeFusion")
    p.add_argument("--public",        action="store_true",
                   help="Make the repo public (default: private).")

    p.add_argument("--stage_mode",
                   choices=["hardlink", "copy", "symlink"],
                   default="hardlink",
                   help="How to materialize NIfTI files in the staging dir.  "
                        "'hardlink' (default) is near-instant if the source and "
                        "staging dirs are on the same filesystem (falls back to "
                        "copy automatically if not).  'copy' duplicates bytes "
                        "(slow but portable).  'symlink' makes pointers.")
    p.add_argument("--symlink",       action="store_true",
                   help="Deprecated; equivalent to --stage_mode symlink.")
    p.add_argument("--manifest_csv",  type=Path, default=None,
                   help="Path to data/manifest/manifest.csv (from stage 10).  "
                        "When provided, manifest.csv + manifest.json are copied "
                        "into the staging dir (filtered to the sample subset for "
                        "Phase 2).  Without this, the dataset will not include "
                        "the LSTV / cv_fold columns.")
    p.add_argument("--gate", action="store_true",
                   help="Re-run the orientation gate during staging.  Default: "
                        "skip — reuse stage 9's audit from --orient_audit_path.  "
                        "Pass --gate only if you suspect canonical/ was modified "
                        "after stage 9 ran.")
    p.add_argument("--orient_audit_path", type=Path, default=None,
                   help="Path to stage 9's orientation audit manifest.  "
                        "Default: <canonical_dir>/../orientation/orient_audit_manifest.json")
    p.add_argument("--lstv_audit_path", type=Path, default=None,
                   help="Path to stage 8's LSTV audit manifest.  Joined into "
                        "radiologist_review.csv for the per-scan evidence "
                        "strings.  Default: <canonical_dir>/../lstv/lstv_audit_manifest.json")
    p.add_argument("--no_upload",     action="store_true",
                   help="Stage only; don't push to HF.  Useful for inspection.")

    p.add_argument("--sample_n",       type=int, default=0,
                   help="If >0, also stage and upload a sample of N top-scoring scans.")
    p.add_argument("--sample_repo_id", type=str, default=None,
                   help="HF repo for the sample.  Default: <repo_id>-Sample")
    p.add_argument("--sample_staging_dir", type=Path, default=None,
                   help="Sample staging dir.  Default: <staging_dir>_sample")

    # ─── LSTV/anomaly-only export (Phase 3) ──────────────────────────────
    p.add_argument("--lstv_repo_id", type=str, default=None,
                   help="If set, also stage and upload an LSTV/anomaly-only "
                        "subset (every scan with lstv_class in "
                        "{t13_supernumerary, lumbarization, truncated}) to "
                        "this HF repo.  Skipped when unset.")
    p.add_argument("--lstv_pretty_name", type=str, default=None,
                   help="Pretty name for LSTV repo.  Default: <pretty_name>-LSTV")
    p.add_argument("--lstv_staging_dir", type=Path, default=None,
                   help="LSTV staging dir.  Default: <staging_dir>_lstv")

    p.add_argument("--workers",       type=int, default=8)
    p.add_argument("--log_level",     default="INFO",
                   choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    if not args.canonical_dir.is_dir():
        log.error("canonical_dir not found: %s", args.canonical_dir)
        return 1
    if args.corrected_dir and not args.corrected_dir.is_dir():
        log.error("corrected_dir not found: %s", args.corrected_dir)
        return 1

    args.staging_dir.mkdir(parents=True, exist_ok=True)

    # Resolve stage_mode (--symlink is a deprecated alias).
    effective_stage_mode = "symlink" if args.symlink else args.stage_mode

    # Default orientation audit path: data/orientation/orient_audit_manifest.json
    # (sibling of canonical/)
    if args.orient_audit_path is None:
        args.orient_audit_path = (
            args.canonical_dir.parent / "orientation" / "orient_audit_manifest.json"
        )
    # Default LSTV audit path: data/lstv/lstv_audit_manifest.json
    if args.lstv_audit_path is None:
        args.lstv_audit_path = (
            args.canonical_dir.parent / "lstv" / "lstv_audit_manifest.json"
        )

    # =========================================================================
    # PHASE 1 — full dataset
    # =========================================================================
    log.info("=" * 72)
    log.info("PHASE 1 — full dataset export to %s (stage_mode=%s)",
             args.repo_id, effective_stage_mode)
    log.info("=" * 72)

    try:
        stats = stage_dataset(
            canonical_dir=args.canonical_dir,
            corrected_dir=args.corrected_dir,
            staging_dir=args.staging_dir,
            unify_manifest_path=args.unify_manifest,
            preview_dir=args.preview_dir,
            workers=args.workers,
            stage_mode=effective_stage_mode,
            run_gate=args.gate,
            orient_audit_path=args.orient_audit_path,
            lstv_audit_path=args.lstv_audit_path,
            manifest_csv=args.manifest_csv,
        )
    except RuntimeError as e:
        log.error("Staging aborted: %s", e)
        return 2

    write_readme(args.staging_dir, args.dataset_pretty_name, args.repo_id, stats)

    log.info("-" * 72)
    log.info("Phase 1 staging summary")
    log.info("-" * 72)
    log.info("  n_scans:      %d", stats["n_scans"])
    log.info("  n_patients:   %d", stats["n_patients"])
    log.info("  splits:       train=%d  val=%d  test=%d",
             stats["n_train"], stats["n_val"], stats["n_test"])
    log.info("  n_corrected:  %d", stats["n_corrected"])
    log.info("  staging_dir:  %s", args.staging_dir.resolve())

    if not args.no_upload:
        try:
            url = upload_to_hf(args.staging_dir, args.repo_id,
                               private=not args.public)
            log.info("Phase 1 done. URL: %s", url)
        except Exception as e:
            log.error("Phase 1 upload failed: %s: %s", type(e).__name__, e)
            log.error("Staging dir preserved at %s — fix and rerun.",
                      args.staging_dir)
            return 3
    else:
        log.info("Phase 1: --no_upload set; staged at %s but NOT pushed.",
                 args.staging_dir.resolve())

    # =========================================================================
    # PHASE 2 — sample (if requested)
    # =========================================================================
    if args.sample_n <= 0:
        log.info("Sample export skipped (--sample_n=%d).", args.sample_n)
        return 0

    sample_repo_id = args.sample_repo_id or f"{args.repo_id}-Sample"
    sample_staging = (args.sample_staging_dir or
                      args.staging_dir.parent / f"{args.staging_dir.name}_sample")
    sample_pretty_name = f"{args.dataset_pretty_name}-Sample"

    log.info("=" * 72)
    log.info("PHASE 2 — sample export (top %d) to %s",
             args.sample_n, sample_repo_id)
    log.info("=" * 72)

    # Pick top-N sample scans from the manifest (instant — no mask I/O).
    # Falls back to mask scanning only if manifest is unavailable.
    if args.manifest_csv and args.manifest_csv.exists():
        sample_ids, scoring_results = select_sample_ids_from_manifest(
            args.manifest_csv, n=args.sample_n,
        )
    else:
        log.warning("No --manifest_csv provided; falling back to slow mask scan.")
        sample_ids, scoring_results = select_sample_ids(
            args.canonical_dir, args.corrected_dir,
            n=args.sample_n, workers=args.workers,
        )

    sample_staging.mkdir(parents=True, exist_ok=True)
    # Persist the scoring results so reviewers can see why these N were picked.
    (sample_staging / "sample_selection.json").write_text(json.dumps({
        "criteria": "top-N by unique vertebra-label count, "
                    "veridah-corrected tiebreak",
        "source":       "manifest.csv" if args.manifest_csv else "mask-scan",
        "n_requested":  args.sample_n,
        "n_selected":   len(sample_ids),
        "selected_ids": sample_ids,
        "all_scores":   scoring_results,
    }, indent=2))

    try:
        sample_stats = stage_dataset(
            canonical_dir=args.canonical_dir,
            corrected_dir=args.corrected_dir,
            staging_dir=sample_staging,
            unify_manifest_path=args.unify_manifest,
            preview_dir=args.preview_dir,
            workers=args.workers,
            stage_mode=effective_stage_mode,
            subset_series_ids=sample_ids,
            run_gate=False,                          # never re-gate for the sample
            orient_audit_path=args.orient_audit_path,
            lstv_audit_path=args.lstv_audit_path,
            manifest_csv=args.manifest_csv,
        )
    except RuntimeError as e:
        log.error("Sample staging aborted: %s", e)
        return 4

    write_readme(sample_staging, sample_pretty_name, sample_repo_id, sample_stats,
                 is_sample=True, parent_repo=args.repo_id)

    log.info("-" * 72)
    log.info("Phase 2 staging summary")
    log.info("-" * 72)
    log.info("  n_scans:      %d", sample_stats["n_scans"])
    log.info("  n_patients:   %d", sample_stats["n_patients"])
    log.info("  splits:       train=%d  val=%d  test=%d",
             sample_stats["n_train"], sample_stats["n_val"], sample_stats["n_test"])
    log.info("  n_corrected:  %d", sample_stats["n_corrected"])
    log.info("  staging_dir:  %s", sample_staging.resolve())

    if not args.no_upload:
        try:
            url = upload_to_hf(sample_staging, sample_repo_id,
                               private=not args.public)
            log.info("Phase 2 done. URL: %s", url)
        except Exception as e:
            log.error("Phase 2 upload failed: %s: %s", type(e).__name__, e)
            log.error("Staging dir preserved at %s — fix and rerun.",
                      sample_staging)
            return 5
    else:
        log.info("Phase 2: --no_upload set; staged at %s but NOT pushed.",
                 sample_staging.resolve())

    # ────────────────────────── PHASE 3 ──────────────────────────────────
    # LSTV/anomaly-only subset.  Skipped unless --lstv_repo_id is set.
    if not args.lstv_repo_id:
        return 0

    lstv_repo_id = args.lstv_repo_id
    lstv_pretty_name = (args.lstv_pretty_name
                         or f"{args.dataset_pretty_name}-LSTV")
    lstv_staging = (args.lstv_staging_dir
                     or args.staging_dir.parent / f"{args.staging_dir.name}_lstv")
    lstv_staging.mkdir(parents=True, exist_ok=True)

    log.info("=" * 72)
    log.info("PHASE 3 — LSTV/anomaly-only export to %s (stage_mode=%s)",
             lstv_repo_id, effective_stage_mode)
    log.info("=" * 72)

    if not args.manifest_csv or not args.manifest_csv.exists():
        log.error("Phase 3 requires --manifest_csv pointing at "
                  "data/manifest/manifest.csv.  Skipping LSTV export.")
        return 6

    lstv_ids, lstv_records = select_anomaly_ids_from_manifest(args.manifest_csv)
    if not lstv_ids:
        log.warning("No anomaly cases found in manifest; skipping Phase 3.")
        return 0

    (lstv_staging / "lstv_selection.json").write_text(json.dumps({
        "criteria":       "lstv_class in {t13_supernumerary, lumbarization, truncated}",
        "n_selected":     len(lstv_ids),
        "selected_ids":   lstv_ids,
        "all_records":    lstv_records,
    }, indent=2))

    try:
        lstv_stats = stage_dataset(
            canonical_dir=args.canonical_dir,
            corrected_dir=args.corrected_dir,
            staging_dir=lstv_staging,
            unify_manifest_path=args.unify_manifest,
            preview_dir=args.preview_dir,
            workers=args.workers,
            stage_mode=effective_stage_mode,
            subset_series_ids=lstv_ids,
            run_gate=False,
            orient_audit_path=args.orient_audit_path,
            lstv_audit_path=args.lstv_audit_path,
            manifest_csv=args.manifest_csv,
        )
    except RuntimeError as e:
        log.error("LSTV staging aborted: %s", e)
        return 7

    write_readme(lstv_staging, lstv_pretty_name, lstv_repo_id, lstv_stats,
                 is_sample=True, parent_repo=args.repo_id)

    log.info("-" * 72)
    log.info("Phase 3 staging summary")
    log.info("-" * 72)
    log.info("  n_scans:      %d", lstv_stats["n_scans"])
    log.info("  n_patients:   %d", lstv_stats["n_patients"])
    log.info("  splits:       train=%d  val=%d  test=%d",
             lstv_stats["n_train"], lstv_stats["n_val"], lstv_stats["n_test"])
    log.info("  n_corrected:  %d", lstv_stats["n_corrected"])
    log.info("  staging_dir:  %s", lstv_staging.resolve())

    if not args.no_upload:
        try:
            url = upload_to_hf(lstv_staging, lstv_repo_id,
                               private=not args.public)
            log.info("Phase 3 done. URL: %s", url)
        except Exception as e:
            log.error("Phase 3 upload failed: %s: %s", type(e).__name__, e)
            log.error("Staging dir preserved at %s — fix and rerun.",
                      lstv_staging)
            return 8
    else:
        log.info("Phase 3: --no_upload set; staged at %s but NOT pushed.",
                 lstv_staging.resolve())

    return 0


if __name__ == "__main__":
    sys.exit(main())
