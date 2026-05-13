"""
reorient.py — Reorient unified scans to canonical PIR orientation.

Each input scan in data/unified/scan-<id>/ has CT and mask in whatever
orientation TUM published.  This module produces a canonical view at
data/canonical/scan-<id>/ where:

  - The CT and mask are reoriented to PIR (Posterior, Inferior, Right)
  - Axis 0 = P direction (toward back)
  - Axis 1 = I direction (toward feet) — vertebrae stack along this axis
  - Axis 2 = R direction (toward right side)
  - The affine matrix reflects the new orientation
  - The data array is permuted/flipped accordingly

PIR is chosen because:
  - It's the standard radiological convention for spine imaging
  - It matches the orientation TUM uses internally for annotation
    (recovered empirically from a subset of subjects)
  - All downstream stages can then assume a fixed axis interpretation:
    spine_axis = 1, label progression increases caudally along axis 1

Input layout (from unify):
    data/unified/scan-verse014/
        scan-verse014_ct.nii.gz       -> symlink to raw file
        scan-verse014_msk.nii.gz      -> symlink
        scan-verse014_snp.png         -> symlink
        scan-verse014_meta.json

Output layout (this stage):
    data/canonical/scan-verse014/
        scan-verse014_ct.nii.gz       -> reoriented data (real file, not symlink)
        scan-verse014_msk.nii.gz      -> reoriented data (real file)
        scan-verse014_snp.png         -> symlink to original snapshot
        scan-verse014_meta.json       -> updated meta with new shape/affine
                                          plus original_paths for provenance

The reorientation is implemented via nibabel.orientations.apply_orientation,
which permutes and flips axes only — no resampling, no interpolation.  The
data values are preserved exactly; only their array positions change.
"""

from __future__ import annotations

import argparse
import json
import logging
import shutil
import sys
import time
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import nibabel as nib
from nibabel.orientations import (
    axcodes2ornt, io_orientation, ornt_transform, apply_orientation,
    inv_ornt_aff,
)

log = logging.getLogger("verse.reorient")


# =============================================================================
# constants
# =============================================================================

TARGET_ORIENTATION: tuple[str, str, str] = ("P", "I", "R")
TARGET_ORNT = axcodes2ornt(TARGET_ORIENTATION)

PROGRESS_EVERY_N_SCANS = 10
PROGRESS_TIME_INTERVAL_SECONDS = 10.0


# =============================================================================
# core reorientation
# =============================================================================

def reorient_image(img: "nib.Nifti1Image") -> tuple["nib.Nifti1Image", list[list[float]]]:
    """Return (reoriented_image, transform) where transform documents the axis
    permutation/flip applied.

    Uses nibabel.orientations.apply_orientation, which permutes and flips axes
    but does not resample.  Voxel values are preserved bit-for-bit.

    The transform is a 3x2 array where row i is [axis_in_source, flip_sign]:
      - axis_in_source is the source axis that ends up at position i
      - flip_sign is +1 (no flip) or -1 (flipped)
    """
    data = np.asarray(img.dataobj)
    current_ornt = io_orientation(img.affine)
    xform = ornt_transform(current_ornt, TARGET_ORNT)
    new_data = apply_orientation(data, xform)
    new_affine = img.affine @ inv_ornt_aff(xform, data.shape)
    new_img = nib.Nifti1Image(new_data, new_affine, header=img.header)
    return new_img, xform.tolist()


def reorient_scan(scan_dir: Path, out_root: Path) -> dict[str, Any]:
    """Reorient one scan-dir to PIR; return summary dict.

    Top-level worker for the ProcessPoolExecutor.
    """
    series_id = scan_dir.name.replace("scan-", "")
    out_dir = out_root / f"scan-{series_id}"
    out_dir.mkdir(parents=True, exist_ok=True)

    meta_path = scan_dir / f"scan-{series_id}_meta.json"
    if not meta_path.exists():
        return {"series_id": series_id, "error": f"no meta.json at {meta_path}"}
    meta = json.loads(meta_path.read_text())

    paths = meta.get("source_paths", {})
    ct_path  = paths.get("ct")
    msk_path = paths.get("msk")
    snp_path = paths.get("snp")
    if not ct_path or not msk_path:
        return {"series_id": series_id,
                "error": f"missing required ct or msk path in {meta_path}"}

    # Load and reorient CT
    ct_img = nib.load(ct_path)
    ct_pir, ct_xform = reorient_image(ct_img)
    new_ct_path = out_dir / f"scan-{series_id}_ct.nii.gz"
    nib.save(ct_pir, str(new_ct_path))

    # Load and reorient mask
    msk_img = nib.load(msk_path)
    msk_pir, msk_xform = reorient_image(msk_img)
    new_msk_path = out_dir / f"scan-{series_id}_msk.nii.gz"
    nib.save(msk_pir, str(new_msk_path))

    # Snapshot doesn't need reorientation — just symlink across for completeness
    new_snp_path: str | None = None
    if snp_path and Path(snp_path).exists():
        new_snp_path = str(out_dir / f"scan-{series_id}_snp.png")
        if Path(new_snp_path).exists() or Path(new_snp_path).is_symlink():
            Path(new_snp_path).unlink()
        Path(new_snp_path).symlink_to(Path(snp_path).absolute())

    # New shape / spacing
    new_shape   = tuple(int(s) for s in ct_pir.shape)
    new_spacing = tuple(float(np.linalg.norm(ct_pir.affine[:3, i])) for i in range(3))

    # Build updated meta — keep most fields, update paths and add provenance
    new_meta = dict(meta)   # shallow copy
    new_meta["source_paths"] = {
        "ct":  str(new_ct_path.absolute()),
        "msk": str(new_msk_path.absolute()),
    }
    if new_snp_path:
        new_meta["source_paths"]["snp"] = new_snp_path
    new_meta["original_paths"] = {
        "ct":  ct_path,
        "msk": msk_path,
    }
    if snp_path:
        new_meta["original_paths"]["snp"] = snp_path
    new_meta["orientation"] = "".join(TARGET_ORIENTATION)
    new_meta["shape"]   = list(new_shape)
    new_meta["spacing"] = list(new_spacing)
    new_meta["reorient_xform_ct"]  = ct_xform
    new_meta["reorient_xform_msk"] = msk_xform
    new_meta["version"] = "0.5.0"

    new_meta_path = out_dir / f"scan-{series_id}_meta.json"
    new_meta_path.write_text(json.dumps(new_meta, indent=2))

    return {
        "series_id":   series_id,
        "patient_id":  meta.get("patient_id"),
        "shape":       list(new_shape),
        "spacing":     list(new_spacing),
        "xform_ct":    ct_xform,
        "xform_msk":   msk_xform,
        "out_dir":     str(out_dir),
        "snp":         new_snp_path is not None,
    }


# =============================================================================
# parallel orchestration
# =============================================================================

def _reorient_one(args: tuple[str, str]) -> dict[str, Any]:
    """Worker entry — string paths for pickle-friendliness."""
    scan_dir_str, out_root_str = args
    try:
        return reorient_scan(Path(scan_dir_str), Path(out_root_str))
    except Exception as e:
        return {"series_id": Path(scan_dir_str).name.replace("scan-", ""),
                "error": f"{type(e).__name__}: {e}"}


def _flush_logs() -> None:
    for h in log.handlers or logging.getLogger().handlers:
        try:
            h.flush()
        except Exception:
            pass


def reorient_all(unified_dir: Path, out_root: Path,
                 workers: int = 8) -> list[dict[str, Any]]:
    """Reorient every scan-* in unified_dir to PIR, output at out_root."""
    out_root.mkdir(parents=True, exist_ok=True)
    scan_dirs = sorted(d for d in unified_dir.iterdir()
                       if d.is_dir() and d.name.startswith("scan-"))
    log.info("Reorienting %d scans -> %s (workers=%d, target=%s)",
             len(scan_dirs), out_root, workers, "".join(TARGET_ORIENTATION))
    _flush_logs()

    results: list[dict[str, Any]] = []
    n_done = 0
    last_log_count = 0
    last_log_time = time.monotonic()
    start = last_log_time

    work_items = [(str(d), str(out_root)) for d in scan_dirs]

    with ProcessPoolExecutor(max_workers=workers) as ex:
        futures = {ex.submit(_reorient_one, item): item for item in work_items}
        for fut in as_completed(futures):
            r = fut.result()
            results.append(r)
            n_done += 1
            if "error" in r:
                log.warning("  %s: %s", r["series_id"], r["error"])

            since_count = n_done - last_log_count
            since_time = time.monotonic() - last_log_time
            if since_count >= PROGRESS_EVERY_N_SCANS or since_time >= PROGRESS_TIME_INTERVAL_SECONDS:
                elapsed = time.monotonic() - start
                rate = n_done / elapsed if elapsed > 0 else 0.0
                remaining = len(scan_dirs) - n_done
                eta = remaining / rate if rate > 0 else 0.0
                log.info("  progress: %d/%d (%.1f%%)  %.1f scans/s  ETA %ds",
                         n_done, len(scan_dirs),
                         100 * n_done / len(scan_dirs),
                         rate, int(eta))
                _flush_logs()
                last_log_count = n_done
                last_log_time = time.monotonic()

    results.sort(key=lambda r: r["series_id"])

    n_ok  = sum(1 for r in results if "error" not in r)
    n_err = len(results) - n_ok
    log.info("Reorient done: %d ok, %d failed", n_ok, n_err)
    return results


def write_reorient_manifest(results: list[dict[str, Any]], out_root: Path) -> Path:
    """Aggregate per-scan results into canonical_manifest.json."""
    n_ok  = sum(1 for r in results if "error" not in r)
    n_err = sum(1 for r in results if "error" in r)

    # Summarize by shape characteristic (interesting because reorient
    # often permutes shapes from e.g. (512, 512, 40) sagittal to (?, ?, ?))
    by_shape_signature: dict[str, int] = defaultdict(int)
    for r in results:
        if "shape" in r:
            sig = "_".join(str(s) for s in r["shape"])
            by_shape_signature[sig] += 1

    manifest = {
        "version":            "0.5.0",
        "target_orientation": "".join(TARGET_ORIENTATION),
        "n_scans":            len(results),
        "n_ok":               n_ok,
        "n_failed":           n_err,
        "by_pir_shape": {
            k: v for k, v in sorted(by_shape_signature.items(),
                                    key=lambda kv: -kv[1])[:20]
        },
        "errors": [
            {"series_id": r["series_id"], "error": r["error"]}
            for r in results if "error" in r
        ],
        "scans": [r for r in results if "error" not in r],
    }
    out_path = out_root / "canonical_manifest.json"
    out_path.write_text(json.dumps(manifest, indent=2))
    return out_path


# =============================================================================
# CLI
# =============================================================================

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="verse-reorient",
        description=(
            "Reorient unified scans to canonical PIR orientation, producing "
            "data/canonical/scan-*/ with reoriented CT and mask NIfTIs."
        ),
    )
    p.add_argument("--unified_dir", type=Path, required=True,
                   help="Path to data/unified/ (input).")
    p.add_argument("--out_dir", type=Path, required=True,
                   help="Path to data/canonical/ (output).")
    p.add_argument("--workers", type=int, default=8,
                   help="Parallel ProcessPoolExecutor workers (default 8).")
    p.add_argument("--log_level", default="INFO",
                   choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    if not args.unified_dir.is_dir():
        log.error("unified_dir not found: %s", args.unified_dir)
        return 1

    results = reorient_all(args.unified_dir, args.out_dir, workers=args.workers)
    manifest_path = write_reorient_manifest(results, args.out_dir)
    log.info("Wrote %s", manifest_path)

    n_err = sum(1 for r in results if "error" in r)
    return 0 if n_err == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
