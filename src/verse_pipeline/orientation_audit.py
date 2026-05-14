"""
orientation_audit.py — verify every canonical (or corrected) scan is PIR.

The reorient stage of the pipeline transforms every NIfTI to PIR canonical:
  axis 0 = P (anterior → posterior)
  axis 1 = I (superior → inferior — the spine axis)
  axis 2 = R (left → right)

This audit reads each scan's affine matrix and uses nibabel.aff2axcodes()
to confirm the orientation tuple is exactly ('P', 'I', 'R').  Any scan whose
CT or mask returns a different tuple is a bug and would render inconsistently.

If the audit reports 100% PIR-correct, then all renders WILL look visually
consistent (head at top, anterior on left in sagittal, anterior on top in
axial), provided visualize.py / visualize_corrections.py use the same display
transformations.

Output: data/orientation/orientation_audit.json
        data/orientation/orientation_audit.csv
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import numpy as np
import nibabel as nib

log = logging.getLogger("verse.orientation_audit")

EXPECTED_AXCODES: tuple[str, str, str] = ("P", "I", "R")


# =============================================================================
# per-scan audit
# =============================================================================

def audit_scan(scan_dir_str: str) -> dict[str, Any]:
    scan_dir = Path(scan_dir_str)
    series_id = scan_dir.name.replace("scan-", "")
    meta_path = scan_dir / f"scan-{series_id}_meta.json"

    if not meta_path.exists():
        return {"series_id": series_id, "error": f"no meta.json at {meta_path}"}

    meta = json.loads(meta_path.read_text())

    ct_path  = Path(meta["source_paths"]["ct"])
    msk_path = Path(meta["source_paths"]["msk"])
    if not ct_path.exists():
        return {"series_id": series_id, "error": f"CT not on disk: {ct_path}"}
    if not msk_path.exists():
        return {"series_id": series_id, "error": f"MSK not on disk: {msk_path}"}

    try:
        ct_img  = nib.load(str(ct_path))
        msk_img = nib.load(str(msk_path))
        ct_axcodes  = nib.aff2axcodes(ct_img.affine)
        msk_axcodes = nib.aff2axcodes(msk_img.affine)
    except Exception as e:
        return {"series_id": series_id, "error": f"{type(e).__name__}: {e}"}

    # Spacing for downstream sanity (signed values from affine norms)
    ct_spacing  = [float(np.linalg.norm(ct_img.affine[:3, k]))  for k in range(3)]
    msk_spacing = [float(np.linalg.norm(msk_img.affine[:3, k])) for k in range(3)]

    ct_pir  = (ct_axcodes  == EXPECTED_AXCODES)
    msk_pir = (msk_axcodes == EXPECTED_AXCODES)
    shape_match = (tuple(int(s) for s in ct_img.shape[:3]) ==
                   tuple(int(s) for s in msk_img.shape[:3]))

    return {
        "series_id":   series_id,
        "ct_path":     str(ct_path),
        "msk_path":    str(msk_path),
        "ct_axcodes":  list(ct_axcodes),
        "msk_axcodes": list(msk_axcodes),
        "ct_shape":    list(ct_img.shape[:3]),
        "msk_shape":   list(msk_img.shape[:3]),
        "ct_spacing_mm":  ct_spacing,
        "msk_spacing_mm": msk_spacing,
        "ct_is_pir":   ct_pir,
        "msk_is_pir":  msk_pir,
        "shape_match": shape_match,
        "passes":      ct_pir and msk_pir and shape_match,
    }


# =============================================================================
# orchestration
# =============================================================================

def _flush_logs() -> None:
    for h in log.handlers or logging.getLogger().handlers:
        try:    h.flush()
        except Exception: pass


def audit_all(input_dir: Path, workers: int = 8) -> list[dict[str, Any]]:
    scan_dirs = sorted(d for d in input_dir.iterdir()
                       if d.is_dir() and d.name.startswith("scan-"))
    log.info("Auditing orientation of %d scans in %s", len(scan_dirs), input_dir)

    results: list[dict[str, Any]] = []
    n_done = 0
    last_log_time = time.monotonic()
    start = last_log_time

    with ProcessPoolExecutor(max_workers=workers) as ex:
        futures = {ex.submit(audit_scan, str(d)): d for d in scan_dirs}
        for fut in as_completed(futures):
            r = fut.result()
            results.append(r)
            n_done += 1
            if "error" in r:
                log.warning("  %s: %s", r["series_id"], r["error"])
            if time.monotonic() - last_log_time >= 5.0:
                elapsed = time.monotonic() - start
                rate = n_done / elapsed if elapsed > 0 else 0.0
                log.info("  progress: %d/%d  %.1f scans/s",
                         n_done, len(scan_dirs), rate)
                _flush_logs()
                last_log_time = time.monotonic()

    results.sort(key=lambda r: r.get("series_id", ""))
    log.info("Orientation audit done: %d scans", len(results))
    return results


def summarize(results: list[dict[str, Any]]) -> dict[str, Any]:
    n_errors        = sum(1 for r in results if "error" in r)
    n_passes        = sum(1 for r in results if r.get("passes"))
    n_ct_pir_fail   = sum(1 for r in results if "error" not in r and not r.get("ct_is_pir"))
    n_msk_pir_fail  = sum(1 for r in results if "error" not in r and not r.get("msk_is_pir"))
    n_shape_fail    = sum(1 for r in results if "error" not in r and not r.get("shape_match"))

    mismatched = [r for r in results if "error" not in r and not r.get("passes")]

    return {
        "n_scans":         len(results),
        "n_errors":        n_errors,
        "n_passes_PIR":    n_passes,
        "n_ct_not_PIR":    n_ct_pir_fail,
        "n_msk_not_PIR":   n_msk_pir_fail,
        "n_shape_mismatch": n_shape_fail,
        "mismatched_subjects": [
            {"series_id":   r["series_id"],
             "ct_axcodes":  r.get("ct_axcodes"),
             "msk_axcodes": r.get("msk_axcodes"),
             "ct_shape":    r.get("ct_shape"),
             "msk_shape":   r.get("msk_shape")}
            for r in mismatched
        ],
    }


def write_outputs(results: list[dict[str, Any]],
                  summary: dict[str, Any],
                  input_dir: Path,
                  out_dir: Path) -> tuple[Path, Path]:
    out_dir.mkdir(parents=True, exist_ok=True)

    manifest_path = out_dir / "orientation_audit.json"
    manifest_path.write_text(json.dumps({
        "version":           "0.1.0",
        "input_dir":         str(input_dir.resolve()),
        "expected_axcodes":  list(EXPECTED_AXCODES),
        "summary":           summary,
        "subjects":          results,
    }, indent=2))

    csv_path = out_dir / "orientation_audit.csv"
    fieldnames = [
        "series_id", "passes",
        "ct_is_pir", "msk_is_pir", "shape_match",
        "ct_axcodes", "msk_axcodes",
        "ct_shape", "msk_shape",
        "error",
    ]
    with csv_path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in results:
            w.writerow({
                "series_id":   r.get("series_id", ""),
                "passes":      r.get("passes", False),
                "ct_is_pir":   r.get("ct_is_pir", ""),
                "msk_is_pir":  r.get("msk_is_pir", ""),
                "shape_match": r.get("shape_match", ""),
                "ct_axcodes":  ",".join(r.get("ct_axcodes", [])) if r.get("ct_axcodes") else "",
                "msk_axcodes": ",".join(r.get("msk_axcodes", [])) if r.get("msk_axcodes") else "",
                "ct_shape":    ",".join(str(x) for x in r.get("ct_shape", [])),
                "msk_shape":   ",".join(str(x) for x in r.get("msk_shape", [])),
                "error":       r.get("error", ""),
            })

    return manifest_path, csv_path


def print_summary(summary: dict[str, Any]) -> None:
    log.info("=" * 72)
    log.info("ORIENTATION AUDIT SUMMARY")
    log.info("=" * 72)
    log.info("Expected axcodes: %s", EXPECTED_AXCODES)
    log.info("")
    log.info("Total scans:              %d", summary["n_scans"])
    log.info("PASS (CT+MSK both PIR):   %d", summary["n_passes_PIR"])
    log.info("")
    log.info("Failures:")
    log.info("  CT not PIR:             %d", summary["n_ct_not_PIR"])
    log.info("  Mask not PIR:           %d", summary["n_msk_not_PIR"])
    log.info("  Shape mismatch:         %d", summary["n_shape_mismatch"])
    log.info("  Read errors:            %d", summary["n_errors"])
    log.info("")
    if summary["mismatched_subjects"]:
        log.info("First %d mismatched subjects:",
                 min(10, len(summary["mismatched_subjects"])))
        for r in summary["mismatched_subjects"][:10]:
            log.info("  %s  ct=%s  msk=%s",
                     r["series_id"], r["ct_axcodes"], r["msk_axcodes"])
    else:
        log.info("All scans are PIR-oriented — renders WILL be visually consistent.")
    log.info("=" * 72)


# =============================================================================
# CLI
# =============================================================================

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="verse-orientation-audit",
        description="Verify every scan in input_dir is PIR-oriented.",
    )
    p.add_argument("--input_dir", type=Path, required=True,
                   help="Path to data/canonical/ or data/corrected/.")
    p.add_argument("--out_dir",   type=Path, required=True,
                   help="Where to write orientation_audit.{json,csv}.")
    p.add_argument("--workers",   type=int, default=8)
    p.add_argument("--strict",    action="store_true",
                   help="Exit non-zero if ANY scan fails PIR check.")
    p.add_argument("--log_level", default="INFO",
                   choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    if not args.input_dir.is_dir():
        log.error("input_dir not found: %s", args.input_dir)
        return 1

    results = audit_all(args.input_dir, workers=args.workers)
    summary = _ = summarize(results)
    manifest_path, csv_path = write_outputs(results, summary,
                                             args.input_dir, args.out_dir)
    log.info("Wrote %s", manifest_path)
    log.info("Wrote %s", csv_path)

    print_summary(summary)

    if args.strict and summary["n_passes_PIR"] < summary["n_scans"]:
        log.error("STRICT mode: %d / %d scans failed PIR check",
                  summary["n_scans"] - summary["n_passes_PIR"], summary["n_scans"])
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
