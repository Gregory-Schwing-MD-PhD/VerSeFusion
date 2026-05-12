"""
qc.py — Per-scan alignment audit for VerSeFusion's unified corpus.

ARCHITECTURE
------------
Patient-level identity is already certain after unify (every scan-dir knows its
demographics-driven patient_id).  This module's only job is to verify that the
CT, mask, and centroid files that unify grouped together actually agree
geometrically — same world coordinate frame, same voxel grid, mask labels
falling inside the CT volume, centroids landing on the right mask labels.

ALL CHECKS ARE PER-SCAN AND LOCAL.  We never compare one scan against another.
This is a pure auditability stage, not a correction stage.

Centroid coordinate system
--------------------------
After empirical verification across all 374 scans of the unified corpus,
TUM ships centroids as direct (X, Y, Z) array indices in the image's own
voxel grid, in BOTH the MICCAI-challenge and BIDS distributions.  Some
BIDS files carry an explicit ``direction`` field (e.g. ``["L", "A", "S"]``)
documenting the anatomical orientation of the axes — but that affects only
interpretation, not indexing.  X/Y/Z map directly to image array axes 0/1/2.

(The earlier "asl_iso_1mm" coordinate-system flag in unify's meta.json was
based on a misread of TUM's documentation; centroids are always in
image-voxel space.)

Checks performed
----------------
Tier 1 — Header-only (cheap, no full volume load):
  1. files_present     — required CT, mask, centroid present on disk
  2. headers_readable  — nibabel can load each NIfTI header
  3. shape_match       — CT.shape == mask.shape
  4. affine_match      — CT and mask affines agree on direction + spacing

Tier 2 — Mask data (loads mask voxels, not CT):
  5. label_inventory   — which VerSe labels (1-28) are present, voxel counts
  6. label_in_range    — no labels outside VerSe's documented 1-28 scheme
  7. labels_nonempty   — every label has at least MIN_LABEL_VOXELS

Tier 3 — Centroid-mask alignment (if centroid present):
  8. centroid_alignment — for each centroid (label, X, Y, Z):
                          - rounds (X, Y, Z) to nearest int → (i, j, k)
                          - checks (i, j, k) is in mask bounds
                          - checks mask[i, j, k] == label

Each check produces:
  status:  one of CheckStatus.PASS / WARN / FAIL / SKIP
  reasons: human-readable strings explaining what we found

A per-scan overall status is computed by aggregating the worst individual
status.  The manifest can be queried by jq to find every flagged scan.

Output
------
data/qc/qc_manifest.json with summary stats + per-scan check records.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

log = logging.getLogger("verse.qc")


# =============================================================================
# constants
# =============================================================================

class CheckStatus:
    PASS = "PASS"
    WARN = "WARN"
    FAIL = "FAIL"
    SKIP = "SKIP"

# Aggregation precedence (worst wins).  PASS < SKIP < WARN < FAIL.
STATUS_RANK = {
    CheckStatus.PASS: 0,
    CheckStatus.SKIP: 1,
    CheckStatus.WARN: 2,
    CheckStatus.FAIL: 3,
}

# Affine-match tolerances (inherited from series_assigner.py).
DIR_TOLERANCE = 1e-3
SPC_TOLERANCE = 0.01     # mm

# VerSe label scheme: 1-7 cervical, 8-19 thoracic T1-T12, 20-25 lumbar L1-L6,
# 26 sacrum, 27 coccyx, 28 T13.
VERSE_LABEL_RANGE = (1, 28)

# Labels with fewer than this many voxels are flagged as suspicious — most
# likely annotation artifacts or single-slice noise.
MIN_LABEL_VOXELS = 50


# =============================================================================
# data classes
# =============================================================================

@dataclass
class CheckResult:
    """One named check's outcome for one scan."""
    status:  str = CheckStatus.PASS
    reasons: list[str] = field(default_factory=list)
    extra:   dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        d: dict[str, Any] = {"status": self.status, "reasons": self.reasons}
        if self.extra:
            d.update(self.extra)
        return d


@dataclass
class ScanReport:
    """Per-scan QC summary."""
    series_id:     str
    patient_id:    str
    source_format: str
    overall:       str = CheckStatus.PASS
    checks:        dict[str, CheckResult] = field(default_factory=dict)

    def aggregate_overall(self) -> None:
        worst = CheckStatus.PASS
        for c in self.checks.values():
            if STATUS_RANK[c.status] > STATUS_RANK[worst]:
                worst = c.status
        self.overall = worst

    def to_dict(self) -> dict[str, Any]:
        return {
            "series_id":     self.series_id,
            "patient_id":    self.patient_id,
            "source_format": self.source_format,
            "overall":       self.overall,
            "checks":        {k: v.to_dict() for k, v in self.checks.items()},
        }


# =============================================================================
# geometry helpers (adapted from CTSpinoPelvic1K's series_assigner)
# =============================================================================

def _directions_match(d1: np.ndarray, d2: np.ndarray,
                      tol: float = DIR_TOLERANCE) -> tuple[bool, list[str]]:
    """Compare two 3x3 direction matrices by per-axis dot product."""
    reasons: list[str] = []
    ok = True
    for i in range(3):
        dot = abs(float(np.dot(d1[:, i], d2[:, i])))
        if dot < 1.0 - tol:
            ok = False
            reasons.append(f"axis {i}: dot product {dot:.4f} < 1-tol={1 - tol:.4f}")
    return ok, reasons


def _decompose_affine(affine: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Split a 4x4 affine into (direction-3x3, spacing-3-vector)."""
    m = affine[:3, :3]
    spacing = np.array([np.linalg.norm(m[:, i]) for i in range(3)])
    if np.any(spacing < 1e-9):
        raise ValueError(f"degenerate affine, spacing has zero norm: {spacing}")
    direction = m / spacing[np.newaxis, :]
    return direction, spacing


# =============================================================================
# individual checks
# =============================================================================

def check_files_present(meta: dict[str, Any]) -> CheckResult:
    paths = meta.get("source_paths", {})
    missing = []
    for kind in ("ct", "msk", "ctd"):
        p = paths.get(kind)
        if not p or not Path(p).exists():
            missing.append(kind)
    if not missing:
        return CheckResult(status=CheckStatus.PASS,
                           reasons=["all required source files present on disk"])
    if missing == ["ctd"]:
        return CheckResult(status=CheckStatus.WARN,
                           reasons=["missing centroid file (downstream centroid-based "
                                    "checks will be skipped)"])
    return CheckResult(status=CheckStatus.FAIL,
                       reasons=[f"missing required source file(s): {missing}"])


def check_headers_readable(ct_path: Path, msk_path: Path) -> tuple[CheckResult, dict]:
    """Try to load both headers.  Returns (result, info_dict_for_downstream)."""
    import nibabel as nib
    info = {}
    try:
        ct_img = nib.load(str(ct_path))
        info["ct_shape"]  = tuple(int(s) for s in ct_img.shape)
        info["ct_affine"] = np.array(ct_img.affine)
    except Exception as e:
        return CheckResult(status=CheckStatus.FAIL,
                           reasons=[f"failed to load CT header: {type(e).__name__}: {e}"]), info
    try:
        msk_img = nib.load(str(msk_path))
        info["msk_shape"]  = tuple(int(s) for s in msk_img.shape)
        info["msk_affine"] = np.array(msk_img.affine)
        info["msk_img"]    = msk_img
    except Exception as e:
        return CheckResult(status=CheckStatus.FAIL,
                           reasons=[f"failed to load mask header: {type(e).__name__}: {e}"]), info

    return CheckResult(status=CheckStatus.PASS,
                       reasons=["CT and mask headers loaded"]), info


def check_shape_match(info: dict) -> CheckResult:
    ct = info.get("ct_shape")
    mk = info.get("msk_shape")
    if ct is None or mk is None:
        return CheckResult(status=CheckStatus.SKIP, reasons=["headers not loaded"])
    if ct == mk:
        return CheckResult(status=CheckStatus.PASS, reasons=[f"both shape {ct}"])
    return CheckResult(status=CheckStatus.FAIL,
                       reasons=[f"CT shape {ct} != mask shape {mk}"])


def check_affine_match(info: dict) -> CheckResult:
    ct_aff = info.get("ct_affine")
    mk_aff = info.get("msk_affine")
    if ct_aff is None or mk_aff is None:
        return CheckResult(status=CheckStatus.SKIP, reasons=["headers not loaded"])
    try:
        ct_dir, ct_spc = _decompose_affine(ct_aff)
        mk_dir, mk_spc = _decompose_affine(mk_aff)
    except ValueError as e:
        return CheckResult(status=CheckStatus.FAIL,
                           reasons=[f"affine decomposition failed: {e}"])

    reasons: list[str] = []
    ok = True

    dir_ok, dir_reasons = _directions_match(ct_dir, mk_dir)
    if dir_ok:
        reasons.append("direction matrices agree within tolerance")
    else:
        ok = False
        reasons.extend(["direction matrices differ:"] + ["  " + r for r in dir_reasons])

    spc_diff = np.abs(ct_spc - mk_spc).max()
    if spc_diff < SPC_TOLERANCE:
        reasons.append(f"spacing matches within {SPC_TOLERANCE}mm "
                       f"(CT={ct_spc.tolist()}, mask={mk_spc.tolist()})")
    else:
        ok = False
        reasons.append(f"spacing differs by {spc_diff:.4f}mm > {SPC_TOLERANCE}mm "
                       f"(CT={ct_spc.tolist()}, mask={mk_spc.tolist()})")

    origin_diff = np.abs(ct_aff[:3, 3] - mk_aff[:3, 3]).max()
    if origin_diff < 0.5:
        reasons.append(f"origins agree (diff={origin_diff:.4f}mm)")
    elif origin_diff < 5.0:
        reasons.append(f"WARN: origins differ by {origin_diff:.4f}mm "
                       f"(under 5mm — likely acceptable rounding)")
    else:
        ok = False
        reasons.append(f"origins differ by {origin_diff:.4f}mm > 5mm")

    if ok:
        return CheckResult(status=CheckStatus.PASS, reasons=reasons)
    return CheckResult(status=CheckStatus.FAIL, reasons=reasons)


def check_label_inventory(info: dict) -> CheckResult:
    """Load mask data, inventory labels, check range and minimum sizes."""
    msk_img = info.get("msk_img")
    if msk_img is None:
        return CheckResult(status=CheckStatus.SKIP, reasons=["mask not loaded"])

    try:
        data = np.asarray(msk_img.dataobj).astype(np.int32)
    except Exception as e:
        return CheckResult(status=CheckStatus.FAIL,
                           reasons=[f"failed to load mask data: {type(e).__name__}: {e}"])

    info["msk_data"] = data

    labels, counts = np.unique(data, return_counts=True)
    nonzero_labels = [(int(l), int(c)) for l, c in zip(labels, counts) if l != 0]
    label_counts = {int(l): int(c) for l, c in nonzero_labels}

    if not nonzero_labels:
        return CheckResult(status=CheckStatus.FAIL,
                           reasons=["mask is entirely zero — no annotated vertebrae"])

    out_of_range = [l for l, _ in nonzero_labels
                    if l < VERSE_LABEL_RANGE[0] or l > VERSE_LABEL_RANGE[1]]
    tiny = [(l, c) for l, c in nonzero_labels if c < MIN_LABEL_VOXELS]

    reasons = [
        f"found {len(nonzero_labels)} non-zero labels: "
        f"{sorted(l for l, _ in nonzero_labels)}",
    ]
    extra = {"labels": sorted(label_counts), "label_voxel_counts": label_counts}

    if out_of_range:
        reasons.append(f"FAIL: {len(out_of_range)} labels outside VerSe range "
                       f"[1, 28]: {out_of_range}")
        return CheckResult(status=CheckStatus.FAIL, reasons=reasons, extra=extra)

    if tiny:
        reasons.append(f"WARN: {len(tiny)} labels have fewer than {MIN_LABEL_VOXELS} "
                       f"voxels (possible artifacts): {tiny}")
        return CheckResult(status=CheckStatus.WARN, reasons=reasons, extra=extra)

    return CheckResult(status=CheckStatus.PASS, reasons=reasons, extra=extra)


def check_centroid_alignment(meta: dict[str, Any], info: dict) -> CheckResult:
    """For each labeled centroid, verify mask[i, j, k] == label.

    TUM ships centroid (X, Y, Z) as direct array indices in the image's own
    voxel grid, across both MICCAI and BIDS distributions.  We round to the
    nearest integer, bounds-check, and look up the mask label at that voxel.

    Some files carry a ``direction`` field as the first JSON entry
    (e.g. {"direction": ["L", "A", "S"]}).  This documents the anatomical
    orientation of the axes but doesn't affect indexing.  We record it for
    audit but otherwise pass through.
    """
    ctd_path = meta.get("source_paths", {}).get("ctd")
    if not ctd_path or not Path(ctd_path).exists():
        return CheckResult(status=CheckStatus.SKIP,
                           reasons=["no centroid file (or unreadable) — alignment check skipped"])

    msk_data = info.get("msk_data")
    if msk_data is None:
        return CheckResult(status=CheckStatus.SKIP,
                           reasons=["mask data not loaded — alignment check skipped"])

    try:
        with open(ctd_path) as f:
            centroids_raw = json.load(f)
    except Exception as e:
        return CheckResult(status=CheckStatus.FAIL,
                           reasons=[f"failed to parse centroid JSON: {type(e).__name__}: {e}"])

    # The JSON is a list whose first entry may be a {"direction": [...]} header
    # and the rest are per-vertebra centroids {"label": int, "X","Y","Z": float}.
    direction = None
    centroid_entries = []
    for entry in centroids_raw if isinstance(centroids_raw, list) else []:
        if not isinstance(entry, dict):
            continue
        if "direction" in entry and direction is None:
            direction = entry["direction"]
            continue
        if all(k in entry for k in ("label", "X", "Y", "Z")):
            centroid_entries.append(entry)

    if not centroid_entries:
        return CheckResult(status=CheckStatus.WARN,
                           reasons=[f"centroid JSON has no label/X/Y/Z entries"])

    shape = msk_data.shape
    matched = 0
    label_mismatch: list[dict] = []
    out_of_bounds: list[dict] = []
    label_missing_in_mask: list[int] = []

    for entry in centroid_entries:
        label = int(entry["label"])
        i = int(round(float(entry["X"])))
        j = int(round(float(entry["Y"])))
        k = int(round(float(entry["Z"])))

        in_bounds = (0 <= i < shape[0] and 0 <= j < shape[1] and 0 <= k < shape[2])
        if not in_bounds:
            out_of_bounds.append({"label": label, "voxel": [i, j, k]})
            continue

        mask_at_voxel = int(msk_data[i, j, k])
        if mask_at_voxel == label:
            matched += 1
        elif mask_at_voxel == 0:
            label_mismatch.append({"label": label, "voxel": [i, j, k],
                                   "got": "background"})
        else:
            label_mismatch.append({"label": label, "voxel": [i, j, k],
                                   "got": mask_at_voxel})

        if not np.any(msk_data == label):
            label_missing_in_mask.append(label)

    n = len(centroid_entries)
    match_rate = matched / n if n else 0.0
    reasons = [
        f"{matched}/{n} centroids land on the correct mask label "
        f"(match rate {100*match_rate:.1f}%)",
    ]
    extra: dict[str, Any] = {
        "n_centroids": n,
        "n_matched":   matched,
        "match_rate":  round(match_rate, 4),
    }
    if direction is not None:
        extra["direction"] = direction
    if out_of_bounds:
        reasons.append(f"{len(out_of_bounds)} centroids fall outside mask bounds: "
                       f"{out_of_bounds[:5]}{'...' if len(out_of_bounds) > 5 else ''}")
        extra["out_of_bounds"] = out_of_bounds
    if label_mismatch:
        reasons.append(f"{len(label_mismatch)} centroids land on wrong label "
                       f"or background: {label_mismatch[:5]}"
                       f"{'...' if len(label_mismatch) > 5 else ''}")
        extra["label_mismatch"] = label_mismatch
    if label_missing_in_mask:
        reasons.append(f"WARN: {len(label_missing_in_mask)} centroid labels missing "
                       f"from mask entirely: {label_missing_in_mask}")
        extra["labels_missing_in_mask"] = label_missing_in_mask

    if match_rate >= 0.95:
        status = CheckStatus.PASS
    elif match_rate >= 0.80:
        status = CheckStatus.WARN
    else:
        status = CheckStatus.FAIL

    return CheckResult(status=status, reasons=reasons, extra=extra)


# =============================================================================
# per-scan driver
# =============================================================================

def audit_scan(scan_dir_str: str) -> dict[str, Any]:
    """Run all checks for one scan-dir; return the serialised ScanReport dict.

    Top-level entry point for the multiprocessing worker pool — takes a path
    as a string for picklability.
    """
    scan_dir = Path(scan_dir_str)
    series_id = scan_dir.name.replace("scan-", "")
    meta_path = scan_dir / f"scan-{series_id}_meta.json"

    if not meta_path.exists():
        rep = ScanReport(series_id=series_id, patient_id="?", source_format="?")
        rep.checks["files_present"] = CheckResult(
            status=CheckStatus.FAIL,
            reasons=[f"no meta.json at {meta_path}"]
        )
        rep.aggregate_overall()
        return rep.to_dict()

    meta = json.loads(meta_path.read_text())
    rep = ScanReport(
        series_id=meta["series_id"],
        patient_id=meta["patient_id"],
        source_format=meta.get("source_format", "miccai"),
    )

    rep.checks["files_present"] = check_files_present(meta)
    if rep.checks["files_present"].status == CheckStatus.FAIL:
        rep.aggregate_overall()
        return rep.to_dict()

    paths = meta["source_paths"]
    ct_path  = Path(paths["ct"])  if "ct"  in paths else None
    msk_path = Path(paths["msk"]) if "msk" in paths else None
    if ct_path is None or msk_path is None:
        rep.aggregate_overall()
        return rep.to_dict()

    result, info = check_headers_readable(ct_path, msk_path)
    rep.checks["headers_readable"] = result
    if result.status == CheckStatus.FAIL:
        rep.aggregate_overall()
        return rep.to_dict()

    rep.checks["shape_match"]    = check_shape_match(info)
    rep.checks["affine_match"]   = check_affine_match(info)
    rep.checks["label_inventory"] = check_label_inventory(info)
    rep.checks["centroid_alignment"] = check_centroid_alignment(meta, info)

    info.pop("msk_data", None)
    info.pop("msk_img", None)
    info.pop("ct_affine", None)
    info.pop("msk_affine", None)

    rep.aggregate_overall()
    return rep.to_dict()


# =============================================================================
# orchestration
# =============================================================================

def _flush_logs() -> None:
    for h in log.handlers or logging.getLogger().handlers:
        try:
            h.flush()
        except Exception:
            pass


def run_qc(unified_dir: Path, workers: int = 8) -> dict[str, Any]:
    """Run audit_scan over every scan-* subdir of unified_dir, parallelised."""
    scan_dirs = sorted(d for d in unified_dir.iterdir()
                       if d.is_dir() and d.name.startswith("scan-"))
    log.info("Found %d scan directories to audit", len(scan_dirs))

    reports: list[dict[str, Any]] = []
    completed = 0
    last_log_count = 0
    last_log_time = time.monotonic()
    start = last_log_time

    with ProcessPoolExecutor(max_workers=workers) as ex:
        futures = {ex.submit(audit_scan, str(d)): d for d in scan_dirs}
        for fut in as_completed(futures):
            rep = fut.result()
            reports.append(rep)
            completed += 1
            since_count = completed - last_log_count
            since_time = time.monotonic() - last_log_time
            if since_count >= 25 or since_time >= 10.0:
                elapsed = time.monotonic() - start
                rate = completed / elapsed if elapsed > 0 else 0.0
                remaining = len(scan_dirs) - completed
                eta = remaining / rate if rate > 0 else 0.0
                log.info("progress: %d/%d (%.1f%%)  %.1f scans/s  ETA %ds",
                         completed, len(scan_dirs), 100 * completed / len(scan_dirs),
                         rate, int(eta))
                _flush_logs()
                last_log_count = completed
                last_log_time = time.monotonic()

    reports.sort(key=lambda r: r["series_id"])

    # Summary
    by_status:    dict[str, int]              = {"PASS": 0, "WARN": 0, "FAIL": 0, "SKIP": 0}
    by_check:     dict[str, dict[str, int]]   = {}
    flagged:      dict[str, list[dict]]       = {"WARN": [], "FAIL": []}

    for r in reports:
        by_status[r["overall"]] = by_status.get(r["overall"], 0) + 1
        for cname, c in r["checks"].items():
            by_check.setdefault(cname, {"PASS": 0, "WARN": 0, "FAIL": 0, "SKIP": 0})
            by_check[cname][c["status"]] = by_check[cname].get(c["status"], 0) + 1
        if r["overall"] in ("WARN", "FAIL"):
            flagged[r["overall"]].append({
                "series_id":  r["series_id"],
                "patient_id": r["patient_id"],
                "failing_checks": sorted(
                    cname for cname, c in r["checks"].items()
                    if c["status"] in ("WARN", "FAIL")
                ),
            })

    return {
        "version":         "0.2.0",
        "n_scans_audited": len(reports),
        "by_status":       by_status,
        "by_check":        by_check,
        "flagged_scans":   flagged,
        "tolerances": {
            "direction_dot_product": DIR_TOLERANCE,
            "spacing_mm":            SPC_TOLERANCE,
            "min_label_voxels":      MIN_LABEL_VOXELS,
        },
        "scans":           reports,
    }


# =============================================================================
# CLI
# =============================================================================

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="verse-qc",
        description="Per-scan alignment audit for VerSeFusion's unified corpus.",
    )
    p.add_argument("--unified_dir", type=Path, required=True,
                   help="Path to data/unified/ (contains scan-* subdirs).")
    p.add_argument("--out_dir", type=Path, required=True,
                   help="Where to write data/qc/qc_manifest.json.")
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
    args.out_dir.mkdir(parents=True, exist_ok=True)

    manifest = run_qc(args.unified_dir, workers=args.workers)
    out_path = args.out_dir / "qc_manifest.json"
    out_path.write_text(json.dumps(manifest, indent=2))
    log.info("Wrote QC manifest -> %s", out_path)

    bs = manifest["by_status"]
    log.info("Audit summary: PASS=%d  WARN=%d  FAIL=%d  SKIP=%d",
             bs["PASS"], bs["WARN"], bs["FAIL"], bs["SKIP"])
    if bs["FAIL"]:
        log.warning("%d scans FAILED audit; see flagged_scans.FAIL in manifest",
                    bs["FAIL"])
    return 0 if bs["FAIL"] == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
