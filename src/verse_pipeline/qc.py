"""
qc.py — Per-scan QC audit for canonically reoriented VerSeFusion scans.

Runs on the output of reorient.py (data/canonical/scan-*/).  All inputs are
in PIR orientation, which lets us hardcode anatomical assumptions:
  - axis 0 = P direction (toward back)
  - axis 1 = I direction (toward feet) — VERTEBRAE STACK ALONG THIS AXIS
  - axis 2 = R direction (toward right side)

ALL CHECKS ARE PER-SCAN AND LOCAL.  This is a pure auditability stage.

Checks performed
----------------
Tier 1 — Header-only:
  1. files_present     — required CT, mask present on disk
  2. headers_readable  — nibabel can load each NIfTI header
  3. shape_match       — CT.shape == mask.shape
  4. affine_match      — CT and mask affines agree on direction + spacing
  5. orientation_pir   — both images report PIR orientation in their affines

Tier 2 — Mask data:
  6. label_inventory   — labels fall in VerSe range [1, 28], no tiny artifacts
  7. label_continuity  — label CoMs are monotonic along axis 1 (I direction):
                         label 1 (C1) is most superior (lowest axis-1 voxel),
                         label 24+ (lumbar/sacrum) is most inferior (highest)

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

STATUS_RANK = {
    CheckStatus.PASS: 0,
    CheckStatus.SKIP: 1,
    CheckStatus.WARN: 2,
    CheckStatus.FAIL: 3,
}

DIR_TOLERANCE = 1e-3
SPC_TOLERANCE = 0.01    # mm

VERSE_LABEL_RANGE = (1, 28)
MIN_LABEL_VOXELS  = 50

# After PIR reorientation, axis 1 = I direction = head-to-foot.
# VerSe labels increase from C1 (top, most superior) downward; increasing
# axis 1 voxel index = more inferior.  Therefore label CoMs should
# monotonically INCREASE in axis 1 as label number increases.
SPINE_AXIS = 1
EXPECTED_DIRECTION = "ascending"

MAX_LABEL_INVERSIONS = 1


# =============================================================================
# data classes
# =============================================================================

@dataclass
class CheckResult:
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
# geometry helpers
# =============================================================================

def _directions_match(d1: np.ndarray, d2: np.ndarray,
                      tol: float = DIR_TOLERANCE) -> tuple[bool, list[str]]:
    reasons: list[str] = []
    ok = True
    for i in range(3):
        dot = abs(float(np.dot(d1[:, i], d2[:, i])))
        if dot < 1.0 - tol:
            ok = False
            reasons.append(f"axis {i}: dot product {dot:.4f} < 1-tol={1 - tol:.4f}")
    return ok, reasons


def _decompose_affine(affine: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    m = affine[:3, :3]
    spacing = np.array([np.linalg.norm(m[:, i]) for i in range(3)])
    if np.any(spacing < 1e-9):
        raise ValueError(f"degenerate affine, spacing has zero norm: {spacing}")
    direction = m / spacing[np.newaxis, :]
    return direction, spacing


def _affine_orientation_codes(affine: np.ndarray) -> tuple[str, str, str]:
    """Determine which anatomical direction each array axis points to.

    Returns a 3-tuple like ('P', 'I', 'R') indicating the dominant world
    direction for each array axis.
    """
    from nibabel.orientations import io_orientation
    ornt = io_orientation(affine)
    # ornt is (3, 2): row i = [source_axis, flip].  The OUTPUT here is which
    # RAS axis each array axis maps to.  But we want PIR-style codes.
    # io_orientation actually returns mapping of array_axis -> RAS_axis,
    # so we read it directly.
    ras_axis_codes = ["R", "A", "S"]
    flipped_codes  = ["L", "P", "I"]
    codes = []
    for source_axis, flip in ornt:
        idx = int(source_axis)
        if flip > 0:
            codes.append(ras_axis_codes[idx])
        else:
            codes.append(flipped_codes[idx])
    return tuple(codes)


# =============================================================================
# individual checks
# =============================================================================

def check_files_present(meta: dict[str, Any]) -> CheckResult:
    paths = meta.get("source_paths", {})
    missing_required = [k for k in ("ct", "msk")
                        if not paths.get(k) or not Path(paths[k]).exists()]
    if missing_required:
        return CheckResult(status=CheckStatus.FAIL,
                           reasons=[f"missing required source file(s): {missing_required}"])
    return CheckResult(status=CheckStatus.PASS,
                       reasons=["CT and mask present on disk"])


def check_headers_readable(ct_path: Path, msk_path: Path) -> tuple[CheckResult, dict]:
    import nibabel as nib
    info: dict = {}
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
        reasons.append(f"spacing differs by {spc_diff:.4f}mm > {SPC_TOLERANCE}mm")

    origin_diff = np.abs(ct_aff[:3, 3] - mk_aff[:3, 3]).max()
    if origin_diff < 0.5:
        reasons.append(f"origins agree (diff={origin_diff:.4f}mm)")
    elif origin_diff < 5.0:
        reasons.append(f"WARN: origins differ by {origin_diff:.4f}mm (acceptable)")
    else:
        ok = False
        reasons.append(f"origins differ by {origin_diff:.4f}mm > 5mm")

    return CheckResult(status=CheckStatus.PASS if ok else CheckStatus.FAIL,
                       reasons=reasons)


def check_orientation_pir(info: dict) -> CheckResult:
    """Verify the CT affine reports PIR orientation.  This is the cornerstone
    check after reorient; everything downstream assumes PIR.
    """
    ct_aff = info.get("ct_affine")
    if ct_aff is None:
        return CheckResult(status=CheckStatus.SKIP, reasons=["CT header not loaded"])
    try:
        codes = _affine_orientation_codes(ct_aff)
    except Exception as e:
        return CheckResult(status=CheckStatus.FAIL,
                           reasons=[f"could not derive orientation: {type(e).__name__}: {e}"])

    expected = ("P", "I", "R")
    extra = {"orientation_codes": list(codes), "expected": list(expected)}
    if codes == expected:
        return CheckResult(status=CheckStatus.PASS,
                           reasons=[f"orientation = {''.join(codes)}"],
                           extra=extra)
    return CheckResult(status=CheckStatus.FAIL,
                       reasons=[f"orientation = {''.join(codes)}, expected {''.join(expected)}"],
                       extra=extra)


def check_label_inventory(info: dict) -> CheckResult:
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
        reasons.append(f"FAIL: {len(out_of_range)} labels outside VerSe range [1, 28]: {out_of_range}")
        return CheckResult(status=CheckStatus.FAIL, reasons=reasons, extra=extra)

    if tiny:
        reasons.append(f"WARN: {len(tiny)} labels have < {MIN_LABEL_VOXELS} voxels: {tiny}")
        return CheckResult(status=CheckStatus.WARN, reasons=reasons, extra=extra)

    return CheckResult(status=CheckStatus.PASS, reasons=reasons, extra=extra)


def check_label_continuity(info: dict) -> CheckResult:
    """Verify per-label CoMs are monotonic along the I axis (axis 1 in PIR).

    Since axis 1 is the inferior direction and VerSe labels increase
    caudally (C1=1 most superior, L6/sacrum highest), CoM[axis 1] should
    increase monotonically with label number.
    """
    msk_data = info.get("msk_data")
    if msk_data is None:
        return CheckResult(status=CheckStatus.SKIP, reasons=["mask data not loaded"])

    label_voxel_counts: dict[int, int] = info.get("label_voxel_counts") or {}
    labels_present = sorted(l for l in label_voxel_counts if l > 0)
    if len(labels_present) < 2:
        return CheckResult(status=CheckStatus.SKIP,
                           reasons=[f"only {len(labels_present)} label(s) present"])

    coms: dict[int, tuple[float, float, float]] = {}
    for label in labels_present:
        coords = np.argwhere(msk_data == label)
        if len(coords) > 0:
            com = coords.mean(axis=0)
            coms[label] = tuple(float(v) for v in com)

    # Axis spreads (for diagnostic — we expect spine to be axis 1)
    com_array = np.array(list(coms.values()))
    spreads = com_array.ptp(axis=0)
    dominant_axis = int(np.argmax(spreads))

    sorted_labels = sorted(coms)
    axis_positions = [coms[l][SPINE_AXIS] for l in sorted_labels]
    diffs = np.diff(axis_positions)

    n_ascending = int((diffs > 0).sum())
    n_descending = int((diffs < 0).sum())
    n_inversions = (n_descending if EXPECTED_DIRECTION == "ascending"
                    else n_ascending)

    reasons = [
        f"{len(coms)} labels; checking continuity along axis {SPINE_AXIS} (I direction)",
        f"label-ordered CoMs run {EXPECTED_DIRECTION}? "
        f"{n_ascending} steps up, {n_descending} steps down ({n_inversions} inversions)",
    ]
    if dominant_axis != SPINE_AXIS:
        reasons.append(f"WARN: dominant CoM spread is on axis {dominant_axis} "
                       f"(spread={spreads[dominant_axis]:.1f}), not the expected "
                       f"spine axis {SPINE_AXIS} (spread={spreads[SPINE_AXIS]:.1f}). "
                       f"Possibly a single-vertebra scan or reorient anomaly.")

    extra = {
        "spine_axis":          SPINE_AXIS,
        "dominant_axis":       dominant_axis,
        "axis_spreads_voxels": [float(s) for s in spreads],
        "n_inversions":        n_inversions,
        "label_coms_voxel":    {str(l): list(c) for l, c in coms.items()},
    }

    if dominant_axis != SPINE_AXIS and spreads[SPINE_AXIS] < 5.0:
        # The spine isn't oriented along axis 1; reorient may have failed
        return CheckResult(status=CheckStatus.FAIL,
                           reasons=reasons + ["FAIL: spine extent on expected axis < 5 voxels"],
                           extra=extra)
    if n_inversions <= MAX_LABEL_INVERSIONS:
        return CheckResult(status=CheckStatus.PASS, reasons=reasons, extra=extra)
    if n_inversions <= 3:
        reasons.append(f"WARN: {n_inversions} > {MAX_LABEL_INVERSIONS} inversions")
        return CheckResult(status=CheckStatus.WARN, reasons=reasons, extra=extra)
    reasons.append(f"FAIL: {n_inversions} inversions suggest label assignment errors")
    return CheckResult(status=CheckStatus.FAIL, reasons=reasons, extra=extra)


# =============================================================================
# per-scan driver
# =============================================================================

def audit_scan(scan_dir_str: str) -> dict[str, Any]:
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
    ct_path  = Path(paths["ct"])
    msk_path = Path(paths["msk"])

    result, info = check_headers_readable(ct_path, msk_path)
    rep.checks["headers_readable"] = result
    if result.status == CheckStatus.FAIL:
        rep.aggregate_overall()
        return rep.to_dict()

    rep.checks["shape_match"]     = check_shape_match(info)
    rep.checks["affine_match"]    = check_affine_match(info)
    rep.checks["orientation_pir"] = check_orientation_pir(info)

    li = check_label_inventory(info)
    rep.checks["label_inventory"] = li
    if li.extra and "label_voxel_counts" in li.extra:
        info["label_voxel_counts"] = li.extra["label_voxel_counts"]

    rep.checks["label_continuity"] = check_label_continuity(info)

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


def run_qc(input_dir: Path, workers: int = 8) -> dict[str, Any]:
    scan_dirs = sorted(d for d in input_dir.iterdir()
                       if d.is_dir() and d.name.startswith("scan-"))
    log.info("Found %d scan directories to audit in %s", len(scan_dirs), input_dir)

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

    by_status: dict[str, int]            = {"PASS": 0, "WARN": 0, "FAIL": 0, "SKIP": 0}
    by_check:  dict[str, dict[str, int]] = {}
    flagged:   dict[str, list[dict]]     = {"WARN": [], "FAIL": []}

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
        "version":         "0.4.0",
        "input_dir":       str(input_dir),
        "n_scans_audited": len(reports),
        "by_status":       by_status,
        "by_check":        by_check,
        "flagged_scans":   flagged,
        "spine_axis":          SPINE_AXIS,
        "expected_direction":  EXPECTED_DIRECTION,
        "tolerances": {
            "direction_dot_product":   DIR_TOLERANCE,
            "spacing_mm":              SPC_TOLERANCE,
            "min_label_voxels":        MIN_LABEL_VOXELS,
            "max_label_inversions":    MAX_LABEL_INVERSIONS,
        },
        "scans":           reports,
    }


# =============================================================================
# CLI
# =============================================================================

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="verse-qc",
        description="Per-scan QC for canonical (PIR-reoriented) VerSe scans.",
    )
    p.add_argument("--input_dir", type=Path, required=True,
                   help="Directory of scan-* (typically data/canonical/).")
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

    if not args.input_dir.is_dir():
        log.error("input_dir not found: %s", args.input_dir)
        return 1
    args.out_dir.mkdir(parents=True, exist_ok=True)

    manifest = run_qc(args.input_dir, workers=args.workers)
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
