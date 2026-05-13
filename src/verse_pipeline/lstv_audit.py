"""
lstv_audit.py — label-based LSTV / TLTV classification for the corrected dataset.

Examines the segmentation labels present in each scan's mask and assigns:
  * LSTV class: lumbarization | sacralization | normal | lsj_fov_truncated | no_lumbar
  * TLTV class: t13_supernumerary | t12_absent | normal_thoracolumbar
                 | tlj_fov_truncated | no_thoracic

Important: This is a LABEL-BASED CLASSIFICATION ONLY.  The output identifies
CANDIDATES for downstream Castellvi grading and morphological review — not
confirmed LSTV diagnoses.  A scan labeled with L6 (label 25) reflects the
annotator's decision to call that vertebra a lumbar variant; whether it
matches Castellvi type I/II/III/IV requires visual review of the lumbosacral
junction (lateralized transverse process, articulation with sacral ala, etc.).

LSTV decision logic
-------------------
Given the set of labels in each mask, examine the lumbosacral region:

  has_l4 = 23 in labels        has_l5 = 24 in labels
  has_l6 = 25 in labels        has_sacrum = 26 in labels

  lumbarization        : L6 (25) is present.  The extra lumbar is anatomically
                         S1 that failed to fuse with the sacrum.
  sacralization        : L5 (24) is absent AND L4 (23) is present AND sacrum
                         is present.  Last lumbar fused with sacrum (only 4
                         lumbars segmented).  The L4 and sacrum requirement
                         confirms the lumbosacral junction is within the FOV.
  normal               : L1-L5 present, no L6.
  lsj_fov_truncated    : Cannot determine — sacrum or L4 missing (FOV doesn't
                         clearly cover the lumbosacral junction).
  no_lumbar            : No lumbar labels at all (focused C-spine or T-spine).

TLTV decision logic
-------------------
  has_t11 = 18 in labels       has_t12 = 19 in labels
  has_t13 = 28 in labels       has_l1  = 20 in labels

  t13_supernumerary    : T13 (28) is present.
  t12_absent           : T12 (19) absent AND T11 (18) and L1 (20) both present
                         (TLJ in FOV).  Missing thoracic anomaly.
  normal_thoracolumbar : T11, T12, L1 present, no T13.
  tlj_fov_truncated    : T11 or L1 missing — can't determine.
  no_thoracic          : No thoracic labels at all.

Output
------
data/lstv/lstv_audit_manifest.json    — full per-subject + summary
data/lstv/lstv_audit_summary.csv      — flat per-subject CSV for spreadsheet review
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import sys
import time
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any

import numpy as np
import nibabel as nib

log = logging.getLogger("verse.lstv_audit")


# =============================================================================
# VerSe label scheme
# =============================================================================

VERSE_LABEL_NAMES: dict[int, str] = {
    1: "C1", 2: "C2", 3: "C3", 4: "C4", 5: "C5", 6: "C6", 7: "C7",
    8: "T1", 9: "T2", 10: "T3", 11: "T4", 12: "T5", 13: "T6",
    14: "T7", 15: "T8", 16: "T9", 17: "T10", 18: "T11", 19: "T12",
    20: "L1", 21: "L2", 22: "L3", 23: "L4", 24: "L5", 25: "L6",
    26: "sacrum", 27: "coccyx", 28: "T13",
}

CERVICAL_LABELS = {1, 2, 3, 4, 5, 6, 7}
THORACIC_LABELS = {8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19}   # T1-T12
LUMBAR_LABELS   = {20, 21, 22, 23, 24, 25}                          # L1-L6
SACRUM = 26
COCCYX = 27
T13    = 28

L1, L2, L3, L4, L5, L6 = 20, 21, 22, 23, 24, 25
T11, T12 = 18, 19


# =============================================================================
# classification
# =============================================================================

@dataclass
class SubjectAudit:
    series_id:        str
    labels_present:   list[int]
    n_labels:         int

    # Boolean presence flags
    has_l1:           bool = False
    has_l4:           bool = False
    has_l5:           bool = False
    has_l6:           bool = False
    has_t11:          bool = False
    has_t12:          bool = False
    has_t13:          bool = False
    has_sacrum:       bool = False
    has_any_lumbar:   bool = False
    has_any_thoracic: bool = False

    # FOV completeness
    lsj_fov_complete: bool = False
    tlj_fov_complete: bool = False

    # Classifications
    lstv_class:       str = "unknown"
    lstv_evidence:    str = ""
    tltv_class:       str = "unknown"
    tltv_evidence:    str = ""

    # Provenance
    veridah_applied:  bool = False
    veridah_correction_type: str | None = None


def _classify_lstv(s: SubjectAudit) -> None:
    """Set s.lstv_class and s.lstv_evidence."""
    if not s.has_any_lumbar:
        s.lstv_class = "no_lumbar"
        s.lstv_evidence = "no lumbar labels in mask (focused C-spine or T-spine)"
        return

    if s.has_l6:
        s.lstv_class = "lumbarization"
        s.lstv_evidence = (
            "L6 (label 25) is present — extra lumbar segment, likely "
            "unfused S1.  Castellvi grading required to confirm phenotype."
        )
        return

    # L6 absent.  To call sacralization we need to confirm LSJ is in FOV:
    # L4 present AND sacrum present.  Otherwise we can't tell if L5 is
    # missing because it's sacralized or just outside FOV.
    if not (s.has_l4 and s.has_sacrum):
        s.lstv_class = "lsj_fov_truncated"
        s.lstv_evidence = (
            f"L6 absent; cannot confirm sacralization because "
            f"L4={s.has_l4} sacrum={s.has_sacrum} — lumbosacral junction "
            f"not fully in FOV."
        )
        return

    # LSJ is in FOV.  L5 presence decides.
    if not s.has_l5:
        s.lstv_class = "sacralization"
        s.lstv_evidence = (
            "L5 (label 24) absent but L4 (23) and sacrum (26) present "
            "(LSJ in FOV) — last lumbar fused with sacrum.  Castellvi "
            "grading required to confirm phenotype."
        )
        return

    # L5 present, no L6, LSJ in FOV
    s.lstv_class = "normal"
    s.lstv_evidence = "L1-L5 present, no L6, LSJ in FOV"


def _classify_tltv(s: SubjectAudit) -> None:
    """Set s.tltv_class and s.tltv_evidence."""
    if not s.has_any_thoracic and not s.has_t13:
        s.tltv_class = "no_thoracic"
        s.tltv_evidence = "no thoracic labels in mask (focused L-spine or C-spine)"
        return

    if s.has_t13:
        s.tltv_class = "t13_supernumerary"
        s.tltv_evidence = (
            "T13 (label 28) is present — supernumerary thoracic vertebra. "
            "Should articulate with ribs."
        )
        return

    # T13 absent.  Confirm TLJ in FOV (need T11 and L1 to bracket it).
    if not s.tlj_fov_complete:
        s.tltv_class = "tlj_fov_truncated"
        s.tltv_evidence = (
            f"T13 absent; cannot confirm T12-absent anomaly because "
            f"T11={s.has_t11} L1={s.has_l1} — thoracolumbar junction "
            f"not fully bracketed."
        )
        return

    # TLJ in FOV.  T12 presence decides.
    if not s.has_t12:
        s.tltv_class = "t12_absent"
        s.tltv_evidence = (
            "T12 (label 19) absent but T11 (18) and L1 (20) present "
            "(TLJ in FOV) — missing thoracic vertebra."
        )
        return

    s.tltv_class = "normal_thoracolumbar"
    s.tltv_evidence = "T11, T12, L1 present, no T13"


def classify_subject(series_id: str, labels_present: set[int],
                     veridah_applied: bool = False,
                     veridah_correction_type: str | None = None) -> SubjectAudit:
    s = SubjectAudit(
        series_id=series_id,
        labels_present=sorted(labels_present),
        n_labels=len(labels_present),
    )

    s.has_l1         = L1  in labels_present
    s.has_l4         = L4  in labels_present
    s.has_l5         = L5  in labels_present
    s.has_l6         = L6  in labels_present
    s.has_t11        = T11 in labels_present
    s.has_t12        = T12 in labels_present
    s.has_t13        = T13 in labels_present
    s.has_sacrum     = SACRUM in labels_present
    s.has_any_lumbar   = bool(labels_present & LUMBAR_LABELS)
    s.has_any_thoracic = bool(labels_present & THORACIC_LABELS)

    s.lsj_fov_complete = s.has_l4 and s.has_sacrum
    s.tlj_fov_complete = s.has_t11 and s.has_l1

    s.veridah_applied = veridah_applied
    s.veridah_correction_type = veridah_correction_type

    _classify_lstv(s)
    _classify_tltv(s)
    return s


# =============================================================================
# per-scan worker
# =============================================================================

def audit_scan(scan_dir_str: str) -> dict[str, Any]:
    scan_dir = Path(scan_dir_str)
    series_id = scan_dir.name.replace("scan-", "")
    meta_path = scan_dir / f"scan-{series_id}_meta.json"

    if not meta_path.exists():
        return {"series_id": series_id, "error": f"no meta.json at {meta_path}"}
    meta = json.loads(meta_path.read_text())

    msk_path = Path(meta["source_paths"]["msk"])
    if not msk_path.exists():
        return {"series_id": series_id, "error": f"mask not on disk: {msk_path}"}

    # Stream the mask in chunks along its largest axis to keep peak memory
    # low.  Full-body VerSe scans can be 512×512×1500+ — loading the whole
    # volume in 8 parallel workers OOMs even with 16G allocated.  At chunk
    # size 32 slices, peak per-worker is ~tens of MB.
    try:
        img = nib.load(str(msk_path))
        shape = tuple(int(s) for s in img.shape[:3])
        biggest_axis = int(np.argmax(shape))
        chunk_size = 32
        labels_present: set[int] = set()
        for start in range(0, shape[biggest_axis], chunk_size):
            end = min(start + chunk_size, shape[biggest_axis])
            slicer = [slice(None)] * 3
            slicer[biggest_axis] = slice(start, end)
            chunk = np.asarray(img.dataobj[tuple(slicer)])
            uniq = np.unique(chunk)
            for v in uniq:
                if v != 0:
                    labels_present.add(int(v))
            del chunk
    except Exception as e:
        return {"series_id": series_id, "error": f"{type(e).__name__}: {e}"}

    veridah_applied = bool(meta.get("veridah_applied", False))
    veridah_type    = meta.get("veridah_correction_type")

    s = classify_subject(series_id, labels_present,
                         veridah_applied=veridah_applied,
                         veridah_correction_type=veridah_type)
    return asdict(s)


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
    log.info("Auditing %d scans in %s", len(scan_dirs), input_dir)

    results: list[dict[str, Any]] = []
    n_done = 0
    last_log_count = 0
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
            if n_done - last_log_count >= 25 or time.monotonic() - last_log_time >= 10.0:
                elapsed = time.monotonic() - start
                rate = n_done / elapsed if elapsed > 0 else 0.0
                remaining = len(scan_dirs) - n_done
                eta = remaining / rate if rate > 0 else 0.0
                log.info("  progress: %d/%d  %.1f scans/s  ETA %ds",
                         n_done, len(scan_dirs), rate, int(eta))
                _flush_logs()
                last_log_count = n_done
                last_log_time = time.monotonic()

    results.sort(key=lambda r: r.get("series_id", ""))
    log.info("Audit done: %d scans", len(results))
    return results


# =============================================================================
# summary / manifest / CSV
# =============================================================================

def _summarize(results: list[dict[str, Any]]) -> dict[str, Any]:
    lstv_counts: Counter[str] = Counter()
    tltv_counts: Counter[str] = Counter()
    cross_tab:   Counter[tuple[str, str]] = Counter()
    by_lstv_cohort: dict[str, list[str]] = defaultdict(list)
    by_tltv_cohort: dict[str, list[str]] = defaultdict(list)

    n_has_l6 = 0
    n_lacks_l5_lsj_in_fov = 0
    n_has_t13 = 0
    n_lacks_t12_tlj_in_fov = 0

    for r in results:
        if "error" in r:
            continue
        lc = r.get("lstv_class", "unknown")
        tc = r.get("tltv_class", "unknown")
        lstv_counts[lc] += 1
        tltv_counts[tc] += 1
        cross_tab[(lc, tc)] += 1
        by_lstv_cohort[lc].append(r["series_id"])
        by_tltv_cohort[tc].append(r["series_id"])

        if r.get("has_l6"):
            n_has_l6 += 1
        if (not r.get("has_l5")) and r.get("lsj_fov_complete"):
            n_lacks_l5_lsj_in_fov += 1
        if r.get("has_t13"):
            n_has_t13 += 1
        if (not r.get("has_t12")) and r.get("tlj_fov_complete") and not r.get("has_t13"):
            n_lacks_t12_tlj_in_fov += 1

    return {
        "n_scans":            len(results),
        "n_errors":           sum(1 for r in results if "error" in r),

        "headline_counts": {
            "has_L6_label":                   n_has_l6,
            "lacks_L5_label_with_LSJ_in_FOV": n_lacks_l5_lsj_in_fov,
            "has_T13_label":                  n_has_t13,
            "lacks_T12_label_with_TLJ_in_FOV": n_lacks_t12_tlj_in_fov,
        },

        "lstv_class_counts": dict(lstv_counts),
        "tltv_class_counts": dict(tltv_counts),
        "cross_tab": {
            f"{lc}__{tc}": n for (lc, tc), n in sorted(cross_tab.items())
        },

        "cohorts": {
            "lstv": {k: sorted(v) for k, v in by_lstv_cohort.items()},
            "tltv": {k: sorted(v) for k, v in by_tltv_cohort.items()},
        },
    }


def write_manifest(results: list[dict[str, Any]], summary: dict[str, Any],
                   input_dir: Path, out_dir: Path) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    manifest = {
        "version":   "0.1.0",
        "input_dir": str(input_dir.resolve()),
        "n_scans":   len(results),
        "summary":   summary,
        "subjects":  results,
    }
    out_path = out_dir / "lstv_audit_manifest.json"
    out_path.write_text(json.dumps(manifest, indent=2))
    return out_path


def write_csv(results: list[dict[str, Any]], out_dir: Path) -> Path:
    """Flat CSV for spreadsheet review.  One row per subject."""
    out_path = out_dir / "lstv_audit_summary.csv"
    fieldnames = [
        "series_id", "n_labels", "labels_present",
        "lstv_class", "lstv_evidence",
        "tltv_class", "tltv_evidence",
        "has_l1", "has_l4", "has_l5", "has_l6",
        "has_t11", "has_t12", "has_t13", "has_sacrum",
        "lsj_fov_complete", "tlj_fov_complete",
        "veridah_applied", "veridah_correction_type",
    ]
    with out_path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in results:
            if "error" in r:
                continue
            row = {k: r.get(k, "") for k in fieldnames}
            row["labels_present"] = ",".join(str(l) for l in r.get("labels_present", []))
            w.writerow(row)
    return out_path


def print_summary(summary: dict[str, Any]) -> None:
    log.info("=" * 72)
    log.info("DATASET AUDIT SUMMARY")
    log.info("=" * 72)
    log.info("Total scans: %d  (errors: %d)",
             summary["n_scans"], summary["n_errors"])
    log.info("")
    log.info("Headline counts:")
    for k, v in summary["headline_counts"].items():
        log.info("  %-40s %d", k, v)
    log.info("")
    log.info("LSTV class:")
    for k, v in sorted(summary["lstv_class_counts"].items(), key=lambda kv: -kv[1]):
        log.info("  %-22s %d", k, v)
    log.info("")
    log.info("TLTV class:")
    for k, v in sorted(summary["tltv_class_counts"].items(), key=lambda kv: -kv[1]):
        log.info("  %-22s %d", k, v)
    log.info("")
    log.info("Cross-tabulation (LSTV × TLTV):")
    for k, v in sorted(summary["cross_tab"].items(), key=lambda kv: -kv[1]):
        if v > 0:
            log.info("  %-50s %d", k, v)
    log.info("=" * 72)


# =============================================================================
# CLI
# =============================================================================

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="verse-lstv-audit",
        description="Label-based LSTV / TLTV classification for the corrected dataset.",
    )
    p.add_argument("--input_dir", type=Path, required=True,
                   help="Path to data/corrected/ (or data/canonical/ for pre-VERIDAH stats).")
    p.add_argument("--out_dir",   type=Path, required=True,
                   help="Where to write lstv_audit_manifest.json and lstv_audit_summary.csv.")
    p.add_argument("--workers",   type=int, default=8)
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
    summary = _summarize(results)

    manifest_path = write_manifest(results, summary, args.input_dir, args.out_dir)
    csv_path      = write_csv(results, args.out_dir)
    log.info("Wrote %s", manifest_path)
    log.info("Wrote %s", csv_path)

    print_summary(summary)
    return 0


if __name__ == "__main__":
    sys.exit(main())
