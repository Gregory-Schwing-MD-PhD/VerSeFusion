"""Apply VERIDAH (Möller et al. 2026) manual label corrections to canonical VerSe.

Input  : data/canonical/scan-<id>/   (PIR-reoriented CT + mask)
Output : data/corrected/scan-<id>/   (PIR + Möller corrections applied to mask)

For each subject in configs/veridah_corrections.csv:
  * t13_shift:       L1→T13, L2→L1, …  (most common, ~14 cases)
  * label_override:  full re-sequence via mask-CoM-sorted label order (3 cases)
  * advisory_only:   no remap, just record TLTV / stump-rib flags

For everyone else: pass-through (symlinks to canonical).

The original veridah.py paired LabelOverride lists against TUM's centroid
JSON file.  We dropped centroids in unify (see chunk 1 limitations), so this
version pairs the override list against labels sorted by their mask
center-of-mass along axis 1 (the I direction in PIR).  That order is the
cranial→caudal spine order, which is what Möller's override sequences
follow.  This is more robust than using TUM's centroid file, since the
centroid file uses inconsistent coordinate conventions across the corpus
but mask-derived CoM is unambiguous.

Möller's note on T13 column (private correspondence):
  "T13=1 was initially set for verse559 but we then realized the surrounding
  labels were wrong and added a full LabelOverride.  Since LabelOverride
  takes priority over T13, we never reset T13.  In short, ignore T13 if
  LabelOverride is set."

References
----------
Möller H. et al. 2026.  VERIDAH: Solving Enumeration Anomaly Aware Vertebra
Labeling across Imaging Sequences.  arXiv:2601.14066.
"""

from __future__ import annotations

import argparse
import ast
import csv
import json
import logging
import re
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import nibabel as nib

log = logging.getLogger("verse.veridah")


# =============================================================================
# CSV loader
# =============================================================================

@dataclass
class Correction:
    """One row of the VERIDAH corrections table, normalized."""
    fid:             str
    candidate_subjects: list[str]      # ordered candidates to match against canonical scan-dirs
    dataset:         str
    vert:            int
    has_t11:         bool
    has_t13:         bool
    label_override:  list[int] | None
    tltv:            bool
    sr_left:         bool
    sr_right:        bool

    @property
    def correction_type(self) -> str:
        # LabelOverride takes priority over T13 per Möller's clarification.
        if self.label_override is not None:
            return "label_override"
        if self.has_t13:
            return "t13_shift"
        if self.has_t11:
            return "t11_shift"
        return "advisory_only"


_SUBJECT_CORE_RE = re.compile(r"^(verse\d+|gl\d+)")
_SPLIT_RE        = re.compile(r"split-(verse\d+|gl\d+)")


def _candidate_subjects(fid: str) -> list[str]:
    """All possible series_ids this fid might refer to.

    The CSV uses fids like:
        verse011                       → ["verse011"]
        verse403_split-verse255        → ["verse403", "verse255"]  (cross-release)
        verse509_dir-iso               → ["verse509"]              (image variant)
        verse642_dir-sag               → ["verse642"]
        gl003                          → ["gl003"]

    Unify chose ONE release per patient (preferring v20), so a CSV row with
    split-verse255 could end up matching either scan-verse403/ (if v19 won)
    or scan-verse255/ (if v20 won).  We return both candidates in CSV order
    and the caller checks which directory actually exists.
    """
    out: list[str] = []
    primary = _SUBJECT_CORE_RE.match(fid)
    if primary:
        out.append(primary.group(1))
    split_match = _SPLIT_RE.search(fid)
    if split_match:
        other = split_match.group(1)
        if other not in out:
            out.append(other)
    return out


def load_corrections(csv_path: Path) -> dict[str, Correction]:
    """Read corrections CSV; return {fid: Correction}."""
    corrections: dict[str, Correction] = {}
    with csv_path.open(newline="") as f:
        for row in csv.DictReader(f):
            fid = row["fid"].strip()
            override_raw = (row.get("LabelOverride") or "").strip()
            override: list[int] | None = None
            if override_raw:
                try:
                    override = [int(x) for x in ast.literal_eval(override_raw)]
                except (ValueError, SyntaxError):
                    log.warning("Could not parse LabelOverride for %s: %r", fid, override_raw)

            try:
                vert = int(row["vert"])
            except (ValueError, KeyError):
                vert = -1

            def _bool(field: str) -> bool:
                v = (row.get(field) or "0").strip()
                return v == "1"

            c = Correction(
                fid                = fid,
                candidate_subjects = _candidate_subjects(fid),
                dataset            = (row.get("dataset") or "").strip(),
                vert               = vert,
                has_t11            = _bool("T11"),
                has_t13            = _bool("T13"),
                label_override     = override,
                tltv               = _bool("TLTV"),
                sr_left            = _bool("SR_l"),
                sr_right           = _bool("SR_r"),
            )
            corrections[fid] = c
    log.info("Loaded %d VERIDAH corrections from %s", len(corrections), csv_path)
    return corrections


# =============================================================================
# remap construction
# =============================================================================

def _t13_shift_remap(mask_labels: Iterable[int]) -> dict[int, int]:
    """T13-shift remap.

      20 (was L1)   → 28 (T13)
      21..25 (lumbar) → lumbar - 1   (L2→L1, L3→L2, …)
    """
    remap: dict[int, int] = {}
    for lbl in mask_labels:
        if lbl == 20:
            remap[20] = 28
        elif 21 <= lbl <= 25:
            remap[lbl] = lbl - 1
    return remap


def _t11_shift_remap(mask_labels: Iterable[int]) -> dict[int, int]:
    """T11-shift remap (missing thoracic; not used in current VERIDAH table)."""
    remap: dict[int, int] = {}
    for lbl in mask_labels:
        if 20 <= lbl <= 24:
            remap[lbl] = lbl + 1
    return remap


class LabelOverrideMismatch(Exception):
    """Raised when mask label count differs from Möller's override list length."""
    def __init__(self, subject: str, n_mask: int, n_override: int):
        self.subject = subject
        self.n_mask = n_mask
        self.n_override = n_override
        super().__init__(
            f"{subject}: mask has {n_mask} non-zero labels but LabelOverride "
            f"has {n_override} entries — refusing to pair, since min(len) silently "
            f"mis-labels one end of the spine."
        )


def _spatial_order_from_mask(mask_data: np.ndarray,
                             labels: list[int]) -> list[int]:
    """Return labels sorted cranial→caudal by mask CoM along the I axis.

    Assumes the mask is in canonical PIR orientation (axis 1 = inferior).
    Smaller axis-1 CoM = more superior = more cranial = comes first.
    """
    coms = {}
    for label in labels:
        coords = np.argwhere(mask_data == label)
        if len(coords) > 0:
            coms[label] = float(coords[:, 1].mean())   # axis 1 = I
    return sorted(coms, key=lambda l: coms[l])


def _label_override_remap(
    correction: Correction,
    mask_data:  np.ndarray,
    mask_labels: list[int],
    *,
    allow_length_mismatch: bool = False,
) -> dict[int, int]:
    """Pair the override list with mask-CoM spatial order to build a remap."""
    assert correction.label_override is not None
    ordered_old = _spatial_order_from_mask(mask_data, mask_labels)
    new_seq = correction.label_override

    if len(ordered_old) != len(new_seq):
        if not allow_length_mismatch:
            raise LabelOverrideMismatch(
                correction.fid, len(ordered_old), len(new_seq),
            )
        log.warning(
            "Length mismatch for %s: mask has %d labels, override has %d — "
            "pairing from cranial end (legacy min-len behaviour)",
            correction.fid, len(ordered_old), len(new_seq),
        )

    n = min(len(ordered_old), len(new_seq))
    remap: dict[int, int] = {}
    for old, new in zip(ordered_old[:n], new_seq[:n]):
        if old != new:
            remap[old] = new
    return remap


def build_remap(
    correction: Correction,
    mask_data: np.ndarray,
    mask_labels: list[int],
    *,
    allow_length_mismatch: bool = False,
) -> dict[int, int]:
    """Dispatch to the right remap-builder for this correction type."""
    if correction.label_override is not None:
        return _label_override_remap(
            correction, mask_data, mask_labels,
            allow_length_mismatch=allow_length_mismatch,
        )
    if correction.has_t13:
        return _t13_shift_remap(mask_labels)
    if correction.has_t11:
        return _t11_shift_remap(mask_labels)
    return {}


# =============================================================================
# apply
# =============================================================================

def apply_remap_to_mask(mask_data: np.ndarray, remap: dict[int, int]) -> np.ndarray:
    """Apply a label remap to a mask via a LUT."""
    if not remap:
        return mask_data.copy()
    max_val = int(max(int(mask_data.max()),
                      max(remap.keys()), max(remap.values())))
    lut = np.arange(max_val + 1, dtype=np.int32)
    for old, new in remap.items():
        lut[old] = new
    clipped = np.clip(mask_data.astype(np.int32), 0, max_val)
    return lut[clipped].astype(np.uint8)


# =============================================================================
# per-scan orchestration
# =============================================================================

@dataclass
class CorrectionRecord:
    """Per-scan result for the manifest."""
    series_id:          str
    veridah_applied:    bool
    correction_type:    str
    fid:                str | None = None
    remap:              dict[int, int] = field(default_factory=dict)
    tltv:               bool = False
    sr_left:            bool = False
    sr_right:           bool = False
    error:              str | None = None


def _passthrough_symlinks(scan_in: Path, scan_out: Path) -> None:
    """Symlink all files from scan_in to scan_out (used for uncorrected subjects)."""
    scan_out.mkdir(parents=True, exist_ok=True)
    for src in sorted(scan_in.iterdir()):
        dst = scan_out / src.name
        if dst.exists() or dst.is_symlink():
            dst.unlink()
        dst.symlink_to(src.absolute())


def correct_scan(
    scan_in: Path,
    scan_out: Path,
    correction: Correction | None,
    *,
    allow_length_mismatch: bool = False,
) -> CorrectionRecord:
    """Apply correction (or pass through) for one canonical scan-dir."""
    series_id = scan_in.name.replace("scan-", "")
    meta_path_in = scan_in / f"scan-{series_id}_meta.json"
    if not meta_path_in.exists():
        return CorrectionRecord(
            series_id=series_id, veridah_applied=False,
            correction_type="error",
            error=f"no meta.json at {meta_path_in}",
        )
    meta = json.loads(meta_path_in.read_text())

    # ---- advisory_only or no correction → pass-through --------------------
    if correction is None or correction.correction_type == "advisory_only":
        _passthrough_symlinks(scan_in, scan_out)
        # Update meta even on passthrough so downstream knows if advisory flags were set
        new_meta = dict(meta)
        new_meta["veridah_applied"]   = False
        new_meta["veridah_correction_type"] = (correction.correction_type
                                               if correction else "none")
        if correction is not None:
            new_meta["veridah_fid"]      = correction.fid
            new_meta["veridah_tltv"]     = correction.tltv
            new_meta["veridah_sr_left"]  = correction.sr_left
            new_meta["veridah_sr_right"] = correction.sr_right
        new_meta["version"] = "0.6.0"
        meta_path_out = scan_out / f"scan-{series_id}_meta.json"
        if meta_path_out.exists() or meta_path_out.is_symlink():
            meta_path_out.unlink()
        meta_path_out.write_text(json.dumps(new_meta, indent=2))
        return CorrectionRecord(
            series_id=series_id, veridah_applied=False,
            correction_type=(correction.correction_type if correction else "none"),
            fid=(correction.fid if correction else None),
            tltv=(correction.tltv if correction else False),
            sr_left=(correction.sr_left if correction else False),
            sr_right=(correction.sr_right if correction else False),
        )

    # ---- need to actually remap labels ------------------------------------
    paths = meta.get("source_paths", {})
    ct_path  = paths.get("ct")
    msk_path = paths.get("msk")
    snp_path = paths.get("snp")
    if not msk_path or not Path(msk_path).exists():
        return CorrectionRecord(
            series_id=series_id, veridah_applied=False,
            correction_type=correction.correction_type, fid=correction.fid,
            error="canonical mask not on disk",
        )

    mask_img = nib.load(msk_path)
    mask_data = np.asarray(mask_img.dataobj).astype(np.int32)
    mask_labels = sorted({int(v) for v in np.unique(mask_data) if v != 0})

    try:
        remap = build_remap(
            correction, mask_data, mask_labels,
            allow_length_mismatch=allow_length_mismatch,
        )
    except LabelOverrideMismatch as e:
        log.error("Skipping %s: %s", series_id, e)
        _passthrough_symlinks(scan_in, scan_out)
        return CorrectionRecord(
            series_id=series_id, veridah_applied=False,
            correction_type=correction.correction_type, fid=correction.fid,
            tltv=correction.tltv, sr_left=correction.sr_left, sr_right=correction.sr_right,
            error=f"length_mismatch:mask={e.n_mask},override={e.n_override}",
        )

    if not remap:
        # nothing to change
        _passthrough_symlinks(scan_in, scan_out)
        return CorrectionRecord(
            series_id=series_id, veridah_applied=False,
            correction_type=correction.correction_type, fid=correction.fid,
            tltv=correction.tltv, sr_left=correction.sr_left, sr_right=correction.sr_right,
        )

    # ---- write corrected mask, symlink CT + snp ---------------------------
    scan_out.mkdir(parents=True, exist_ok=True)
    new_mask_data = apply_remap_to_mask(mask_data, remap)
    new_mask = nib.Nifti1Image(new_mask_data, mask_img.affine, header=mask_img.header)
    new_mask.set_data_dtype(np.uint8)
    out_msk = scan_out / f"scan-{series_id}_msk.nii.gz"
    if out_msk.exists() or out_msk.is_symlink():
        out_msk.unlink()
    nib.save(new_mask, str(out_msk))

    # Symlink CT (unchanged) and snp (unchanged)
    if ct_path and Path(ct_path).exists():
        dst = scan_out / f"scan-{series_id}_ct.nii.gz"
        if dst.exists() or dst.is_symlink():
            dst.unlink()
        dst.symlink_to(Path(ct_path).absolute())
    if snp_path and Path(snp_path).exists():
        dst = scan_out / f"scan-{series_id}_snp.png"
        if dst.exists() or dst.is_symlink():
            dst.unlink()
        dst.symlink_to(Path(snp_path).absolute())

    # ---- write updated meta -----------------------------------------------
    new_meta = dict(meta)
    new_meta["source_paths"] = dict(meta["source_paths"])
    new_meta["source_paths"]["msk"] = str(out_msk.absolute())
    new_meta["original_paths"] = dict(meta.get("original_paths", {}))
    new_meta["original_paths"]["msk_canonical"] = msk_path
    new_meta["veridah_applied"]         = True
    new_meta["veridah_fid"]             = correction.fid
    new_meta["veridah_correction_type"] = correction.correction_type
    new_meta["veridah_remap"]           = {str(k): int(v) for k, v in remap.items()}
    new_meta["veridah_tltv"]            = correction.tltv
    new_meta["veridah_sr_left"]         = correction.sr_left
    new_meta["veridah_sr_right"]        = correction.sr_right
    new_meta["version"]                 = "0.6.0"
    meta_path_out = scan_out / f"scan-{series_id}_meta.json"
    if meta_path_out.exists() or meta_path_out.is_symlink():
        meta_path_out.unlink()
    meta_path_out.write_text(json.dumps(new_meta, indent=2))

    log.info("Corrected %s (type=%s, remap=%s)",
             series_id, correction.correction_type, remap)

    return CorrectionRecord(
        series_id=series_id, veridah_applied=True,
        correction_type=correction.correction_type, fid=correction.fid,
        remap={int(k): int(v) for k, v in remap.items()},
        tltv=correction.tltv, sr_left=correction.sr_left, sr_right=correction.sr_right,
    )


# =============================================================================
# parallel orchestration
# =============================================================================

def _correct_one(args: tuple[str, str, dict | None, bool]) -> dict[str, Any]:
    scan_in_str, scan_out_str, correction_dict, allow_mismatch = args
    correction = None
    if correction_dict is not None:
        correction = Correction(**correction_dict)
    try:
        rec = correct_scan(
            Path(scan_in_str), Path(scan_out_str), correction,
            allow_length_mismatch=allow_mismatch,
        )
        return rec.__dict__
    except Exception as e:
        return {"series_id": Path(scan_in_str).name.replace("scan-", ""),
                "veridah_applied": False,
                "correction_type": "error",
                "error": f"{type(e).__name__}: {e}",
                "remap": {}, "tltv": False, "sr_left": False, "sr_right": False,
                "fid": correction.fid if correction else None}


def _flush_logs() -> None:
    for h in log.handlers or logging.getLogger().handlers:
        try:
            h.flush()
        except Exception:
            pass


def _build_subject_to_correction(
    corrections: dict[str, Correction],
    scan_dirs: list[Path],
) -> dict[str, Correction]:
    """Match CSV fids to actual canonical scan-dirs.

    Each correction has 1-2 candidate subject names; pick the first that
    matches an existing scan-dir.
    """
    available = {d.name.replace("scan-", "") for d in scan_dirs}
    out: dict[str, Correction] = {}
    unmatched: list[str] = []
    for fid, c in corrections.items():
        match = None
        for cand in c.candidate_subjects:
            if cand in available:
                match = cand
                break
        if match is None:
            unmatched.append(fid)
            continue
        if match in out:
            log.warning("Multiple CSV fids match %s (kept %s, ignoring %s)",
                        match, out[match].fid, fid)
            continue
        out[match] = c
    if unmatched:
        log.warning("%d CSV corrections did not match any canonical scan: %s",
                    len(unmatched), unmatched)
    return out


def correct_all(
    in_dir: Path,
    out_dir: Path,
    corrections: dict[str, Correction],
    *,
    workers: int = 8,
    allow_length_mismatch: bool = False,
) -> list[CorrectionRecord]:
    out_dir.mkdir(parents=True, exist_ok=True)
    scan_dirs = sorted(d for d in in_dir.iterdir()
                       if d.is_dir() and d.name.startswith("scan-"))
    log.info("Processing %d scans (corrections available for %d)",
             len(scan_dirs), len(corrections))

    subject_to_correction = _build_subject_to_correction(corrections, scan_dirs)

    work_items: list[tuple[str, str, dict | None, bool]] = []
    for d in scan_dirs:
        sid = d.name.replace("scan-", "")
        correction = subject_to_correction.get(sid)
        correction_dict = (correction.__dict__ if correction is not None else None)
        work_items.append((str(d), str(out_dir / d.name),
                           correction_dict, allow_length_mismatch))

    results: list[CorrectionRecord] = []
    n_done = 0
    last_log_count = 0
    last_log_time = time.monotonic()
    start = last_log_time

    with ProcessPoolExecutor(max_workers=workers) as ex:
        futures = {ex.submit(_correct_one, item): item for item in work_items}
        for fut in as_completed(futures):
            r_dict = fut.result()
            results.append(CorrectionRecord(**{
                k: v for k, v in r_dict.items()
                if k in CorrectionRecord.__dataclass_fields__
            }))
            n_done += 1
            if n_done - last_log_count >= 25 or time.monotonic() - last_log_time >= 10.0:
                elapsed = time.monotonic() - start
                rate = n_done / elapsed if elapsed > 0 else 0.0
                remaining = len(scan_dirs) - n_done
                eta = remaining / rate if rate > 0 else 0.0
                log.info("progress: %d/%d (%.1f%%)  %.1f scans/s  ETA %ds",
                         n_done, len(scan_dirs),
                         100 * n_done / len(scan_dirs), rate, int(eta))
                _flush_logs()
                last_log_count = n_done
                last_log_time = time.monotonic()

    results.sort(key=lambda r: r.series_id)
    n_corrected = sum(1 for r in results if r.veridah_applied)
    n_errors    = sum(1 for r in results if r.error)
    log.info("Veridah done: %d corrected, %d passed through, %d errors",
             n_corrected, len(results) - n_corrected - n_errors, n_errors)
    return results


def write_veridah_manifest(
    results: list[CorrectionRecord],
    out_dir: Path,
    csv_path: Path,
) -> Path:
    n_corrected = sum(1 for r in results if r.veridah_applied)
    by_type: dict[str, int] = {}
    for r in results:
        if r.veridah_applied:
            by_type[r.correction_type] = by_type.get(r.correction_type, 0) + 1

    manifest = {
        "version":     "0.6.0",
        "csv_source":  str(csv_path.resolve()),
        "n_scans":     len(results),
        "n_corrected": n_corrected,
        "n_passthrough": len(results) - n_corrected,
        "by_correction_type": by_type,
        "corrections": [
            {
                "series_id":         r.series_id,
                "fid":               r.fid,
                "veridah_applied":   r.veridah_applied,
                "correction_type":   r.correction_type,
                "remap":             {str(k): v for k, v in r.remap.items()},
                "tltv":              r.tltv,
                "sr_left":           r.sr_left,
                "sr_right":          r.sr_right,
                "error":             r.error,
            }
            for r in results if r.fid is not None or r.veridah_applied or r.error
        ],
    }
    out_path = out_dir / "veridah_manifest.json"
    out_path.write_text(json.dumps(manifest, indent=2))
    return out_path


# =============================================================================
# CLI
# =============================================================================

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="verse-veridah",
        description=(
            "Apply Möller 2026 VERIDAH manual label corrections to canonical "
            "(PIR-reoriented) VerSe scans.  Input: data/canonical/.  Output: "
            "data/corrected/ with corrected mask + updated meta.json for the "
            "~25 affected subjects, and symlinks for everyone else."
        ),
    )
    p.add_argument("--in_dir",  type=Path, required=True,
                   help="Canonical tree (typically data/canonical/).")
    p.add_argument("--out_dir", type=Path, required=True,
                   help="Where to write the corrected tree.")
    p.add_argument("--corrections_csv", type=Path, required=True,
                   help="Path to configs/veridah_corrections.csv.")
    p.add_argument("--workers", type=int, default=8)
    p.add_argument("--allow_length_mismatch", action="store_true",
                   help="Fall back to legacy min(len) pairing for LabelOverride "
                        "rows whose mask label count differs from the override "
                        "length.  Default: error and pass through uncorrected.")
    p.add_argument("--log_level", default="INFO",
                   choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    if not args.in_dir.is_dir():
        log.error("in_dir not found: %s", args.in_dir)
        return 1
    if not args.corrections_csv.exists():
        log.error("corrections_csv not found: %s", args.corrections_csv)
        return 1

    corrections = load_corrections(args.corrections_csv)
    results = correct_all(args.in_dir, args.out_dir, corrections,
                          workers=args.workers,
                          allow_length_mismatch=args.allow_length_mismatch)
    manifest_path = write_veridah_manifest(results, args.out_dir,
                                           args.corrections_csv)
    log.info("Wrote %s", manifest_path)

    n_err = sum(1 for r in results if r.error)
    return 0 if n_err == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
