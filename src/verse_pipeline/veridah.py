"""Apply VERIDAH (Moeller et al. 2026) manual label corrections to VerSe.

The VerSe team's original OSF segmentation masks contain a small number of
mislabeled enumeration anomalies — predominantly T13 (extra thoracic
vertebra) cases where the supernumerary thoracic was annotated as L1 with
all subsequent lumbar labels shifted down.  The VERIDAH team
(Hendrik Moeller et al., TUM) manually reviewed VerSe and published a
table of 25 corrections; everything else in VerSe their team judged
correct.

This module:

  * Loads ``configs/veridah_corrections.csv``.
  * Builds a per-subject ``{old_label: new_label}`` remap from each row.
  * Applies the remap to a NIfTI segmentation mask via a LUT.
  * Applies the same remap to the matching centroid JSON.
  * Records advisory flags (TLTV, stump-rib left/right) for downstream use.

Two correction types are supported:

  T13-shift  (most common, 14 cases)
      Original mask labels the supernumerary thoracic vertebra as L1=20 and
      slides subsequent lumbar labels down by one.  Remap:

          20 (was L1) -> 28 (T13)
          21 (was L2) -> 20 (L1)
          22 (was L3) -> 21 (L2)
          ...
          25 (was L6) -> 24 (L5)   [if originally present]

  LabelOverride  (3 cases)
      Full sequence replacement; the OSF labels were so wrong the whole
      spine numbering was re-seeded.  The override is a Python-list
      literal of new label values in cranial -> caudal order; we pair it
      with the spatial-order list of labels from the centroid JSON to
      construct an explicit remap.

CSV schema:

    fid            "verse014" or "verse559_dir-sag"  (subject id, may include dir-* variant)
    dataset        "dataset-verse19" | "dataset-verse20"
    vert           reference label that's the subject of the correction (mostly 20 = L1)
    T11            1 if missing-thoracic anomaly (none in the published table)
    T13            1 if extra-thoracic anomaly (apply T13-shift if set)
    LabelOverride  Python-list literal of full new label sequence (or empty)
    TLTV           advisory — last thoracic is a transitional vertebra
    SR_l, SR_r     advisory — stump rib left/right

References
----------
Moeller H. et al. 2026.  VERIDAH: Solving Enumeration Anomaly Aware Vertebra
Labeling across Imaging Sequences.  arXiv:2601.14066.
"""

from __future__ import annotations

import argparse
import ast
import csv
import json
import logging
import shutil
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable

import nibabel as nib
import numpy as np

from verse_pipeline.utils.centroid_json import (
    Centroid,
    CentroidFile,
    parse_centroid_json,
    write_centroid_json,
)

log = logging.getLogger("verse.veridah")


# =============================================================================
# CSV loader
# =============================================================================

@dataclass
class Correction:
    """One row of the VERIDAH corrections table, normalized."""
    fid:            str                # raw fid from the CSV (e.g. "verse559_dir-sag")
    subject:       str                 # canonical subject id stripped of dir-* variant
    dataset:       str                 # "dataset-verse19" | "dataset-verse20"
    vert:          int
    has_t11:       bool
    has_t13:       bool
    label_override: list[int] | None   # None unless a full re-sequence was provided
    tltv:          bool
    sr_left:       bool
    sr_right:      bool

    @property
    def correction_type(self) -> str:
        if self.label_override is not None:
            return "label_override"
        if self.has_t13:
            return "t13_shift"
        if self.has_t11:
            return "t11_shift"
        return "advisory_only"


def _norm_subject(fid: str) -> str:
    """Strip any ``_dir-*`` variant suffix from the fid to get the bare subject id."""
    # CSV uses bare "verseNNN"; the unified tree uses "sub-verseNNN".  We
    # return the bare form ("verseNNN") to match what `discover_subjects`
    # returns as the subject string.
    return fid.split("_dir-")[0]


def load_corrections(csv_path: Path) -> dict[str, Correction]:
    """Read the corrections CSV; return ``{subject_id: Correction}``.

    If a subject appears with multiple ``dir-*`` rows, only the first wins
    (advisory; the spreadsheet has a couple of these).
    """
    corrections: dict[str, Correction] = {}
    with csv_path.open(newline="") as f:
        for row in csv.DictReader(f):
            fid = row["fid"].strip()
            subject = _norm_subject(fid)
            override_raw = (row.get("LabelOverride") or "").strip()
            override: list[int] | None = None
            if override_raw:
                try:
                    override = [int(x) for x in ast.literal_eval(override_raw)]
                except (ValueError, SyntaxError):
                    log.warning("Could not parse LabelOverride for %s: %r", fid, override_raw)
                    override = None

            c = Correction(
                fid=fid,
                subject=subject,
                dataset=row["dataset"].strip(),
                vert=int(row["vert"]),
                has_t11=row.get("T11", "0").strip() == "1",
                has_t13=row.get("T13", "0").strip() == "1",
                label_override=override,
                tltv=row.get("TLTV", "0").strip() == "1",
                sr_left=row.get("SR_l", "0").strip() == "1",
                sr_right=row.get("SR_r", "0").strip() == "1",
            )

            if subject in corrections:
                log.warning(
                    "Duplicate correction rows for subject %s — keeping first (%s)",
                    subject, corrections[subject].fid,
                )
                continue
            corrections[subject] = c
    log.info("Loaded %d VERIDAH corrections from %s", len(corrections), csv_path)
    return corrections


# =============================================================================
# Remap construction
# =============================================================================

def _t13_shift_remap(mask_labels: Iterable[int]) -> dict[int, int]:
    """Build the canonical T13-shift remap.

    Given the set of labels actually present in the OSF mask, build:
        original L1 (20) -> 28 (T13)
        original L>=2    -> label - 1   (L2 becomes L1, etc.)
    Labels outside the lumbar range are untouched.
    """
    remap: dict[int, int] = {}
    for lbl in mask_labels:
        if lbl == 20:
            remap[20] = 28
        elif 21 <= lbl <= 25:
            remap[lbl] = lbl - 1
        # cervical (1-7), thoracic (8-19), already-28 (T13), sacrum (26), etc:
        # left alone (no entry in the dict means "keep as-is" downstream).
    return remap


def _t11_shift_remap(mask_labels: Iterable[int]) -> dict[int, int]:
    """Symmetric remap for missing thoracic (T11) cases.

    Original mask labels the last thoracic (12 originally) as the now-missing
    one; everything below shifts up.  Listed for completeness; not used by
    any row in the current VERIDAH table.
    """
    remap: dict[int, int] = {}
    for lbl in mask_labels:
        if 20 <= lbl <= 24:
            remap[lbl] = lbl + 1
    return remap


def _label_override_remap(
    correction: Correction,
    centroid: CentroidFile,
) -> dict[int, int]:
    """Pair the override list with the centroid spatial order to build a remap.

    The centroid JSON in PIR (or any orientation; we assume the original
    OSF orientation here) lists vertebrae in cranial->caudal spatial order.
    The override list also runs cranial->caudal.  Match them index-for-index.
    """
    assert correction.label_override is not None
    ordered_old = [c.label for c in centroid.centroids]
    new_seq = correction.label_override

    if len(ordered_old) != len(new_seq):
        log.error(
            "Length mismatch for %s: centroid has %d vertebrae, "
            "LabelOverride has %d entries — using min(len)",
            correction.subject, len(ordered_old), len(new_seq),
        )

    n = min(len(ordered_old), len(new_seq))
    remap: dict[int, int] = {}
    for old, new in zip(ordered_old[:n], new_seq[:n]):
        if old != new:
            remap[old] = new
    return remap


def build_remap(
    correction: Correction,
    mask_labels: set[int],
    centroid: CentroidFile,
) -> dict[int, int]:
    """Dispatch to the right remap-builder for this correction type."""
    if correction.label_override is not None:
        return _label_override_remap(correction, centroid)
    if correction.has_t13:
        return _t13_shift_remap(mask_labels)
    if correction.has_t11:
        return _t11_shift_remap(mask_labels)
    return {}   # advisory-only row


# =============================================================================
# Apply
# =============================================================================

def apply_remap_to_mask(mask_data: np.ndarray, remap: dict[int, int]) -> np.ndarray:
    """Apply a label remap to a NIfTI mask via a LUT.  Returns a new array."""
    if not remap:
        return mask_data.copy()

    max_val = int(max(mask_data.max(), max(remap.keys()), max(remap.values())))
    lut = np.arange(max_val + 1, dtype=np.int32)
    for old, new in remap.items():
        lut[old] = new
    return lut[np.clip(mask_data, 0, max_val)].astype(np.uint8)


def apply_remap_to_centroid(
    centroid: CentroidFile,
    remap: dict[int, int],
) -> CentroidFile:
    """Apply the same remap to every centroid label in the JSON."""
    if not remap:
        return centroid
    new_centroids = tuple(
        Centroid(
            label=remap.get(c.label, c.label),
            x=c.x, y=c.y, z=c.z,
        )
        for c in centroid.centroids
    )
    return CentroidFile(
        direction=centroid.direction,
        centroids=new_centroids,
        raw_path=centroid.raw_path,
    )


# =============================================================================
# Per-subject orchestration
# =============================================================================

@dataclass
class CorrectionRecord:
    """Per-subject result row for the manifest."""
    subject:          str
    veridah_corrected: bool
    correction_type:  str
    remap:            dict[int, int] = field(default_factory=dict)
    tltv:             bool = False
    sr_left:          bool = False
    sr_right:         bool = False
    error:            str | None = None


def correct_subject(
    subj_in: Path,
    subj_out: Path,
    correction: Correction | None,
) -> CorrectionRecord:
    """Apply corrections (if any) to one subject; otherwise pass through.

    Pass-through means symlinking everything from subj_in into subj_out
    without modification.  When corrections exist we materialise corrected
    mask + centroid JSON and symlink the CT (unchanged) and snp (if any).
    """
    subject = subj_in.name.removeprefix("sub-")
    subj_out.mkdir(parents=True, exist_ok=True)

    ct = next(iter(subj_in.glob("*_ct.nii.gz")), None)
    msk = next(iter(subj_in.glob("*_msk.nii.gz")), None)
    ctd = next(iter(subj_in.glob("*_ctd.json")), None)
    snp = next(iter(subj_in.glob("*_snp.png")), None)

    def _passthrough() -> None:
        for src in (ct, msk, ctd, snp):
            if src is None:
                continue
            dst = subj_out / src.name
            if dst.exists() or dst.is_symlink():
                dst.unlink()
            dst.symlink_to(src.resolve())

    if correction is None or correction.correction_type == "advisory_only":
        _passthrough()
        return CorrectionRecord(
            subject=subject,
            veridah_corrected=False,
            correction_type=correction.correction_type if correction else "none",
            tltv=correction.tltv if correction else False,
            sr_left=correction.sr_left if correction else False,
            sr_right=correction.sr_right if correction else False,
        )

    if msk is None or ctd is None:
        return CorrectionRecord(
            subject=subject,
            veridah_corrected=False,
            correction_type=correction.correction_type,
            error="missing mask or centroid file",
        )

    # ---- build remap from actual mask labels + centroid order --------------
    mask_img = nib.load(str(msk))
    mask_data = np.asanyarray(mask_img.dataobj).astype(np.int32, copy=False)
    mask_labels = {int(v) for v in np.unique(mask_data).tolist() if v != 0}

    centroid = parse_centroid_json(ctd)
    remap = build_remap(correction, mask_labels, centroid)

    if not remap:
        # No actual remap to apply (e.g. override list matched original).
        _passthrough()
        return CorrectionRecord(
            subject=subject,
            veridah_corrected=False,
            correction_type=correction.correction_type,
            tltv=correction.tltv,
            sr_left=correction.sr_left,
            sr_right=correction.sr_right,
        )

    # ---- materialise corrected mask ---------------------------------------
    new_mask_data = apply_remap_to_mask(mask_data, remap)
    new_mask = nib.Nifti1Image(new_mask_data, mask_img.affine, header=mask_img.header)
    new_mask.set_data_dtype(np.uint8)
    out_msk = subj_out / msk.name
    if out_msk.exists() or out_msk.is_symlink():
        out_msk.unlink()
    nib.save(new_mask, str(out_msk))

    # ---- materialise corrected centroid JSON ------------------------------
    new_centroid = apply_remap_to_centroid(centroid, remap)
    out_ctd = subj_out / ctd.name
    if out_ctd.exists() or out_ctd.is_symlink():
        out_ctd.unlink()
    write_centroid_json(out_ctd, new_centroid)

    # ---- symlink the rest --------------------------------------------------
    for src in (ct, snp):
        if src is None:
            continue
        dst = subj_out / src.name
        if dst.exists() or dst.is_symlink():
            dst.unlink()
        dst.symlink_to(src.resolve())

    log.info(
        "Corrected %s  (type=%s, remap=%s)",
        subject, correction.correction_type, remap,
    )

    return CorrectionRecord(
        subject=subject,
        veridah_corrected=True,
        correction_type=correction.correction_type,
        remap=remap,
        tltv=correction.tltv,
        sr_left=correction.sr_left,
        sr_right=correction.sr_right,
    )


# =============================================================================
# CLI
# =============================================================================

DEFAULT_CSV = Path(__file__).resolve().parents[2] / "configs" / "veridah_corrections.csv"


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="verse-veridah-correct",
        description=(
            "Apply Moeller 2026 VERIDAH manual label corrections to VerSeFusion. "
            "Input is the post-unify tree (data/unified/); output is data/corrected/ "
            "with corrected mask + centroid for the ~25 affected subjects, and "
            "symlinks for everyone else."
        ),
    )
    p.add_argument("--in_dir",  type=Path, required=True, help="Post-unify tree (data/unified).")
    p.add_argument("--out_dir", type=Path, required=True, help="Where to write corrected/.")
    p.add_argument(
        "--corrections_csv",
        type=Path,
        default=DEFAULT_CSV,
        help=f"Path to corrections CSV (default: {DEFAULT_CSV}).",
    )
    p.add_argument("--log_level", default="INFO")
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    corrections = load_corrections(args.corrections_csv)
    subdirs = sorted(d for d in args.in_dir.glob("sub-*") if d.is_dir())
    log.info(
        "Processing %d subjects (corrections defined for %d)",
        len(subdirs), len(corrections),
    )

    records: list[CorrectionRecord] = []
    n_corrected = 0
    for d in subdirs:
        subject = d.name.removeprefix("sub-")
        correction = corrections.get(subject)
        rec = correct_subject(d, args.out_dir / d.name, correction)
        records.append(rec)
        if rec.veridah_corrected:
            n_corrected += 1

    # ---- manifest ----------------------------------------------------------
    manifest = {
        "version":         "0.1.0",
        "csv_source":      str(args.corrections_csv.resolve()),
        "n_subjects":      len(records),
        "n_corrected":     n_corrected,
        "n_in_csv":        len(corrections),
        "subjects": [
            {
                "subject":          r.subject,
                "veridah_corrected": r.veridah_corrected,
                "correction_type":   r.correction_type,
                "remap": {str(k): v for k, v in r.remap.items()},
                "tltv":              r.tltv,
                "sr_left":           r.sr_left,
                "sr_right":          r.sr_right,
                "error":             r.error,
            }
            for r in records
        ],
    }
    args.out_dir.mkdir(parents=True, exist_ok=True)
    out_path = args.out_dir / "veridah_manifest.json"
    out_path.write_text(json.dumps(manifest, indent=2))
    log.info("Wrote %s  (%d/%d subjects corrected)", out_path, n_corrected, len(records))

    # ---- audit: were any CSV subjects not found in the input tree? --------
    input_subjects = {r.subject for r in records}
    missing = [s for s in corrections if s not in input_subjects]
    if missing:
        log.warning(
            "%d corrections in CSV had no matching subject in --in_dir: %s",
            len(missing), missing,
        )

    return 0


if __name__ == "__main__":
    sys.exit(main())
