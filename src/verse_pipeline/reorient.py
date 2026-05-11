"""Reorient every CT, mask, and centroid set to **PIR**.

This matches the CTSpinoPelvic1K convention so that:

  * any model trained on CTSpinoPelvic1K can be evaluated on VerSeFusion
    without an inference-time reorientation step;
  * VerSeFusion can be used as a pretraining corpus and downstream
    fine-tuning on CTSpinoPelvic1K does not need to switch frames.

For every subject in ``--in_dir``:

  1. load the CT, compute the reorientation plan, save the PIR CT;
  2. apply the *same plan* to the mask, save the PIR mask (uint8);
  3. apply the *same plan* to every centroid voxel coordinate, save the
     PIR centroid JSON with ``direction = "PIR"``.

Inputs are taken from the flat layout produced by ``verse-unify``::

    <in_dir>/sub-verseNNN/
        sub-verseNNN_ct.nii.gz
        sub-verseNNN_msk.nii.gz
        sub-verseNNN_ctd.json

Outputs go to the same flat layout under ``--out_dir``.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from dataclasses import dataclass
from pathlib import Path

import nibabel as nib

from verse_pipeline.utils.centroid_json import (
    Centroid,
    CentroidFile,
    parse_centroid_json,
    write_centroid_json,
)
from verse_pipeline.utils.nifti import (
    TARGET_AXCODES_PIR,
    build_reorient_plan,
    current_axcodes,
    reorient_centroid_voxel,
    reorient_image,
)

log = logging.getLogger("verse.reorient")


# =============================================================================
# per-subject reorient
# =============================================================================

@dataclass
class ReorientResult:
    subject:        str
    src_axcodes:    str
    tgt_axcodes:    str
    skipped:        bool       # True if already in PIR
    error:          str | None = None


def reorient_subject(
    subj_in: Path,
    subj_out: Path,
) -> ReorientResult:
    """Reorient one subject in-place under ``subj_out``."""
    subject = subj_in.name.removeprefix("sub-")
    subj_out.mkdir(parents=True, exist_ok=True)

    ct_path = next(iter(subj_in.glob("*_ct.nii.gz")), None)
    msk_path = next(iter(subj_in.glob("*_msk.nii.gz")), None)
    ctd_path = next(iter(subj_in.glob("*_ctd.json")), None)
    snp_path = next(iter(subj_in.glob("*_snp.png")), None)

    if ct_path is None or msk_path is None or ctd_path is None:
        return ReorientResult(
            subject=subject,
            src_axcodes="?",
            tgt_axcodes="".join(TARGET_AXCODES_PIR),
            skipped=False,
            error="missing one of ct/msk/ctd",
        )

    # ---- compute the plan from the CT (mask must share the affine) ----------
    ct_img = nib.load(str(ct_path))
    src_axcodes = current_axcodes(ct_img)
    src_str = "".join(src_axcodes)
    tgt_str = "".join(TARGET_AXCODES_PIR)

    if src_axcodes == TARGET_AXCODES_PIR:
        # Already PIR — symlink / copy through.
        log.info("[%s] already PIR — passing through", subject)
        for src in (ct_path, msk_path, ctd_path) + ((snp_path,) if snp_path else ()):
            dst = subj_out / src.name.replace(f"sub-{subject}_", f"sub-{subject}_")
            if dst.exists() or dst.is_symlink():
                dst.unlink()
            dst.symlink_to(src.resolve())
        return ReorientResult(subject=subject, src_axcodes=src_str, tgt_axcodes=tgt_str, skipped=True)

    plan = build_reorient_plan(ct_img, TARGET_AXCODES_PIR)

    # ---- CT -----------------------------------------------------------------
    new_ct, _ = reorient_image(ct_img, TARGET_AXCODES_PIR)
    out_ct = subj_out / f"sub-{subject}_ct.nii.gz"
    nib.save(new_ct, str(out_ct))

    # ---- mask ---------------------------------------------------------------
    msk_img = nib.load(str(msk_path))
    new_msk, _ = reorient_image(msk_img, TARGET_AXCODES_PIR)
    # preserve uint8 / int16 dtype rather than letting it widen to float
    msk_data = new_msk.get_fdata().astype(msk_img.get_data_dtype(), copy=False)
    new_msk = nib.Nifti1Image(msk_data, new_msk.affine, header=new_msk.header)
    out_msk = subj_out / f"sub-{subject}_msk.nii.gz"
    nib.save(new_msk, str(out_msk))

    # ---- centroids ----------------------------------------------------------
    ctd = parse_centroid_json(ctd_path)
    new_centroids = tuple(
        Centroid(
            label=c.label,
            **dict(zip(("x", "y", "z"), reorient_centroid_voxel((c.x, c.y, c.z), plan))),
        )
        for c in ctd.centroids
    )
    out_ctd = subj_out / f"sub-{subject}_ctd.json"
    write_centroid_json(out_ctd, CentroidFile(direction=tgt_str, centroids=new_centroids))

    # ---- preview PNG passes through unchanged ------------------------------
    if snp_path is not None:
        out_snp = subj_out / snp_path.name
        if out_snp.exists() or out_snp.is_symlink():
            out_snp.unlink()
        out_snp.symlink_to(snp_path.resolve())

    return ReorientResult(subject=subject, src_axcodes=src_str, tgt_axcodes=tgt_str, skipped=False)


# =============================================================================
# CLI
# =============================================================================

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="verse-reorient",
        description="Reorient every CT/mask/centroid set to PIR.",
    )
    p.add_argument("--in_dir",  type=Path, required=True)
    p.add_argument("--out_dir", type=Path, required=True)
    p.add_argument("--log_level", default="INFO")
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    subdirs = sorted(d for d in args.in_dir.glob("sub-*") if d.is_dir())
    log.info("Reorienting %d subject(s) → %s", len(subdirs), args.out_dir)

    results: list[dict] = []
    for d in subdirs:
        res = reorient_subject(d, args.out_dir / d.name)
        results.append({
            "subject":     res.subject,
            "src_axcodes": res.src_axcodes,
            "tgt_axcodes": res.tgt_axcodes,
            "skipped":     res.skipped,
            "error":       res.error,
        })
        tag = "(passthrough)" if res.skipped else ""
        suffix = f"  ERROR: {res.error}" if res.error else ""
        log.info("[%s] %s -> %s %s%s", res.subject, res.src_axcodes, res.tgt_axcodes, tag, suffix)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    (args.out_dir / "reorient_manifest.json").write_text(json.dumps({
        "n_subjects":   len(results),
        "target":       "".join(TARGET_AXCODES_PIR),
        "n_passthrough": sum(1 for r in results if r["skipped"]),
        "n_errors":     sum(1 for r in results if r["error"]),
        "subjects":     results,
    }, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
