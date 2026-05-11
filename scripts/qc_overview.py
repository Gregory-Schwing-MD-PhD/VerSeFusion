#!/usr/bin/env python3
"""Regenerate VerSe-style preview PNGs (`*_snp.png`) for QA.

VerSe ships an overview PNG alongside each subject — a mid-sagittal CT slice
with overlaid coloured vertebra masks and centroid markers — that the
challenge organisers used as a visual QA aid.  After reorientation /
crosswalk the original PNG no longer matches the data, so this script
regenerates a fresh one per subject.

Usage:
    python scripts/qc_overview.py \\
        --in_dir  data/reoriented \\
        --out_dir data/reoriented \\
        --suffix _snp_pir.png

Per subject, writes ``<out_dir>/sub-verseNNN/sub-verseNNN<suffix>``.
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt   # noqa: E402
import nibabel as nib              # noqa: E402
import numpy as np                 # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from verse_pipeline.utils.centroid_json import parse_centroid_json  # noqa: E402

log = logging.getLogger("verse.qc")


# =============================================================================
# rendering
# =============================================================================

def _mid_sagittal_slice(vol: np.ndarray) -> np.ndarray:
    """Return the middle slice along the RIGHT axis of a PIR volume.

    For axcodes (P, I, R) the third axis is left-right, so the mid-sagittal
    slice is at ``vol.shape[2] // 2``.  Output shape is (P, I).
    """
    s = vol.shape[2] // 2
    return vol[:, :, s]


def _render_subject(
    ct_path: Path,
    msk_path: Path,
    ctd_path: Path,
    out_path: Path,
) -> None:
    ct = nib.load(str(ct_path)).get_fdata()
    msk = nib.load(str(msk_path)).get_fdata()
    centroid = parse_centroid_json(ctd_path)

    ct_slice = _mid_sagittal_slice(ct)
    msk_slice = _mid_sagittal_slice(msk)

    fig, ax = plt.subplots(figsize=(6, 9), dpi=120)
    ax.imshow(ct_slice.T, cmap="gray", origin="lower", aspect="auto")
    masked = np.ma.masked_where(msk_slice == 0, msk_slice)
    ax.imshow(masked.T, cmap="tab20", alpha=0.45, origin="lower", aspect="auto")

    # Plot centroids whose right-axis (z) coord is near the mid slice.
    mid_z = ct.shape[2] // 2
    band = max(8, ct.shape[2] // 40)
    for c in centroid.centroids:
        if abs(c.z - mid_z) <= band:
            ax.plot(c.x, c.y, "o", markersize=4, markeredgecolor="white", markerfacecolor="red")
            ax.text(c.x + 3, c.y, str(c.label), color="white", fontsize=7)

    ax.set_axis_off()
    ax.set_title(out_path.stem, fontsize=9)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, bbox_inches="tight", pad_inches=0.05)
    plt.close(fig)


# =============================================================================
# CLI
# =============================================================================

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="qc-overview")
    p.add_argument("--in_dir",  type=Path, required=True)
    p.add_argument("--out_dir", type=Path, required=True)
    p.add_argument("--suffix",  default="_snp_pir.png",
                   help="Output filename suffix (default _snp_pir.png).")
    p.add_argument("--log_level", default="INFO")
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    subdirs = sorted(d for d in args.in_dir.glob("sub-*") if d.is_dir())
    log.info("QC overview for %d subject(s)", len(subdirs))

    for d in subdirs:
        sub = d.name
        ct  = next(iter(d.glob("*_ct.nii.gz")), None)
        msk = next(iter(d.glob("*_msk.nii.gz")), None)
        ctd = next(iter(d.glob("*_ctd.json")), None)
        if ct is None or msk is None or ctd is None:
            log.warning("Skip %s — missing one of ct/msk/ctd", sub)
            continue
        out = args.out_dir / sub / f"{sub}{args.suffix}"
        try:
            _render_subject(ct, msk, ctd, out)
            log.info("Wrote %s", out)
        except Exception as e:        # rendering errors should not abort the batch
            log.error("Failed %s: %s", sub, e)
    return 0


if __name__ == "__main__":
    sys.exit(main())
