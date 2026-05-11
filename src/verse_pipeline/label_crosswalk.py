"""VerSe ↔ CTSpinoPelvic1K label crosswalk.

The crosswalk is loaded from ``configs/label_scheme.yaml``.  Two modes are
exposed:

  * ``forward`` — VerSe → CTSpinoPelvic1K 10-class (drops cervical/thoracic).
  * ``reverse`` — CTSpinoPelvic1K → VerSe (loses hip labels; rarely useful).

Forward crosswalk is the relevant direction for **external validation of
CTSpinoPelvic1K-trained models on VerSeFusion**: the model predicts in the
CTSpinoPelvic1K scheme, ground-truth VerSe masks must be remapped into the
same scheme before DSC / surface-distance metrics are computed.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import nibabel as nib
import numpy as np
import yaml

log = logging.getLogger("verse.crosswalk")

CONFIG_DEFAULT = Path(__file__).resolve().parents[2] / "configs" / "label_scheme.yaml"


# =============================================================================
# load
# =============================================================================

def load_crosswalk(config_path: Path | None = None) -> tuple[dict[int, int], dict[int, int]]:
    """Return (forward, reverse) label-mapping dicts."""
    cfg = yaml.safe_load((config_path or CONFIG_DEFAULT).read_text())
    fwd = {int(k): int(v) for k, v in cfg["crosswalk"]["verse_to_ctspinopelvic1k"].items()}
    rev = {int(k): int(v) for k, v in cfg["crosswalk"]["ctspinopelvic1k_to_verse"].items()}
    return fwd, rev


# =============================================================================
# apply
# =============================================================================

def apply_mapping(mask: np.ndarray, mapping: dict[int, int], *, default: int = 0) -> np.ndarray:
    """Apply an int→int mapping to a label volume.

    Labels not present in ``mapping`` are remapped to ``default`` (0 by default,
    i.e. background — used for VerSe cervical/thoracic which have no
    CTSpinoPelvic1K equivalent).
    """
    src_max = int(mask.max())
    lut = np.full(src_max + 1, default, dtype=np.int32)
    for k, v in mapping.items():
        if 0 <= k <= src_max:
            lut[k] = v
    return lut[np.clip(mask, 0, src_max)].astype(np.uint8)


def crosswalk_file(
    in_path: Path,
    out_path: Path,
    mapping: dict[int, int],
) -> dict[str, list[int]]:
    """Run the crosswalk on one NIfTI mask and write the result.

    Returns the unique-label histogram before/after, for QA logging.
    """
    img = nib.load(str(in_path))
    src = np.asanyarray(img.dataobj).astype(np.int32, copy=False)
    dst = apply_mapping(src, mapping)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    new_img = nib.Nifti1Image(dst, img.affine, header=img.header)
    new_img.set_data_dtype(np.uint8)
    nib.save(new_img, str(out_path))

    return {
        "labels_in":  sorted(int(x) for x in np.unique(src).tolist()),
        "labels_out": sorted(int(x) for x in np.unique(dst).tolist()),
    }


# =============================================================================
# CLI
# =============================================================================

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="verse-crosswalk",
        description="Remap VerSe vertebra labels into the CTSpinoPelvic1K 10-class scheme (or reverse).",
    )
    p.add_argument("--in_dir",  type=Path, required=True, help="Directory of sub-* subdirs.")
    p.add_argument("--out_dir", type=Path, required=True)
    p.add_argument(
        "--direction",
        choices=["forward", "reverse"],
        default="forward",
        help="forward = VerSe → CTSpinoPelvic1K (default).",
    )
    p.add_argument("--config", type=Path, default=None)
    p.add_argument("--log_level", default="INFO")
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    fwd, rev = load_crosswalk(args.config)
    mapping = fwd if args.direction == "forward" else rev
    log.info("Crosswalk direction: %s  (%d entries)", args.direction, len(mapping))

    subdirs = sorted(d for d in args.in_dir.glob("sub-*") if d.is_dir())
    records: list[dict] = []
    for d in subdirs:
        src = next(iter(d.glob("*_msk.nii.gz")), None)
        if src is None:
            log.warning("No mask in %s — skipping", d)
            continue
        dst = args.out_dir / d.name / src.name
        hist = crosswalk_file(src, dst, mapping)
        records.append({"subject": d.name, **hist, "dst": str(dst)})

    args.out_dir.mkdir(parents=True, exist_ok=True)
    (args.out_dir / "crosswalk_manifest.json").write_text(json.dumps({
        "direction": args.direction,
        "mapping":   {str(k): v for k, v in mapping.items()},
        "n_subjects": len(records),
        "subjects":   records,
    }, indent=2))
    log.info("Wrote %s  (%d subjects)", args.out_dir / "crosswalk_manifest.json", len(records))
    return 0


if __name__ == "__main__":
    sys.exit(main())
