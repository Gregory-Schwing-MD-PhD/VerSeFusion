"""Export VerSeFusion to a HuggingFace-compatible flat directory.

The export mirrors the CTSpinoPelvic1K HF layout so the same downstream
trainers can be pointed at either dataset::

    <out_dir>/
      ct/<sub-verseNNN>.nii.gz
      labels/<sub-verseNNN>.nii.gz
      centroids/<sub-verseNNN>.json
      placed_manifest.json            (copied from the reoriented tree)
      splits/cv_5fold.json
      splits/test.json
      dataset_card.md
      LICENSE.txt

Files are staged via copy (default) or symlink so the export dir is
self-contained and ready to ``huggingface-cli upload`` or load via
``datasets.load_dataset("imagefolder", data_dir=...)`` with a custom loader.
"""

from __future__ import annotations

import argparse
import json
import logging
import shutil
import sys
from pathlib import Path
from typing import Literal

log = logging.getLogger("verse.hf_export")


# =============================================================================
# staging
# =============================================================================

def stage_file(src: Path, dst: Path, mode: Literal["copy", "symlink"]) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists() or dst.is_symlink():
        dst.unlink()
    if mode == "symlink":
        dst.symlink_to(src.resolve())
    else:
        shutil.copy2(src, dst)


def stage_all(
    in_dir: Path,
    out_dir: Path,
    *,
    mode: Literal["copy", "symlink"] = "copy",
) -> int:
    subdirs = sorted(d for d in in_dir.glob("sub-*") if d.is_dir())
    for d in subdirs:
        sub = d.name  # "sub-verseNNN"
        ct = next(iter(d.glob("*_ct.nii.gz")), None)
        msk = next(iter(d.glob("*_msk.nii.gz")), None)
        ctd = next(iter(d.glob("*_ctd.json")), None)
        if ct is None or msk is None or ctd is None:
            log.warning("Skipping %s — missing one of ct/msk/ctd", sub)
            continue
        stage_file(ct,  out_dir / "ct"       / f"{sub}.nii.gz", mode)
        stage_file(msk, out_dir / "labels"    / f"{sub}.nii.gz", mode)
        stage_file(ctd, out_dir / "centroids" / f"{sub}.json",   mode)
    return len(subdirs)


# =============================================================================
# dataset card
# =============================================================================

DATASET_CARD = """---
license: cc-by-sa-4.0
language:
  - en
pretty_name: VerSeFusion (VerSe 2019 + 2020, dedup'd)
size_categories:
  - n<1K
task_categories:
  - image-segmentation
tags:
  - medical-imaging
  - ct
  - spine
  - vertebrae
  - segmentation
---

# VerSeFusion

A reproducible unification of the **VerSe 2019** and **VerSe 2020** CT
vertebrae segmentation datasets into a single subject-keyed corpus with:

* The 105-image subject overlap deduplicated (VerSe20 wins by default).
* All CTs, masks, and centroid annotations reoriented to **PIR**.
* Per-subject metadata including explicit enumeration-anomaly flags
  (LSTV, T13) derived from the canonical VerSe label scheme.
* 5-fold cross-validation splits stratified on anomaly category.

## Layout

| path             | type                                              |
|------------------|---------------------------------------------------|
| `ct/`            | CT volumes, NIfTI `<sub-verseNNN>.nii.gz`         |
| `labels/`        | Vertebra masks, NIfTI `<sub-verseNNN>.nii.gz`     |
| `centroids/`     | Centroid JSON, voxel coordinates in PIR space      |
| `placed_manifest.json` | per-subject metadata                         |
| `splits/`        | `cv_5fold.json`, `test.json`                       |

## Label scheme (28-class)

| value  | structure                                |
|-------:|------------------------------------------|
| 0      | background                               |
| 1–7    | C1–C7                                    |
| 8–19   | T1–T12                                   |
| 20–24  | L1–L5                                    |
| 25     | L6 (lumbarized LSTV)                     |
| 26     | sacrum (defined in scheme, unannotated)  |
| 27     | coccyx (defined in scheme, unannotated)  |
| 28     | T13 (extra thoracic, cranial anomaly)    |

## Citation

If you use VerSeFusion, please cite the original VerSe papers:

  * Sekuboyina A. et al., *VerSe: A Vertebrae Labelling and Segmentation
    Benchmark for Multi-detector CT Images.*  Medical Image Analysis (2021).
  * Löffler M. et al., *A Vertebral Segmentation Dataset with Fracture Grading.*
    Radiology AI (2020).
  * Liebl H. and Schinz D. et al., *A Computed Tomography Vertebral Segmentation
    Dataset with Anatomical Variations and Multi-Vendor Scanner Data.* (2021).

## License

The underlying VerSe data is licensed under **CC-BY-SA 4.0**.  This export
inherits the same license and must be redistributed under the same terms with
appropriate attribution.
"""


# =============================================================================
# CLI
# =============================================================================

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="verse-hf-export",
        description="Build a HuggingFace-compatible flat directory under data/hf_export/.",
    )
    p.add_argument("--in_dir",   type=Path, required=True, help="Reoriented tree (data/reoriented).")
    p.add_argument("--out_dir",  type=Path, required=True, help="Output flat directory.")
    p.add_argument("--mode",     choices=["copy", "symlink"], default="copy")
    p.add_argument("--manifest", type=Path, default=None, help="placed_manifest.json (default sibling of --in_dir).")
    p.add_argument("--splits",   type=Path, default=None, help="splits/ directory.")
    p.add_argument("--log_level", default="INFO")
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    args.out_dir.mkdir(parents=True, exist_ok=True)
    n = stage_all(args.in_dir, args.out_dir, mode=args.mode)
    log.info("Staged %d subjects to %s", n, args.out_dir)

    # carry manifest
    manifest_path = args.manifest or (args.in_dir / "placed_manifest.json")
    if manifest_path.is_file():
        shutil.copy2(manifest_path, args.out_dir / "placed_manifest.json")
        log.info("Copied %s", manifest_path)

    # carry splits
    splits_root = args.splits or (args.in_dir / "splits")
    if splits_root.is_dir():
        dst_splits = args.out_dir / "splits"
        dst_splits.mkdir(exist_ok=True)
        for f in splits_root.glob("*.json"):
            shutil.copy2(f, dst_splits / f.name)
        log.info("Copied splits from %s", splits_root)

    # dataset card + license
    (args.out_dir / "dataset_card.md").write_text(DATASET_CARD)
    (args.out_dir / "LICENSE.txt").write_text(
        "VerSeFusion derivative data is licensed under CC-BY-SA 4.0 inherited "
        "from upstream VerSe.  See dataset_card.md for citations.\n"
    )
    log.info("Wrote dataset_card.md and LICENSE.txt")
    return 0


if __name__ == "__main__":
    sys.exit(main())
