"""Build the per-subject ``placed_manifest.json`` for VerSeFusion.

Per-subject record::

    {
      "sub-verse014": {
        "subject":  "verse014",
        "source":   "verse20",                     # release the files came from
        "split":    "training",                    # within that release
        "paths": {
            "ct":  "data/reoriented/sub-verse014/sub-verse014_ct.nii.gz",
            "msk": "data/reoriented/sub-verse014/sub-verse014_msk.nii.gz",
            "ctd": "data/reoriented/sub-verse014/sub-verse014_ctd.json"
        },
        "image": {
            "shape":   [512, 512, 200],
            "spacing": [0.8, 0.8, 1.0],
            "axcodes": "PIR"
        },
        "centroids": {
            "n":       24,
            "labels":  [1, 2, ..., 25],
            "lumbar_n":   6,
            "thoracic_n": 12,
            "cervical_n": 6
        },
        "anomaly": {
            "has_lstv":  true,
            "has_t13":   false,
            "category":  "lstv"
        }
      },
      ...
    }

The manifest is written next to the data tree (default
``data/reoriented/placed_manifest.json``) and is the single source of truth
for the splits / hf_export / nnunet_wandb_variant stages.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Any

import nibabel as nib

from verse_pipeline.lstv import flags_from_centroid
from verse_pipeline.utils.centroid_json import parse_centroid_json
from verse_pipeline.utils.nifti import current_axcodes

log = logging.getLogger("verse.manifest")


# =============================================================================
# per-subject inspection
# =============================================================================

def _read_image_meta(ct_path: Path) -> dict[str, Any]:
    img = nib.load(str(ct_path))
    spacing = tuple(float(s) for s in img.header.get_zooms()[:3])
    return {
        "shape":   list(int(s) for s in img.shape[:3]),
        "spacing": list(spacing),
        "axcodes": "".join(current_axcodes(img)),
    }


def _read_unify_record(unify_manifest: Path, subject: str) -> dict[str, Any] | None:
    """Look up provenance for one subject from unify_manifest.json (best-effort)."""
    if not unify_manifest.is_file():
        return None
    try:
        records = json.loads(unify_manifest.read_text()).get("subjects", [])
    except json.JSONDecodeError:
        return None
    for rec in records:
        if rec.get("subject") == subject:
            return rec
    return None


def build_subject_record(
    subj_dir: Path,
    *,
    unify_manifest: Path | None,
) -> dict[str, Any] | None:
    """Build one manifest record from a sub-verseNNN/ directory."""
    subject = subj_dir.name.removeprefix("sub-")
    ct = next(iter(subj_dir.glob("*_ct.nii.gz")), None)
    msk = next(iter(subj_dir.glob("*_msk.nii.gz")), None)
    ctd = next(iter(subj_dir.glob("*_ctd.json")), None)

    if ct is None or msk is None or ctd is None:
        log.warning("Skipping %s — missing one of ct/msk/ctd", subj_dir)
        return None

    centroid = parse_centroid_json(ctd)
    flags = flags_from_centroid(centroid)
    image_meta = _read_image_meta(ct)

    src_meta = _read_unify_record(unify_manifest, subject) if unify_manifest else None

    return {
        "subject":  subject,
        "source":   src_meta["chosen_release"] if src_meta else None,
        "split":    src_meta["chosen_split"]   if src_meta else None,
        "paths": {
            "ct":  str(ct),
            "msk": str(msk),
            "ctd": str(ctd),
        },
        "image": image_meta,
        "centroids": {
            "n":         centroid.n_vertebrae,
            "labels":    centroid.labels,
            "lumbar_n":  flags.lumbar_n,
            "thoracic_n": flags.thoracic_n,
            "cervical_n": flags.cervical_n,
        },
        "anomaly": {
            "has_lstv": flags.has_lstv,
            "has_t13":  flags.has_t13,
            "category": flags.category,
        },
    }


# =============================================================================
# CLI
# =============================================================================

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="verse-manifest",
        description="Build placed_manifest.json over a (unified or reoriented) subject tree.",
    )
    p.add_argument("--in_dir",   type=Path, required=True, help="Directory of sub-* subdirs.")
    p.add_argument("--out_path", type=Path, required=True, help="Output JSON path.")
    p.add_argument(
        "--unify_manifest",
        type=Path,
        default=None,
        help="Optional unify_manifest.json to enrich each record with source/split.",
    )
    p.add_argument("--log_level", default="INFO")
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    # Default to a sibling unify_manifest.json if not supplied.
    unify_manifest = args.unify_manifest or (args.in_dir.parent / "unified" / "unify_manifest.json")

    subdirs = sorted(d for d in args.in_dir.glob("sub-*") if d.is_dir())
    log.info("Building manifest over %d subject(s) from %s", len(subdirs), args.in_dir)

    records: dict[str, Any] = {}
    for d in subdirs:
        rec = build_subject_record(d, unify_manifest=unify_manifest)
        if rec is None:
            continue
        records[f"sub-{rec['subject']}"] = rec

    payload = {
        "version":     "0.1.0",
        "in_dir":      str(args.in_dir.resolve()),
        "n_subjects":  len(records),
        "subjects":    records,
    }
    args.out_path.parent.mkdir(parents=True, exist_ok=True)
    args.out_path.write_text(json.dumps(payload, indent=2))
    log.info("Wrote %s  (%d subjects)", args.out_path, len(records))
    return 0


if __name__ == "__main__":
    sys.exit(main())
