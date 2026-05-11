"""Detect enumeration anomalies (LSTV, T13) from the VerSe centroid JSON.

VerSe labels enumeration anomalies *explicitly* in the segmentation mask
itself, using two specific label values:

    25 → L6 (a lumbarized lumbosacral transitional vertebra)
    28 → T13 (a supernumerary thoracic vertebra)

so detection reduces to membership tests over the centroid label set.  No
geometric / morphometric analysis is needed (unlike SPINEPS/VERIDAH-based
detection on datasets that *don't* carry these labels).
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

from verse_pipeline.utils.bids import discover_subjects
from verse_pipeline.utils.centroid_json import CentroidFile, parse_centroid_json

LSTV_LABEL = 25
T13_LABEL = 28

log = logging.getLogger("verse.lstv")


# =============================================================================
# detection
# =============================================================================

@dataclass(frozen=True)
class AnomalyFlags:
    """Per-subject enumeration-anomaly flags."""
    has_lstv:    bool
    has_t13:     bool
    lumbar_n:    int   # count of labels 20-25
    thoracic_n:  int   # count of labels 8-19 + 28
    cervical_n:  int   # count of labels 1-7
    is_normal:   bool  # neither LSTV nor T13

    @property
    def category(self) -> str:
        """One of 'normal' | 'lstv' | 't13' | 'both'."""
        if self.has_lstv and self.has_t13:
            return "both"
        if self.has_lstv:
            return "lstv"
        if self.has_t13:
            return "t13"
        return "normal"


def flags_from_centroid(file: CentroidFile) -> AnomalyFlags:
    labels = set(file.labels)
    has_lstv = LSTV_LABEL in labels
    has_t13 = T13_LABEL in labels

    cerv = sum(1 for c in file.centroids if 1 <= c.label <= 7)
    thor = sum(1 for c in file.centroids if 8 <= c.label <= 19 or c.label == T13_LABEL)
    lumb = sum(1 for c in file.centroids if 20 <= c.label <= 25)

    return AnomalyFlags(
        has_lstv=has_lstv,
        has_t13=has_t13,
        lumbar_n=lumb,
        thoracic_n=thor,
        cervical_n=cerv,
        is_normal=not (has_lstv or has_t13),
    )


# =============================================================================
# CLI: audit
# =============================================================================

def _audit_from_manifest(manifest_path: Path) -> dict[str, int]:
    """Count LSTV/T13/normal subjects from a pre-built placed_manifest.json."""
    data = json.loads(manifest_path.read_text())
    cats: Counter[str] = Counter()
    for sub in data["subjects"].values():
        cats[sub["anomaly"]["category"]] += 1
    return dict(cats)


def _audit_from_dir(unified_dir: Path) -> dict[str, int]:
    """Live-count by walking subjects + reading each centroid JSON."""
    subjects = discover_subjects(unified_dir)
    if not subjects:
        # Fall back to flat layout (post-unify): sub-XXX directories at top level.
        cats: Counter[str] = Counter()
        for subdir in sorted(unified_dir.glob("sub-*")):
            ctd_files = list(subdir.glob("*_ctd.json"))
            if not ctd_files:
                cats["missing_ctd"] += 1
                continue
            file = parse_centroid_json(ctd_files[0])
            cats[flags_from_centroid(file).category] += 1
        return dict(cats)

    cats = Counter()
    for sub, files in subjects.items():
        if files.ctd is None:
            cats["missing_ctd"] += 1
            continue
        file = parse_centroid_json(files.ctd)
        cats[flags_from_centroid(file).category] += 1
    return dict(cats)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="verse-lstv",
        description="Audit / detect LSTV (label 25) and T13 (label 28) anomalies.",
    )
    p.add_argument("--audit", action="store_true", help="Print anomaly counts and exit.")
    p.add_argument("--manifest", type=Path, help="Read counts from placed_manifest.json.")
    p.add_argument("--in_dir",   type=Path, help="Walk a unified/reoriented dir directly.")
    p.add_argument("--log_level", default="INFO")
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    if not args.audit:
        log.error("Currently only --audit is implemented.")
        return 2

    if args.manifest:
        counts = _audit_from_manifest(args.manifest)
    elif args.in_dir:
        counts = _audit_from_dir(args.in_dir)
    else:
        log.error("Pass --manifest <path> or --in_dir <path>.")
        return 2

    total = sum(counts.values())
    print(f"\nVerSeFusion anomaly audit  (n={total})")
    print("-" * 40)
    for cat in ("normal", "lstv", "t13", "both", "missing_ctd"):
        if cat in counts:
            print(f"  {cat:14s}  {counts[cat]:4d}  ({100 * counts[cat] / max(total,1):5.1f}%)")
    print()
    return 0


if __name__ == "__main__":
    sys.exit(main())
