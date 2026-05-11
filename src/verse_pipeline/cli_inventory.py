"""Print a quick inventory of the manifest to stdout."""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="verse-inventory")
    p.add_argument("--manifest", type=Path, required=True)
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    data = json.loads(args.manifest.read_text())
    subjects: dict = data["subjects"]

    by_source: Counter[str] = Counter()
    by_split:  Counter[tuple[str, str]] = Counter()
    by_anom:   Counter[str] = Counter()
    lumbar_n:  Counter[int] = Counter()

    for rec in subjects.values():
        src = rec.get("source") or "unknown"
        spl = rec.get("split")  or "unknown"
        by_source[src] += 1
        by_split[(src, spl)] += 1
        by_anom[rec["anomaly"]["category"]] += 1
        lumbar_n[rec["centroids"]["lumbar_n"]] += 1

    print(f"\nVerSeFusion inventory — {len(subjects)} subjects total")
    print("=" * 60)

    print("\nBy source:")
    for src, n in sorted(by_source.items()):
        print(f"  {src:12s}  {n:4d}")

    print("\nBy source/split:")
    for (src, spl), n in sorted(by_split.items()):
        print(f"  {src:12s} / {spl:12s}  {n:4d}")

    print("\nBy anomaly category:")
    for cat in ("normal", "lstv", "t13", "both"):
        if cat in by_anom:
            print(f"  {cat:12s}  {by_anom[cat]:4d}")

    print("\nLumbar vertebra count distribution:")
    for k in sorted(lumbar_n):
        print(f"  L-count={k}  {lumbar_n[k]:4d}")

    print()
    return 0


if __name__ == "__main__":
    sys.exit(main())
