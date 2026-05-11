"""5-fold stratified CV splits for VerSeFusion.

Subjects are stratified by **anomaly category** (``normal | lstv | t13 | both``)
so every fold sees representative examples of each enumeration anomaly.

Output schema (``splits/cv_5fold.json``) is intentionally identical in spirit
to CTSpinoPelvic1K's, so the same downstream training scripts can consume
both datasets::

    {
      "version": "v1",
      "n_folds": 5,
      "seed":    20260511,
      "stratify_axes": ["anomaly.category"],
      "folds": [
        {
          "fold": 0,
          "train": ["sub-verse001", ...],
          "val":   ["sub-verse014", ...]
        },
        ...
      ]
    }

For a true held-out test set, copy a subset of subjects out *before*
splitting (e.g. all VerSe20 test-split subjects) and pass them via
``--holdout``; those are written to ``splits/test.json``.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from collections import Counter
from pathlib import Path

import numpy as np
from sklearn.model_selection import StratifiedKFold

log = logging.getLogger("verse.splits")


# =============================================================================
# helpers
# =============================================================================

def load_manifest(path: Path) -> dict:
    return json.loads(path.read_text())


def select_holdout(manifest: dict, mode: str) -> set[str]:
    """Return a set of sub-* ids to hold out from CV.

    Modes:
      * ``none``           — no holdout, all subjects enter the CV pool.
      * ``verse20_test``   — every subject whose source/split is verse20/test.
      * a comma-separated list of ``sub-*`` ids.
    """
    if mode == "none":
        return set()
    if mode == "verse20_test":
        return {
            sub_id for sub_id, rec in manifest["subjects"].items()
            if rec.get("source") == "verse20" and rec.get("split") == "test"
        }
    if "," in mode or mode.startswith("sub-"):
        return {s.strip() for s in mode.split(",") if s.strip()}
    raise SystemExit(f"Unknown --holdout mode: {mode!r}")


# =============================================================================
# split
# =============================================================================

def build_splits(
    manifest: dict,
    n_folds: int,
    seed: int,
    holdout: set[str],
) -> dict:
    pool = [
        (sub_id, rec["anomaly"]["category"])
        for sub_id, rec in manifest["subjects"].items()
        if sub_id not in holdout
    ]
    pool.sort(key=lambda kv: kv[0])  # deterministic order

    ids = np.array([sub_id for sub_id, _ in pool])
    strata = np.array([cat for _, cat in pool])

    log.info("CV pool: %d subjects  |  holdout: %d", len(ids), len(holdout))
    log.info("Stratum distribution:  %s", dict(Counter(strata.tolist())))

    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)
    folds = []
    for fold_i, (train_idx, val_idx) in enumerate(skf.split(ids, strata)):
        folds.append({
            "fold":  fold_i,
            "train": ids[train_idx].tolist(),
            "val":   ids[val_idx].tolist(),
        })

    return {
        "version":        "v1",
        "n_folds":        n_folds,
        "seed":           seed,
        "stratify_axes":  ["anomaly.category"],
        "folds":          folds,
    }


# =============================================================================
# CLI
# =============================================================================

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="verse-splits",
        description="5-fold stratified CV splits over a VerSeFusion manifest.",
    )
    p.add_argument("--manifest", type=Path, required=True)
    p.add_argument("--out_dir",  type=Path, required=True)
    p.add_argument("--n_folds",  type=int, default=5)
    p.add_argument("--seed",     type=int, default=20260511)
    p.add_argument(
        "--holdout",
        default="verse20_test",
        help="Holdout selection: 'none' | 'verse20_test' | 'sub-A,sub-B,...'.",
    )
    p.add_argument("--log_level", default="INFO")
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    manifest = load_manifest(args.manifest)
    holdout = select_holdout(manifest, args.holdout)

    splits = build_splits(manifest, args.n_folds, args.seed, holdout)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    (args.out_dir / "cv_5fold.json").write_text(json.dumps(splits, indent=2))
    (args.out_dir / "test.json").write_text(json.dumps({
        "version": "v1",
        "subjects": sorted(holdout),
        "n":       len(holdout),
        "source":  args.holdout,
    }, indent=2))

    log.info("Wrote %s and %s", args.out_dir / "cv_5fold.json", args.out_dir / "test.json")
    return 0


if __name__ == "__main__":
    sys.exit(main())
