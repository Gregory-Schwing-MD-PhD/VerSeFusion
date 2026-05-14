"""
verse_pipeline.splits_builder — patient-level 5-fold CV with LSTV stratification.

Consumes data/manifest/manifest.csv and emits data/manifest/splits_5fold.json.

By default, ALL 374 scans go through 5-fold CV (VerSe's original
training/validation/test column is ignored — those were challenge
splits, not ML splits).  Use one of the folds as your test set for
final reporting, or all folds for CV averaging.  Override with
`--use_verse_test` if you specifically want to preserve VerSe's
test_set as held-out.

SCHEMA (version 2, May 2026)
============================
Patient-level splits.  Each patient_id is assigned to exactly one fold;
all scans belonging to that patient go together (no patient leakage
across train/val).  For paired patients (~19 cases in VerSeFusion), this
means both scans live in the same fold.

LSTV TAXONOMY
=============
4-way `lstv_class` derived from the LSTV audit (resolved upstream by
verse_pipeline.manifest_builder using the audit's own categoricals,
not boolean re-derivation):

  Class               n_expected   Resolution
  ------------------- -----------  --------------------------------------
  t13_supernumerary   ~18          tltv_class_audit == "t13_supernumerary"
  lumbarization       ~44          lstv_class_audit == "lumbarization"
  truncated            ~6          tltv_class_audit == "t12_absent"
  normal             ~306          otherwise

Per-patient class: WORST-CASE across the patient's scans
                   (t13 > lumb > trunc > normal).

ROUND-ROBIN STRATIFIED K-FOLD
=============================
Per-stratum round-robin (sklearn's StratifiedKFold can't handle strata
with n < n_folds — the 6 truncated patients would crash it).  Each
stratum is shuffled (seeded) then dealt out across folds, so a stratum
of size 6 ends up with 2/1/1/1/1 patients across 5 folds, etc.

OUTPUT (splits_5fold.json)
==========================
{
  "schema_version":     2,
  "n_patients":         355,
  "n_folds":            5,
  "seed":               42,
  "strata_scheme":      "lstv_class_4way",
  "use_verse_test":     false,
  "subtypes":           [...],
  "subtype_counts":     {...},
  "test_patients":      [...],          # empty unless --use_verse_test
  "test_series_ids":    [...],          # empty unless --use_verse_test
  "folds": [
    {
      "fold":            0,
      "train_patients":  [...],
      "val_patients":    [...],
      "train_series_ids":[...],
      "val_series_ids":  [...]
    },
    ...
  ],
  "patient_subtypes": {patient_id: subtype, ...},
  "patient_attrs":    {patient_id: {n_scans, lstv_class, ...}, ...}
}
"""

from __future__ import annotations

import argparse
import json
import logging
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import pandas as pd

log = logging.getLogger("verse.splits_builder")


SCHEMA_VERSION = 3
SUBTYPES = ("t13_supernumerary", "lumbarization", "truncated", "normal")
CLASS_PRIORITY = {c: i for i, c in enumerate(SUBTYPES)}     # lower = worse-case


def _aggregate_per_patient(df: pd.DataFrame) -> dict[str, dict[str, Any]]:
    """Collapse multi-scan patients to a single attribute dict.

    Worst-case LSTV class (per CLASS_PRIORITY) and `any` over the
    veridah_applied flag, since either signals "interesting" patient.
    """
    by_pat: dict[str, dict[str, Any]] = {}
    for pid, group in df.groupby("patient_id"):
        if pd.isna(pid) or pid == "":
            continue
        classes = [c for c in group["lstv_class"] if pd.notna(c)]
        worst = min(classes, key=lambda c: CLASS_PRIORITY.get(c, 99)) if classes else "normal"
        by_pat[str(pid)] = {
            "n_scans":         int(len(group)),
            "series_ids":      sorted(group["series_id"].astype(str).tolist()),
            "lstv_class":      worst,
            "any_veridah":     bool(group["veridah_applied"].any()),
            "scan_lstv_classes": [str(c) for c in group["lstv_class"]],
        }
    return by_pat


def _round_robin_stratified_kfold(
    patients_by_class: dict[str, list[str]],
    n_folds:          int,
    seed:             int,
) -> dict[str, int]:
    """Per-stratum round-robin assignment.  Returns {patient_id: fold_idx}.

    Each stratum is shuffled (seeded) then dealt out 0,1,...,n-1,0,1,...
    across folds.  Patients in strata smaller than n_folds end up in the
    lowest-index folds, which is fine — we don't pretend to perfectly
    stratify cohorts smaller than n_folds, just spread them.
    """
    rng = random.Random(seed)
    assignment: dict[str, int] = {}
    for cls in sorted(patients_by_class.keys()):
        ids = sorted(patients_by_class[cls])
        rng.shuffle(ids)
        for i, pid in enumerate(ids):
            assignment[pid] = i % n_folds
    return assignment


def _stratified_test_holdout(
    patients_by_class: dict[str, list[str]],
    test_fraction:     float,
    seed:              int,
) -> tuple[set[str], dict[str, list[str]]]:
    """Pull `test_fraction` of each class out for the test set.

    Patient-level; size proportional to class size with floor 1 (so even
    `truncated` with 6 patients gets at least 1 test case) and ceiling
    n_class-1 (so no class is entirely consumed by the test split — at
    least 1 patient per class must remain for CV).

    Returns (test_patient_set, remaining_patients_by_class).  The seed
    is offset from the fold-assignment seed so the test holdout doesn't
    deterministically alias the fold-0 val set.
    """
    rng = random.Random(seed + 9973)   # distinct stream from fold assignment
    test_ids: set[str] = set()
    remaining: dict[str, list[str]] = {}
    for cls in sorted(patients_by_class.keys()):
        ids = sorted(patients_by_class[cls])
        rng.shuffle(ids)
        if not ids:
            remaining[cls] = []
            continue
        n_test = round(len(ids) * test_fraction)
        n_test = max(1, n_test) if test_fraction > 0 else 0
        n_test = min(n_test, len(ids) - 1)   # never consume the class
        n_test = max(0, n_test)
        test_ids.update(ids[:n_test])
        remaining[cls] = ids[n_test:]
    return test_ids, remaining


def build_splits(
    manifest_csv:   Path,
    output_path:    Path,
    n_folds:        int = 5,
    seed:           int = 42,
    test_fraction:  float = 0.0,
    use_verse_test: bool = False,
) -> dict[str, Any]:
    if not manifest_csv.exists():
        raise FileNotFoundError(
            f"manifest.csv not found at {manifest_csv}.  Run "
            f"`make manifest-slurm` first."
        )
    if use_verse_test and test_fraction > 0:
        raise ValueError(
            "--use_verse_test and --test_fraction are mutually exclusive: "
            "the former preserves VerSe's challenge-era test column verbatim; "
            "the latter resamples test from all subjects."
        )
    if not 0.0 <= test_fraction < 1.0:
        raise ValueError(f"--test_fraction must be in [0, 1), got {test_fraction}")

    df = pd.read_csv(manifest_csv)
    log.info("Loaded manifest: %d scans, %d patients",
             len(df), df["patient_id"].nunique())

    # ─── decide test holdout ─────────────────────────────────────────────
    test_patients:   list[str]
    test_series_ids: list[str]
    if use_verse_test:
        is_test = df["split"] == "test"
        test_patients   = sorted(df[is_test]["patient_id"].dropna().astype(str).unique().tolist())
        test_series_ids = sorted(df[is_test]["series_id"].astype(str).unique().tolist())
        cv_df = df[~is_test]
        test_mode = "verse_native"
        log.info("--use_verse_test: holding VerSe's test column (%d patients, %d scans)",
                 len(test_patients), len(test_series_ids))
    elif test_fraction > 0:
        # Resample test patient-level, stratified by lstv_class.  Do this BEFORE
        # the k-fold step so the same patients never leak into both pools.
        all_attrs = _aggregate_per_patient(df)
        by_class_all: dict[str, list[str]] = defaultdict(list)
        for pid, a in all_attrs.items():
            by_class_all[a["lstv_class"]].append(pid)
        test_set, remaining_by_class = _stratified_test_holdout(
            by_class_all, test_fraction, seed,
        )
        test_patients   = sorted(test_set)
        test_series_ids = sorted(s for pid in test_patients
                                   for s in all_attrs[pid]["series_ids"])
        cv_df = df[~df["patient_id"].astype(str).isin(test_set)]
        test_mode = f"resampled_fraction={test_fraction}"
        log.info("--test_fraction=%.2f: stratified holdout — "
                 "%d test patients (%d scans)",
                 test_fraction, len(test_patients), len(test_series_ids))
        # Per-class test breakdown
        from collections import Counter
        log.info("  test set by class: %s",
                 dict(Counter(all_attrs[p]["lstv_class"] for p in test_patients)))
    else:
        test_patients, test_series_ids = [], []
        cv_df = df
        test_mode = "none"
        log.info("test_fraction=0 and use_verse_test=False: folding ALL %d patients "
                 "(%d scans) into CV; no held-out test.",
                 df["patient_id"].nunique(), len(df))

    patient_attrs = _aggregate_per_patient(cv_df)
    log.info("CV pool: %d patients (%d scans)",
             len(patient_attrs), len(cv_df))

    # Group patients by their (worst-case) LSTV class
    by_class: dict[str, list[str]] = defaultdict(list)
    for pid, a in patient_attrs.items():
        by_class[a["lstv_class"]].append(pid)
    log.info("Patient-level LSTV class counts:")
    for cls in SUBTYPES:
        log.info("  %-22s %d", cls, len(by_class.get(cls, [])))
    extra = {c: len(v) for c, v in by_class.items() if c not in SUBTYPES}
    if extra:
        log.warning("  unrecognised classes: %s", extra)

    # Build map: patient_id -> series_ids
    pid_to_series: dict[str, list[str]] = {
        pid: a["series_ids"] for pid, a in patient_attrs.items()
    }

    # Assign each patient to a val-fold (round-robin within stratum)
    val_assignment = _round_robin_stratified_kfold(by_class, n_folds, seed)

    # Build folds: train = patients with val-fold != k, val = patients with val-fold == k
    folds: list[dict[str, Any]] = []
    for k in range(n_folds):
        train_pids = sorted([p for p, vf in val_assignment.items() if vf != k])
        val_pids   = sorted([p for p, vf in val_assignment.items() if vf == k])
        train_series = sorted(s for p in train_pids for s in pid_to_series[p])
        val_series   = sorted(s for p in val_pids   for s in pid_to_series[p])
        folds.append({
            "fold":             k,
            "train_patients":   train_pids,
            "val_patients":     val_pids,
            "train_series_ids": train_series,
            "val_series_ids":   val_series,
        })

    # Per-fold cross-tab for the log
    log.info("=" * 72)
    log.info("Per-fold LSTV distribution (val side)")
    log.info("=" * 72)
    header = "  %-3s  %-10s  %-10s" % ("fold", "n_train", "n_val")
    for cls in SUBTYPES:
        header += "  %-12s" % cls[:12]
    log.info(header)
    for f in folds:
        val_counts = Counter(patient_attrs[p]["lstv_class"] for p in f["val_patients"])
        row = "  %-3d  %-10d  %-10d" % (f["fold"], len(f["train_patients"]), len(f["val_patients"]))
        for cls in SUBTYPES:
            row += "  %-12d" % val_counts.get(cls, 0)
        log.info(row)
    log.info("=" * 72)

    doc = {
        "schema_version":  SCHEMA_VERSION,
        "n_patients":      len(patient_attrs),
        "n_test_patients": len(test_patients),
        "n_folds":         n_folds,
        "seed":            seed,
        "strata_scheme":   "lstv_class_4way",
        "test_mode":       test_mode,
        "test_fraction":   float(test_fraction),
        "use_verse_test":  bool(use_verse_test),
        "subtypes":        list(SUBTYPES),
        "subtype_counts":  {cls: len(by_class.get(cls, [])) for cls in SUBTYPES},
        "test_patients":   test_patients,
        "test_series_ids": test_series_ids,
        "folds":           folds,
        "patient_subtypes": {pid: a["lstv_class"] for pid, a in patient_attrs.items()},
        "patient_attrs":   patient_attrs,
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(doc, indent=2, default=str))
    log.info("Wrote %s", output_path)
    return doc


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                  formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--manifest_csv", type=Path, required=True,
                   help="Path to data/manifest/manifest.csv from stage 10a")
    p.add_argument("--out",          type=Path, required=True,
                   help="Output path for splits_5fold.json")
    p.add_argument("--n_folds",      type=int, default=5)
    p.add_argument("--seed",         type=int, default=42)

    test_grp = p.add_mutually_exclusive_group()
    test_grp.add_argument("--test_fraction", type=float, default=0.15,
                          help="Patient-level stratified test holdout fraction "
                               "(default 0.15).  Set 0 to fold all subjects "
                               "into CV with no held-out test.  Combined with "
                               "--n_folds=5 this gives roughly 70%% train / "
                               "15%% val / 15%% test per fold.")
    test_grp.add_argument("--use_verse_test", action="store_true",
                          help="Preserve VerSe's challenge-era test column "
                               "verbatim (~113 scans).  Mutually exclusive "
                               "with --test_fraction.")
    p.add_argument("--log_level",    default="INFO",
                   choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    args = p.parse_args(argv)
    logging.basicConfig(
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        level=args.log_level,
    )
    try:
        build_splits(args.manifest_csv, args.out,
                     n_folds=args.n_folds, seed=args.seed,
                     test_fraction=args.test_fraction,
                     use_verse_test=args.use_verse_test)
    except Exception as e:
        log.error("Splits build failed: %s: %s", type(e).__name__, e)
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
