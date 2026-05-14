"""
verse_pipeline.manifest_builder — per-scan metadata aggregation.

Aggregates per-subject metadata from canonical/, corrected/, lstv/, and
unified/ into a flat tabular manifest.  Splits (5-fold CV) are produced
by `verse_pipeline.splits_builder` and live in a separate file
(splits_5fold.json) so the splits can be regenerated with different
strata or seeds without rebuilding the manifest.

Output
------
- manifest.csv          flat tabular form (pandas-friendly, nnU-Net-friendly)
- manifest.json         lossless (labels_present remains a list)
- manifest_summary.json cross-tabs by split × lstv_class
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from collections import Counter
from pathlib import Path
from typing import Any

import pandas as pd

log = logging.getLogger("verse.manifest_builder")


def _load_unify_manifest(path: Path) -> dict[str, dict[str, Any]]:
    """Tolerant loader for the unify manifest.

    Returns {series_id: {patient_id, split, source_dataset}}.

    Accepts any of the following top-level layouts and is forgiving about
    per-record key naming (series_id / scan_id / subject_id / id):
        {"subjects": [...]}
        {"scans":    [...]}
        {"records":  [...]}
        {"entries":  [...]}
        {"items":    [...]}
        [{...}, {...}, ...]                       # flat list
        {"<sid>": {...}, "<sid2>": {...}, ...}    # dict keyed by series_id
    """
    if not path.exists():
        raise FileNotFoundError(f"unify manifest not found: {path}")
    raw = json.loads(path.read_text())

    records: list[dict] = []
    detected = "unknown"
    if isinstance(raw, list):
        records = [r for r in raw if isinstance(r, dict)]
        detected = "top-level list"
    elif isinstance(raw, dict):
        for key in ("subjects", "scans", "records", "entries", "items"):
            v = raw.get(key)
            if isinstance(v, list):
                records = [r for r in v if isinstance(r, dict)]
                detected = f"key='{key}'"
                break
        if not records:
            # Maybe dict keyed by series_id
            for k, v in raw.items():
                if isinstance(v, dict):
                    rec = dict(v)
                    rec.setdefault("series_id", rec.get("series_id") or k)
                    records.append(rec)
            if records:
                detected = "dict keyed by series_id"

    out: dict[str, dict[str, Any]] = {}
    for sub in records:
        sid = (sub.get("series_id") or sub.get("scan_id")
               or sub.get("subject_id") or sub.get("id"))
        if not sid:
            continue
        out[str(sid)] = {
            "patient_id": (sub.get("patient_id") or sub.get("patient")
                            or sub.get("subject")),
            "split": (sub.get("split") or sub.get("verse_split")
                      or sub.get("partition") or "unknown"),
            "source_dataset": (sub.get("source_dataset") or sub.get("verse_dataset")
                                or sub.get("dataset") or sub.get("source")
                                or sub.get("chosen_release")),
            "source_format":  sub.get("source_format"),
            "age":            sub.get("age"),
            "sex":            sub.get("sex"),
            "patient_pos":    sub.get("patient_pos") or sub.get("patient_position")
                                or sub.get("position"),
        }
    log.info("Loaded unify manifest: %d subjects  (layout: %s)", len(out), detected)
    if not out and records:
        log.warning("Found %d records but none had a series_id-like field.  "
                    "Available keys on first record: %s",
                    len(records), sorted(records[0].keys())[:20])
    return out


def _load_canonical_meta(canonical_dir: Path, series_id: str) -> dict[str, Any]:
    p = canonical_dir / f"scan-{series_id}" / f"scan-{series_id}_meta.json"
    if not p.exists():
        return {}
    return json.loads(p.read_text())


def _load_veridah(corrected_dir: Path | None) -> dict[str, dict[str, Any]]:
    if corrected_dir is None:
        return {}
    p = corrected_dir / "veridah_manifest.json"
    if not p.exists():
        return {}
    vm = json.loads(p.read_text())
    out: dict[str, dict[str, Any]] = {}
    for c in vm.get("corrections", []):
        sid = c.get("series_id")
        if not sid:
            continue
        out[sid] = {
            "veridah_applied": bool(c.get("veridah_applied", False)),
            "veridah_action":  c.get("action"),
            "veridah_kind":    c.get("kind") or c.get("category"),
        }
    log.info("Loaded veridah manifest: %d corrections", len(out))
    return out


def _load_lstv_audit(path: Path) -> dict[str, dict[str, Any]]:
    """Returns {series_id: {...flat record from audit...}}.

    The audit's per-record schema (lowercase booleans + categorical strings):
        has_l1/4/5/6, has_t11/12/13, has_sacrum
        has_any_lumbar, has_any_thoracic
        lsj_fov_complete, tlj_fov_complete
        lstv_class:  no_lumbar | lumbarization | lsj_fov_truncated | ...
        tltv_class:  normal_thoracolumbar | t13_supernumerary |
                     t12_absent | tlj_fov_truncated | no_thoracic | ...
        labels_present (list[int]), n_labels, lstv_evidence, tltv_evidence
        veridah_applied, veridah_correction_type
    """
    if not path.exists():
        log.warning("LSTV audit manifest not found: %s — LSTV columns will be empty", path)
        return {}
    data = json.loads(path.read_text())
    out: dict[str, dict[str, Any]] = {}
    items = data.get("subjects") or data.get("results") or data.get("entries") or []
    for entry in items:
        sid = entry.get("series_id")
        if not sid:
            continue
        labels = entry.get("labels_present") or []
        out[sid] = {
            "labels_present":  sorted({int(v) for v in labels}),
            "n_labels":        int(entry.get("n_labels") or len(set(labels))),
            "has_T13":         bool(entry.get("has_t13", False)),
            "has_L6":          bool(entry.get("has_l6", False)),
            # Cohort flag from the audit's own categoricals (NOT a re-derivation
            # from has_t12/t11 — the audit knows whether T12 absence is real
            # vs FOV-truncated).
            "lacks_T12_TLJ_in_FOV": entry.get("tltv_class") == "t12_absent",
            "lstv_class_audit": entry.get("lstv_class"),
            "tltv_class_audit": entry.get("tltv_class"),
            "lstv_evidence":   entry.get("lstv_evidence"),
            "tltv_evidence":   entry.get("tltv_evidence"),
        }
    log.info("Loaded LSTV audit: %d subjects", len(out))
    return out


def _compute_lstv_class(lstv_class_audit: str | None,
                         tltv_class_audit: str | None,
                         has_t13: bool,
                         has_l6: bool) -> str:
    """Map the audit's two categoricals into our 4-way stratification class.

    Priority (a scan can have both transitional vertebrae and FOV issues —
    we keep the most clinically relevant):
        t13_supernumerary  — audit's tltv_class says so
        lumbarization      — audit's lstv_class says so
        truncated          — t12 is genuinely absent (NOT just FOV-truncated)
        normal             — everything else (incl. fov-truncated scans that
                              still segment fine — they're not anomalous)

    Bools (has_t13/has_l6) are belt-and-suspenders fallback in case the
    audit's categoricals are missing on a record.
    """
    if tltv_class_audit == "t13_supernumerary" or has_t13:
        return "t13_supernumerary"
    if lstv_class_audit == "lumbarization" or has_l6:
        return "lumbarization"
    if tltv_class_audit == "t12_absent":
        return "truncated"
    return "normal"


def _assemble_rows(
    canonical_dir: Path,
    unify_rows: dict[str, dict[str, Any]],
    veridah:     dict[str, dict[str, Any]],
    lstv_audit:  dict[str, dict[str, Any]],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for sid, base in unify_rows.items():
        meta = _load_canonical_meta(canonical_dir, sid)
        vd   = veridah.get(sid, {})
        lst  = lstv_audit.get(sid, {})

        shape = meta.get("shape") or (None, None, None)
        spc   = meta.get("spacing") or meta.get("spacing_mm") or (None, None, None)

        has_t13   = bool(lst.get("has_T13", False))
        has_l6    = bool(lst.get("has_L6", False))
        lacks_t12 = bool(lst.get("lacks_T12_TLJ_in_FOV", False))
        lstv_cls  = _compute_lstv_class(
            lst.get("lstv_class_audit"),
            lst.get("tltv_class_audit"),
            has_t13, has_l6,
        )

        row = {
            "series_id":            sid,
            "patient_id":           base.get("patient_id"),
            "split":                base.get("split"),
            "source_dataset":       base.get("source_dataset"),
            "source_format":        base.get("source_format") or meta.get("source_format"),
            "shape_p":              shape[0] if shape else None,
            "shape_i":              shape[1] if shape else None,
            "shape_r":              shape[2] if shape else None,
            "spacing_p_mm":         spc[0] if spc else None,
            "spacing_i_mm":         spc[1] if spc else None,
            "spacing_r_mm":         spc[2] if spc else None,
            "age":                  base.get("age") if base.get("age") is not None
                                      else meta.get("age"),
            "sex":                  base.get("sex") or meta.get("sex"),
            "patient_pos":          base.get("patient_pos")
                                      or meta.get("patient_pos")
                                      or meta.get("patient_position"),
            "veridah_applied":      bool(vd.get("veridah_applied", False)),
            "veridah_action":       vd.get("veridah_action"),
            "veridah_kind":         vd.get("veridah_kind"),
            "n_labels":             lst.get("n_labels"),
            "labels_present":       json.dumps(lst.get("labels_present", []))
                                     if "labels_present" in lst else None,
            "has_T13":              has_t13,
            "has_L6":                has_l6,
            "lacks_T12_TLJ_in_FOV": lacks_t12,
            "lstv_class":           lstv_cls,
            "lstv_class_audit":     lst.get("lstv_class_audit"),
            "tltv_class_audit":     lst.get("tltv_class_audit"),
            "ct_relative_path":     f"scans/{sid}/ct.nii.gz",
            "mask_relative_path":   f"scans/{sid}/mask.nii.gz",
        }
        rows.append(row)
    return rows


def _summarize(df: pd.DataFrame) -> dict[str, Any]:
    summary: dict[str, Any] = {
        "n_total":         int(len(df)),
        "n_patients":      int(df["patient_id"].nunique()),
        "splits":          dict(Counter(df["split"])),
        "lstv_class":      dict(Counter(df["lstv_class"])),
        "source_dataset":  dict(Counter(df["source_dataset"].dropna())),
        "source_format":   dict(Counter(df["source_format"].dropna())),
        "veridah_applied": int(df["veridah_applied"].sum()),
        "lstv_per_split":  {},
    }
    for split, sub in df.groupby("split"):
        summary["lstv_per_split"][str(split)] = dict(Counter(sub["lstv_class"]))
    return summary


def _print_summary(summary: dict[str, Any]) -> None:
    log.info("=" * 72)
    log.info("MANIFEST SUMMARY")
    log.info("=" * 72)
    log.info("  total scans:        %d", summary["n_total"])
    log.info("  unique patients:    %d", summary["n_patients"])
    log.info("  veridah_applied:    %d", summary["veridah_applied"])
    log.info("  splits:             %s", summary["splits"])
    log.info("  lstv_class:         %s", summary["lstv_class"])
    log.info("  source_dataset:     %s", summary["source_dataset"])
    log.info("  source_format:      %s", summary["source_format"])
    log.info("-" * 72)
    log.info("LSTV class × split")
    log.info("-" * 72)
    classes = sorted({c for d in summary["lstv_per_split"].values() for c in d})
    splits  = sorted(summary["lstv_per_split"].keys())
    header = "  %-22s" % "lstv_class"
    for sp in splits:
        header += "  %-12s" % sp
    log.info(header)
    for cls in classes:
        row = "  %-22s" % cls
        for sp in splits:
            row += "  %-12d" % summary["lstv_per_split"].get(sp, {}).get(cls, 0)
        log.info(row)
    log.info("=" * 72)
    log.info("Next: run `make splits-slurm` to generate 5-fold stratified CV splits.")
    log.info("=" * 72)


def build_manifest(
    canonical_dir:    Path,
    corrected_dir:    Path | None,
    unify_manifest:   Path,
    lstv_audit:       Path,
    output_dir:       Path,
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    unify_rows = _load_unify_manifest(unify_manifest)
    veridah    = _load_veridah(corrected_dir)
    lstv       = _load_lstv_audit(lstv_audit)

    rows = _assemble_rows(canonical_dir, unify_rows, veridah, lstv)
    if not rows:
        raise RuntimeError(
            "No rows assembled.  This almost always means the unify manifest "
            "had 0 usable subjects — check the 'Loaded unify manifest' line "
            f"above.  Inspect {unify_manifest} and report its top-level keys "
            "if the permissive loader still doesn't recognize it."
        )
    df = pd.DataFrame(rows).sort_values("series_id").reset_index(drop=True)
    log.info("Assembled manifest: %d rows × %d columns", len(df), df.shape[1])

    csv_path = output_dir / "manifest.csv"
    df.to_csv(csv_path, index=False)
    log.info("Wrote %s", csv_path)

    json_rows = df.to_dict(orient="records")
    for r in json_rows:
        if r.get("labels_present"):
            try:
                r["labels_present"] = json.loads(r["labels_present"])
            except (TypeError, json.JSONDecodeError):
                pass
    manifest_json = {
        "schema_version": 1,
        "dataset_name":   "VerSeFusion",
        "n_subjects":     len(df),
        "subjects":       json_rows,
    }
    json_path = output_dir / "manifest.json"
    json_path.write_text(json.dumps(manifest_json, indent=2, default=str))
    log.info("Wrote %s", json_path)

    summary = _summarize(df)
    (output_dir / "manifest_summary.json").write_text(
        json.dumps(summary, indent=2, default=str)
    )
    log.info("Wrote %s", output_dir / "manifest_summary.json")
    _print_summary(summary)
    return summary


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                  formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--canonical_dir",  type=Path, required=True)
    p.add_argument("--corrected_dir",  type=Path, default=None)
    p.add_argument("--unify_manifest", type=Path, required=True)
    p.add_argument("--lstv_audit",     type=Path, required=True)
    p.add_argument("--output_dir",     type=Path, required=True)
    p.add_argument("--log_level",      default="INFO",
                   choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    args = p.parse_args(argv)
    logging.basicConfig(
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        level=args.log_level,
    )
    try:
        build_manifest(
            canonical_dir=args.canonical_dir,
            corrected_dir=args.corrected_dir,
            unify_manifest=args.unify_manifest,
            lstv_audit=args.lstv_audit,
            output_dir=args.output_dir,
        )
    except Exception as e:
        log.error("Manifest build failed: %s: %s", type(e).__name__, e)
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
