"""Unify MICCAI-format VerSe downloads into per-scan canonical directories.

Input layout (post-download):

    data/raw/verse19/
        ├── (folders from OSF, MICCAI-flat or BIDS-nested)
        │   ├── verse014.nii.gz
        │   ├── verse014_seg.nii.gz
        │   ├── verse014_ctd.json
        │   └── ...
    data/raw/verse20/
        ├── ...

Output layout (data/unified/):

    data/unified/
    ├── scan-verse014/
    │   ├── scan-verse014_ct.nii.gz       -> symlink to raw file
    │   ├── scan-verse014_msk.nii.gz      -> symlink
    │   ├── scan-verse014_ctd.json        -> symlink
    │   ├── scan-verse014_snp.png         -> symlink
    │   └── scan-verse014_meta.json       (generated)
    ├── scan-verse090/                    (scan 1 of patient verse400)
    ├── scan-verse155/                    (scan 2 of patient verse400)
    └── scan-gl003/

The meta.json captures:
  - series_id, patient_id (linking sibling scans of the same patient)
  - chosen_release (verse19 / verse20)
  - other_releases (cross-release siblings, recorded for provenance)
  - split (training / validation / test, derived from raw path)
  - position ("1 of 2", etc.)
  - sex, age (from TUM demographics)
  - source_paths (kind -> raw filesystem path)
  - source_format ("miccai" or "bids", from filename style)
  - centroid_coord_system (always "voxel" — see note below)

A note on coordinate systems
----------------------------
TUM ships centroids as direct (X, Y, Z) array indices in the image's own
voxel grid, across BOTH the MICCAI-challenge and BIDS distributions
(empirically verified across all 374 scans of the unified corpus).

Some BIDS files carry an explicit ``direction`` field
(e.g. ``["L", "A", "S"]``) documenting the anatomical orientation of the
image axes — but it affects only interpretation, not indexing.

The ``centroid_coord_system`` field in meta.json is retained for forward
compatibility but is currently always ``"voxel"``.  Downstream stages
(reorient, veridah) use centroids as direct array indices.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path

from verse_pipeline.utils.demographics import (
    DemographicRow, index_by_series, load_demographics,
)
from verse_pipeline.utils.miccai import (
    MICCAIFile, parse_path, required_kinds,
)

log = logging.getLogger("verse.unify")

# When a scan appears in both v19 and v20, which release's copy wins?
PREFERRED_RELEASE = "verse20"

# Determines the canonical split tag from the OSF directory name.
SPLIT_PATTERNS: tuple[tuple[str, str], ...] = (
    ("train", "training"),
    ("valid", "validation"),
    ("test",  "test"),
)

# Progress-logging cadence.
DISCOVER_PROGRESS_EVERY_N_FILES = 500
MATERIALISE_PROGRESS_EVERY_N_SCANS = 25
PROGRESS_TIME_INTERVAL_SECONDS = 10.0


def _flush_logs() -> None:
    """Force-flush log handlers so SLURM .err files show output promptly."""
    for h in log.handlers or logging.getLogger().handlers:
        try:
            h.flush()
        except Exception:
            pass


# =============================================================================
# discovery — walk raw MICCAI tree, group files by (release, series_id)
# =============================================================================

@dataclass
class RawScan:
    """All files for one series_id within one release."""
    release:   str
    series_id: str
    split:     str
    files:     dict[str, MICCAIFile] = field(default_factory=dict)

    def is_complete(self) -> bool:
        return all(k in self.files for k in required_kinds())

    def missing_kinds(self) -> list[str]:
        return [k for k in required_kinds() if k not in self.files]


def _split_for_path(path: Path) -> str:
    """Infer the train/valid/test split from a path component."""
    for part in path.parts:
        lower = part.lower()
        for needle, canonical in SPLIT_PATTERNS:
            if needle in lower:
                return canonical
    return "?"


def discover_raw(raw_root: Path) -> dict[tuple[str, str], RawScan]:
    """Walk raw_root recursively, grouping recognised files by (release, series_id).

    Logs progress every DISCOVER_PROGRESS_EVERY_N_FILES files or every
    PROGRESS_TIME_INTERVAL_SECONDS seconds, whichever comes first.
    """
    out: dict[tuple[str, str], RawScan] = {}
    if not raw_root.is_dir():
        log.error("Raw dir not found: %s", raw_root)
        return out

    for release_dir in sorted(raw_root.iterdir()):
        if not release_dir.is_dir():
            continue
        release = release_dir.name
        if release not in ("verse19", "verse20"):
            log.debug("Skipping unknown release dir: %s", release_dir)
            continue

        log.info("Walking release %s at %s …", release, release_dir)
        _flush_logs()

        n_files_seen = 0
        n_parsed = 0
        last_log_count = 0
        last_log_time = time.monotonic()
        start = last_log_time

        for path in release_dir.rglob("*"):
            if not path.is_file():
                continue
            n_files_seen += 1
            parsed = parse_path(path)
            if parsed is not None:
                n_parsed += 1
                key = (release, parsed.series_id)
                if key not in out:
                    out[key] = RawScan(
                        release=release,
                        series_id=parsed.series_id,
                        split=_split_for_path(path),
                    )
                if parsed.kind in out[key].files:
                    log.warning("Duplicate %s for %s/%s: keeping %s, ignoring %s",
                                parsed.kind, release, parsed.series_id,
                                out[key].files[parsed.kind].path, path)
                else:
                    out[key].files[parsed.kind] = parsed

            since_count = n_files_seen - last_log_count
            since_time = time.monotonic() - last_log_time
            if (since_count >= DISCOVER_PROGRESS_EVERY_N_FILES
                    or since_time >= PROGRESS_TIME_INTERVAL_SECONDS):
                n_scans_so_far = sum(1 for (r, _) in out if r == release)
                rate = n_files_seen / (time.monotonic() - start) if (time.monotonic() - start) > 0 else 0.0
                log.info("  [%s] %d files seen, %d parsed, %d scans grouped (%.0f files/s)",
                         release, n_files_seen, n_parsed, n_scans_so_far, rate)
                _flush_logs()
                last_log_count = n_files_seen
                last_log_time = time.monotonic()

        n_scans = sum(1 for (r, _) in out if r == release)
        log.info("Release %s done: %d files scanned, %d MICCAI-recognised, %d scans grouped",
                 release, n_files_seen, n_parsed, n_scans)
        _flush_logs()

    return out


# =============================================================================
# unification — pick canonical copy per series_id, materialise scan-dir
# =============================================================================

@dataclass
class UnifiedScan:
    """One canonical scan directory output."""
    series_id:              str
    patient_id:             str
    chosen_release:         str
    other_releases:         list[str]
    split:                  str
    position:               str
    sex:                    str
    age:                    int | None
    in_v19:                 bool
    in_v20:                 bool
    source_paths:           dict[str, str]
    missing_kinds:          list[str]
    out_dir:                Path
    source_format:          str
    centroid_coord_system:  str


def _detect_source_format(source_paths: dict[str, str]) -> str:
    """Return 'bids' if any file's name carries BIDS markers, else 'miccai'."""
    for path in source_paths.values():
        name = Path(path).name
        if name.startswith("sub-") or "_dir-" in name:
            return "bids"
    return "miccai"


def _choose_release(
    raws: dict[tuple[str, str], RawScan],
    series_id: str,
    in_v19: bool,
    in_v20: bool,
) -> tuple[str | None, list[str]]:
    """Pick the canonical release for this series_id; return (chosen, others)."""
    candidates: list[str] = []
    for release in ("verse19", "verse20"):
        if (release, series_id) in raws:
            candidates.append(release)
    if not candidates:
        return None, []

    complete = [r for r in candidates if raws[(r, series_id)].is_complete()]
    if complete:
        chosen = (PREFERRED_RELEASE if PREFERRED_RELEASE in complete else complete[0])
    elif PREFERRED_RELEASE in candidates:
        chosen = PREFERRED_RELEASE
    else:
        chosen = candidates[0]

    others = [r for r in candidates if r != chosen]
    return chosen, others


def materialise_scan(
    raw: RawScan,
    demo: DemographicRow,
    others_releases: list[str],
    out_root: Path,
) -> UnifiedScan:
    """Create out_root/scan-<series_id>/ with symlinks + meta.json.

    Uses .absolute() (no FS access) instead of .resolve() (which does a stat
    on every path component and is slow on distributed filesystems).
    """
    scan_dir = out_root / f"scan-{raw.series_id}"
    scan_dir.mkdir(parents=True, exist_ok=True)

    ext_map = {"ct":  "ct.nii.gz",  "msk": "msk.nii.gz",
               "ctd": "ctd.json",   "snp": "snp.png"}

    source_paths: dict[str, str] = {}
    for kind, src in raw.files.items():
        dst_name = f"scan-{raw.series_id}_{ext_map[kind]}"
        dst = scan_dir / dst_name
        if dst.exists() or dst.is_symlink():
            dst.unlink()
        src_abs = src.path.absolute()
        dst.symlink_to(src_abs)
        source_paths[kind] = str(src_abs)

    source_format = _detect_source_format(source_paths)
    centroid_coord_system = "voxel"   # TUM ships centroids as direct array indices

    meta = {
        "series_id":             raw.series_id,
        "patient_id":            demo.patient_id,
        "chosen_release":        raw.release,
        "other_releases":        others_releases,
        "split":                 raw.split,
        "position":              demo.position,
        "in_v19":                demo.in_v19,
        "in_v20":                demo.in_v20,
        "sex":                   demo.sex,
        "age":                   demo.age,
        "source_paths":          source_paths,
        "missing_kinds":         raw.missing_kinds(),
        "source_format":         source_format,
        "centroid_coord_system": centroid_coord_system,
        "version":               "0.3.0",
    }
    meta_path = scan_dir / f"scan-{raw.series_id}_meta.json"
    meta_path.write_text(json.dumps(meta, indent=2))

    return UnifiedScan(
        series_id=raw.series_id,
        patient_id=demo.patient_id,
        chosen_release=raw.release,
        other_releases=others_releases,
        split=raw.split,
        position=demo.position,
        sex=demo.sex,
        age=demo.age,
        in_v19=demo.in_v19,
        in_v20=demo.in_v20,
        source_paths=source_paths,
        missing_kinds=raw.missing_kinds(),
        out_dir=scan_dir,
        source_format=source_format,
        centroid_coord_system=centroid_coord_system,
    )


def unify(
    raw_root: Path,
    out_root: Path,
    demographics_path: Path,
) -> list[UnifiedScan]:
    """Top-level unify: discover raw, walk demographics, materialise per-scan dirs."""
    out_root.mkdir(parents=True, exist_ok=True)

    demographics = load_demographics(demographics_path)
    demo_idx = index_by_series(demographics)
    log.info("Demographics: %d series across %d patients",
             len(demographics), len({r.patient_id for r in demographics}))
    _flush_logs()

    raws = discover_raw(raw_root)
    log.info("Raw discovery: %d (release, series_id) tuples across both releases",
             len(raws))
    _flush_logs()

    disk_series = {sid for (_, sid) in raws}
    demo_series = set(demo_idx)
    missing_from_disk = demo_series - disk_series
    extra_on_disk = disk_series - demo_series

    if missing_from_disk:
        log.warning("%d series in demographics but NOT in raw data: %s",
                    len(missing_from_disk), sorted(missing_from_disk)[:10])
    if extra_on_disk:
        log.warning("%d series in raw data but NOT in demographics: %s",
                    len(extra_on_disk), sorted(extra_on_disk)[:10])
    _flush_logs()

    log.info("Materialising %d scan directories…", len(demographics))
    _flush_logs()

    unified: list[UnifiedScan] = []
    skipped = 0
    last_log_count = 0
    last_log_time = time.monotonic()
    start = last_log_time

    for i, demo in enumerate(demographics, 1):
        chosen, others = _choose_release(raws, demo.series_id,
                                         demo.in_v19, demo.in_v20)
        if chosen is None:
            log.warning("No raw files for series %s (patient %s) — skipping",
                        demo.series_id, demo.patient_id)
            skipped += 1
        else:
            raw = raws[(chosen, demo.series_id)]
            unified.append(materialise_scan(raw, demo, others, out_root))

        since_count = i - last_log_count
        since_time = time.monotonic() - last_log_time
        if (since_count >= MATERIALISE_PROGRESS_EVERY_N_SCANS
                or since_time >= PROGRESS_TIME_INTERVAL_SECONDS):
            elapsed = time.monotonic() - start
            rate = i / elapsed if elapsed > 0 else 0.0
            remaining = len(demographics) - i
            eta = remaining / rate if rate > 0 else 0.0
            log.info("  progress: %d/%d (%.1f%%)  unified=%d skipped=%d  "
                     "%.1f scans/s  ETA %ds",
                     i, len(demographics), 100 * i / len(demographics),
                     len(unified), skipped, rate, int(eta))
            _flush_logs()
            last_log_count = i
            last_log_time = time.monotonic()

    log.info("Materialise done: %d unified, %d skipped (no raw files)",
             len(unified), skipped)
    _flush_logs()
    return unified


# =============================================================================
# manifest writer
# =============================================================================

def write_unify_manifest(unified: list[UnifiedScan], out_root: Path) -> Path:
    by_release        = defaultdict(int)
    by_split          = defaultdict(int)
    by_patient        = defaultdict(list)
    by_source_format  = defaultdict(int)
    n_complete   = 0
    n_image_only = 0
    n_msk_only   = 0
    n_other      = 0

    for s in unified:
        by_release[s.chosen_release] += 1
        by_split[s.split] += 1
        by_patient[s.patient_id].append(s.series_id)
        by_source_format[s.source_format] += 1
        m = set(s.missing_kinds)
        if not m:
            n_complete += 1
        elif m == {"msk", "ctd"} or m == {"msk", "ctd", "snp"}:
            n_image_only += 1
        elif m == {"ct"}:
            n_msk_only += 1
        else:
            n_other += 1

    multi_patients = {p: scans for p, scans in by_patient.items() if len(scans) > 1}

    manifest = {
        "version":              "0.3.0",
        "source_format":        "miccai_with_bids_fallback",
        "preferred_release":    PREFERRED_RELEASE,
        "n_scans":              len(unified),
        "n_patients":           len(by_patient),
        "n_multi_series":       len(multi_patients),
        "by_release":           dict(by_release),
        "by_split":             dict(by_split),
        "by_source_format":     dict(by_source_format),
        "completeness": {
            "n_complete":       n_complete,
            "n_image_only":     n_image_only,
            "n_msk_only":       n_msk_only,
            "n_other_partial":  n_other,
        },
        "multi_series_patients": multi_patients,
        "scans": [
            {
                "series_id":             s.series_id,
                "patient_id":            s.patient_id,
                "chosen_release":        s.chosen_release,
                "other_releases":        s.other_releases,
                "split":                 s.split,
                "position":              s.position,
                "in_v19":                s.in_v19,
                "in_v20":                s.in_v20,
                "sex":                   s.sex,
                "age":                   s.age,
                "missing_kinds":         s.missing_kinds,
                "source_paths":          s.source_paths,
                "out_dir":               str(s.out_dir),
                "source_format":         s.source_format,
                "centroid_coord_system": s.centroid_coord_system,
            }
            for s in unified
        ],
    }
    path = out_root / "unify_manifest.json"
    path.write_text(json.dumps(manifest, indent=2))
    return path


# =============================================================================
# CLI
# =============================================================================

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="verse-unify",
        description=(
            "Unify MICCAI-format VerSe downloads into per-scan canonical "
            "directories.  Each scan becomes data/unified/scan-<series_id>/ "
            "with symlinks to raw files plus a meta.json."
        ),
    )
    p.add_argument("--raw_dir", type=Path, required=True,
                   help="Where the downloader placed raw MICCAI files.")
    p.add_argument("--out_dir", type=Path, required=True,
                   help="Where to write data/unified/scan-*.")
    p.add_argument("--demographics", type=Path, required=True,
                   help="Path to configs/verse_demographics.csv.")
    p.add_argument("--log_level", default="INFO",
                   choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    unified = unify(args.raw_dir, args.out_dir, args.demographics)
    manifest_path = write_unify_manifest(unified, args.out_dir)
    log.info("Wrote %s", manifest_path)
    log.info("Unified %d scan(s) across %d patient(s)",
             len(unified), len({s.patient_id for s in unified}))
    return 0


if __name__ == "__main__":
    sys.exit(main())
