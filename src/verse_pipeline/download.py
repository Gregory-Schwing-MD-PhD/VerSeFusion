"""Download VerSe in MICCAI-challenge format from OSF, with BIDS fallback.

Primary source — MICCAI-challenge format (osf.io/923ap and osf.io/b2wxj).  This
is the original release used in the VerSe challenges, with flat per-image
filenames and uniform centroid coordinate system (1 mm isotropic ASL).

Fallback source — BIDS subject-based format (osf.io/jtfa5 and osf.io/4skx2).
A small handful of subjects (~6) appear in TUM's published demographics but
are missing from the MICCAI nodes.  Those same subjects DO exist in the BIDS
mirrors.  Rather than maintaining a hardcoded list (which would go stale if
TUM ever publishes them to MICCAI), we auto-discover the gap by comparing
MICCAI listings against the demographics CSV.

Pipeline
--------
Phase 1: Listing (serial, throttled).
    - Walk both MICCAI nodes.
    - Determine which demographic series_ids ended up on neither MICCAI node.
    - If any are missing, walk both BIDS nodes and harvest just those subjects.

Phase 2: Fetching (parallel via ThreadPoolExecutor).
    - All MICCAI files + BIDS-fallback files for missing subjects.
    - Each file's manifest entry tagged with source: "miccai" or "bids_fallback".
    - BIDS-fallback files are kept with their original ``sub-`` prefix; the
      unify stage's parser handles both naming conventions.

Resumable: files already present at expected size are counted as cached.
Manifest is written once at the end, no concurrent-update races.

Usage
-----
    python -m verse_pipeline.download --out_dir data/raw
    python -m verse_pipeline.download --out_dir data/raw --workers 16
    python -m verse_pipeline.download --out_dir data/raw --no_bids_fallback
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import re
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from pathlib import Path

import requests

# We use the miccai parser to determine each MICCAI file's series_id during
# the gap-discovery phase.  Demographics gives us the authoritative list of
# expected series_ids; the diff tells us what's missing.
from verse_pipeline.utils.demographics import load_demographics, index_by_series
from verse_pipeline.utils.miccai import parse_filename as parse_miccai_filename


# =============================================================================
# constants
# =============================================================================

OSF_API_BASE = "https://api.osf.io/v2"

# Primary source: MICCAI-format nodes (flat, image-series-based layout).
MICCAI_RELEASES: dict[str, str] = {
    "verse19": "923ap",
    "verse20": "b2wxj",
}

# Fallback source: BIDS subject-based mirrors.  Walked only for subjects
# missing from MICCAI listings.
BIDS_RELEASES: dict[str, str] = {
    "verse19": "jtfa5",
    "verse20": "4skx2",
}

CHUNK_SIZE = 1 << 20
TIMEOUT_SECONDS = 120
USER_AGENT = "VerSeFusion/0.3.0 (miccai+bids-fallback)"

OSF_LIST_THROTTLE_SECONDS = 0.8
OSF_LIST_MAX_RETRIES = 6
DOWNLOAD_MAX_RETRIES = 4
DEFAULT_WORKERS = 8

# BIDS filename: sub-verseNNN_<entities>_<kind>.<ext>
# We use this to (a) extract series_id during fallback discovery, and (b)
# rename downloaded BIDS files into MICCAI-style names post-download.
_BIDS_FILENAME_RE = re.compile(
    r"^sub-(?P<series>verse\d+|gl\d+)(?P<rest>.*)$",
    re.IGNORECASE,
)


log = logging.getLogger("verse.download")


# =============================================================================
# OSF tree walker — serial, throttled, retries
# =============================================================================

@dataclass
class OSFEntry:
    kind:              str           # "file" | "folder"
    name:              str
    materialized_path: str
    size_bytes:        int | None
    download_url:      str | None
    folder_url:        str | None
    release:           str = ""
    node_id:           str = ""
    source:            str = ""      # "miccai" | "bids_fallback"


def _osf_get_json(url: str, session: requests.Session) -> dict:
    """GET an OSF API URL; throttle + 429 exponential backoff."""
    delay = 2.0
    for attempt in range(OSF_LIST_MAX_RETRIES + 1):
        time.sleep(OSF_LIST_THROTTLE_SECONDS)
        r = session.get(url, timeout=TIMEOUT_SECONDS, headers={"User-Agent": USER_AGENT})
        if r.status_code == 429:
            retry_after = r.headers.get("Retry-After")
            wait = float(retry_after) if retry_after and retry_after.isdigit() else delay
            if attempt < OSF_LIST_MAX_RETRIES:
                log.warning("OSF 429 on %s — sleeping %.1fs (retry %d/%d)",
                            url, wait, attempt + 1, OSF_LIST_MAX_RETRIES)
                time.sleep(wait)
                delay = min(delay * 2, 60.0)
                continue
        r.raise_for_status()
        return r.json()
    raise requests.HTTPError(f"OSF retries exhausted: {url}")


def _parse_entry(item: dict) -> OSFEntry:
    a = item["attributes"]
    kind = a.get("kind", "file")
    name = a.get("name", "?")
    mpath = a.get("materialized_path") or a.get("path") or f"/{name}"
    size = a.get("size")
    links = item.get("links", {})
    download_url = links.get("download") if kind == "file" else None
    folder_url = None
    if kind == "folder":
        try:
            folder_url = item["relationships"]["files"]["links"]["related"]["href"]
        except KeyError:
            folder_url = None
    return OSFEntry(
        kind=kind, name=name, materialized_path=mpath,
        size_bytes=int(size) if size is not None else None,
        download_url=download_url, folder_url=folder_url,
    )


def list_storage_recursive(release: str, node_id: str,
                           session: requests.Session) -> list[OSFEntry]:
    """Recursively walk osfstorage on <node_id>; return all file-kind entries."""
    seed = f"{OSF_API_BASE}/nodes/{node_id}/files/osfstorage/?page[size]=100"
    files: list[OSFEntry] = []
    queue: list[str] = [seed]
    visited: set[str] = set()
    failed: set[str] = set()

    def _drain(q: list[str]) -> None:
        n = 0
        while q:
            url = q.pop()
            if url in visited:
                continue
            visited.add(url)
            try:
                payload = _osf_get_json(url, session)
            except requests.RequestException as e:
                log.error("Failed to list %s: %s", url, e)
                failed.add(url)
                continue
            for item in payload.get("data", []):
                entry = _parse_entry(item)
                entry.release = release
                entry.node_id = node_id
                if entry.kind == "file":
                    files.append(entry)
                elif entry.kind == "folder" and entry.folder_url:
                    q.append(entry.folder_url)
            nxt = payload.get("links", {}).get("next")
            if nxt:
                q.append(nxt)
            n += 1
            if n % 10 == 0:
                log.info("  [%s/%s] %d API requests, %d files, %d folders queued",
                         release, node_id, n, len(files), len(q))

    _drain(queue)

    if failed:
        log.warning("Retrying %d folder listing(s) that failed on first pass…",
                    len(failed))
        retry = list(failed)
        failed.clear()
        for u in retry:
            visited.discard(u)
        _drain(retry)

    if failed:
        log.warning("%d folder listing(s) STILL failed after retry — "
                    "inventory may be incomplete.", len(failed))

    log.info("  [%s/%s] listing DONE: %d files", release, node_id, len(files))
    return files


# =============================================================================
# series-id extraction
# =============================================================================

def _miccai_series_id(name: str) -> str | None:
    """Extract series_id from a MICCAI filename, or None if unparseable."""
    parsed = parse_miccai_filename(name)
    return parsed[0] if parsed else None


def _bids_series_id(name: str) -> str | None:
    """Extract series_id from a BIDS filename like ``sub-verse014_dir-iso_ct.nii.gz``."""
    m = _BIDS_FILENAME_RE.match(name)
    return m.group("series").lower() if m else None


def series_ids_from_listing(entries: list[OSFEntry], source: str) -> set[str]:
    """Return the set of distinct series_ids present in this listing."""
    ids: set[str] = set()
    for e in entries:
        if source == "miccai":
            sid = _miccai_series_id(e.name)
        elif source == "bids":
            sid = _bids_series_id(e.name)
        else:
            sid = None
        if sid:
            ids.add(sid)
    return ids


# =============================================================================
# per-file download — runs in worker threads
# =============================================================================

_thread_local = threading.local()


def _session_for_thread() -> requests.Session:
    s = getattr(_thread_local, "session", None)
    if s is None:
        s = requests.Session()
        s.headers.update({"User-Agent": USER_AGENT})
        _thread_local.session = s
    return s


def _local_path(entry: OSFEntry, raw_dir: Path) -> Path:
    """Mirror OSF folder layout under <raw_dir>/<release>/."""
    rel = entry.materialized_path.lstrip("/")
    return raw_dir / entry.release / rel


def download_one(entry: OSFEntry, raw_dir: Path) -> dict:
    """Download a single file; return a manifest record."""
    dst = _local_path(entry, raw_dir)
    dst.parent.mkdir(parents=True, exist_ok=True)

    base = {
        "release":    entry.release,
        "node_id":    entry.node_id,
        "source":     entry.source,
        "remote":     entry.materialized_path,
        "local":      str(dst),
        "size_bytes": entry.size_bytes,
    }

    if dst.is_file():
        if entry.size_bytes is None or dst.stat().st_size == entry.size_bytes:
            return {**base, "status": "cached"}
        log.warning("Local %s size %d != remote %d; re-downloading",
                    dst, dst.stat().st_size, entry.size_bytes)
        dst.unlink()

    if not entry.download_url:
        log.error("No download URL for %s", entry.materialized_path)
        return {**base, "status": "failed"}

    part = dst.with_suffix(dst.suffix + ".part")
    if part.exists():
        try:
            part.unlink()
        except OSError:
            pass

    session = _session_for_thread()
    last_err = None
    backoff = 2.0
    for attempt in range(DOWNLOAD_MAX_RETRIES + 1):
        try:
            with session.get(entry.download_url, stream=True,
                             timeout=TIMEOUT_SECONDS,
                             headers={"User-Agent": USER_AGENT}) as r:
                if r.status_code == 429 and attempt < DOWNLOAD_MAX_RETRIES:
                    wait = float(r.headers.get("Retry-After") or backoff)
                    log.warning("429 on %s — sleeping %.1fs (retry %d/%d)",
                                entry.name, wait, attempt + 1, DOWNLOAD_MAX_RETRIES)
                    time.sleep(wait)
                    backoff = min(backoff * 2, 30.0)
                    continue
                r.raise_for_status()
                with part.open("wb") as f:
                    for chunk in r.iter_content(chunk_size=CHUNK_SIZE):
                        if chunk:
                            f.write(chunk)
            os.replace(part, dst)
            return {**base, "status": "downloaded"}
        except requests.RequestException as e:
            last_err = e
            if attempt < DOWNLOAD_MAX_RETRIES:
                log.warning("Download retry %d/%d for %s after %s: %s",
                            attempt + 1, DOWNLOAD_MAX_RETRIES, entry.name,
                            type(e).__name__, e)
                time.sleep(backoff)
                backoff = min(backoff * 2, 30.0)

    log.error("Download failed for %s after %d attempts: %s",
              entry.materialized_path, DOWNLOAD_MAX_RETRIES + 1, last_err)
    if part.exists():
        try:
            part.unlink()
        except OSError:
            pass
    return {**base, "status": "failed"}


# =============================================================================
# orchestration
# =============================================================================

@dataclass
class DownloadReport:
    files:        list[dict] = field(default_factory=list)
    n_downloaded: int = 0
    n_cached:     int = 0
    n_failed:     int = 0


def discover_and_plan(demographics_path: Path, use_bids_fallback: bool,
                      releases: list[str]) -> tuple[list[OSFEntry], set[str], set[str]]:
    """Phase 1+2: list MICCAI, compute per-kind completeness gap, list BIDS fallback.

    Trigger BIDS fallback for any subject missing one or more REQUIRED kinds
    in MICCAI — not just subjects missing entirely.  MICCAI sometimes ships
    partial coverage (e.g. CT + mask but no centroid) which used to fall
    through the previous "zero coverage" check.

    For each subject with missing kinds, we walk the BIDS nodes and harvest
    ONLY files matching the missing kinds, so MICCAI's good files take
    precedence (and we don't double-download).

    Returns:
        all_files: every OSFEntry to fetch (MICCAI + BIDS supplements)
        miccai_ids: set of series_ids with ≥1 file in MICCAI (for logging)
        recovered_ids: set of series_ids that got ≥1 file from BIDS fallback
    """
    from collections import defaultdict

    all_files: list[OSFEntry] = []
    miccai_ids: set[str] = set()
    recovered_ids: set[str] = set()

    # The kinds required for a subject to be considered "complete".
    REQUIRED_KINDS = {"ct", "msk", "ctd"}

    rows = load_demographics(demographics_path)
    expected_ids = {r.series_id for r in rows}
    log.info("Demographics: %d expected series_ids", len(expected_ids))

    # Track which kinds are present in MICCAI for each series_id, per release.
    # We key by series_id alone (not release) because the per-release split is
    # handled later by unify; what matters here is "is any release shipping
    # this kind for this subject from MICCAI?"
    miccai_kinds: dict[str, set[str]] = defaultdict(set)

    with requests.Session() as session:
        session.headers.update({"User-Agent": USER_AGENT})

        # --- Phase 1: list MICCAI primary nodes ---
        for release in releases:
            node_id = MICCAI_RELEASES[release]
            log.info("=" * 70)
            log.info("Listing MICCAI %s (node %s)…", release, node_id)
            files = list_storage_recursive(release, node_id, session)
            for f in files:
                f.source = "miccai"
                parsed = parse_miccai_filename(f.name)
                if parsed:
                    sid, kind = parsed
                    miccai_ids.add(sid)
                    miccai_kinds[sid].add(kind)
            all_files.extend(files)

        # --- Phase 2: compute per-subject missing-kinds map ---
        # subject_gaps[sid] is the set of REQUIRED kinds missing for this
        # subject across all MICCAI releases.
        subject_gaps: dict[str, set[str]] = {}
        for sid in expected_ids:
            present = miccai_kinds.get(sid, set()) & REQUIRED_KINDS
            missing = REQUIRED_KINDS - present
            if missing:
                subject_gaps[sid] = missing

        n_complete    = len(expected_ids) - len(subject_gaps)
        n_partial     = sum(1 for ks in subject_gaps.values() if ks != REQUIRED_KINDS)
        n_zero        = sum(1 for ks in subject_gaps.values() if ks == REQUIRED_KINDS)
        log.info("MICCAI completeness: %d/%d subjects have all required kinds",
                 n_complete, len(expected_ids))
        log.info("  %d subjects partially covered (some kinds present, some missing)",
                 n_partial)
        log.info("  %d subjects with zero MICCAI files", n_zero)

        if not subject_gaps:
            log.info("No subjects need BIDS fallback.")
            return all_files, miccai_ids, recovered_ids

        # Show specifics for partial-coverage cases (helps debugging).
        partial_examples = sorted([sid for sid, ks in subject_gaps.items()
                                   if ks != REQUIRED_KINDS])[:15]
        if partial_examples:
            log.info("Partial-coverage subjects (first 15): %s",
                     [(sid, sorted(subject_gaps[sid])) for sid in partial_examples])

        if not use_bids_fallback:
            log.info("BIDS fallback disabled by --no_bids_fallback; "
                     "incomplete subjects will not be filled in.")
            return all_files, miccai_ids, recovered_ids

        # --- Phase 3: BIDS fallback, filtered per-subject AND per-kind ---
        log.info("=" * 70)
        log.info("BIDS fallback ENABLED — listing BIDS nodes for the %d "
                 "incomplete subjects…", len(subject_gaps))
        recovered_kinds: dict[str, set[str]] = defaultdict(set)
        for release in releases:
            bids_node = BIDS_RELEASES[release]
            log.info("Listing BIDS %s (node %s)…", release, bids_node)
            files = list_storage_recursive(release, bids_node, session)

            relevant: list[OSFEntry] = []
            for f in files:
                sid = _bids_series_id(f.name)
                if sid not in subject_gaps:
                    continue
                parsed = parse_miccai_filename(f.name)
                if parsed is None:
                    continue
                _, kind = parsed
                # Only take BIDS files for kinds the subject actually needs;
                # if MICCAI already has the CT, we don't want BIDS's CT too.
                if kind not in subject_gaps[sid]:
                    continue
                # And only the first BIDS hit per (sid, kind) — BIDS can have
                # several with different entities (dir-iso vs dir-sag etc.).
                if kind in recovered_kinds[sid]:
                    continue
                f.source = "bids_fallback"
                relevant.append(f)
                recovered_kinds[sid].add(kind)
                recovered_ids.add(sid)
            log.info("  [%s/BIDS] retained %d files for %d subjects (per-kind filtered)",
                     release, len(relevant),
                     len({_bids_series_id(f.name) for f in relevant}))
            all_files.extend(relevant)

        # Report what we recovered vs still missing.
        still_missing = {sid: gaps - recovered_kinds.get(sid, set())
                         for sid, gaps in subject_gaps.items()}
        still_missing = {sid: ks for sid, ks in still_missing.items() if ks}
        n_fully_recovered = sum(
            1 for sid, gaps in subject_gaps.items()
            if not (gaps - recovered_kinds.get(sid, set()))
        )

        log.info("BIDS fallback summary:")
        log.info("  fully recovered: %d/%d subjects", n_fully_recovered, len(subject_gaps))
        log.info("  files added:     %d", sum(len(v) for v in recovered_kinds.values()))
        if still_missing:
            log.warning("  %d subjects STILL missing kinds after fallback:",
                        len(still_missing))
            for sid in sorted(still_missing):
                log.warning("    %s: missing %s", sid, sorted(still_missing[sid]))

    return all_files, miccai_ids, recovered_ids


def fetch_all_parallel(files: list[OSFEntry], raw_dir: Path, workers: int,
                       progress_interval_files: int = 25,
                       progress_interval_seconds: float = 30.0,
                       ) -> DownloadReport:
    """Phase 3: parallel downloads with periodic progress logged to stdout."""
    report = DownloadReport()
    if not files:
        return report

    total_files = len(files)
    total_bytes = sum(f.size_bytes or 0 for f in files)
    log.info("Fetching %d files (%.2f GB) with %d workers…",
             total_files, total_bytes / 1e9, workers)

    bytes_done = 0
    completed = 0
    last_log_count = 0
    last_log_time = time.monotonic()
    start_time = last_log_time

    def _flush_progress() -> None:
        nonlocal last_log_count, last_log_time
        elapsed = time.monotonic() - start_time
        rate_files = completed / elapsed if elapsed > 0 else 0.0
        remaining = total_files - completed
        eta_seconds = (remaining / rate_files) if rate_files > 0 else 0.0
        pct = (completed / total_files) * 100 if total_files else 0
        log.info(
            "progress: %d/%d (%.1f%%)  dl=%d cached=%d failed=%d  "
            "%.0f MB  %.1f files/s  ETA %s",
            completed, total_files, pct,
            report.n_downloaded, report.n_cached, report.n_failed,
            bytes_done / 1e6, rate_files, _fmt_eta(eta_seconds),
        )
        for h in log.handlers or logging.getLogger().handlers:
            try:
                h.flush()
            except Exception:
                pass
        last_log_count = completed
        last_log_time = time.monotonic()

    with ThreadPoolExecutor(max_workers=workers, thread_name_prefix="dl") as ex:
        futures = {ex.submit(download_one, f, raw_dir): f for f in files}
        for fut in as_completed(futures):
            rec = fut.result()
            report.files.append(rec)
            status = rec["status"]
            if status == "downloaded":
                report.n_downloaded += 1
                bytes_done += rec.get("size_bytes") or 0
            elif status == "cached":
                report.n_cached += 1
            else:
                report.n_failed += 1
            completed += 1

            since_count = completed - last_log_count
            since_time = time.monotonic() - last_log_time
            if (since_count >= progress_interval_files
                    or since_time >= progress_interval_seconds):
                _flush_progress()

    _flush_progress()
    return report


def _fmt_eta(seconds: float) -> str:
    if seconds <= 0:
        return "0s"
    seconds = int(seconds)
    h, rem = divmod(seconds, 3600)
    m, s = divmod(rem, 60)
    if h:
        return f"{h}h{m:02d}m{s:02d}s"
    if m:
        return f"{m}m{s:02d}s"
    return f"{s}s"


# =============================================================================
# CLI
# =============================================================================

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="verse-download",
        description=(
            "Download VerSe in MICCAI-challenge format from OSF nodes "
            "923ap and b2wxj.  Auto-recovers missing subjects from BIDS-format "
            "mirrors (jtfa5, 4skx2) when MICCAI's coverage is incomplete."
        ),
    )
    p.add_argument("--out_dir", type=Path, required=True,
                   help="Destination directory; verse19/ verse20/ subdirs are created.")
    p.add_argument("--release", action="append", choices=list(MICCAI_RELEASES),
                   help="Limit to specific release(s).  Pass multiple times.  Default: both.")
    p.add_argument("--workers", type=int, default=DEFAULT_WORKERS,
                   help=f"Parallel download workers (default {DEFAULT_WORKERS}).")
    p.add_argument("--demographics", type=Path,
                   default=Path("configs/verse_demographics.csv"),
                   help="Path to TUM demographics CSV (used to detect MICCAI gaps).")
    p.add_argument("--no_bids_fallback", action="store_true",
                   help="Skip the BIDS-mirror fallback for missing subjects.")
    p.add_argument("--log_level", default="INFO",
                   choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    releases = args.release or list(MICCAI_RELEASES)
    args.out_dir.mkdir(parents=True, exist_ok=True)

    files, miccai_ids, recovered_ids = discover_and_plan(
        args.demographics,
        use_bids_fallback=not args.no_bids_fallback,
        releases=releases,
    )
    if not files:
        log.error("No files discovered; aborting.")
        return 1

    report = fetch_all_parallel(files, args.out_dir, workers=args.workers)

    manifest_path = args.out_dir / "download_manifest.json"
    manifest_path.write_text(json.dumps({
        "source":            "osf_miccai_with_bids_fallback",
        "miccai_node_map":   MICCAI_RELEASES,
        "bids_node_map":     BIDS_RELEASES,
        "workers":           args.workers,
        "n_files":           len(report.files),
        "n_downloaded":      report.n_downloaded,
        "n_cached":          report.n_cached,
        "n_failed":          report.n_failed,
        "miccai_series_ids": sorted(miccai_ids),
        "bids_fallback_ids": sorted(recovered_ids),
        "files":             report.files,
        "version":           "0.3.0",
    }, indent=2))
    log.info("Wrote manifest -> %s", manifest_path)
    log.info("Summary: %d downloaded, %d cached, %d failed | "
             "MICCAI covers %d series, BIDS fallback recovered %d",
             report.n_downloaded, report.n_cached, report.n_failed,
             len(miccai_ids), len(recovered_ids))

    if report.n_failed:
        log.error("%d file(s) failed — re-run to retry (downloads are resumable).",
                  report.n_failed)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
