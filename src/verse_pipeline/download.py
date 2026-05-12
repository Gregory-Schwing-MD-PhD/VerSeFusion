"""Download VerSe in MICCAI-challenge format from OSF.

The MICCAI-format distribution is hosted at:

    VerSe 2019: https://osf.io/923ap/
    VerSe 2020: https://osf.io/b2wxj/

This is the *original* release format used in the MICCAI challenges, with
flat per-image filenames (``verseNNN.nii.gz``, ``verseNNN_seg.nii.gz``,
``verseNNN_ctd.json``, ``verseNNN_snapshot.png``).  We use it instead of
the BIDS subject-based mirrors (``jtfa5``, ``4skx2``) because:

  * The MICCAI release is what TUM originally published; using it gives us
    auditable provenance back to the canonical source.
  * Patient-level deduplication is left to the user, performed in the unify
    stage with explicit reference to TUM's published demographic table.
  * Centroid coordinates are uniformly in 1 mm isotropic ASL space across
    both releases (the BIDS form is in per-image voxel space, which mixes
    coordinate frames).

The downloader walks each OSF node via the public REST API, fetches every
file it advertises, and writes outcomes to a per-file manifest.  Behaviour
identical in spirit to the previous BIDS downloader:

  * Pre-request throttle (0.8 s) to stay under OSF's ~100 req/min limit.
  * HTTP 429 retry with exponential backoff (2 s -> 4 s -> 8 s, up to 6 attempts).
  * Atomic writes via ``.part`` files.
  * Resumable: already-downloaded files matching expected size are skipped.
  * Per-listing progress beacons so SLURM jobs don't look hung.

Usage
-----
    python -m verse_pipeline.download --out_dir data/raw
    python -m verse_pipeline.download --out_dir data/raw --release verse20
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

import requests
from tqdm import tqdm


# =============================================================================
# constants
# =============================================================================

OSF_API_BASE = "https://api.osf.io/v2"

# MICCAI-challenge-format nodes (flat, image-series-based layout).
RELEASES: dict[str, str] = {
    "verse19": "923ap",
    "verse20": "b2wxj",
}

CHUNK_SIZE = 1 << 20
TIMEOUT_SECONDS = 120
USER_AGENT = "VerSeFusion/0.2.0 (miccai)"

# Stay well under OSF's ~100 req/min unauthenticated limit.
OSF_LIST_THROTTLE_SECONDS = 0.8
OSF_LIST_MAX_RETRIES = 6

log = logging.getLogger("verse.download")


# =============================================================================
# OSF tree walker
# =============================================================================

@dataclass
class OSFEntry:
    kind:              str            # "file" | "folder"
    name:              str
    materialized_path: str
    size_bytes:        int | None
    download_url:      str | None
    folder_url:        str | None     # listing URL for folder contents


def _osf_get_json(url: str, session: requests.Session) -> dict:
    """GET an OSF API URL, with per-request throttle + 429 exponential backoff."""
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


def list_storage_recursive(node_id: str, session: requests.Session) -> list[OSFEntry]:
    """Recursively walk osfstorage on <node_id>; return all file-kind entries.

    Failed folder listings are retried once at the end of the walk; persistent
    failures are warned about but not fatal.
    """
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
                if entry.kind == "file":
                    files.append(entry)
                elif entry.kind == "folder" and entry.folder_url:
                    q.append(entry.folder_url)
            nxt = payload.get("links", {}).get("next")
            if nxt:
                q.append(nxt)
            n += 1
            if n % 10 == 0:
                log.info("Listing progress: %d requests, %d files, %d folders queued",
                         n, len(files), len(q))

    _drain(queue)

    if failed:
        log.warning("Retrying %d folder listing(s) that failed on first pass…", len(failed))
        retry = list(failed)
        failed.clear()
        for u in retry:
            visited.discard(u)
        _drain(retry)

    if failed:
        log.warning("%d folder listing(s) STILL failed after retry — inventory may be incomplete.",
                    len(failed))

    return files


# =============================================================================
# per-file download
# =============================================================================

def _local_path(entry: OSFEntry, release_root: Path) -> Path:
    """Mirror OSF folder layout under <release_root>."""
    rel = entry.materialized_path.lstrip("/")
    return release_root / rel


def download_file(entry: OSFEntry, release_root: Path,
                  session: requests.Session) -> tuple[Path, str]:
    """Download one file; return (local_path, status).

    Status: "downloaded" | "cached" | "failed".
    """
    dst = _local_path(entry, release_root)
    dst.parent.mkdir(parents=True, exist_ok=True)

    if dst.is_file():
        if entry.size_bytes is None or dst.stat().st_size == entry.size_bytes:
            return dst, "cached"
        log.warning("Local %s size %d != remote %d; re-downloading",
                    dst, dst.stat().st_size, entry.size_bytes)
        dst.unlink()

    part = dst.with_suffix(dst.suffix + ".part")
    if part.exists():
        part.unlink()

    if not entry.download_url:
        log.error("No download URL for %s", entry.materialized_path)
        return dst, "failed"

    try:
        with session.get(entry.download_url, stream=True, timeout=TIMEOUT_SECONDS,
                         headers={"User-Agent": USER_AGENT}) as r:
            r.raise_for_status()
            total = int(r.headers.get("Content-Length", "0")) or entry.size_bytes or 0
            with part.open("wb") as f, tqdm(
                total=total if total > 0 else None,
                unit="B", unit_scale=True, unit_divisor=1024,
                desc=entry.name, leave=False,
            ) as pbar:
                for chunk in r.iter_content(chunk_size=CHUNK_SIZE):
                    if not chunk:
                        continue
                    f.write(chunk)
                    pbar.update(len(chunk))
    except requests.RequestException as e:
        log.error("Download failed for %s: %s", entry.materialized_path, e)
        return dst, "failed"

    os.replace(part, dst)
    return dst, "downloaded"


# =============================================================================
# orchestration
# =============================================================================

@dataclass
class DownloadReport:
    files:         list[dict] = field(default_factory=list)
    n_downloaded:  int = 0
    n_cached:      int = 0
    n_failed:      int = 0


def fetch_release(release: str, node_id: str, raw_dir: Path,
                  session: requests.Session, report: DownloadReport) -> None:
    release_root = raw_dir / release
    release_root.mkdir(parents=True, exist_ok=True)

    log.info("=" * 70)
    log.info("Release: %s  (MICCAI OSF node: %s)", release, node_id)
    log.info("Listing files via OSF REST API…")

    files = list_storage_recursive(node_id, session)
    log.info("Discovered %d file(s) on OSF for %s", len(files), release)
    if not files:
        log.error("OSF returned no files for node %s — aborting %s", node_id, release)
        return

    for entry in files:
        dst, status = download_file(entry, release_root, session)
        report.files.append({
            "release":    release,
            "node_id":    node_id,
            "remote":     entry.materialized_path,
            "local":      str(dst),
            "size_bytes": entry.size_bytes,
            "status":     status,
        })
        if status == "downloaded":
            report.n_downloaded += 1
        elif status == "cached":
            report.n_cached += 1
        else:
            report.n_failed += 1


# =============================================================================
# CLI
# =============================================================================

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="verse-download",
        description=(
            "Download VerSe in MICCAI-challenge format from OSF nodes "
            "923ap (VerSe19) and b2wxj (VerSe20).  Use --release to limit."
        ),
    )
    p.add_argument("--out_dir", type=Path, required=True,
                   help="Destination directory.  Subdirs verse19/ verse20/ are created.")
    p.add_argument("--release", action="append", choices=list(RELEASES),
                   help="Limit to specific release(s).  Pass multiple times.  Default: both.")
    p.add_argument("--log_level", default="INFO",
                   choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    releases = args.release or list(RELEASES)
    log.info("Planning MICCAI-format pull for releases: %s", releases)
    args.out_dir.mkdir(parents=True, exist_ok=True)

    report = DownloadReport()
    with requests.Session() as session:
        session.headers.update({"User-Agent": USER_AGENT})
        for release in releases:
            fetch_release(release, RELEASES[release], args.out_dir, session, report)

    manifest_path = args.out_dir / "download_manifest.json"
    manifest_path.write_text(json.dumps({
        "source":       "osf_miccai",
        "osf_node_map": RELEASES,
        "n_files":      len(report.files),
        "n_downloaded": report.n_downloaded,
        "n_cached":     report.n_cached,
        "n_failed":     report.n_failed,
        "files":        report.files,
        "version":      "0.2.0",
    }, indent=2))
    log.info("Wrote manifest -> %s", manifest_path)
    log.info("Summary: %d downloaded, %d cached, %d failed",
             report.n_downloaded, report.n_cached, report.n_failed)

    if report.n_failed:
        log.error("%d file(s) failed — re-run to retry (downloads are resumable).",
                  report.n_failed)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
