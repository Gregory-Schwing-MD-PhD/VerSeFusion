"""Download the VerSe 2019 + VerSe 2020 BIDS-restructured archives from OSF.

The original S3 distribution at ``s3.bonescreen.de`` has been offline since
May 2024 (see https://github.com/anjany/verse/issues/17).  The same
BIDS-restructured ("subject-based") data is available on OSF under
separate child nodes that the upstream README never linked to:

    VerSe 2019 subject-based:  https://osf.io/jtfa5/
    VerSe 2020 subject-based:  https://osf.io/4skx2/

This module talks to OSF's public REST API directly (no osfclient or other
new dependency — uses the same ``requests`` already in the stack):

    https://api.osf.io/v2/nodes/<node_id>/files/osfstorage/

The downloader:

  * Recurses the OSF folder tree, downloading every file to a local path
    that mirrors the remote layout.
  * Is resumable — already-downloaded files (matching expected size) are
    skipped on re-run.
  * Atomically renames each file (``.part`` -> final) so a partial download
    is never confused for a complete one.
  * Extracts any ``.zip`` files in place, gated on a per-zip
    ``.extracted`` marker so unpack is idempotent.
  * Writes a ``download_manifest.json`` summarising every file fetched.

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
import zipfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable

import requests
from tqdm import tqdm

# =============================================================================
# constants
# =============================================================================

OSF_API_BASE = "https://api.osf.io/v2"

# release id -> OSF node id holding the subject-based (BIDS-restructured) data.
# These ids were surfaced in https://github.com/anjany/verse/issues/17 (the
# upstream maintainers host both the MICCAI-challenge layout AND the
# BIDS-restructured layout on OSF, but only link to the MICCAI nodes in the
# README; the BIDS-restructured child nodes are jtfa5 / 4skx2).
RELEASES: dict[str, str] = {
    "verse19": "jtfa5",
    "verse20": "4skx2",
}

CHUNK_SIZE = 1 << 20            # 1 MiB
TIMEOUT_SECONDS = 120
USER_AGENT = "VerSeFusion/0.1.0 (https://github.com/gschwing/VerSeFusion)"

# OSF rate-limits its API to ~100 req/min per IP for unauthenticated callers.
# Sleep this long between listing requests to stay well under the cap, and
# back off on HTTP 429 with exponential delay.
OSF_LIST_THROTTLE_SECONDS = 0.8
OSF_LIST_MAX_RETRIES = 6

log = logging.getLogger("verse.download")


# =============================================================================
# OSF tree walker
# =============================================================================

@dataclass
class OSFEntry:
    """One file or folder returned by the OSF REST API."""
    kind:        str               # "file" | "folder"
    name:        str
    materialized_path: str         # absolute path within osfstorage, e.g. "/01_training/raw.../foo.nii.gz"
    size_bytes:  int | None        # None for folders
    download_url: str | None       # None for folders
    folder_url:  str | None        # listing URL for the folder's contents


def _osf_get_json(url: str, session: requests.Session) -> dict:
    """GET an OSF API URL, with throttle + exponential backoff on 429.

    OSF returns HTTP 429 if the unauthenticated rate limit (~100 req/min) is
    breached.  We sleep ``OSF_LIST_THROTTLE_SECONDS`` before every request to
    stay under the cap, and on 429 we back off (2s, 4s, 8s, ...) up to
    ``OSF_LIST_MAX_RETRIES`` times before giving up.
    """
    import time

    delay = 2.0
    for attempt in range(OSF_LIST_MAX_RETRIES + 1):
        time.sleep(OSF_LIST_THROTTLE_SECONDS)
        r = session.get(url, timeout=TIMEOUT_SECONDS, headers={"User-Agent": USER_AGENT})
        if r.status_code == 429:
            # honour Retry-After if the server sent one, else exponential
            retry_after = r.headers.get("Retry-After")
            wait = float(retry_after) if retry_after and retry_after.isdigit() else delay
            if attempt < OSF_LIST_MAX_RETRIES:
                log.warning(
                    "OSF 429 on %s — sleeping %.1fs (retry %d/%d)",
                    url, wait, attempt + 1, OSF_LIST_MAX_RETRIES,
                )
                time.sleep(wait)
                delay = min(delay * 2, 60.0)
                continue
        r.raise_for_status()
        return r.json()
    # Exhausted retries; raise so the caller logs and continues.
    raise requests.HTTPError(f"OSF rate limit exceeded after {OSF_LIST_MAX_RETRIES} retries: {url}")


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
        kind=kind,
        name=name,
        materialized_path=mpath,
        size_bytes=int(size) if size is not None else None,
        download_url=download_url,
        folder_url=folder_url,
    )


def list_storage_recursive(node_id: str, session: requests.Session) -> list[OSFEntry]:
    """Walk every file in <node_id>'s osfstorage, recursing into folders.

    Returns a flat list of file-kind ``OSFEntry`` records (folders are not
    emitted; only files).  Failed folder listings are retried at the end of
    the walk; if any still fail after the retry pass, a warning is logged
    with the count so the caller knows the inventory may be incomplete.
    """
    seed_url = f"{OSF_API_BASE}/nodes/{node_id}/files/osfstorage/?page[size]=100"
    files: list[OSFEntry] = []
    queue: list[str] = [seed_url]
    visited: set[str] = set()
    failed: set[str] = set()

    def _drain(q: list[str]) -> None:
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

            # JSON:API pagination
            next_url = payload.get("links", {}).get("next")
            if next_url:
                q.append(next_url)

    _drain(queue)

    # Retry pass — give any 429'd folders a second chance now that the
    # initial burst has subsided.
    if failed:
        log.warning(
            "Retrying %d folder listing(s) that failed on first pass…", len(failed),
        )
        retry_queue = list(failed)
        failed.clear()
        # Re-allow these URLs (clear them from visited so _drain refetches).
        for u in retry_queue:
            visited.discard(u)
        _drain(retry_queue)

    if failed:
        log.warning(
            "%d folder listing(s) STILL failed after retry — inventory may be incomplete. "
            "Re-run `make download-slurm` to pick up missed files.",
            len(failed),
        )

    return files


# =============================================================================
# per-file download
# =============================================================================

def _local_path(entry: OSFEntry, release_root: Path) -> Path:
    """Compute the local destination for an OSF file, mirroring remote layout."""
    rel = entry.materialized_path.lstrip("/")
    return release_root / rel


def download_file(
    entry: OSFEntry,
    release_root: Path,
    session: requests.Session,
) -> tuple[Path, str]:
    """Download one OSF file.  Returns (local_path, status).

    Status values:
        "downloaded" — fetched from OSF this run.
        "cached"     — already present locally at the expected size.
        "failed"     — fetch errored; .part file (if any) left behind.
    """
    dst = _local_path(entry, release_root)
    dst.parent.mkdir(parents=True, exist_ok=True)

    # already-complete file?
    if dst.is_file():
        if entry.size_bytes is None or dst.stat().st_size == entry.size_bytes:
            return dst, "cached"
        log.warning(
            "Local %s size %d != remote %d; re-downloading",
            dst, dst.stat().st_size, entry.size_bytes,
        )
        dst.unlink()

    part = dst.with_suffix(dst.suffix + ".part")
    if part.exists():
        part.unlink()

    if not entry.download_url:
        log.error("No download URL for %s", entry.materialized_path)
        return dst, "failed"

    try:
        with session.get(
            entry.download_url,
            stream=True,
            timeout=TIMEOUT_SECONDS,
            headers={"User-Agent": USER_AGENT},
        ) as r:
            r.raise_for_status()
            total = int(r.headers.get("Content-Length", "0")) or entry.size_bytes or 0
            with part.open("wb") as f, tqdm(
                total=total if total > 0 else None,
                unit="B",
                unit_scale=True,
                unit_divisor=1024,
                desc=entry.name,
                leave=False,
            ) as pbar:
                for chunk in r.iter_content(chunk_size=CHUNK_SIZE):
                    if not chunk:
                        continue
                    f.write(chunk)
                    pbar.update(len(chunk))
    except requests.RequestException as e:
        log.error("Download failed for %s: %s", entry.materialized_path, e)
        return dst, "failed"

    os.replace(part, dst)        # atomic
    return dst, "downloaded"


# =============================================================================
# zip extraction (post-download)
# =============================================================================

def extract_zips(release_root: Path) -> list[Path]:
    """Extract every .zip under <release_root> in place, idempotently.

    Each zip leaves a sibling .extracted marker so re-runs are no-ops.
    Returns the list of zips actually extracted on this call.
    """
    extracted: list[Path] = []
    for zip_path in sorted(release_root.rglob("*.zip")):
        marker = zip_path.with_suffix(zip_path.suffix + ".extracted")
        if marker.exists():
            continue
        try:
            with zipfile.ZipFile(zip_path) as zf:
                bad = zf.testzip()
                if bad is not None:
                    log.error("Zip CRC failure at %s in %s", bad, zip_path)
                    continue
                members = zf.namelist()
                for m in tqdm(members, desc=f"extract {zip_path.name}", leave=False):
                    zf.extract(m, zip_path.parent)
            marker.write_text(json.dumps({"members": len(members)}, indent=2))
            extracted.append(zip_path)
            log.info("Extracted %s", zip_path)
        except zipfile.BadZipFile as e:
            log.error("BadZipFile for %s: %s", zip_path, e)
    return extracted


# =============================================================================
# orchestration
# =============================================================================

@dataclass
class DownloadReport:
    files: list[dict] = field(default_factory=list)
    n_downloaded: int = 0
    n_cached:     int = 0
    n_failed:     int = 0
    n_extracted_zips: int = 0


def fetch_release(
    release: str,
    node_id: str,
    raw_dir: Path,
    session: requests.Session,
    report: DownloadReport,
) -> None:
    release_root = raw_dir / release
    release_root.mkdir(parents=True, exist_ok=True)

    log.info("=" * 70)
    log.info("Release: %s  (OSF node: %s)", release, node_id)
    log.info("Listing files via OSF REST API…")

    files = list_storage_recursive(node_id, session)
    log.info("Discovered %d file(s) on OSF for %s", len(files), release)

    if not files:
        log.error("OSF returned no files for node %s — aborting %s", node_id, release)
        return

    for entry in files:
        dst, status = download_file(entry, release_root, session)
        report.files.append({
            "release":   release,
            "node_id":   node_id,
            "remote":    entry.materialized_path,
            "local":     str(dst),
            "size_bytes": entry.size_bytes,
            "status":    status,
        })
        if status == "downloaded":
            report.n_downloaded += 1
        elif status == "cached":
            report.n_cached += 1
        else:
            report.n_failed += 1

    log.info("Extracting any .zip files under %s…", release_root)
    extracted = extract_zips(release_root)
    report.n_extracted_zips += len(extracted)


# =============================================================================
# CLI
# =============================================================================

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="verse-download",
        description=(
            "Download the VerSe 2019 + VerSe 2020 BIDS-restructured data "
            "from OSF (the s3.bonescreen.de endpoint has been offline since "
            "May 2024; see https://github.com/anjany/verse/issues/17)."
        ),
    )
    p.add_argument(
        "--out_dir",
        type=Path,
        required=True,
        help="Destination directory.  Subdirs verse19/ and verse20/ are created.",
    )
    p.add_argument(
        "--release",
        action="append",
        choices=list(RELEASES),
        help="Limit to specific release(s).  Pass multiple times.  Default: both.",
    )
    p.add_argument(
        "--log_level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
    )
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    releases = args.release or list(RELEASES)
    log.info("Planning OSF pull for releases: %s", releases)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    report = DownloadReport()

    with requests.Session() as session:
        session.headers.update({"User-Agent": USER_AGENT})
        for release in releases:
            node_id = RELEASES[release]
            fetch_release(release, node_id, args.out_dir, session, report)

    # ---- summary manifest ---------------------------------------------------
    manifest_path = args.out_dir / "download_manifest.json"
    manifest_path.write_text(json.dumps({
        "source":           "osf",
        "osf_node_map":     RELEASES,
        "n_files":          len(report.files),
        "n_downloaded":     report.n_downloaded,
        "n_cached":         report.n_cached,
        "n_failed":         report.n_failed,
        "n_extracted_zips": report.n_extracted_zips,
        "files":            report.files,
        "version":          "0.2.0",
    }, indent=2))
    log.info("Wrote manifest -> %s", manifest_path)

    log.info(
        "Summary: %d downloaded, %d cached, %d failed, %d zips extracted",
        report.n_downloaded, report.n_cached, report.n_failed, report.n_extracted_zips,
    )

    if report.n_failed:
        log.error("%d file(s) failed — re-run to retry (downloads are resumable).", report.n_failed)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
