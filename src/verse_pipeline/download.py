"""Download the VerSe 2019 + VerSe 2020 BIDS-restructured archives.

The recommended distribution lives on the TUM bonescreen S3 bucket:

    https://s3.bonescreen.de/public/VerSe-complete/dataset-verse{19,20}{training,validation,test}.zip

Each release ships as three zips (training / validation / test).  All six
zips together total ~30 GB and are mirrored from the canonical OSF
repositories (osf.io/nqjyw, osf.io/t98fz) with the *restructured* annotation
format the maintainers recommend.

This script is:
    * resumable — uses HTTP Range headers; partial .part files are reused
    * idempotent — already-downloaded archives are skipped
    * atomic — files are written to <name>.part and renamed only on success
    * integrity-checked — sha256 hashes are recorded in a manifest
    * unpack-once — extraction is gated on a .extracted marker

Usage
-----
    python -m verse_pipeline.download --out_dir data/raw
    python -m verse_pipeline.download --out_dir data/raw --release verse20
    python -m verse_pipeline.download --out_dir data/raw --skip_extract
"""

from __future__ import annotations

import argparse
import hashlib
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

S3_BASE = "https://s3.bonescreen.de/public/VerSe-complete"

# (release, split) -> zip filename on S3
RELEASES: dict[str, list[tuple[str, str]]] = {
    "verse19": [
        ("training",   "dataset-verse19training.zip"),
        ("validation", "dataset-verse19validation.zip"),
        ("test",       "dataset-verse19test.zip"),
    ],
    "verse20": [
        ("training",   "dataset-verse20training.zip"),
        ("validation", "dataset-verse20validation.zip"),
        ("test",       "dataset-verse20test.zip"),
    ],
}

CHUNK_SIZE = 1024 * 1024  # 1 MiB
TIMEOUT_SECONDS = 120
USER_AGENT = "VerSeFusion/0.1.0 (https://github.com/gschwing/VerSeFusion)"

log = logging.getLogger("verse.download")


# =============================================================================
# dataclasses
# =============================================================================

@dataclass
class Archive:
    """One S3 zip to fetch."""
    release: str          # "verse19" | "verse20"
    split:   str          # "training" | "validation" | "test"
    filename: str         # e.g. "dataset-verse19training.zip"

    @property
    def url(self) -> str:
        return f"{S3_BASE}/{self.filename}"

    def out_zip(self, raw_dir: Path) -> Path:
        return raw_dir / self.release / "downloads" / self.filename

    def part_zip(self, raw_dir: Path) -> Path:
        return self.out_zip(raw_dir).with_suffix(self.out_zip(raw_dir).suffix + ".part")

    def extracted_marker(self, raw_dir: Path) -> Path:
        # one marker per archive — uniqueness comes from the filename stem.
        stem = self.filename.removesuffix(".zip")
        return raw_dir / self.release / "downloads" / f".{stem}.extracted"

    def extract_root(self, raw_dir: Path) -> Path:
        return raw_dir / self.release / self.split


@dataclass
class DownloadReport:
    """Per-run summary written to <raw_dir>/download_manifest.json."""
    archives: list[dict] = field(default_factory=list)

    def add(self, archive: Archive, *, status: str, size_bytes: int, sha256: str | None) -> None:
        self.archives.append({
            "release":     archive.release,
            "split":       archive.split,
            "filename":    archive.filename,
            "url":         archive.url,
            "status":      status,                # "downloaded" | "cached" | "extracted" | "skipped" | "failed"
            "size_bytes":  size_bytes,
            "sha256":      sha256,
        })


# =============================================================================
# core download
# =============================================================================

def _http_size(url: str, session: requests.Session) -> int | None:
    """HEAD the URL to discover Content-Length.  Returns None if unavailable."""
    try:
        r = session.head(url, allow_redirects=True, timeout=TIMEOUT_SECONDS)
        r.raise_for_status()
        cl = r.headers.get("Content-Length")
        return int(cl) if cl is not None else None
    except (requests.RequestException, ValueError):
        return None


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(CHUNK_SIZE), b""):
            h.update(chunk)
    return h.hexdigest()


def _download_resumable(url: str, part_path: Path, session: requests.Session) -> int:
    """Resumable download to <part_path>.  Returns the final byte count.

    If <part_path> already exists, sends a Range: bytes=<n>- header and appends
    the remaining bytes.  Falls back to a fresh download if the server replies
    with 200 instead of 206 (i.e. ignores the Range header).
    """
    part_path.parent.mkdir(parents=True, exist_ok=True)
    existing = part_path.stat().st_size if part_path.exists() else 0

    headers = {"User-Agent": USER_AGENT}
    if existing > 0:
        headers["Range"] = f"bytes={existing}-"
        log.info("Resuming download from byte %d: %s", existing, part_path.name)

    with session.get(url, headers=headers, stream=True, timeout=TIMEOUT_SECONDS) as r:
        r.raise_for_status()

        # If we asked for a Range but got a 200, the server doesn't support
        # resume — restart from zero.
        mode = "ab"
        if existing > 0 and r.status_code == 200:
            log.warning("Server ignored Range header — restarting download fresh: %s", url)
            existing = 0
            part_path.unlink(missing_ok=True)
            mode = "wb"

        total = int(r.headers.get("Content-Length", "0")) + existing
        pbar = tqdm(
            total=total if total > 0 else None,
            initial=existing,
            unit="B",
            unit_scale=True,
            unit_divisor=1024,
            desc=part_path.name,
            leave=False,
        )

        with part_path.open(mode) as f:
            for chunk in r.iter_content(chunk_size=CHUNK_SIZE):
                if not chunk:
                    continue
                f.write(chunk)
                pbar.update(len(chunk))
        pbar.close()

    return part_path.stat().st_size


def _verify_zip(zip_path: Path) -> bool:
    """Return True iff <zip_path> opens cleanly and its CRCs check out."""
    try:
        with zipfile.ZipFile(zip_path) as zf:
            bad = zf.testzip()
            if bad is not None:
                log.error("Zip CRC failure at %s in %s", bad, zip_path)
                return False
        return True
    except zipfile.BadZipFile as e:
        log.error("BadZipFile for %s: %s", zip_path, e)
        return False


def _extract_zip(zip_path: Path, dest_root: Path) -> None:
    """Extract <zip_path> into <dest_root>.  Caller must guarantee idempotency."""
    dest_root.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path) as zf:
        members = zf.namelist()
        for m in tqdm(members, desc=f"extract {zip_path.name}", leave=False):
            zf.extract(m, dest_root)


# =============================================================================
# orchestration
# =============================================================================

def fetch_archive(
    archive: Archive,
    raw_dir: Path,
    session: requests.Session,
    *,
    extract: bool,
    report: DownloadReport,
) -> None:
    """Download + verify + (optionally) extract one archive."""
    out_zip = archive.out_zip(raw_dir)
    part_zip = archive.part_zip(raw_dir)
    marker = archive.extracted_marker(raw_dir)
    extract_root = archive.extract_root(raw_dir)

    # ---- 1. download ----------------------------------------------------------
    if out_zip.exists():
        log.info("Already have %s (size=%d)", out_zip.name, out_zip.stat().st_size)
        status_download = "cached"
    else:
        try:
            _download_resumable(archive.url, part_zip, session)
        except requests.RequestException as e:
            log.error("Download failed for %s: %s", archive.url, e)
            report.add(archive, status="failed", size_bytes=0, sha256=None)
            return

        if not _verify_zip(part_zip):
            log.error("Zip verification failed — leaving .part for inspection: %s", part_zip)
            report.add(archive, status="failed", size_bytes=part_zip.stat().st_size, sha256=None)
            return

        os.replace(part_zip, out_zip)   # atomic rename
        status_download = "downloaded"

    size = out_zip.stat().st_size
    sha = _sha256_file(out_zip)
    log.info("OK %s  size=%d  sha256=%s", out_zip.name, size, sha[:16])

    # ---- 2. extract -----------------------------------------------------------
    if not extract:
        report.add(archive, status=status_download, size_bytes=size, sha256=sha)
        return

    if marker.exists():
        log.info("Already extracted: %s", archive.filename)
        report.add(archive, status="extracted", size_bytes=size, sha256=sha)
        return

    try:
        _extract_zip(out_zip, extract_root)
        marker.write_text(json.dumps({"sha256": sha, "size_bytes": size}, indent=2))
        report.add(archive, status="extracted", size_bytes=size, sha256=sha)
    except (zipfile.BadZipFile, OSError) as e:
        log.error("Extraction failed for %s: %s", out_zip, e)
        report.add(archive, status="failed", size_bytes=size, sha256=sha)


def select_archives(releases: Iterable[str]) -> list[Archive]:
    archives: list[Archive] = []
    for release in releases:
        if release not in RELEASES:
            raise SystemExit(f"Unknown release: {release}.  Pick from {list(RELEASES)}")
        for split, filename in RELEASES[release]:
            archives.append(Archive(release=release, split=split, filename=filename))
    return archives


# =============================================================================
# CLI
# =============================================================================

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="verse-download",
        description="Download the VerSe 2019 + VerSe 2020 BIDS-restructured archives.",
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
        "--skip_extract",
        action="store_true",
        help="Download only; leave zips packed.",
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
    archives = select_archives(releases)
    log.info("Planning download of %d archive(s) across %d release(s)", len(archives), len(releases))

    args.out_dir.mkdir(parents=True, exist_ok=True)
    report = DownloadReport()

    with requests.Session() as session:
        session.headers.update({"User-Agent": USER_AGENT})
        for archive in archives:
            log.info("=" * 70)
            log.info("Archive: %s/%s  (%s)", archive.release, archive.split, archive.filename)
            log.info("URL:     %s", archive.url)
            fetch_archive(
                archive,
                raw_dir=args.out_dir,
                session=session,
                extract=not args.skip_extract,
                report=report,
            )

    # ---- final manifest ------------------------------------------------------
    manifest_path = args.out_dir / "download_manifest.json"
    manifest_path.write_text(json.dumps({
        "archives": report.archives,
        "n_archives": len(report.archives),
        "out_dir": str(args.out_dir.resolve()),
        "version": "0.1.0",
    }, indent=2))
    log.info("Wrote manifest -> %s", manifest_path)

    n_failed = sum(1 for a in report.archives if a["status"] == "failed")
    if n_failed:
        log.error("%d / %d archives failed.  See %s.", n_failed, len(report.archives), manifest_path)
        return 1
    log.info("All %d archives OK.", len(report.archives))
    return 0


if __name__ == "__main__":
    sys.exit(main())
