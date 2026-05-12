"""Parse MICCAI-challenge-format VerSe filenames.

The MICCAI release of VerSe (osf.io/923ap for v19, osf.io/b2wxj for v20)
uses flat per-image filenames rather than the BIDS subject-based layout:

    Image:          verseNNN.nii.gz             (or verseNNN_CT-orient.nii.gz in v20)
    Mask:           verseNNN_seg.nii.gz         (or verseNNN_CT-orient_seg.nii.gz)
    Centroids:      verseNNN_ctd.json
    Snapshot:       verseNNN_snapshot.png

Glocker-cohort subjects use ``gl`` instead of ``verse``:

    Image:          gl003.nii.gz
    Mask:           gl003_seg.nii.gz
    ...

VerSe19 uses the short form (``verse014.nii.gz``).  VerSe20 sometimes adds a
``_CT-orient`` infix; this module accepts both.  Returns the bare series ID
(e.g. ``verse014`` or ``gl003``) regardless of which variant the filename used.

Used by the unify stage to group MICCAI files into per-scan tuples and by the
downloader to validate that every expected file has a partner.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path


# Series ID core: "verse" or "gl" followed by 1-4 digits.
_SERIES_CORE_RE = re.compile(r"^(verse\d+|gl\d+)")

# Canonical kinds derived from MICCAI filename suffix patterns.
_SUFFIX_PATTERNS: tuple[tuple[re.Pattern[str], str], ...] = (
    # snapshot first (most specific suffix)
    (re.compile(r"(?:_CT-orient)?_snapshot\.png$"),               "snp"),
    # centroid
    (re.compile(r"(?:_CT-orient)?_ctd\.json$"),                   "ctd"),
    # mask (must come before bare image regex, since "_seg" is more specific)
    (re.compile(r"(?:_CT-orient)?_seg\.nii\.gz$"),                "msk"),
    # bare image — must NOT contain _seg, _ctd, _snapshot
    (re.compile(r"(?:_CT-orient)?\.nii\.gz$"),                    "ct"),
)


@dataclass(frozen=True)
class MICCAIFile:
    """One MICCAI-format file with its parsed identifiers."""
    path:      Path        # full path on disk (may not exist yet during planning)
    series_id: str         # e.g. "verse014", "gl003"
    kind:      str         # one of: "ct", "msk", "ctd", "snp"


def parse_filename(name: str) -> tuple[str, str] | None:
    """Parse a MICCAI filename; return (series_id, kind) or None if unrecognized.

    Examples
    --------
    >>> parse_filename("verse014.nii.gz")
    ('verse014', 'ct')
    >>> parse_filename("verse014_seg.nii.gz")
    ('verse014', 'msk')
    >>> parse_filename("verse014_ctd.json")
    ('verse014', 'ctd')
    >>> parse_filename("verse014_snapshot.png")
    ('verse014', 'snp')
    >>> parse_filename("verse500_CT-orient.nii.gz")
    ('verse500', 'ct')
    >>> parse_filename("verse500_CT-orient_seg.nii.gz")
    ('verse500', 'msk')
    >>> parse_filename("gl003.nii.gz")
    ('gl003', 'ct')
    >>> parse_filename("readme.txt") is None
    True
    """
    m = _SERIES_CORE_RE.match(name)
    if not m:
        return None
    series_id = m.group(1)
    remainder = name[m.end():]

    for pattern, kind in _SUFFIX_PATTERNS:
        if pattern.fullmatch(remainder):
            return (series_id, kind)
    return None


def parse_path(path: Path) -> MICCAIFile | None:
    """Return a structured ``MICCAIFile`` for the given path, or ``None``."""
    parsed = parse_filename(path.name)
    if not parsed:
        return None
    series_id, kind = parsed
    return MICCAIFile(path=path, series_id=series_id, kind=kind)


def required_kinds() -> tuple[str, ...]:
    """The three file kinds required for downstream segmentation training."""
    return ("ct", "msk", "ctd")
