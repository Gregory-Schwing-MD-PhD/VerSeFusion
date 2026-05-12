"""Parse MICCAI-challenge-format VerSe filenames (with BIDS-fallback support).

VerSe data lives on OSF in two parallel formats.  This module parses both so
the unify stage can ingest files from either source uniformly.

MICCAI-challenge format (the primary source: osf.io/923ap, osf.io/b2wxj)
------------------------------------------------------------------------

VerSe 2019 (clean, predictable):

    verse014.nii.gz                         (image)
    verse014_seg.nii.gz                     (mask)
    verse014_ctd.json                       (centroids)
    verse014_snapshot.png                   (preview)

VerSe 2020 (multiple orientation infixes, different ctd/snp suffixes):

    verse512_CT-iso.nii.gz                  (image, sagittal-iso resampled)
    verse512_CT-iso_seg.nii.gz              (mask)
    verse512_CT-iso_iso-ctd.json            (centroids, 1mm-iso ASL space)
    verse512_CT-iso.png                     (preview)
    verse512_CT-iso-w.png                   (wireframe variant — NOT consumed)

VerSe 2020 also re-releases multi-series patient scans with the canonical
patient ID prefixed to each filename:

    verse400_verse090_CT-iso.nii.gz         (patient verse400, scan verse090)
    verse400_verse090_CT-iso_iso-ctd.json
    ...

BIDS subject-based format (fallback source: osf.io/jtfa5, osf.io/4skx2)
-----------------------------------------------------------------------

Used by download.py's auto-recovery for subjects missing from MICCAI:

    sub-verse500_dir-iso_ct.nii.gz                 (image)
    sub-verse500_dir-iso_seg-vert_msk.nii.gz       (mask)
    sub-verse500_dir-iso_seg-subreg_ctd.json       (centroids in voxel space!)
    sub-verse500_dir-iso_seg-vert_snp.png          (preview)

Multi-series patients in BIDS:

    sub-verse400_split-verse090_dir-iso_ct.nii.gz  (canonical+series IDs)

We accept BIDS filenames with or without the leading ``sub-`` prefix; the
prefix gets stripped during parsing.  Note that BIDS centroids are in
per-image voxel space, while MICCAI centroids are in 1 mm isotropic ASL
space — chunk 2's reorient stage dispatches on this when transforming.

Conventions handled
-------------------
Series IDs:
  - Lower- or upper-case ``verse\\d+`` or ``gl\\d+``
  - Optional ``sub-`` prefix (BIDS)
  - Optional ``split-`` prefix on the second series ID (BIDS multi-series)
  - Always normalised to lowercase on return

Orientation / entity infixes (zero or more, in any order):
  - ``_CT-iso``, ``_CT-sag``, ``_CT-ax``, ``_CT-cor``  (MICCAI v20)
  - ``_CT_ax``                                          (MICCAI v20 underscore variant)
  - ``_dir-iso``, ``_dir-sag``, ``_dir-ax``             (BIDS direction entity)
  - ``_seg-vert``, ``_seg-subreg``                      (BIDS segmentation-type entity)
  - Any other ``_<key>[-_]<value>``                     (defensive)

Kind suffix variants:
  - Image:    ``.nii.gz`` (MICCAI bare) or ``_ct.nii.gz`` (BIDS)
  - Mask:     ``_seg.nii.gz`` (MICCAI) or ``_msk.nii.gz`` (BIDS)
  - Centroid: ``_ctd.json`` (v19) / ``_iso-ctd.json`` (MICCAI v20) / ``_ctd.json`` (BIDS)
  - Snapshot: ``_snapshot.png`` (v19) / ``.png`` (MICCAI v20) / ``_snp.png`` (BIDS)

Wireframe variants (``-w.png``) are intentionally NOT matched.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

# Series ID extractor.  Captures up to two consecutive verseN/glN tokens at
# the start of the filename, with optional BIDS-style ``sub-`` and ``split-``
# prefixes:
#   one ID  → single-series scan: a == series_id
#   two IDs → multi-series scan: a is the canonical patient_id, b is the
#                                original MICCAI series_id
# Case-insensitive so GL003, gl003, Verse014 all parse the same.
_SERIES_RE = re.compile(
    r"^(?:sub-)?"
    r"(?P<a>verse\d+|gl\d+)"
    r"(?:_(?:split-)?(?P<b>verse\d+|gl\d+))?"
    r"(?P<rest>.*)$",
    re.IGNORECASE,
)

# Entity infix: zero or more BIDS-style or MICCAI-style entity tags.
# Each segment is _<key><sep><value> where:
#   key:   one letter followed by optional alphanumeric (e.g. "CT", "dir", "seg")
#   sep:   - or _
#   value: alphanumeric (e.g. "iso", "vert", "ax")
_INFIX = r"(?:_[A-Za-z][A-Za-z0-9]*[-_][A-Za-z0-9]+)*"

# Kind suffix patterns.  Order matters when two patterns could match the same
# tail; more specific patterns first.  The infix is shared across all of them.
_SUFFIX_PATTERNS: tuple[tuple[re.Pattern[str], str], ...] = (
    # snapshot variants
    (re.compile(_INFIX + r"_snp\.png$"),                "snp"),  # BIDS
    (re.compile(_INFIX + r"_snapshot\.png$"),           "snp"),  # MICCAI v19
    (re.compile(_INFIX + r"\.png$"),                    "snp"),  # MICCAI v20
    # centroid variants
    (re.compile(_INFIX + r"_iso-ctd\.json$"),           "ctd"),  # MICCAI v20
    (re.compile(_INFIX + r"_ctd\.json$"),               "ctd"),  # v19 + BIDS
    # mask variants
    (re.compile(_INFIX + r"_msk\.nii\.gz$"),            "msk"),  # BIDS
    (re.compile(_INFIX + r"_seg\.nii\.gz$"),            "msk"),  # MICCAI
    # image variants
    (re.compile(_INFIX + r"_ct\.nii\.gz$"),             "ct"),   # BIDS
    (re.compile(_INFIX + r"\.nii\.gz$"),                "ct"),   # MICCAI
)


@dataclass(frozen=True)
class MICCAIFile:
    """One parsed VerSe file with its source-agnostic identifiers."""
    path:      Path
    series_id: str     # always lowercase (e.g. "verse014", "gl003", "verse090")
    kind:      str     # one of: "ct", "msk", "ctd", "snp"


def parse_filename(name: str) -> tuple[str, str] | None:
    """Parse a VerSe filename (MICCAI or BIDS); return (series_id, kind) or None.

    Series ID is always returned in lowercase.  For multi-series patient
    files (e.g. ``verse400_verse090_CT-iso.nii.gz`` or BIDS-style
    ``sub-verse400_split-verse090_dir-iso_ct.nii.gz``), the *second* ID
    is returned as the series_id — that's the original MICCAI scan
    identifier we use as the canonical key.

    Examples
    --------
    MICCAI v19 single-series:

    >>> parse_filename("verse014.nii.gz")
    ('verse014', 'ct')
    >>> parse_filename("verse014_seg.nii.gz")
    ('verse014', 'msk')
    >>> parse_filename("verse014_ctd.json")
    ('verse014', 'ctd')
    >>> parse_filename("verse014_snapshot.png")
    ('verse014', 'snp')

    MICCAI v20 single-series with orientation infixes:

    >>> parse_filename("verse512_CT-iso.nii.gz")
    ('verse512', 'ct')
    >>> parse_filename("verse512_CT-iso_seg.nii.gz")
    ('verse512', 'msk')
    >>> parse_filename("verse512_CT-iso_iso-ctd.json")
    ('verse512', 'ctd')
    >>> parse_filename("verse512_CT-iso.png")
    ('verse512', 'snp')
    >>> parse_filename("verse580_CT-sag_iso-ctd.json")
    ('verse580', 'ctd')

    MICCAI v20 Glocker (uppercase in filename, normalised to lowercase):

    >>> parse_filename("GL003.nii.gz")
    ('gl003', 'ct')
    >>> parse_filename("GL017_CT_ax.nii.gz")
    ('gl017', 'ct')
    >>> parse_filename("GL090_CT-ax_iso-ctd.json")
    ('gl090', 'ctd')
    >>> parse_filename("GL003_iso-ctd.json")
    ('gl003', 'ctd')

    MICCAI v20 multi-series patient files:

    >>> parse_filename("verse400_verse090_CT-iso.nii.gz")
    ('verse090', 'ct')
    >>> parse_filename("verse400_verse155_CT-iso_seg.nii.gz")
    ('verse155', 'msk')
    >>> parse_filename("verse403_verse208_CT-sag_iso-ctd.json")
    ('verse208', 'ctd')

    BIDS-format fallback files (with sub- prefix, dir/seg entities):

    >>> parse_filename("sub-verse500_dir-iso_ct.nii.gz")
    ('verse500', 'ct')
    >>> parse_filename("sub-verse500_dir-iso_seg-vert_msk.nii.gz")
    ('verse500', 'msk')
    >>> parse_filename("sub-verse500_dir-iso_seg-subreg_ctd.json")
    ('verse500', 'ctd')
    >>> parse_filename("sub-verse500_dir-iso_seg-vert_snp.png")
    ('verse500', 'snp')

    BIDS-format with sub- already stripped (e.g. by an upstream rename):

    >>> parse_filename("verse500_dir-iso_ct.nii.gz")
    ('verse500', 'ct')
    >>> parse_filename("verse500_dir-iso_seg-vert_msk.nii.gz")
    ('verse500', 'msk')

    BIDS multi-series patients:

    >>> parse_filename("sub-verse400_split-verse090_dir-iso_ct.nii.gz")
    ('verse090', 'ct')

    Wireframe variants (-w.png) are intentionally NOT matched:

    >>> parse_filename("verse512_CT-iso-w.png") is None
    True
    >>> parse_filename("GL003-w.png") is None
    True

    Random files NOT matched:

    >>> parse_filename("readme.txt") is None
    True
    >>> parse_filename("license.txt") is None
    True
    >>> parse_filename("dataset_description.json") is None
    True
    """
    m = _SERIES_RE.match(name)
    if not m:
        return None

    a = m.group("a").lower()
    b = m.group("b")
    rest = m.group("rest")

    # Multi-series files (MICCAI or BIDS) have BOTH a patient ID and a series
    # ID; the second (b) is the original MICCAI series identifier we want.
    series_id = b.lower() if b is not None else a

    for pattern, kind in _SUFFIX_PATTERNS:
        if pattern.fullmatch(rest):
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
