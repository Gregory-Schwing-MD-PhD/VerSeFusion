"""Parse BIDS-style VerSe filenames.

VerSe uses a flat subset of BIDS entity-suffix-extension naming:

    sub-verse000_dir-orient_ct.nii.gz             (CT image)
    sub-verse000_dir-orient_seg-vert_msk.nii.gz   (vertebra segmentation)
    sub-verse000_dir-orient_seg-subreg_ctd.json   (centroid coordinates)
    sub-verse000_dir-orient_seg-vert_snp.png      (annotation preview PNG)

Some subjects carry an optional ``split`` entity (e.g. ``split-01``) when
they have multiple acquisitions.  The full layout per archive is::

    <archive>/{rawdata,derivatives}/sub-verseNNN/<file>

This module is intentionally I/O-free: it accepts paths or filenames and
produces typed records that downstream stages compose into manifests.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

# =============================================================================
# regex
# =============================================================================

# BIDS-ish: key1-val1_key2-val2_..._suffix.ext (.ext may be .nii.gz / .json / .png)
_ENTITY_RE = re.compile(r"^(?P<key>[a-zA-Z0-9]+)-(?P<val>[a-zA-Z0-9]+)$")

_KNOWN_SUFFIXES: dict[str, tuple[str, ...]] = {
    "ct.nii.gz":          ("ct",          "nii.gz"),
    "msk.nii.gz":         ("msk",         "nii.gz"),
    "ctd.json":           ("ctd",         "json"),
    "snp.png":            ("snp",         "png"),
}

FileKind = Literal["ct", "msk", "ctd", "snp"]


# =============================================================================
# data
# =============================================================================

@dataclass(frozen=True)
class BIDSName:
    """Structured view of a VerSe BIDS-style filename."""

    subject:     str                 # "verse000"
    entities:    dict[str, str]      # e.g. {"dir": "orient", "split": "01", "seg": "vert"}
    kind:        FileKind            # "ct" | "msk" | "ctd" | "snp"
    extension:   str                 # "nii.gz" | "json" | "png"
    original:    str                 # the raw basename

    # ----- ergonomic accessors ------------------------------------------------
    @property
    def sub_id(self) -> str:
        """Canonical subject id, ``sub-<subject>``."""
        return f"sub-{self.subject}"

    @property
    def split(self) -> str | None:
        return self.entities.get("split")

    @property
    def direction(self) -> str | None:
        return self.entities.get("dir")

    @property
    def seg(self) -> str | None:
        return self.entities.get("seg")

    @property
    def is_image(self) -> bool:
        return self.kind == "ct"

    @property
    def is_mask(self) -> bool:
        return self.kind == "msk"

    @property
    def is_centroid_json(self) -> bool:
        return self.kind == "ctd"


# =============================================================================
# parser
# =============================================================================

def parse_bids_name(name: str | Path) -> BIDSName:
    """Parse one BIDS-style basename.

    Raises ``ValueError`` if the name does not match the VerSe layout.
    """
    basename = Path(name).name

    # Identify the longest matching suffix.
    suffix_match: tuple[str, str] | None = None
    stem: str | None = None
    for marker, (kind, ext) in _KNOWN_SUFFIXES.items():
        if basename.endswith(f"_{marker}"):
            stem = basename[: -(len(marker) + 1)]  # strip "_<marker>"
            suffix_match = (kind, ext)
            break

    if suffix_match is None or stem is None:
        raise ValueError(f"Unrecognised VerSe suffix in: {basename}")

    kind, ext = suffix_match

    # The stem is now "sub-verseNNN_<entity1>_<entity2>_..._seg-<region>" (or
    # without the seg- if the file is the CT itself).
    parts = stem.split("_")
    if not parts or not parts[0].startswith("sub-"):
        raise ValueError(f"Missing sub-* prefix in: {basename}")

    subject = parts[0].removeprefix("sub-")
    entities: dict[str, str] = {}
    for token in parts[1:]:
        m = _ENTITY_RE.match(token)
        if not m:
            raise ValueError(f"Malformed entity token {token!r} in: {basename}")
        entities[m.group("key")] = m.group("val")

    return BIDSName(
        subject=subject,
        entities=entities,
        kind=kind,  # type: ignore[arg-type]
        extension=ext,
        original=basename,
    )


# =============================================================================
# subject grouping
# =============================================================================

@dataclass(frozen=True)
class SubjectFiles:
    """The four canonical files for one VerSe subject (some may be missing)."""

    subject:  str
    ct:       Path | None
    msk:      Path | None
    ctd:      Path | None
    snp:      Path | None

    @property
    def sub_id(self) -> str:
        return f"sub-{self.subject}"

    @property
    def is_complete(self) -> bool:
        return all(p is not None for p in (self.ct, self.msk, self.ctd))


def discover_subjects(release_root: Path) -> dict[str, SubjectFiles]:
    """Walk an extracted VerSe archive root and group files by subject.

    Expected layout::

        <release_root>/rawdata/sub-verseNNN/*.nii.gz
        <release_root>/derivatives/sub-verseNNN/*.{nii.gz,json,png}

    Returns ``{subject_id: SubjectFiles}``.  ``release_root`` may be the split
    root (``.../verse20/training``) or any directory that contains ``rawdata``
    and ``derivatives`` siblings.
    """
    rawdata = release_root / "rawdata"
    derivatives = release_root / "derivatives"

    accum: dict[str, dict[FileKind, Path]] = {}

    def _ingest(root: Path) -> None:
        if not root.is_dir():
            return
        for path in root.rglob("*"):
            if not path.is_file():
                continue
            try:
                bids = parse_bids_name(path.name)
            except ValueError:
                continue
            accum.setdefault(bids.subject, {})[bids.kind] = path

    _ingest(rawdata)
    _ingest(derivatives)

    subjects: dict[str, SubjectFiles] = {}
    for sub, files in accum.items():
        subjects[sub] = SubjectFiles(
            subject=sub,
            ct=files.get("ct"),
            msk=files.get("msk"),
            ctd=files.get("ctd"),
            snp=files.get("snp"),
        )
    return subjects
