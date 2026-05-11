"""Read and write the VerSe centroid annotation JSON.

VerSe ships per-subject centroid coordinates alongside each segmentation
mask::

    sub-verse000_dir-orient_seg-subreg_ctd.json

The file is a JSON array whose first element is a small header record (the
orientation triple, e.g. ``{"direction": "PIR"}``) followed by one record
per labelled vertebra::

    [
      {"direction": "PIR"},
      {"label": 20, "X":  64.5, "Y": 132.2, "Z":  45.8},
      {"label": 21, "X":  62.1, "Y": 145.7, "Z":  46.4},
      ...
    ]

Coordinates are in **voxels** in the image space of the corresponding CT.
Labels follow the VerSe 28-class scheme (see ``configs/label_scheme.yaml``).

Some files in the wild omit the header; some swap key casing
(``"X"``/``"x"``); some have an extra ``"verts"`` wrapper.  This parser
normalises all of those.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any

# =============================================================================
# dataclasses
# =============================================================================

@dataclass(frozen=True)
class Centroid:
    """One labelled vertebra centroid in voxel space."""
    label: int
    x: float
    y: float
    z: float

    def as_dict(self) -> dict[str, Any]:
        return {"label": self.label, "X": self.x, "Y": self.y, "Z": self.z}


@dataclass(frozen=True)
class CentroidFile:
    """Complete parsed centroid JSON for a single subject."""
    direction: str | None             # e.g. "PIR" — the orientation the coords are in
    centroids: tuple[Centroid, ...]
    raw_path:  Path | None = None

    @property
    def labels(self) -> list[int]:
        return [c.label for c in self.centroids]

    @property
    def n_vertebrae(self) -> int:
        return len(self.centroids)


# =============================================================================
# parse
# =============================================================================

def _norm_key(d: dict[str, Any], options: tuple[str, ...]) -> Any:
    """Look up first matching key (case-insensitively) from a dict."""
    lc = {k.lower(): v for k, v in d.items()}
    for opt in options:
        if opt.lower() in lc:
            return lc[opt.lower()]
    return None


def parse_centroid_json(path: Path | str) -> CentroidFile:
    """Parse a VerSe centroid JSON, tolerating known variants.

    Raises ``ValueError`` if the file is structurally unusable.
    """
    p = Path(path)
    with p.open() as f:
        raw = json.load(f)

    # Some files wrap everything in {"verts": [...]}.
    if isinstance(raw, dict) and "verts" in raw:
        records = raw["verts"]
        # Direction can also live at top-level.
        top_direction = raw.get("direction") or raw.get("dir")
    elif isinstance(raw, list):
        records = raw
        top_direction = None
    else:
        raise ValueError(f"Unrecognised centroid JSON structure in {p}")

    direction: str | None = top_direction
    centroids: list[Centroid] = []
    for rec in records:
        if not isinstance(rec, dict):
            continue

        # Header record carries the orientation.
        d = _norm_key(rec, ("direction", "dir"))
        if d is not None and "label" not in {k.lower() for k in rec}:
            direction = str(d)
            continue

        label = _norm_key(rec, ("label",))
        x = _norm_key(rec, ("X", "x"))
        y = _norm_key(rec, ("Y", "y"))
        z = _norm_key(rec, ("Z", "z"))
        if label is None or x is None or y is None or z is None:
            # Skip malformed rows rather than blow up the whole file.
            continue

        centroids.append(Centroid(
            label=int(label),
            x=float(x),
            y=float(y),
            z=float(z),
        ))

    return CentroidFile(
        direction=direction,
        centroids=tuple(centroids),
        raw_path=p,
    )


# =============================================================================
# write
# =============================================================================

def write_centroid_json(path: Path | str, file: CentroidFile) -> None:
    """Serialise a CentroidFile back to disk in the canonical VerSe layout."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)

    out: list[dict[str, Any]] = []
    if file.direction is not None:
        out.append({"direction": file.direction})
    out.extend(c.as_dict() for c in file.centroids)

    with p.open("w") as f:
        json.dump(out, f, indent=2)


# =============================================================================
# label-set queries (used by lstv.py / manifest.py)
# =============================================================================

def has_label(file: CentroidFile, label: int) -> bool:
    return any(c.label == label for c in file.centroids)


def vertebra_count(file: CentroidFile, region: str | None = None) -> int:
    """Count centroids in a region (cervical / thoracic / lumbar) or all."""
    if region is None:
        return file.n_vertebrae

    bands = {
        "cervical": range(1, 8),    # 1-7
        "thoracic": list(range(8, 20)) + [28],   # 8-19 + T13
        "lumbar":   range(20, 26),  # 20-25 (L1-L6)
    }
    if region not in bands:
        raise ValueError(f"Unknown region {region!r}.  Pick from {list(bands)}.")

    band = set(bands[region])
    return sum(1 for c in file.centroids if c.label in band)
