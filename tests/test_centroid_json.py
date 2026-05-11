"""Unit tests for verse_pipeline.utils.centroid_json."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from verse_pipeline.utils.centroid_json import (
    Centroid,
    CentroidFile,
    has_label,
    parse_centroid_json,
    vertebra_count,
    write_centroid_json,
)


# =============================================================================
# fixtures
# =============================================================================

@pytest.fixture
def canonical_ctd_path(tmp_path: Path) -> Path:
    payload = [
        {"direction": "PIR"},
        {"label": 20, "X": 64.5, "Y": 132.2, "Z": 45.8},
        {"label": 21, "X": 62.1, "Y": 145.7, "Z": 46.4},
        {"label": 22, "X": 60.0, "Y": 159.0, "Z": 47.0},
        {"label": 25, "X": 55.0, "Y": 200.0, "Z": 50.0},  # L6 = LSTV
    ]
    p = tmp_path / "sub-verse001_ctd.json"
    p.write_text(json.dumps(payload))
    return p


@pytest.fixture
def lowercase_ctd_path(tmp_path: Path) -> Path:
    payload = [
        {"dir": "PIR"},
        {"label": 20, "x": 64.5, "y": 132.2, "z": 45.8},
        {"label": 28, "x": 30.0, "y": 50.0,  "z": 40.0},  # T13
    ]
    p = tmp_path / "sub-verse002_ctd.json"
    p.write_text(json.dumps(payload))
    return p


@pytest.fixture
def wrapped_ctd_path(tmp_path: Path) -> Path:
    payload = {
        "direction": "PIR",
        "verts": [
            {"label": 20, "X": 1.0, "Y": 2.0, "Z": 3.0},
            {"label": 21, "X": 4.0, "Y": 5.0, "Z": 6.0},
        ],
    }
    p = tmp_path / "sub-verse003_ctd.json"
    p.write_text(json.dumps(payload))
    return p


# =============================================================================
# tests
# =============================================================================

def test_parse_canonical(canonical_ctd_path: Path):
    file = parse_centroid_json(canonical_ctd_path)
    assert file.direction == "PIR"
    assert file.n_vertebrae == 4
    assert file.labels == [20, 21, 22, 25]
    assert file.centroids[0] == Centroid(label=20, x=64.5, y=132.2, z=45.8)


def test_parse_lowercase_keys(lowercase_ctd_path: Path):
    file = parse_centroid_json(lowercase_ctd_path)
    assert file.direction == "PIR"
    assert file.labels == [20, 28]


def test_parse_verts_wrapped(wrapped_ctd_path: Path):
    file = parse_centroid_json(wrapped_ctd_path)
    assert file.direction == "PIR"
    assert file.n_vertebrae == 2
    assert file.labels == [20, 21]


def test_has_label(canonical_ctd_path: Path):
    file = parse_centroid_json(canonical_ctd_path)
    assert has_label(file, 25) is True   # LSTV
    assert has_label(file, 28) is False  # no T13


def test_vertebra_count_by_region(canonical_ctd_path: Path):
    file = parse_centroid_json(canonical_ctd_path)
    assert vertebra_count(file)             == 4
    assert vertebra_count(file, "lumbar")   == 4
    assert vertebra_count(file, "thoracic") == 0
    assert vertebra_count(file, "cervical") == 0


def test_vertebra_count_invalid_region(canonical_ctd_path: Path):
    file = parse_centroid_json(canonical_ctd_path)
    with pytest.raises(ValueError, match="region"):
        vertebra_count(file, "pelvis")


def test_roundtrip(canonical_ctd_path: Path, tmp_path: Path):
    file = parse_centroid_json(canonical_ctd_path)
    out = tmp_path / "out.json"
    write_centroid_json(out, file)

    reloaded = parse_centroid_json(out)
    assert reloaded.direction == file.direction
    assert reloaded.labels == file.labels
    assert len(reloaded.centroids) == len(file.centroids)


def test_empty_centroid_file(tmp_path: Path):
    p = tmp_path / "empty.json"
    p.write_text(json.dumps([{"direction": "PIR"}]))
    file = parse_centroid_json(p)
    assert file.n_vertebrae == 0
    assert file.labels == []


def test_unrecognised_structure_raises(tmp_path: Path):
    p = tmp_path / "bad.json"
    p.write_text(json.dumps(42))   # neither list nor dict-with-verts
    with pytest.raises(ValueError, match="structure"):
        parse_centroid_json(p)
