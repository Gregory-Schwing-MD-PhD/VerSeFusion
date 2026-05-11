"""Unit tests for verse_pipeline.lstv."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from verse_pipeline.lstv import LSTV_LABEL, T13_LABEL, flags_from_centroid
from verse_pipeline.utils.centroid_json import parse_centroid_json


# =============================================================================
# helpers
# =============================================================================

def _write_ctd(path: Path, labels: list[int]) -> Path:
    payload: list[dict] = [{"direction": "PIR"}]
    for i, lbl in enumerate(labels):
        payload.append({"label": lbl, "X": float(i), "Y": float(i), "Z": float(i)})
    path.write_text(json.dumps(payload))
    return path


# =============================================================================
# fixtures
# =============================================================================

@pytest.fixture
def normal_subject(tmp_path: Path) -> Path:
    # T1..T12 + L1..L5 (no anomaly)
    labels = list(range(8, 25))  # 8..24 inclusive
    return _write_ctd(tmp_path / "normal_ctd.json", labels)


@pytest.fixture
def lstv_subject(tmp_path: Path) -> Path:
    labels = list(range(8, 25)) + [LSTV_LABEL]   # add L6
    return _write_ctd(tmp_path / "lstv_ctd.json", labels)


@pytest.fixture
def t13_subject(tmp_path: Path) -> Path:
    labels = list(range(8, 25)) + [T13_LABEL]    # add T13
    return _write_ctd(tmp_path / "t13_ctd.json", labels)


@pytest.fixture
def both_subject(tmp_path: Path) -> Path:
    labels = list(range(8, 25)) + [LSTV_LABEL, T13_LABEL]
    return _write_ctd(tmp_path / "both_ctd.json", labels)


# =============================================================================
# tests
# =============================================================================

def test_flags_normal(normal_subject: Path):
    file = parse_centroid_json(normal_subject)
    flags = flags_from_centroid(file)
    assert flags.is_normal is True
    assert flags.has_lstv is False
    assert flags.has_t13 is False
    assert flags.category == "normal"
    assert flags.lumbar_n == 5         # L1..L5
    assert flags.thoracic_n == 12      # T1..T12
    assert flags.cervical_n == 0


def test_flags_lstv(lstv_subject: Path):
    flags = flags_from_centroid(parse_centroid_json(lstv_subject))
    assert flags.has_lstv is True
    assert flags.has_t13 is False
    assert flags.is_normal is False
    assert flags.category == "lstv"
    assert flags.lumbar_n == 6         # L1..L6


def test_flags_t13(t13_subject: Path):
    flags = flags_from_centroid(parse_centroid_json(t13_subject))
    assert flags.has_lstv is False
    assert flags.has_t13 is True
    assert flags.category == "t13"
    # T13 counts as thoracic
    assert flags.thoracic_n == 13


def test_flags_both(both_subject: Path):
    flags = flags_from_centroid(parse_centroid_json(both_subject))
    assert flags.has_lstv is True
    assert flags.has_t13 is True
    assert flags.is_normal is False
    assert flags.category == "both"


def test_label_constants():
    # Sentinel values must match the configs/default.env contract.
    assert LSTV_LABEL == 25
    assert T13_LABEL == 28
