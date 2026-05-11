"""Unit tests for verse_pipeline.label_crosswalk."""

from __future__ import annotations

import numpy as np

from verse_pipeline.label_crosswalk import apply_mapping, load_crosswalk


# =============================================================================
# tests — loader
# =============================================================================

def test_load_crosswalk_returns_both_directions():
    fwd, rev = load_crosswalk()
    assert isinstance(fwd, dict)
    assert isinstance(rev, dict)
    # Sanity: forward must include L1..L6 and sacrum
    for verse_label in (20, 21, 22, 23, 24, 25, 26):
        assert verse_label in fwd
    # Reverse must include CTSpinoPelvic1K 1..7
    for ctspp_label in (1, 2, 3, 4, 5, 6, 7):
        assert ctspp_label in rev


def test_load_crosswalk_lumbar_mapping_is_offset_by_19():
    fwd, _ = load_crosswalk()
    for verse_label in range(20, 26):
        assert fwd[verse_label] == verse_label - 19


def test_load_crosswalk_lstv_is_six():
    fwd, _ = load_crosswalk()
    assert fwd[25] == 6


def test_load_crosswalk_sacrum_maps_to_seven():
    fwd, _ = load_crosswalk()
    assert fwd[26] == 7


def test_load_crosswalk_reverse_drops_hip_labels():
    _, rev = load_crosswalk()
    # CTSpinoPelvic1K labels 8 and 9 are hips; they have no VerSe equivalent.
    assert 8 not in rev
    assert 9 not in rev


# =============================================================================
# tests — apply_mapping
# =============================================================================

def test_apply_mapping_lumbar_only():
    fwd, _ = load_crosswalk()
    mask = np.array([
        [0, 20, 21],
        [22, 23, 24],
        [25, 0, 0],
    ], dtype=np.int32)
    out = apply_mapping(mask, fwd)
    expected = np.array([
        [0, 1, 2],
        [3, 4, 5],
        [6, 0, 0],
    ], dtype=np.uint8)
    np.testing.assert_array_equal(out, expected)


def test_apply_mapping_drops_unmapped_labels_to_background():
    fwd, _ = load_crosswalk()
    # 5 (cervical C5) and 12 (thoracic T5) are not in the forward map → 0.
    mask = np.array([[5, 12, 28]], dtype=np.int32)
    out = apply_mapping(mask, fwd)
    np.testing.assert_array_equal(out, np.array([[0, 0, 0]], dtype=np.uint8))


def test_apply_mapping_handles_zero_max_volume():
    fwd, _ = load_crosswalk()
    mask = np.zeros((3, 3), dtype=np.int32)
    out = apply_mapping(mask, fwd)
    np.testing.assert_array_equal(out, mask.astype(np.uint8))


def test_apply_mapping_output_dtype_is_uint8():
    fwd, _ = load_crosswalk()
    mask = np.array([20, 25], dtype=np.int32)
    out = apply_mapping(mask, fwd)
    assert out.dtype == np.uint8


def test_apply_mapping_custom_default():
    fwd, _ = load_crosswalk()
    # Use 255 as the default for unmapped labels — useful in QA to spot
    # voxels that fell through the crosswalk.
    mask = np.array([5, 20], dtype=np.int32)
    out = apply_mapping(mask, fwd, default=255)
    np.testing.assert_array_equal(out, np.array([255, 1], dtype=np.uint8))
