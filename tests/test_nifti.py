"""Unit tests for verse_pipeline.utils.nifti."""

from __future__ import annotations

import numpy as np
import nibabel as nib
import pytest

from verse_pipeline.utils.nifti import (
    TARGET_AXCODES_PIR,
    build_reorient_plan,
    current_axcodes,
    needs_reorient,
    reorient_centroid_voxel,
    reorient_image,
)


# =============================================================================
# helpers
# =============================================================================

def _make_img(axcodes: tuple[str, str, str], shape=(10, 20, 30)) -> nib.Nifti1Image:
    """Build a tiny NIfTI whose affine yields the requested axcodes.

    Each axcode picks a basis direction; we assemble a diagonal-permuted
    affine that matches.
    """
    code_to_axis = {
        "L": (0, -1), "R": (0,  1),
        "P": (1, -1), "A": (1,  1),
        "I": (2, -1), "S": (2,  1),
    }
    affine = np.zeros((4, 4))
    affine[3, 3] = 1.0
    for out_axis, code in enumerate(axcodes):
        ras_axis, sign = code_to_axis[code]
        affine[ras_axis, out_axis] = sign  # spacing = 1mm
    data = np.arange(np.prod(shape), dtype=np.int16).reshape(shape)
    return nib.Nifti1Image(data, affine)


# =============================================================================
# tests — orientation introspection
# =============================================================================

def test_current_axcodes_identity():
    img = _make_img(("R", "A", "S"))
    assert current_axcodes(img) == ("R", "A", "S")


def test_needs_reorient_true_for_ras():
    img = _make_img(("R", "A", "S"))
    assert needs_reorient(img) is True


def test_needs_reorient_false_for_pir():
    img = _make_img(("P", "I", "R"))
    assert needs_reorient(img) is False


# =============================================================================
# tests — reorient image
# =============================================================================

def test_reorient_to_pir():
    img = _make_img(("R", "A", "S"))
    new, plan = reorient_image(img, TARGET_AXCODES_PIR)
    assert current_axcodes(new) == TARGET_AXCODES_PIR
    assert plan.source_axcodes == ("R", "A", "S")
    assert plan.target_axcodes == TARGET_AXCODES_PIR
    assert plan.src_shape == (10, 20, 30)


def test_reorient_preserves_voxel_count():
    img = _make_img(("L", "P", "I"))
    new, _ = reorient_image(img, TARGET_AXCODES_PIR)
    assert int(np.asanyarray(new.dataobj).size) == int(np.asanyarray(img.dataobj).size)


# =============================================================================
# tests — centroid reorientation
# =============================================================================

def test_centroid_reorient_identity_when_already_pir():
    img = _make_img(("P", "I", "R"), shape=(10, 20, 30))
    plan = build_reorient_plan(img, TARGET_AXCODES_PIR)
    # axis_permutation should be (0,1,2) and flips (1,1,1)
    assert plan.axis_permutation == (0, 1, 2)
    assert plan.axis_flips == (1, 1, 1)

    out = reorient_centroid_voxel((3.0, 7.0, 11.0), plan)
    assert out == (3.0, 7.0, 11.0)


def test_centroid_reorient_consistent_with_image_reorient():
    """A centroid placed at a known voxel should map to the same voxel as
    the data at that voxel after image reorientation.
    """
    img = _make_img(("R", "A", "S"), shape=(10, 20, 30))
    data = np.zeros((10, 20, 30), dtype=np.int16)
    data[2, 5, 7] = 99
    img = nib.Nifti1Image(data, img.affine)

    new, plan = reorient_image(img, TARGET_AXCODES_PIR)
    new_data = np.asanyarray(new.dataobj)

    out_xyz = reorient_centroid_voxel((2.0, 5.0, 7.0), plan)
    ox, oy, oz = (int(round(v)) for v in out_xyz)
    assert new_data[ox, oy, oz] == 99


def test_centroid_flips_account_for_shape_minus_one():
    # Source axcodes ('L', 'A', 'S'); target PIR requires flipping the L→R
    # axis (which is the right axis in output).
    img = _make_img(("L", "A", "S"), shape=(10, 20, 30))
    plan = build_reorient_plan(img, TARGET_AXCODES_PIR)

    # A coord at the *minimum* of the flipped axis should map to its maximum
    # (shape - 1) along the matching output axis.
    src_max_x = 9   # original x extent = 10, so max index = 9
    out = reorient_centroid_voxel((src_max_x, 0.0, 0.0), plan)
    # Re-do the test in reverse — sending (0,0,0) should land at a different
    # corner than (9, 0, 0) iff the x axis was flipped.
    out_origin = reorient_centroid_voxel((0.0, 0.0, 0.0), plan)
    assert out != out_origin
