"""NIfTI orientation utilities.

VerSeFusion reorients every CT and segmentation mask to **PIR** (Posterior /
Inferior / Right) to match the CTSpinoPelvic1K convention.  This is the
``nibabel`` axcode triple, not the SimpleITK ``DirectionCosines`` matrix.

A matching transformation must be applied to the centroid coordinates
(which live in voxel space), since reorientation permutes and/or flips
voxel axes.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import nibabel as nib
import numpy as np
from nibabel.orientations import (
    aff2axcodes,
    axcodes2ornt,
    inv_ornt_aff,
    io_orientation,
    ornt_transform,
)

TARGET_AXCODES_PIR: tuple[str, str, str] = ("P", "I", "R")


# =============================================================================
# orientation introspection
# =============================================================================

def current_axcodes(img: nib.Nifti1Image) -> tuple[str, str, str]:
    """Return the current axcode triple of a NIfTI image."""
    codes = aff2axcodes(img.affine)
    return tuple(codes)  # type: ignore[return-value]


def needs_reorient(img: nib.Nifti1Image, target: Sequence[str] = TARGET_AXCODES_PIR) -> bool:
    return current_axcodes(img) != tuple(target)


# =============================================================================
# image reorientation
# =============================================================================

@dataclass(frozen=True)
class ReorientPlan:
    """Pre-computed orientation transform — apply to image and centroids both.

    The semantics match nibabel's ``ornt_transform`` output:
    ``transform[i] = (out_axis, flip)`` means **input axis i** becomes
    **output axis out_axis**, optionally flipped (``flip == -1``).
    """

    source_axcodes: tuple[str, str, str]
    target_axcodes: tuple[str, str, str]
    transform:      np.ndarray            # nibabel ornt_transform output
    src_shape:      tuple[int, int, int]

    @property
    def axis_permutation(self) -> tuple[int, int, int]:
        """``out_axis_of_input[i]`` — which output axis the i-th input axis becomes."""
        return tuple(int(self.transform[i, 0]) for i in range(3))  # type: ignore[return-value]

    @property
    def axis_flips(self) -> tuple[int, int, int]:
        """``flip_of_input[i]`` — 1 if input axis i is preserved, -1 if flipped."""
        return tuple(int(self.transform[i, 1]) for i in range(3))  # type: ignore[return-value]


def build_reorient_plan(
    img: nib.Nifti1Image,
    target: Sequence[str] = TARGET_AXCODES_PIR,
) -> ReorientPlan:
    """Compute the source → target orientation transform for an image.

    The same plan must be applied to any voxel-space coordinates (e.g.
    centroids) so they line up with the reoriented image.
    """
    src = current_axcodes(img)
    src_ornt = io_orientation(img.affine)
    tgt_ornt = axcodes2ornt(tuple(target))
    transform = ornt_transform(src_ornt, tgt_ornt)
    return ReorientPlan(
        source_axcodes=src,
        target_axcodes=tuple(target),  # type: ignore[arg-type]
        transform=transform,
        src_shape=tuple(img.shape[:3]),  # type: ignore[arg-type]
    )


def reorient_image(
    img: nib.Nifti1Image,
    target: Sequence[str] = TARGET_AXCODES_PIR,
) -> tuple[nib.Nifti1Image, ReorientPlan]:
    """Reorient a NIfTI image and return both the reoriented image and the plan."""
    plan = build_reorient_plan(img, target)
    new = img.as_reoriented(plan.transform)
    return new, plan


def reorient_to_pir_inplace(path_in: Path, path_out: Path, *, is_mask: bool) -> ReorientPlan:
    """Reorient a NIfTI on disk to PIR and write a new file.

    Masks are reoriented with nearest-neighbour semantics (no resampling — just
    a permutation/flip, which is exactly what ``as_reoriented`` performs;
    voxel values are preserved bitwise).
    """
    img = nib.load(str(path_in))
    new, plan = reorient_image(img, TARGET_AXCODES_PIR)

    path_out.parent.mkdir(parents=True, exist_ok=True)

    # Preserve dtype precisely for masks.
    if is_mask:
        data = np.asanyarray(new.dataobj).astype(np.uint8, copy=False)
        new = nib.Nifti1Image(data, new.affine, header=new.header)
        new.set_data_dtype(np.uint8)

    nib.save(new, str(path_out))
    return plan


# =============================================================================
# centroid reorientation
# =============================================================================

def reorient_centroid_voxel(
    voxel_xyz: tuple[float, float, float],
    plan: ReorientPlan,
) -> tuple[float, float, float]:
    """Apply the same orientation transform to a single voxel-space centroid.

    nibabel encodes the transform as ``transform[i] = (out_axis, flip)``:
    input axis ``i`` becomes output axis ``out_axis``, optionally flipped.
    To transform a voxel coordinate we therefore iterate over **input
    axes**, flip the input coordinate along its own axis if needed, and
    write the result into the matching output slot.
    """
    perm = plan.axis_permutation
    flips = plan.axis_flips
    src_shape = plan.src_shape

    out = [0.0, 0.0, 0.0]
    for in_axis in range(3):
        out_axis = perm[in_axis]
        v = voxel_xyz[in_axis]
        if flips[in_axis] == -1:
            v = (src_shape[in_axis] - 1) - v
        out[out_axis] = v
    return tuple(out)  # type: ignore[return-value]


# =============================================================================
# affine sanity
# =============================================================================

def affines_close(a: np.ndarray, b: np.ndarray, *, atol: float = 1e-4) -> bool:
    """Return True iff two 4x4 affines agree elementwise (used in QA)."""
    return np.allclose(a, b, atol=atol)
