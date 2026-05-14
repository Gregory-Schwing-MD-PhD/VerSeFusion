"""
_render_helpers.py — shared mm-aware panel rendering for visualize.py and
visualize_corrections.py.

Why
---
Earlier renders used `aspect="auto"` for imshow.  This stretches every panel
to fill its axes box, which means:
  - A 200mm-tall lumbar-only scan and a 700mm-tall full-spine scan end up
    the SAME height in the figure.
  - Voxel spacing variations across scans (0.5mm vs 1.5mm slice thickness)
    produce different visual aspect ratios.
The renders look inconsistent across the dataset even when the data is
correctly PIR-oriented.

This module provides one function — `setup_mm_panel` — that:
  1. Maps the image pixels to millimeter coordinates via the `extent` arg.
  2. Uses `aspect="equal"` so 1 mm in the data is 1 mm visually.
  3. Sets a FIXED mm window centered on the mask centroid, so every panel
     of every scan covers the same physical extent (default 600mm × 600mm).
  4. Small scans appear small with black padding; large scans get clipped
     to the window.  Cross-scan, every vertebra now looks the same size.

Centroid markers are drawn in mm coordinates too (via `centroid_to_mm`),
so they land in the right place under the new extent.

PIR storage assumption
----------------------
All inputs are PIR canonical:
  axis 0 = P (anterior → posterior)
  axis 1 = I (superior → inferior — the spine direction)
  axis 2 = R (left → right)

Sagittal panels are transposed for display (head at top, anterior at left).
Coronal and axial keep their natural orientation.
"""

from __future__ import annotations

from typing import Sequence

import numpy as np

# Default fixed window covering the whole spine for most scans.  Override per
# render via the window_mm argument.
DEFAULT_WINDOW_MM: float = 600.0


# =============================================================================
# core: set up a panel with mm-based extent and fixed window
# =============================================================================

def setup_mm_panel(
    ax,
    slice_2d: np.ndarray,
    plane:    int,
    spacing:  Sequence[float],
    center_vox: Sequence[float],
    window_mm:  float = DEFAULT_WINDOW_MM,
) -> tuple[float, float, float, float]:
    """Render slice_2d on `ax` with mm-based axes and a fixed window.

    Args:
        ax:           matplotlib Axes
        slice_2d:     2D array — for sagittal, this MUST be the post-transpose
                      array (shape (I, P)).  For coronal/axial use the raw slice.
        plane:        0=coronal, 1=axial, 2=sagittal
        spacing:      voxel size in mm for each PIR axis (sp_P, sp_I, sp_R)
        center_vox:   PIR voxel coordinates to center the window on.  Pass the
                      mask centroid for QC; for VERIDAH renders, pass the focus
                      labels' centroid.
        window_mm:    side length of the visible window in mm.

    Returns:
        (xlim_lo_mm, xlim_hi_mm, ylim_lo_mm, ylim_hi_mm)
        Useful when drawing additional elements (ROI boxes, markers).

    Coordinate semantics
    --------------------
    For each plane we identify the vertical and horizontal spacing in mm,
    and the PIR axes contributing to the slice's rows/cols.

    coronal (plane=0):  slice has shape (I, R)
        rows = I axis (axis 1) → vertical spacing = spacing[1]
        cols = R axis (axis 2) → horizontal spacing = spacing[2]
        slice center: (I_center, R_center)

    axial (plane=1):    slice has shape (P, R)
        rows = P axis (axis 0) → vertical spacing = spacing[0]
        cols = R axis (axis 2) → horizontal spacing = spacing[2]
        slice center: (P_center, R_center)

    sagittal (plane=2): slice has shape (I, P) AFTER transpose
        rows = I axis (axis 1) → vertical spacing = spacing[1]
        cols = P axis (axis 0) → horizontal spacing = spacing[0]
        slice center: (I_center, P_center)

    `extent` maps image pixels (0..h, 0..w) to data coordinates in mm.
    With origin='upper' (matplotlib default for imshow), extent is
    (left, right, bottom, top).  We pass (0, w*sp_h, h*sp_v, 0) so
    that y=0 corresponds to row 0 (top of image), and y increases
    downward.
    """
    # slice_2d may be 2D (grayscale) or 3D (RGB, shape (H, W, 3)).  We only
    # care about the spatial dims for setting extent and panel limits.
    h_px, w_px = slice_2d.shape[:2]

    if plane == 0:        # Coronal
        sp_v, sp_h = float(spacing[1]), float(spacing[2])
        cy_vox, cx_vox = float(center_vox[1]), float(center_vox[2])
    elif plane == 1:      # Axial
        sp_v, sp_h = float(spacing[0]), float(spacing[2])
        cy_vox, cx_vox = float(center_vox[0]), float(center_vox[2])
    elif plane == 2:      # Sagittal (transposed)
        sp_v, sp_h = float(spacing[1]), float(spacing[0])
        cy_vox, cx_vox = float(center_vox[1]), float(center_vox[0])
    else:
        raise ValueError(f"plane must be 0/1/2, got {plane}")

    # Image pixels → mm coordinates
    extent = (0.0, w_px * sp_h, h_px * sp_v, 0.0)
    ax.imshow(slice_2d, extent=extent, aspect="equal", interpolation="nearest")

    # Fixed window centered on the mask centroid in mm
    cy_mm = cy_vox * sp_v
    cx_mm = cx_vox * sp_h
    half = window_mm / 2.0
    xlim = (cx_mm - half, cx_mm + half)
    ylim = (cy_mm + half, cy_mm - half)  # inverted: row 0 (top) maps to small y
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)

    return (xlim[0], xlim[1], ylim[0], ylim[1])


def centroid_to_mm(
    com_pir: Sequence[float],
    plane:   int,
    spacing: Sequence[float],
) -> tuple[float, float]:
    """Convert a PIR voxel-coord centroid to (x_mm, y_mm) for the given plane."""
    p, i, r = float(com_pir[0]), float(com_pir[1]), float(com_pir[2])
    if plane == 0:
        return (r * float(spacing[2]), i * float(spacing[1]))
    if plane == 1:
        return (r * float(spacing[2]), p * float(spacing[0]))
    if plane == 2:
        return (p * float(spacing[0]), i * float(spacing[1]))
    raise ValueError(f"plane must be 0/1/2, got {plane}")


def bbox_pir_to_mm(
    pir_bbox: Sequence[int] | None,
    plane:    int,
    spacing:  Sequence[float],
    pad_mm:   float = 8.0,
) -> tuple[float, float, float, float] | None:
    """Project a PIR voxel bbox (p_min, i_min, r_min, p_max, i_max, r_max) to
    mm screen coords for the given plane.  Returns (x0, y0, x1, y1) in mm."""
    if pir_bbox is None:
        return None
    p_min, i_min, r_min, p_max, i_max, r_max = pir_bbox
    sp_p, sp_i, sp_r = float(spacing[0]), float(spacing[1]), float(spacing[2])
    if plane == 0:   # x = R, y = I
        return (r_min * sp_r - pad_mm, i_min * sp_i - pad_mm,
                r_max * sp_r + pad_mm, i_max * sp_i + pad_mm)
    if plane == 1:   # x = R, y = P
        return (r_min * sp_r - pad_mm, p_min * sp_p - pad_mm,
                r_max * sp_r + pad_mm, p_max * sp_p + pad_mm)
    # sagittal: x = P, y = I
    return (p_min * sp_p - pad_mm, i_min * sp_i - pad_mm,
            p_max * sp_p + pad_mm, i_max * sp_i + pad_mm)


def mask_centroid_vox(mask_3d: np.ndarray) -> tuple[float, float, float]:
    """Centroid of nonzero voxels in PIR voxel coordinates.  Returns the
    volume center if the mask is empty (so partial-FOV scans without labels
    don't blow up)."""
    coords = np.argwhere(mask_3d > 0)
    if len(coords) == 0:
        return (mask_3d.shape[0] / 2.0,
                mask_3d.shape[1] / 2.0,
                mask_3d.shape[2] / 2.0)
    com = coords.mean(axis=0)
    return (float(com[0]), float(com[1]), float(com[2]))
