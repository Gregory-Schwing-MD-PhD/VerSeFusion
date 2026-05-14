"""
visualize_corrections.py — VERIDAH-corrections review renders.

Same layout as before — 3 rows × 2-or-3 cols depending on subject kind —
with the SAME mm-aware panel rendering as visualize.py (May 2026 change).

Every panel now covers a fixed mm window (default 600mm × 600mm) centered
on the focus labels' centroid.  Aspect ratios are physically correct.  Every
render across the dataset looks comparable in scale, regardless of voxel
spacing or scan extent.

The red ROI box still highlights remapped vertebrae on all three columns
(Raw CT, BEFORE, AFTER) — but now in millimeter coordinates so it lands in
exactly the right anatomical location.

Output: data/corrected/renders/<series_id>_before_after.png
        data/corrected/renders/index.html
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import numpy as np
import nibabel as nib

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib import patheffects as mpe

from verse_pipeline._render_helpers import (
    DEFAULT_WINDOW_MM,
    setup_mm_panel,
    centroid_to_mm,
    bbox_pir_to_mm,
    mask_centroid_vox,
)

log = logging.getLogger("verse.visualize_corrections")


# =============================================================================
# constants
# =============================================================================

_HU_MIN, _HU_MAX = -150, 700
DEFAULT_DPI = 100
SLICE_TOLERANCE_VOX = 10

CENTROID_MARKER_SIZE      = 7
CENTROID_LABEL_FONTSIZE   = 10
CENTROID_LABEL_OFFSET     = (8, 6)
CENTROID_MARKER_EDGEWIDTH = 1.0

ROI_BOX_PAD_MM    = 10.0
ROI_BOX_COLOR     = "#FF3344"
ROI_BOX_LINEWIDTH = 2.0

THORACIC_LABELS_MAX = 19
T13_LABEL           = 28

_VERSE_COLORS: dict[int, tuple[float, float, float, float]] = {
    1:  (0.30, 0.55, 0.95, 0.55),  2:  (0.25, 0.50, 0.90, 0.55),
    3:  (0.20, 0.45, 0.85, 0.55),  4:  (0.20, 0.40, 0.80, 0.55),
    5:  (0.15, 0.35, 0.75, 0.55),  6:  (0.15, 0.30, 0.70, 0.55),
    7:  (0.10, 0.25, 0.65, 0.55),
    8:  (0.10, 0.55, 0.65, 0.55),  9:  (0.10, 0.60, 0.60, 0.55),
    10: (0.10, 0.65, 0.55, 0.55),  11: (0.10, 0.70, 0.50, 0.55),
    12: (0.10, 0.75, 0.45, 0.55),  13: (0.15, 0.78, 0.40, 0.55),
    14: (0.20, 0.80, 0.35, 0.55),  15: (0.30, 0.82, 0.30, 0.55),
    16: (0.40, 0.85, 0.25, 0.55),  17: (0.50, 0.87, 0.20, 0.55),
    18: (0.60, 0.88, 0.15, 0.55),  19: (0.70, 0.90, 0.10, 0.55),
    20: (0.95, 0.85, 0.10, 0.55),  21: (0.95, 0.75, 0.10, 0.55),
    22: (0.95, 0.65, 0.10, 0.55),  23: (0.95, 0.55, 0.10, 0.55),
    24: (0.95, 0.45, 0.10, 0.55),  25: (0.95, 0.35, 0.10, 0.55),
    26: (0.95, 0.20, 0.15, 0.60),
    27: (0.85, 0.20, 0.85, 0.60),
    28: (0.55, 0.20, 0.85, 0.65),
}

_KIND_BORDER = {
    "corrected": "#3B6D11",
    "advisory":  "#888888",
    "rejected":  "#791F1F",
    "passthrough_other": "#888888",
}

_PLANE_AXIS_NAMES = {0: "p", 1: "i", 2: "r"}
_PLANES: list[tuple[int, str]] = [(0, "Coronal"), (1, "Axial"), (2, "Sagittal")]


# =============================================================================
# image helpers
# =============================================================================

def _window(ct: np.ndarray) -> np.ndarray:
    return np.clip((ct - _HU_MIN) / (_HU_MAX - _HU_MIN), 0.0, 1.0)


def _overlay(base_2d: np.ndarray, lbl_2d: np.ndarray,
             cmap: dict[int, tuple[float, float, float, float]]) -> np.ndarray:
    if base_2d.ndim == 2:
        rgb = np.stack([base_2d, base_2d, base_2d], axis=-1).astype(np.float32)
    else:
        rgb = base_2d.astype(np.float32).copy()
    for v, (r, g, b, a) in cmap.items():
        m = (lbl_2d == v)
        if not m.any():
            continue
        rgb[m, 0] = rgb[m, 0] * (1 - a) + r * a
        rgb[m, 1] = rgb[m, 1] * (1 - a) + g * a
        rgb[m, 2] = rgb[m, 2] * (1 - a) + b * a
    return np.clip(rgb, 0.0, 1.0)


def _display_slice(arr2d: np.ndarray, dim: int) -> np.ndarray:
    return arr2d.T if dim == 2 else arr2d


def _safe_slice(arr: np.ndarray, dim: int, idx: int) -> np.ndarray:
    clamped = int(np.clip(idx, 0, arr.shape[dim] - 1))
    s = [slice(None)] * arr.ndim
    s[dim] = clamped
    return arr[tuple(s)]


def _label_coms(msk_data: np.ndarray, labels: list[int]) -> dict[int, tuple[float, float, float]]:
    out: dict[int, tuple[float, float, float]] = {}
    for lbl in labels:
        coords = np.argwhere(msk_data == lbl)
        if len(coords) > 0:
            com = coords.mean(axis=0)
            out[lbl] = tuple(float(v) for v in com)
    return out


def _draw_centroid_markers_mm(ax, coms, dim, mid_idx, spacing) -> None:
    for lbl, com in coms.items():
        if dim == 0:    in_slice = abs(com[0] - mid_idx) < SLICE_TOLERANCE_VOX
        elif dim == 1:  in_slice = abs(com[1] - mid_idx) < SLICE_TOLERANCE_VOX
        else:           in_slice = abs(com[2] - mid_idx) < SLICE_TOLERANCE_VOX
        if not in_slice:
            continue
        x_mm, y_mm = centroid_to_mm(com, dim, spacing)
        ax.plot(x_mm, y_mm, marker="o", color="#FFE000",
                markersize=CENTROID_MARKER_SIZE,
                markeredgecolor="#222",
                markeredgewidth=CENTROID_MARKER_EDGEWIDTH,
                linestyle="")
        ax.annotate(
            str(lbl), xy=(x_mm, y_mm), color="#FFE000",
            fontsize=CENTROID_LABEL_FONTSIZE, weight="bold",
            xytext=CENTROID_LABEL_OFFSET, textcoords="offset points",
            path_effects=[mpe.withStroke(linewidth=2.5, foreground="#111")],
        )


def _draw_roi_box(ax, mm_bbox: tuple[float, float, float, float] | None) -> None:
    if mm_bbox is None:
        return
    x0, y0, x1, y1 = mm_bbox
    rect = mpatches.Rectangle(
        (x0, y0), x1 - x0, y1 - y0,
        linewidth=ROI_BOX_LINEWIDTH, edgecolor=ROI_BOX_COLOR,
        facecolor="none", zorder=10,
    )
    ax.add_patch(rect)


# =============================================================================
# ROI focus
# =============================================================================

def _affected_bbox_pir(msk_before: np.ndarray,
                       focus_labels: set[int]) -> tuple[int, int, int, int, int, int] | None:
    if not focus_labels:
        return None
    mask_focus = np.isin(msk_before, list(focus_labels))
    if not mask_focus.any():
        return None
    coords = np.argwhere(mask_focus)
    return (int(coords[:, 0].min()), int(coords[:, 1].min()), int(coords[:, 2].min()),
            int(coords[:, 0].max()), int(coords[:, 1].max()), int(coords[:, 2].max()))


def _compute_focus_labels(entry: dict[str, Any], kind: str,
                          mask_labels: list[int]) -> set[int]:
    if kind == "corrected":
        remap = entry.get("remap", {})
        return {int(k) for k in remap.keys()}
    if kind == "advisory":
        if T13_LABEL in mask_labels:
            return {T13_LABEL}
        thoracic_in_mask = [l for l in mask_labels if 8 <= l <= THORACIC_LABELS_MAX]
        if thoracic_in_mask:
            return {max(thoracic_in_mask)}
        return set()
    return set()


def _focus_center_vox(msk_before: np.ndarray,
                      focus_labels: set[int]) -> tuple[float, float, float]:
    """Window center: centroid of focus labels if any, else whole-mask centroid."""
    if focus_labels:
        mask_focus = np.isin(msk_before, list(focus_labels))
        coords = np.argwhere(mask_focus)
        if len(coords) > 0:
            com = coords.mean(axis=0)
            return (float(com[0]), float(com[1]), float(com[2]))
    return mask_centroid_vox(msk_before)


# =============================================================================
# classification + headers + caption
# =============================================================================

def _classify_subject(entry: dict[str, Any]) -> str:
    if entry.get("error"):
        return "rejected"
    if entry.get("veridah_applied"):
        return "corrected"
    if entry.get("tltv") or entry.get("sr_left") or entry.get("sr_right"):
        return "advisory"
    return "passthrough_other"


def _flag_str(entry: dict[str, Any]) -> str:
    flags = []
    if entry.get("tltv"):    flags.append("TLTV")
    if entry.get("sr_left"): flags.append("SR_l")
    if entry.get("sr_right"):flags.append("SR_r")
    return ", ".join(flags) if flags else "—"


def _header_for(entry: dict[str, Any], kind: str,
                labels_before: list[int],
                labels_after:  list[int]) -> str:
    sid    = entry["series_id"]
    fid    = entry.get("fid", "?")
    ctype  = entry.get("correction_type", "?")
    flags  = _flag_str(entry)

    if kind == "corrected":
        remap = entry.get("remap", {})
        remap_str = ", ".join(f"{k}→{v}" for k, v in
                              sorted(remap.items(), key=lambda kv: int(kv[0])))
        line2 = (f"remap: {remap_str or '(none)'}   "
                 f"before={labels_before}   after={labels_after}   flags: {flags}")
    elif kind == "rejected":
        err = entry.get("error", "?")
        line2 = f"REJECTED: {err}   labels={labels_before}   flags: {flags}"
    elif kind == "advisory":
        line2 = f"(mask unchanged — only advisory flags recorded)   labels={labels_before}   flags: {flags}"
    else:
        line2 = f"labels={labels_before}   flags: {flags}"

    return f"{sid}   fid={fid}   type={ctype}\n{line2}"


def _reviewer_caption(kind: str, entry: dict[str, Any]) -> str:
    if kind == "corrected":
        ctype = entry.get("correction_type", "")
        if ctype == "t13_shift":
            return (
                "REVIEW: Within the red box, verify the PURPLE vertebra (label 28 = T13) "
                "articulates with RIBS, and the YELLOW vertebrae below (labels 20-24 = L1-L5) "
                "have NO RIBS.  Coronal panel is best for rib counting."
            )
        return (
            "REVIEW: Verify that each THORACIC label (green 8-19, purple 28) articulates "
            "with RIBS, and each LUMBAR label (yellow/orange 20-25) has NO RIBS.  "
            "Coronal panel is best for rib counting."
        )
    if kind == "advisory":
        return (
            "REVIEW: The red box marks the last thoracic vertebra (Möller flagged "
            "TLTV / stump-rib).  Inspect its rib morphology — small / asymmetric / "
            "absent ribs are the diagnostic features."
        )
    if kind == "rejected":
        return (
            "Correction was NOT applied — see header for reason.  "
            "Inspect canonical mask vs. raw CT to decide next steps."
        )
    return "Canonical mask shown."


# =============================================================================
# core renderers (mm-aware)
# =============================================================================

def _render_three_panels(axes_col,
                          ct_data:  np.ndarray,
                          msk_data: np.ndarray | None,
                          slice_by_dim: dict[int, int],
                          coms:     dict[int, tuple[float, float, float]] | None,
                          roi_pir_bbox: tuple[int, int, int, int, int, int] | None,
                          spacing:  tuple[float, float, float],
                          center_vox: tuple[float, float, float],
                          window_mm: float,
                          show_plane_labels: bool) -> None:
    """Draw the three rows (coronal/axial/sagittal) into one column."""
    ct_win = _window(ct_data)

    for row, (dim, plane_name) in enumerate(_PLANES):
        idx = slice_by_dim[dim]
        ct_slice = _display_slice(_safe_slice(ct_win, dim, idx), dim)

        if msk_data is not None:
            msk_slice = _display_slice(_safe_slice(msk_data, dim, idx), dim)
            rgb = _overlay(ct_slice, msk_slice, _VERSE_COLORS)
        else:
            rgb = np.stack([ct_slice, ct_slice, ct_slice], axis=-1)

        setup_mm_panel(axes_col[row], rgb, dim, spacing, center_vox, window_mm)
        axes_col[row].set_xticks([]); axes_col[row].set_yticks([])
        for s in axes_col[row].spines.values(): s.set_visible(False)

        if coms is not None and msk_data is not None:
            _draw_centroid_markers_mm(axes_col[row], coms, dim, idx, spacing)

        roi_mm = bbox_pir_to_mm(roi_pir_bbox, dim, spacing, pad_mm=ROI_BOX_PAD_MM)
        _draw_roi_box(axes_col[row], roi_mm)

        if show_plane_labels:
            axes_col[row].text(
                -0.06, 0.5,
                f"{plane_name}  {_PLANE_AXIS_NAMES[dim]}={idx}",
                transform=axes_col[row].transAxes,
                fontsize=8, color="#aaaaaa",
                rotation=90, va="center", ha="right",
            )


def render_subject(
    series_id: str,
    canonical_dir: Path,
    corrected_dir: Path,
    out_path: Path,
    entry: dict[str, Any],
    dpi: int = DEFAULT_DPI,
    window_mm: float = DEFAULT_WINDOW_MM,
) -> dict[str, Any]:
    kind = _classify_subject(entry)

    canon_scan = canonical_dir / f"scan-{series_id}"
    canon_meta = json.loads((canon_scan / f"scan-{series_id}_meta.json").read_text())
    ct_img    = nib.load(canon_meta["source_paths"]["ct"])
    canon_msk = nib.load(canon_meta["source_paths"]["msk"])
    ct_data    = np.asarray(ct_img.dataobj).astype(np.float32)
    canon_data = np.asarray(canon_msk.dataobj).astype(np.int32)

    mn = tuple(min(a, b) for a, b in zip(ct_data.shape, canon_data.shape))
    ct_data    = ct_data   [:mn[0], :mn[1], :mn[2]]
    canon_data = canon_data[:mn[0], :mn[1], :mn[2]]

    spacing = tuple(float(np.linalg.norm(ct_img.affine[:3, k])) for k in range(3))
    labels_before = sorted(int(l) for l in np.unique(canon_data) if l != 0)
    coms_before = _label_coms(canon_data, labels_before)

    # ROI focus labels and bbox
    focus_labels = _compute_focus_labels(entry, kind, labels_before)
    roi_bbox     = _affected_bbox_pir(canon_data, focus_labels)

    # Window center: focus labels if any (so corrections zoom in on the ROI),
    # otherwise whole-mask centroid.
    center_vox = _focus_center_vox(canon_data, focus_labels)

    # Slice indices at the window center
    p_idx = int(np.clip(round(center_vox[0]), 0, ct_data.shape[0] - 1))
    i_idx = int(np.clip(round(center_vox[1]), 0, ct_data.shape[1] - 1))
    r_idx = int(np.clip(round(center_vox[2]), 0, ct_data.shape[2] - 1))
    slice_by_dim = {0: p_idx, 1: i_idx, 2: r_idx}

    # Load AFTER mask for corrected subjects
    corr_data: np.ndarray | None = None
    labels_after = labels_before
    coms_after = coms_before
    if kind == "corrected":
        corr_scan = corrected_dir / f"scan-{series_id}"
        corr_meta = json.loads((corr_scan / f"scan-{series_id}_meta.json").read_text())
        corr_msk  = nib.load(corr_meta["source_paths"]["msk"])
        corr_data = np.asarray(corr_msk.dataobj).astype(np.int32)
        mn2 = tuple(min(a, b) for a, b in zip(ct_data.shape, corr_data.shape))
        corr_data = corr_data[:mn2[0], :mn2[1], :mn2[2]]
        labels_after = sorted(int(l) for l in np.unique(corr_data) if l != 0)
        coms_after   = _label_coms(corr_data, labels_after)

    # Figure layout
    if kind == "corrected":
        fig, axes = plt.subplots(3, 3, figsize=(14, 13),
                                  gridspec_kw={"hspace": 0.04, "wspace": 0.04})
        col_titles = ["Raw CT", "BEFORE (canonical)", "AFTER (corrected)"]
    else:
        fig, axes = plt.subplots(3, 2, figsize=(10, 13),
                                  gridspec_kw={"hspace": 0.04, "wspace": 0.04})
        col_titles = ["Raw CT", "CANONICAL mask"]

    fig.patch.set_facecolor("#111111")
    try:
        for ax in axes.flat:
            ax.set_facecolor("#111111")
            ax.axis("off")

        for ci, t in enumerate(col_titles):
            axes[0, ci].set_title(t, fontsize=10, color="#cccccc", pad=2)

        # column 0: raw CT (no mask) — best for counting ribs
        _render_three_panels([axes[r, 0] for r in range(3)],
                              ct_data, None, slice_by_dim, None,
                              roi_bbox, spacing, center_vox, window_mm,
                              show_plane_labels=True)

        # column 1: canonical mask (BEFORE for corrected, CANONICAL otherwise)
        _render_three_panels([axes[r, 1] for r in range(3)],
                              ct_data, canon_data, slice_by_dim, coms_before,
                              roi_bbox, spacing, center_vox, window_mm,
                              show_plane_labels=False)

        # column 2: corrected mask (only for corrected)
        if kind == "corrected" and corr_data is not None:
            _render_three_panels([axes[r, 2] for r in range(3)],
                                  ct_data, corr_data, slice_by_dim, coms_after,
                                  roi_bbox, spacing, center_vox, window_mm,
                                  show_plane_labels=False)

        # suptitle / borders / caption
        title_color = "#dddddd"
        if kind == "rejected":   title_color = "#ff6666"
        elif kind == "corrected": title_color = "#a3e565"

        fig.suptitle(
            _header_for(entry, kind, labels_before, labels_after),
            fontsize=11, color=title_color, y=0.997,
        )

        border = _KIND_BORDER.get(kind, "#888888")
        for ax in axes.flat:
            for s in ax.spines.values():
                s.set_edgecolor(border)
                s.set_linewidth(0.8)
                s.set_visible(True)

        caption = _reviewer_caption(kind, entry)
        fig.text(0.5, 0.012, caption, ha="center", va="bottom", fontsize=9,
                 color="#dddddd", style="italic", wrap=True)

        out_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(str(out_path), dpi=dpi, bbox_inches="tight",
                    facecolor=fig.get_facecolor())
    finally:
        plt.close(fig)

    return {
        "series_id":       series_id,
        "fid":             entry.get("fid"),
        "kind":            kind,
        "correction_type": entry.get("correction_type"),
        "strategy":        entry.get("strategy"),
        "remap":           entry.get("remap", {}),
        "tltv":            entry.get("tltv", False),
        "sr_left":         entry.get("sr_left", False),
        "sr_right":        entry.get("sr_right", False),
        "error":           entry.get("error"),
        "labels_before":   labels_before,
        "labels_after":    labels_after,
        "roi_focus_labels": sorted(focus_labels),
        "spacing_mm":      list(spacing),
        "window_mm":       window_mm,
        "out_path":        str(out_path),
    }


# =============================================================================
# parallel orchestration
# =============================================================================

def _render_one(args: tuple[str, str, str, str, dict, int, float]) -> dict[str, Any]:
    series_id, canon_dir_str, corr_dir_str, out_dir_str, entry, dpi, window_mm = args
    out_path = Path(out_dir_str) / f"{series_id}_before_after.png"
    try:
        return render_subject(series_id, Path(canon_dir_str), Path(corr_dir_str),
                              out_path, entry, dpi=dpi, window_mm=window_mm)
    except Exception as e:
        return {"series_id": series_id, "error": f"{type(e).__name__}: {e}"}


def _flush_logs() -> None:
    for h in log.handlers or logging.getLogger().handlers:
        try:    h.flush()
        except Exception: pass


def render_all_csv_subjects(
    canonical_dir: Path,
    corrected_dir: Path,
    out_dir: Path,
    veridah_manifest_path: Path,
    workers: int = 4,
    dpi: int = DEFAULT_DPI,
    window_mm: float = DEFAULT_WINDOW_MM,
    only_kinds: set[str] | None = None,
) -> list[dict[str, Any]]:
    if not veridah_manifest_path.exists():
        log.error("Veridah manifest not found: %s", veridah_manifest_path)
        return []
    manifest = json.loads(veridah_manifest_path.read_text())
    entries = manifest.get("corrections", [])
    if only_kinds is not None:
        entries = [e for e in entries if _classify_subject(e) in only_kinds]

    log.info("Rendering %d subjects (from %s, window=%.0fmm)",
             len(entries), veridah_manifest_path, window_mm)
    if not entries:
        log.warning("No subjects to render.")
        return []

    out_dir.mkdir(parents=True, exist_ok=True)
    work_items = [
        (entry["series_id"], str(canonical_dir), str(corrected_dir),
         str(out_dir), entry, dpi, window_mm)
        for entry in entries
    ]

    results: list[dict[str, Any]] = []
    n_done = 0
    last_log_count = 0
    last_log_time = time.monotonic()
    start = last_log_time

    with ProcessPoolExecutor(max_workers=workers) as ex:
        futures = {ex.submit(_render_one, item): item for item in work_items}
        for fut in as_completed(futures):
            r = fut.result()
            results.append(r)
            n_done += 1
            if "error" in r and "kind" not in r:
                log.warning("  %s: %s", r["series_id"], r["error"])
            if n_done - last_log_count >= 5 or time.monotonic() - last_log_time >= 10.0:
                elapsed = time.monotonic() - start
                rate = n_done / elapsed if elapsed > 0 else 0.0
                remaining = len(entries) - n_done
                eta = remaining / rate if rate > 0 else 0.0
                log.info("  progress: %d/%d (%.1f%%)  %.1f scans/s  ETA %ds",
                         n_done, len(entries), 100 * n_done / len(entries),
                         rate, int(eta))
                _flush_logs()
                last_log_count = n_done
                last_log_time = time.monotonic()

    results.sort(key=lambda r: r.get("series_id", ""))
    by_kind: dict[str, int] = {}
    for r in results:
        k = r.get("kind", "error")
        by_kind[k] = by_kind.get(k, 0) + 1
    log.info("Render done: %s", by_kind)
    return results


# =============================================================================
# HTML gallery
# =============================================================================

GALLERY_HTML = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>VerSeFusion — VERIDAH corrections</title>
<style>
  body { margin:0; padding:24px; background:#111111; color:#dddddd;
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", system-ui, sans-serif; }
  h1 { margin:0 0 4px; font-size:22px; font-weight:500; }
  .subtitle { color:#888; font-size:14px; margin-bottom:8px; }
  .reviewer-note {
    background:#1a1a1a; border:1px solid #FF3344; border-radius:8px;
    padding:12px 16px; margin-bottom:20px; font-size:13px;
    color:#ddd; line-height:1.5;
  }
  .reviewer-note strong { color:#FF6677; }
  .controls {
    position: sticky; top: 0; background: #111111; padding: 10px 0; z-index: 10;
    border-bottom: 1px solid #333; margin-bottom: 20px;
    display: flex; gap: 12px; align-items: center; font-size:13px;
  }
  .controls label { color:#888; }
  .controls select {
    padding: 5px 10px; border: 1px solid #444; border-radius: 6px;
    background: #1a1a1a; color: #dddddd; font-size: 13px;
  }
  .summary {
    display:flex; gap:14px; padding:12px 16px; background:#1a1a1a;
    border:1px solid #333; border-radius:8px; margin-bottom:20px;
    font-size:13px; flex-wrap:wrap;
  }
  .pill { padding:2px 10px; border-radius:12px; font-weight:500; }
  .pill.corrected { background:#1f3d12; color:#a3e565; }
  .pill.advisory  { background:#252525; color:#aaaaaa; }
  .pill.rejected  { background:#3d1212; color:#ff8888; }
  .grid { display:grid; grid-template-columns:1fr; gap:24px; }
  .card { background:#1a1a1a; border:1px solid #333; border-radius:10px;
    overflow:hidden; }
  .card-header {
    padding:12px 18px; border-bottom:1px solid #2a2a2a;
    display:flex; align-items:center; gap:12px; flex-wrap:wrap;
  }
  .card-title { font-size:16px; font-weight:500; color:#dddddd; }
  .card-meta  { font-size:13px; color:#888; font-family:ui-monospace,monospace; }
  .card-error { font-size:13px; color:#ff8888; font-family:ui-monospace,monospace; }
  .card-img-wrap img { display:block; width:100%; height:auto; }
</style>
</head>
<body>
<h1>VerSeFusion — VERIDAH corrections</h1>
<div class="subtitle">All __N_TOTAL__ subjects from Möller's corrections CSV.</div>

<div class="reviewer-note">
  <strong>Reviewer workflow:</strong>
  For each <em>corrected</em> subject, verify within the RED ROI BOX that
  <strong>thoracic</strong> labels (green 8-19, purple 28 = T13) articulate with <strong>RIBS</strong>,
  and <strong>lumbar</strong> labels (yellow/orange 20-25) have <strong>NO RIBS</strong>.
  The Raw CT column (leftmost) is best for counting ribs without color overlay.
  For <em>advisory</em> subjects, the ROI box marks the last thoracic vertebra —
  inspect its rib morphology to confirm Möller's TLTV / stump-rib flag.
</div>

<div class="summary">
  <span>Outcomes:</span>
  <span class="pill corrected">corrected · __N_CORRECTED__</span>
  <span class="pill advisory">advisory · __N_ADVISORY__</span>
  <span class="pill rejected">rejected · __N_REJECTED__</span>
</div>

<div class="controls">
  <label>Kind: <select id="filter-kind">
    <option value="all">all</option>
    <option value="corrected">corrected</option>
    <option value="advisory">advisory</option>
    <option value="rejected">rejected</option>
    <option value="passthrough_other">passthrough (other)</option>
  </select></label>
  <span id="count" style="margin-left:auto;color:#888">_</span>
</div>

<div class="grid" id="grid">
__CARDS__
</div>

<script>
const filter = document.getElementById("filter-kind");
const cards = Array.from(document.querySelectorAll(".card"));
const count = document.getElementById("count");
function apply() {
  const v = filter.value;
  let n = 0;
  cards.forEach(c => {
    const show = (v === "all") || (c.dataset.kind === v);
    c.style.display = show ? "" : "none";
    if (show) n++;
  });
  count.textContent = n + " of " + cards.length + " subjects";
}
filter.addEventListener("change", apply);
apply();
</script>
</body>
</html>"""

CARD_TEMPLATE = """  <div class="card" data-kind="__KIND__">
    <div class="card-header">
      <span class="card-title">__SERIES_ID__</span>
      <span class="pill __KIND_CLASS__">__KIND_LABEL__</span>
      <span class="card-meta">fid=__FID__</span>
      <span class="card-meta">type=__TYPE__</span>
      __EXTRA__
    </div>
    <a class="card-img-wrap" href="__PNG__" target="_blank">
      <img loading="lazy" src="__PNG__" alt="__SERIES_ID__">
    </a>
  </div>"""


def _kind_class(kind: str) -> str:
    return {"corrected": "corrected", "advisory": "advisory",
            "rejected":  "rejected"}.get(kind, "advisory")


def _kind_label(kind: str) -> str:
    return {"corrected": "corrected", "advisory":  "advisory",
            "rejected":  "rejected",  "passthrough_other": "passthrough"}.get(kind, kind)


def build_gallery(out_dir: Path, render_results: list[dict[str, Any]]) -> Path:
    by_kind: dict[str, int] = {}
    for r in render_results:
        k = r.get("kind", "error")
        by_kind[k] = by_kind.get(k, 0) + 1

    cards: list[str] = []
    for r in sorted(render_results, key=lambda x: x.get("series_id", "")):
        if "out_path" not in r:
            continue
        kind = r.get("kind", "?")
        png_rel = Path(r["out_path"]).name

        extras = []
        if kind == "corrected":
            remap = r.get("remap", {})
            remap_str = ", ".join(f"{k}→{v}" for k, v in
                                  sorted(remap.items(), key=lambda kv: int(kv[0])))
            if remap_str:
                extras.append(f'<span class="card-meta">remap: {remap_str}</span>')
        if kind == "rejected":
            extras.append(f'<span class="card-error">REJECTED: {r.get("error", "?")}</span>')
        if r.get("tltv") or r.get("sr_left") or r.get("sr_right"):
            flags = []
            if r.get("tltv"):    flags.append("TLTV")
            if r.get("sr_left"): flags.append("SR_l")
            if r.get("sr_right"):flags.append("SR_r")
            extras.append(f'<span class="card-meta">flags: {", ".join(flags)}</span>')

        cards.append(
            CARD_TEMPLATE
            .replace("__SERIES_ID__", r["series_id"])
            .replace("__KIND__",      kind)
            .replace("__KIND_CLASS__", _kind_class(kind))
            .replace("__KIND_LABEL__", _kind_label(kind))
            .replace("__FID__",       r.get("fid") or "?")
            .replace("__TYPE__",      r.get("correction_type") or "?")
            .replace("__EXTRA__",     "\n      ".join(extras))
            .replace("__PNG__",       png_rel)
        )

    html = (GALLERY_HTML
            .replace("__N_TOTAL__",     str(len(render_results)))
            .replace("__N_CORRECTED__", str(by_kind.get("corrected", 0)))
            .replace("__N_ADVISORY__",  str(by_kind.get("advisory", 0)))
            .replace("__N_REJECTED__",  str(by_kind.get("rejected", 0)))
            .replace("__CARDS__",       "\n".join(cards)))

    out_path = out_dir / "index.html"
    out_path.write_text(html)
    log.info("Wrote gallery: %s  (%d cards)", out_path, len(cards))
    return out_path


# =============================================================================
# CLI
# =============================================================================

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="verse-visualize-corrections",
        description="Per-subject review renders for the VERIDAH corrections CSV.",
    )
    p.add_argument("--canonical_dir", type=Path, required=True)
    p.add_argument("--corrected_dir", type=Path, required=True)
    p.add_argument("--out_dir",       type=Path, required=True)
    p.add_argument("--veridah_manifest", type=Path,
                   help="Default: <corrected_dir>/veridah_manifest.json")
    p.add_argument("--only_kinds", nargs="*",
                   choices=["corrected", "advisory", "rejected", "passthrough_other"])
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--dpi",     type=int, default=DEFAULT_DPI)
    p.add_argument("--window_mm", type=float, default=DEFAULT_WINDOW_MM,
                   help="Physical extent of each panel in mm (default 600).")
    p.add_argument("--log_level", default="INFO")
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    veridah_manifest = args.veridah_manifest or (args.corrected_dir / "veridah_manifest.json")
    if not veridah_manifest.exists():
        log.error("Veridah manifest not found: %s", veridah_manifest)
        return 1

    only_kinds = set(args.only_kinds) if args.only_kinds else None
    results = render_all_csv_subjects(
        args.canonical_dir, args.corrected_dir, args.out_dir,
        veridah_manifest, workers=args.workers, dpi=args.dpi,
        window_mm=args.window_mm, only_kinds=only_kinds,
    )

    index_path = args.out_dir / "renders_manifest.json"
    index_path.write_text(json.dumps({
        "n_rendered": sum(1 for r in results if "out_path" in r),
        "n_failed":   sum(1 for r in results if "out_path" not in r),
        "window_mm":  args.window_mm,
        "renders":    results,
    }, indent=2))
    log.info("Wrote %s", index_path)

    build_gallery(args.out_dir, results)
    return 0


if __name__ == "__main__":
    sys.exit(main())
