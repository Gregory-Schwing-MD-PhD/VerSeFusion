"""
visualize.py — per-scan QC renders for canonical (PIR) VerSe scans.

For each scan in data/canonical/scan-*/, generates a 3-row × 2-col PNG:
  Row 0: Coronal  (slice along axis 0, the P direction)
  Row 1: Axial    (slice along axis 1, the I direction)
  Row 2: Sagittal (slice along axis 2, the R direction — display is transposed)
  Col 0: Raw CT (bone window)
  Col 1: CT + vertebra mask + centroid markers

Stored volumes are PIR.  Sagittal slices are transposed so head appears at top.

Output: data/qc/renders/<series_id>.png
        data/qc/renders/renders_manifest.json
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
from matplotlib import patheffects as mpe

log = logging.getLogger("verse.visualize")


# =============================================================================
# constants
# =============================================================================

_HU_MIN, _HU_MAX = -200, 800
DEFAULT_DPI = 100
SLICE_TOLERANCE_VOX = 10

CENTROID_MARKER_SIZE      = 7
CENTROID_LABEL_FONTSIZE   = 10
CENTROID_LABEL_OFFSET     = (8, 6)
CENTROID_MARKER_EDGEWIDTH = 1.0

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

_PLANE_AXIS_NAMES = {0: "p", 1: "i", 2: "r"}
_PLANES: list[tuple[int, str]] = [(0, "Coronal"), (1, "Axial"), (2, "Sagittal")]


# =============================================================================
# helpers
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


def _choose_slices(ct: np.ndarray, msk: np.ndarray) -> tuple[int, int, int]:
    nonzero = np.argwhere(msk > 0) if msk is not None else None
    if nonzero is None or len(nonzero) == 0:
        return tuple(s // 2 for s in ct.shape)
    p = int(np.clip(int(nonzero[:, 0].mean()), 0, ct.shape[0] - 1))
    i = int(np.clip(int(nonzero[:, 1].mean()), 0, ct.shape[1] - 1))
    r = int(np.clip(int(nonzero[:, 2].mean()), 0, ct.shape[2] - 1))
    return (p, i, r)


def _label_coms(msk_data: np.ndarray, labels: list[int]) -> dict[int, tuple[float, float, float]]:
    out: dict[int, tuple[float, float, float]] = {}
    for lbl in labels:
        coords = np.argwhere(msk_data == lbl)
        if len(coords) > 0:
            com = coords.mean(axis=0)
            out[lbl] = tuple(float(v) for v in com)
    return out


def _centroid_screen_xy(com_pir, dim):
    p, i, r = com_pir
    if dim == 0:  return (r, i)
    if dim == 1:  return (r, p)
    return (p, i)


def _draw_centroid_markers(ax, coms, dim, mid_idx) -> None:
    for lbl, com in coms.items():
        if dim == 0:    in_slice = abs(com[0] - mid_idx) < SLICE_TOLERANCE_VOX
        elif dim == 1:  in_slice = abs(com[1] - mid_idx) < SLICE_TOLERANCE_VOX
        else:           in_slice = abs(com[2] - mid_idx) < SLICE_TOLERANCE_VOX
        if not in_slice:
            continue
        x, y = _centroid_screen_xy(com, dim)
        ax.plot(x, y, marker="o", color="#FFE000",
                markersize=CENTROID_MARKER_SIZE,
                markeredgecolor="#222",
                markeredgewidth=CENTROID_MARKER_EDGEWIDTH,
                linestyle="")
        ax.annotate(
            str(lbl), xy=(x, y), color="#FFE000",
            fontsize=CENTROID_LABEL_FONTSIZE, weight="bold",
            xytext=CENTROID_LABEL_OFFSET, textcoords="offset points",
            path_effects=[mpe.withStroke(linewidth=2.5, foreground="#111")],
        )


# =============================================================================
# per-scan render
# =============================================================================

def render_scan(scan_dir: Path, out_path: Path,
                dpi: int = DEFAULT_DPI) -> dict[str, Any]:
    series_id = scan_dir.name.replace("scan-", "")
    meta_path = scan_dir / f"scan-{series_id}_meta.json"
    if not meta_path.exists():
        raise FileNotFoundError(f"no meta.json at {meta_path}")
    meta = json.loads(meta_path.read_text())

    ct_img  = nib.load(meta["source_paths"]["ct"])
    msk_img = nib.load(meta["source_paths"]["msk"])
    ct_data  = np.asarray(ct_img.dataobj).astype(np.float32)
    msk_data = np.asarray(msk_img.dataobj).astype(np.int32)

    mn = tuple(min(a, b) for a, b in zip(ct_data.shape, msk_data.shape))
    ct_data  = ct_data [:mn[0], :mn[1], :mn[2]]
    msk_data = msk_data[:mn[0], :mn[1], :mn[2]]

    spacing = tuple(float(np.linalg.norm(ct_img.affine[:3, k])) for k in range(3))
    labels_present = sorted(int(l) for l in np.unique(msk_data) if l != 0)
    coms = _label_coms(msk_data, labels_present)

    p_idx, i_idx, r_idx = _choose_slices(ct_data, msk_data)
    slice_by_dim = {0: p_idx, 1: i_idx, 2: r_idx}

    fig, axes = plt.subplots(3, 2, figsize=(9, 13),
                              gridspec_kw={"hspace": 0.04, "wspace": 0.04})
    fig.patch.set_facecolor("#111111")
    for ax in axes.flat:
        ax.set_facecolor("#111111")
        ax.axis("off")

    axes[0, 0].set_title("Raw CT",          fontsize=10, color="#cccccc", pad=2)
    axes[0, 1].set_title("CT + mask + CoM", fontsize=10, color="#cccccc", pad=2)

    ct_win = _window(ct_data)

    for row, (dim, plane_name) in enumerate(_PLANES):
        idx = slice_by_dim[dim]
        ct_slice  = _display_slice(_safe_slice(ct_win,   dim, idx), dim)
        msk_slice = _display_slice(_safe_slice(msk_data, dim, idx), dim)

        rgb_raw = np.stack([ct_slice, ct_slice, ct_slice], axis=-1)
        rgb_ov  = _overlay(ct_slice, msk_slice, _VERSE_COLORS)

        axes[row, 0].imshow(rgb_raw, aspect="auto", interpolation="nearest")
        axes[row, 1].imshow(rgb_ov,  aspect="auto", interpolation="nearest")
        _draw_centroid_markers(axes[row, 1], coms, dim, idx)

        axes[row, 0].text(
            -0.04, 0.5,
            f"{plane_name}  {_PLANE_AXIS_NAMES[dim]}={idx}",
            transform=axes[row, 0].transAxes,
            fontsize=8, color="#aaaaaa",
            rotation=90, va="center", ha="right",
        )

    source_format = meta.get("source_format", "?")
    fig.suptitle(
        f"{series_id}   shape={tuple(msk_data.shape)}   "
        f"spacing=({spacing[0]:.2f}, {spacing[1]:.2f}, {spacing[2]:.2f}) mm   "
        f"labels={len(labels_present)}   src={source_format}",
        fontsize=11, color="#dddddd", y=0.997,
    )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(out_path), dpi=dpi, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    plt.close(fig)

    return {
        "series_id":     series_id,
        "source_format": source_format,
        "n_labels":      len(labels_present),
        "shape":         list(msk_data.shape),
        "out_path":      str(out_path),
    }


# =============================================================================
# parallel orchestration  (unchanged from prior version)
# =============================================================================

def _render_one(args: tuple[str, str, int]) -> dict[str, Any]:
    scan_dir_str, out_dir_str, dpi = args
    scan_dir = Path(scan_dir_str)
    series_id = scan_dir.name.replace("scan-", "")
    out_path = Path(out_dir_str) / f"{series_id}.png"
    try:
        return render_scan(scan_dir, out_path, dpi=dpi)
    except Exception as e:
        return {"series_id": series_id, "error": f"{type(e).__name__}: {e}"}


def _flush_logs() -> None:
    for h in log.handlers or logging.getLogger().handlers:
        try:    h.flush()
        except Exception: pass


def render_many(scan_dirs: list[Path], out_dir: Path,
                workers: int = 4, dpi: int = DEFAULT_DPI) -> list[dict[str, Any]]:
    out_dir.mkdir(parents=True, exist_ok=True)
    log.info("Rendering %d scans -> %s (workers=%d, dpi=%d)",
             len(scan_dirs), out_dir, workers, dpi)
    _flush_logs()

    work_items = [(str(d), str(out_dir), dpi) for d in scan_dirs]
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
            if "error" in r:
                log.warning("  %s: %s", r["series_id"], r["error"])
            if n_done - last_log_count >= 10 or time.monotonic() - last_log_time >= 10.0:
                elapsed = time.monotonic() - start
                rate = n_done / elapsed if elapsed > 0 else 0.0
                remaining = len(scan_dirs) - n_done
                eta = remaining / rate if rate > 0 else 0.0
                log.info("  progress: %d/%d (%.1f%%)  %.1f scans/s  ETA %ds",
                         n_done, len(scan_dirs),
                         100 * n_done / len(scan_dirs), rate, int(eta))
                _flush_logs()
                last_log_count = n_done
                last_log_time = time.monotonic()

    results.sort(key=lambda r: r["series_id"])
    n_ok = sum(1 for r in results if "error" not in r)
    log.info("Render done: %d ok, %d failed", n_ok, len(results) - n_ok)
    return results


# =============================================================================
# CLI
# =============================================================================

def _scans_to_render(args: argparse.Namespace) -> list[Path]:
    all_scans = sorted(d for d in args.input_dir.iterdir()
                       if d.is_dir() and d.name.startswith("scan-"))
    if args.scans:
        wanted = set(args.scans)
        scans = [d for d in all_scans if d.name.replace("scan-", "") in wanted]
    elif args.flagged_from:
        manifest = json.loads(Path(args.flagged_from).read_text())
        flagged_ids = set()
        for status_group in ("WARN", "FAIL"):
            for entry in manifest.get("flagged_scans", {}).get(status_group, []):
                flagged_ids.add(entry["series_id"])
        scans = [d for d in all_scans if d.name.replace("scan-", "") in flagged_ids]
        log.info("Found %d flagged scans in %s", len(scans), args.flagged_from)
    else:
        scans = all_scans
    if args.limit:
        scans = scans[:args.limit]
    return scans


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="verse-visualize",
                                description="Per-scan QC PNGs for canonical (PIR) VerSe scans.")
    p.add_argument("--input_dir",  type=Path, required=True)
    p.add_argument("--out_dir",    type=Path, required=True)
    p.add_argument("--scans",      nargs="*")
    p.add_argument("--flagged_from", type=Path)
    p.add_argument("--limit",      type=int)
    p.add_argument("--workers",    type=int, default=4)
    p.add_argument("--dpi",        type=int, default=DEFAULT_DPI)
    p.add_argument("--log_level",  default="INFO",
                   choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )
    if not args.input_dir.is_dir():
        log.error("input_dir not found: %s", args.input_dir)
        return 1
    scans = _scans_to_render(args)
    if not scans:
        log.error("No scans selected.")
        return 1
    results = render_many(scans, args.out_dir, workers=args.workers, dpi=args.dpi)
    index_path = args.out_dir / "renders_manifest.json"
    index_path.write_text(json.dumps({
        "n_rendered": sum(1 for r in results if "error" not in r),
        "n_failed":   sum(1 for r in results if "error" in r),
        "renders":    results,
    }, indent=2))
    log.info("Wrote %s", index_path)
    return 0


if __name__ == "__main__":
    sys.exit(main())
