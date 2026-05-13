"""
visualize.py — per-scan QC renders for canonically reoriented scans.

For each scan in data/canonical/scan-*/, generates a 3-panel PNG with:
  - CT in bone window, grayscale background
  - Mask labels overlaid in distinct semi-transparent colors
  - Mask-derived center-of-mass markers (yellow dots, labeled by vertebra number)

Inputs are already in PIR orientation (per the reorient stage), so this
module loads the canonical NIfTIs and plots directly — no orientation
handling needed.  Axis 0 = P, axis 1 = I, axis 2 = R.

Usage
-----
    python -m verse_pipeline.visualize \\
        --input_dir data/canonical \\
        --out_dir   data/qc/renders \\
        --limit     10
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
import matplotlib.colors as mcolors

log = logging.getLogger("verse.visualize")


# =============================================================================
# constants
# =============================================================================

CT_DISPLAY_MIN = -200
CT_DISPLAY_MAX = 1500

SLICE_TOLERANCE_VOX = 10
DEFAULT_DPI = 80


# =============================================================================
# rendering
# =============================================================================

def _make_label_cmap() -> mcolors.ListedColormap:
    base  = plt.get_cmap("tab20")(np.linspace(0, 1, 20))
    extra = plt.get_cmap("Set3")(np.linspace(0, 1, 12))
    colors = np.vstack([base, extra])[:29]
    colors[0] = [0, 0, 0, 0]
    return mcolors.ListedColormap(colors)


def _bone_window(ct_data: np.ndarray) -> np.ndarray:
    clipped = np.clip(ct_data, CT_DISPLAY_MIN, CT_DISPLAY_MAX)
    return (clipped - CT_DISPLAY_MIN) / (CT_DISPLAY_MAX - CT_DISPLAY_MIN)


def _best_slice_idx(msk_data: np.ndarray, axis: int) -> int:
    if not np.any(msk_data > 0):
        return msk_data.shape[axis] // 2
    sum_axes = tuple(a for a in range(msk_data.ndim) if a != axis)
    profile = np.sum(msk_data > 0, axis=sum_axes)
    return int(np.argmax(profile))


def mask_centers_of_mass(msk_data: np.ndarray, labels: list[int]) -> dict[int, tuple[float, float, float]]:
    out: dict[int, tuple[float, float, float]] = {}
    for label in labels:
        coords = np.argwhere(msk_data == label)
        if len(coords) == 0:
            continue
        com = coords.mean(axis=0)
        out[label] = tuple(float(v) for v in com)
    return out


def _draw_panel(
    ax,
    ct_slice: np.ndarray,
    msk_slice: np.ndarray,
    label_cmap: mcolors.ListedColormap,
    com_in_plane: list[tuple[int, float, float, bool]],
    title: str,
    xlabel: str,
    ylabel: str,
) -> None:
    """Draw one orthogonal panel.

    com_in_plane: list of (label, screen_x, screen_y, in_or_near_slice_bool)
    """
    ax.imshow(ct_slice.T, cmap="gray", origin="lower", aspect="equal",
              vmin=0, vmax=1, interpolation="nearest")
    masked = np.ma.masked_where(msk_slice == 0, msk_slice).T
    ax.imshow(masked, cmap=label_cmap, alpha=0.45, origin="lower",
              aspect="equal", vmin=0, vmax=28, interpolation="nearest")

    for label, x, y, in_plane in com_in_plane:
        alpha = 1.0 if in_plane else 0.25
        ax.plot(x, y, marker="o", color="#FFD500", markersize=6,
                markeredgecolor="black", markeredgewidth=0.8,
                alpha=alpha, linestyle="")
        if in_plane:
            ax.annotate(str(label), xy=(x, y), color="#FFFF80",
                        fontsize=8, weight="bold",
                        xytext=(7, 7), textcoords="offset points")

    ax.set_title(title, fontsize=11)
    ax.set_xlabel(xlabel, fontsize=9)
    ax.set_ylabel(ylabel, fontsize=9)
    ax.tick_params(labelsize=7)


def render_scan(scan_dir: Path, out_path: Path,
                dpi: int = DEFAULT_DPI) -> dict[str, Any]:
    """Render one canonical scan."""
    series_id = scan_dir.name.replace("scan-", "")
    meta_path = scan_dir / f"scan-{series_id}_meta.json"
    if not meta_path.exists():
        raise FileNotFoundError(f"no meta.json at {meta_path}")
    meta = json.loads(meta_path.read_text())

    ct_path  = Path(meta["source_paths"]["ct"])
    msk_path = Path(meta["source_paths"]["msk"])

    ct_img  = nib.load(str(ct_path))
    msk_img = nib.load(str(msk_path))

    ct_data  = np.asarray(ct_img.dataobj).astype(np.float32)
    msk_data = np.asarray(msk_img.dataobj).astype(np.int32)

    spacing = tuple(float(np.linalg.norm(ct_img.affine[:3, i])) for i in range(3))
    ct_display = _bone_window(ct_data)

    labels_present = sorted(int(l) for l in np.unique(msk_data) if l != 0)
    coms = mask_centers_of_mass(msk_data, labels_present)

    cmap = _make_label_cmap()

    mid_p = _best_slice_idx(msk_data, axis=0)
    mid_i = _best_slice_idx(msk_data, axis=1)
    mid_r = _best_slice_idx(msk_data, axis=2)

    fig, axes = plt.subplots(1, 3, figsize=(16, 7))

    # Sagittal: slice along R (axis 2), view (P, I) plane = (axis 0, axis 1)
    ct_slice  = ct_display[:, :, mid_r]
    msk_slice = msk_data  [:, :, mid_r]
    com_in_plane = [(label, com[0], com[1],
                     abs(com[2] - mid_r) < SLICE_TOLERANCE_VOX)
                    for label, com in coms.items()]
    _draw_panel(axes[0], ct_slice, msk_slice, cmap, com_in_plane,
                title=f"Sagittal  (R-slice {mid_r}/{msk_data.shape[2]})",
                xlabel="← P-axis (posterior →)",
                ylabel="← I-axis (inferior →)")

    # Coronal: slice along P (axis 0), view (R, I) plane = (axis 2, axis 1)
    ct_slice  = ct_display[mid_p, :, :]
    msk_slice = msk_data  [mid_p, :, :]
    com_in_plane = [(label, com[2], com[1],
                     abs(com[0] - mid_p) < SLICE_TOLERANCE_VOX)
                    for label, com in coms.items()]
    _draw_panel(axes[1], ct_slice, msk_slice, cmap, com_in_plane,
                title=f"Coronal  (P-slice {mid_p}/{msk_data.shape[0]})",
                xlabel="← R-axis (right →)",
                ylabel="← I-axis (inferior →)")

    # Axial: slice along I (axis 1), view (P, R) plane = (axis 0, axis 2)
    ct_slice  = ct_display[:, mid_i, :]
    msk_slice = msk_data  [:, mid_i, :]
    com_in_plane = [(label, com[0], com[2],
                     abs(com[1] - mid_i) < SLICE_TOLERANCE_VOX)
                    for label, com in coms.items()]
    _draw_panel(axes[2], ct_slice, msk_slice, cmap, com_in_plane,
                title=f"Axial  (I-slice {mid_i}/{msk_data.shape[1]})",
                xlabel="← P-axis (posterior →)",
                ylabel="← R-axis (right →)")

    fig.suptitle(
        f"{series_id}   source={meta.get('source_format')}   "
        f"shape={msk_data.shape}   "
        f"spacing={tuple(round(s, 3) for s in spacing)}   "
        f"labels={len(labels_present)}   "
        f"orientation={meta.get('orientation', 'PIR')}",
        fontsize=12, y=0.995,
    )
    fig.text(0.5, 0.02,
             "Yellow ● : mask-derived vertebra center-of-mass.  "
             "Mask colors: per-label (distinct).  "
             "Image is canonical PIR.",
             ha="center", fontsize=9, color="#444441")

    plt.tight_layout(rect=[0, 0.04, 1, 0.97])
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(str(out_path), dpi=dpi, bbox_inches="tight", facecolor="white")
    plt.close(fig)

    return {
        "series_id":     series_id,
        "source_format": meta.get("source_format"),
        "n_labels":      len(labels_present),
        "shape":         list(msk_data.shape),
        "out_path":      str(out_path),
    }


# =============================================================================
# parallel orchestration
# =============================================================================

def _render_one(args: tuple[str, str, int]) -> dict[str, Any]:
    scan_dir_str, out_dir_str, dpi = args
    scan_dir = Path(scan_dir_str)
    out_dir  = Path(out_dir_str)
    series_id = scan_dir.name.replace("scan-", "")
    out_path = out_dir / f"{series_id}.png"
    try:
        return render_scan(scan_dir, out_path, dpi=dpi)
    except Exception as e:
        return {"series_id": series_id, "error": f"{type(e).__name__}: {e}"}


def _flush_logs() -> None:
    for h in log.handlers or logging.getLogger().handlers:
        try:
            h.flush()
        except Exception:
            pass


def render_many(scan_dirs: list[Path], out_dir: Path,
                workers: int = 4, dpi: int = DEFAULT_DPI) -> list[dict[str, Any]]:
    out_dir.mkdir(parents=True, exist_ok=True)
    log.info("Rendering %d scans -> %s (workers=%d, dpi=%d)",
             len(scan_dirs), out_dir, workers, dpi)
    _flush_logs()

    results: list[dict[str, Any]] = []
    n_done = 0
    last_log_count = 0
    last_log_time = time.monotonic()
    start = last_log_time

    work_items = [(str(d), str(out_dir), dpi) for d in scan_dirs]

    with ProcessPoolExecutor(max_workers=workers) as ex:
        futures = {ex.submit(_render_one, item): item for item in work_items}
        for fut in as_completed(futures):
            r = fut.result()
            results.append(r)
            n_done += 1
            if "error" in r:
                log.warning("  %s: %s", r["series_id"], r["error"])

            since_count = n_done - last_log_count
            since_time = time.monotonic() - last_log_time
            if since_count >= 10 or since_time >= 10.0:
                elapsed = time.monotonic() - start
                rate = n_done / elapsed if elapsed > 0 else 0.0
                remaining = len(scan_dirs) - n_done
                eta = remaining / rate if rate > 0 else 0.0
                log.info("  progress: %d/%d (%.1f%%)  %.1f scans/s  ETA %ds",
                         n_done, len(scan_dirs),
                         100 * n_done / len(scan_dirs),
                         rate, int(eta))
                _flush_logs()
                last_log_count = n_done
                last_log_time = time.monotonic()

    results.sort(key=lambda r: r["series_id"])

    n_ok  = sum(1 for r in results if "error" not in r)
    n_err = len(results) - n_ok
    log.info("Render done: %d ok, %d failed", n_ok, n_err)
    return results


# =============================================================================
# CLI
# =============================================================================

def _scans_to_render(args: argparse.Namespace) -> list[Path]:
    all_scans = sorted(d for d in args.input_dir.iterdir()
                       if d.is_dir() and d.name.startswith("scan-"))

    if args.scans:
        wanted = set(args.scans)
        scans = [d for d in all_scans
                 if d.name.replace("scan-", "") in wanted]
    elif args.flagged_from:
        manifest = json.loads(Path(args.flagged_from).read_text())
        flagged_ids = set()
        for status_group in ("WARN", "FAIL"):
            for entry in manifest.get("flagged_scans", {}).get(status_group, []):
                flagged_ids.add(entry["series_id"])
        scans = [d for d in all_scans
                 if d.name.replace("scan-", "") in flagged_ids]
        log.info("Found %d flagged scans in %s", len(scans), args.flagged_from)
    else:
        scans = all_scans

    if args.limit:
        scans = scans[: args.limit]

    return scans


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="verse-visualize",
        description="Render per-scan QC PNGs for canonical (PIR) scans.",
    )
    p.add_argument("--input_dir", type=Path, required=True,
                   help="Path to data/canonical/ (contains scan-* subdirs).")
    p.add_argument("--out_dir", type=Path, required=True,
                   help="Where to write PNG files.")
    p.add_argument("--scans", nargs="*",
                   help="Specific series IDs to render (default: all).")
    p.add_argument("--flagged_from", type=Path,
                   help="Render only scans flagged WARN/FAIL in this qc_manifest.json.")
    p.add_argument("--limit", type=int,
                   help="Cap the number of scans rendered.")
    p.add_argument("--workers", type=int, default=4,
                   help="ProcessPoolExecutor workers (default 4).")
    p.add_argument("--dpi", type=int, default=DEFAULT_DPI,
                   help="PNG dpi (default 80; 150 for paper figures).")
    p.add_argument("--log_level", default="INFO",
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

    results = render_many(scans, args.out_dir,
                          workers=args.workers, dpi=args.dpi)

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
