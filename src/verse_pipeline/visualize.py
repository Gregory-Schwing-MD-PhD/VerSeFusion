"""
visualize.py — per-scan QC renders for the unified VerSeFusion corpus.

For each scan, generates a 3-panel PNG (sagittal, coronal, axial mid-slice) with:
  - CT in bone window, grayscale background
  - Vertebra mask labels overlaid in distinct semi-transparent colors
  - TUM-published centroid positions marked with red crosses, labeled by
    vertebra number
  - Actual mask center-of-mass marked with yellow dots, for visual comparison

The centroid coordinate convention is dispatched per-source:
  - MICCAI (no `direction` field): PIR-mm, divide by PIR spacing → PIR voxel
  - BIDS  (`direction=["L","A","S"]`): voxel coords in the LAS-reoriented image,
    which is the raw image's voxel grid → convert to PIR voxel via axis flip+permute

This module is the visual companion to qc.py's centroid_alignment check.
Where qc.py reports a numeric match rate, visualize.py shows you whether the
centroids actually look right on the image.

Usage
-----
    # Render a specific list of scans
    python -m verse_pipeline.visualize \
        --unified_dir data/unified \
        --out_dir     data/qc/renders \
        --scans gl003 verse014 verse512

    # Render the first 10 scans (smoke test)
    python -m verse_pipeline.visualize \
        --unified_dir data/unified \
        --out_dir     data/qc/renders \
        --limit 10

    # Render every scan that QC flagged (read from qc_manifest.json)
    python -m verse_pipeline.visualize \
        --unified_dir   data/unified \
        --out_dir       data/qc/renders \
        --flagged_from  data/qc/qc_manifest.json

    # Render all 374
    python -m verse_pipeline.visualize \
        --unified_dir data/unified \
        --out_dir     data/qc/renders
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
matplotlib.use("Agg")     # non-interactive backend for headless cluster runs
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from nibabel.orientations import (
    axcodes2ornt, io_orientation, ornt_transform, apply_orientation,
    inv_ornt_aff,
)

log = logging.getLogger("verse.visualize")


# =============================================================================
# constants
# =============================================================================

# Bone-window CT display range (HU)
CT_DISPLAY_MIN = -200
CT_DISPLAY_MAX = 1500

# Distance (in voxels) within which a centroid is plotted on a slice
SLICE_TOLERANCE_VOX = 10

# Render DPI — keeps PNGs lightweight for browsing
DEFAULT_DPI = 80


# =============================================================================
# geometry helpers
# =============================================================================

def reorient_to_pir(img: "nib.Nifti1Image") -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return (data_pir, pir_affine, spacing_pir) — reoriented to PIR."""
    data = np.asarray(img.dataobj)
    target_ornt = axcodes2ornt(("P", "I", "R"))
    current_ornt = io_orientation(img.affine)
    xform = ornt_transform(current_ornt, target_ornt)
    data_pir = apply_orientation(data, xform)
    pir_affine = img.affine @ inv_ornt_aff(xform, data.shape)
    spacing_pir = np.array([np.linalg.norm(pir_affine[:3, i]) for i in range(3)])
    return data_pir, pir_affine, spacing_pir


def load_centroids(ctd_path: Path) -> tuple[list[dict], list[str] | None]:
    """Return (centroid_entries, direction_field_or_None)."""
    if not ctd_path or not Path(ctd_path).exists():
        return [], None
    try:
        with open(ctd_path) as f:
            raw = json.load(f)
    except Exception:
        return [], None
    if not isinstance(raw, list):
        return [], None
    direction = None
    entries = []
    for e in raw:
        if not isinstance(e, dict):
            continue
        if "direction" in e and direction is None:
            direction = e["direction"]
            continue
        if all(k in e for k in ("label", "X", "Y", "Z")):
            entries.append({
                "label": int(e["label"]),
                "X": float(e["X"]),
                "Y": float(e["Y"]),
                "Z": float(e["Z"]),
            })
    return entries, direction


def centroids_to_pir_voxel(
    centroids: list[dict],
    direction: list[str] | None,
    pir_shape: tuple[int, int, int],
    spacing_pir: np.ndarray,
) -> list[dict]:
    """Add a 'pir_voxel' field to each centroid entry, dispatching on source format.

    MICCAI centroids (no direction field): PIR-mm, divide by PIR spacing.
    BIDS centroids (direction=["L","A","S"]): voxel coords in the LAS-reoriented image,
      which is the raw image's voxel grid for those subjects.  We convert to
      PIR voxel via axis flip+permute.

    The LAS → PIR mapping:
      LAS axis 0 (L direction) ↔ PIR axis 2 (R direction), flipped
      LAS axis 1 (A direction) ↔ PIR axis 0 (P direction), flipped
      LAS axis 2 (S direction) ↔ PIR axis 1 (I direction), flipped

    So given LAS voxel (X, Y, Z) and PIR shape (P, I, R):
      pir_voxel_p = pir_shape[0] - 1 - LAS_Y
      pir_voxel_i = pir_shape[1] - 1 - LAS_Z
      pir_voxel_r = pir_shape[2] - 1 - LAS_X
    """
    out = []
    for c in centroids:
        if direction is None:
            # MICCAI: PIR-mm
            pir_voxel = (
                c["X"] / spacing_pir[0],
                c["Y"] / spacing_pir[1],
                c["Z"] / spacing_pir[2],
            )
        else:
            # BIDS LAS voxel → PIR voxel
            pir_voxel = (
                pir_shape[0] - 1 - c["Y"],
                pir_shape[1] - 1 - c["Z"],
                pir_shape[2] - 1 - c["X"],
            )
        out.append({**c, "pir_voxel": pir_voxel})
    return out


def label_centers_of_mass(msk_pir: np.ndarray, labels: list[int]) -> dict[int, tuple[float, float, float]]:
    """Return {label: (p, i, r)} for each label present in the mask."""
    out: dict[int, tuple[float, float, float]] = {}
    for label in labels:
        coords = np.argwhere(msk_pir == label)
        if len(coords) == 0:
            continue
        com = coords.mean(axis=0)
        out[label] = tuple(float(v) for v in com)
    return out


# =============================================================================
# rendering
# =============================================================================

def _make_label_cmap() -> mcolors.ListedColormap:
    """Discrete colormap for VerSe labels 0-28 (0 = transparent background)."""
    base = plt.get_cmap("tab20")(np.linspace(0, 1, 20))
    extra = plt.get_cmap("Set3")(np.linspace(0, 1, 12))
    colors = np.vstack([base, extra])[:29]
    colors[0] = [0, 0, 0, 0]   # background transparent
    return mcolors.ListedColormap(colors)


def _bone_window(ct_data: np.ndarray) -> np.ndarray:
    """Clip CT to bone window and normalize 0-1 for display."""
    clipped = np.clip(ct_data, CT_DISPLAY_MIN, CT_DISPLAY_MAX)
    return (clipped - CT_DISPLAY_MIN) / (CT_DISPLAY_MAX - CT_DISPLAY_MIN)


def _best_slice_idx(msk_pir: np.ndarray, axis: int) -> int:
    """Pick the slice index along `axis` with the most labeled voxels."""
    if not np.any(msk_pir > 0):
        return msk_pir.shape[axis] // 2
    sum_axes = tuple(a for a in range(msk_pir.ndim) if a != axis)
    profile = np.sum(msk_pir > 0, axis=sum_axes)
    return int(np.argmax(profile))


def _draw_panel(
    ax,
    ct_slice: np.ndarray,
    msk_slice: np.ndarray,
    label_cmap: mcolors.ListedColormap,
    centroids_in_plane: list[tuple[int, float, float, bool]],
    com_in_plane: list[tuple[int, float, float]],
    title: str,
    xlabel: str,
    ylabel: str,
) -> None:
    """Draw one orthogonal panel.

    centroids_in_plane: list of (label, x, y, in_slice_or_nearby_bool)
    com_in_plane:       list of (label, x, y)  — actual mask CoM
    """
    ax.imshow(ct_slice.T, cmap="gray", origin="lower", aspect="equal",
              vmin=0, vmax=1, interpolation="nearest")
    masked = np.ma.masked_where(msk_slice == 0, msk_slice).T
    ax.imshow(masked, cmap=label_cmap, alpha=0.45, origin="lower",
              aspect="equal", vmin=0, vmax=28, interpolation="nearest")

    # Yellow dots: actual mask CoM
    for label, x, y in com_in_plane:
        ax.plot(x, y, marker="o", color="#FFD500", markersize=5,
                markeredgecolor="black", markeredgewidth=0.5, linestyle="")

    # Red crosses: TUM centroid; faded if not in this slice
    for label, x, y, in_plane in centroids_in_plane:
        alpha = 1.0 if in_plane else 0.25
        ax.plot(x, y, marker="+", color="#E24B4A", markersize=14,
                markeredgewidth=2.0, alpha=alpha, linestyle="")
        if in_plane:
            ax.annotate(str(label), xy=(x, y), color="#FFFF80",
                        fontsize=8, weight="bold",
                        xytext=(6, 6), textcoords="offset points",
                        path_effects=[])

    ax.set_title(title, fontsize=11)
    ax.set_xlabel(xlabel, fontsize=9)
    ax.set_ylabel(ylabel, fontsize=9)
    ax.tick_params(labelsize=7)


def render_scan(scan_dir: Path, out_path: Path,
                dpi: int = DEFAULT_DPI) -> dict[str, Any]:
    """Render one scan; return small summary dict."""
    series_id = scan_dir.name.replace("scan-", "")
    meta_path = scan_dir / f"scan-{series_id}_meta.json"
    if not meta_path.exists():
        raise FileNotFoundError(f"no meta.json at {meta_path}")
    meta = json.loads(meta_path.read_text())

    ct_path  = Path(meta["source_paths"]["ct"])
    msk_path = Path(meta["source_paths"]["msk"])
    ctd_path = meta["source_paths"].get("ctd")

    ct_img  = nib.load(str(ct_path))
    msk_img = nib.load(str(msk_path))

    ct_pir,  _, _           = reorient_to_pir(ct_img)
    msk_pir, _, spacing_pir = reorient_to_pir(msk_img)
    ct_pir  = ct_pir.astype(np.float32)
    msk_pir = msk_pir.astype(np.int32)
    ct_display = _bone_window(ct_pir)

    centroids, direction = load_centroids(Path(ctd_path)) if ctd_path else ([], None)
    centroids = centroids_to_pir_voxel(centroids, direction,
                                       msk_pir.shape, spacing_pir)
    coms = label_centers_of_mass(msk_pir, [c["label"] for c in centroids])

    cmap = _make_label_cmap()

    # Three orthogonal slices — pick by max label coverage
    mid_p = _best_slice_idx(msk_pir, axis=0)   # P axis → coronal-ish view
    mid_i = _best_slice_idx(msk_pir, axis=1)   # I axis → axial view
    mid_r = _best_slice_idx(msk_pir, axis=2)   # R axis → sagittal view

    fig, axes = plt.subplots(1, 3, figsize=(16, 7))

    # --- Sagittal: slice along R, show (P, I) plane ---------------------------
    ct_slice  = ct_display[:, :, mid_r]
    msk_slice = msk_pir   [:, :, mid_r]
    ctd_in_plane = [(c["label"], c["pir_voxel"][0], c["pir_voxel"][1],
                     abs(c["pir_voxel"][2] - mid_r) < SLICE_TOLERANCE_VOX)
                    for c in centroids]
    com_in_plane = [(label, com[0], com[1])
                    for label, com in coms.items()
                    if abs(com[2] - mid_r) < SLICE_TOLERANCE_VOX]
    _draw_panel(axes[0], ct_slice, msk_slice, cmap,
                ctd_in_plane, com_in_plane,
                title=f"Sagittal  (R-slice {mid_r}/{msk_pir.shape[2]})",
                xlabel="← P-axis (posterior →)",
                ylabel="← I-axis (inferior →)")

    # --- Coronal: slice along P, show (R, I) plane ----------------------------
    ct_slice  = ct_display[mid_p, :, :]
    msk_slice = msk_pir   [mid_p, :, :]
    # Coronal axes: x is R, y is I.  But our slice is (I, R) — transpose in _draw_panel.
    # The slice shape is (I_len, R_len) so transpose puts R on x, I on y. Centroids
    # need (R, I) for plotting → use pir_voxel[2], pir_voxel[1].
    ctd_in_plane = [(c["label"], c["pir_voxel"][2], c["pir_voxel"][1],
                     abs(c["pir_voxel"][0] - mid_p) < SLICE_TOLERANCE_VOX)
                    for c in centroids]
    com_in_plane = [(label, com[2], com[1])
                    for label, com in coms.items()
                    if abs(com[0] - mid_p) < SLICE_TOLERANCE_VOX]
    _draw_panel(axes[1], ct_slice, msk_slice, cmap,
                ctd_in_plane, com_in_plane,
                title=f"Coronal  (P-slice {mid_p}/{msk_pir.shape[0]})",
                xlabel="← R-axis (right →)",
                ylabel="← I-axis (inferior →)")

    # --- Axial: slice along I, show (P, R) plane -----------------------------
    ct_slice  = ct_display[:, mid_i, :]
    msk_slice = msk_pir   [:, mid_i, :]
    ctd_in_plane = [(c["label"], c["pir_voxel"][0], c["pir_voxel"][2],
                     abs(c["pir_voxel"][1] - mid_i) < SLICE_TOLERANCE_VOX)
                    for c in centroids]
    com_in_plane = [(label, com[0], com[2])
                    for label, com in coms.items()
                    if abs(com[1] - mid_i) < SLICE_TOLERANCE_VOX]
    _draw_panel(axes[2], ct_slice, msk_slice, cmap,
                ctd_in_plane, com_in_plane,
                title=f"Axial  (I-slice {mid_i}/{msk_pir.shape[1]})",
                xlabel="← P-axis (posterior →)",
                ylabel="← R-axis (right →)")

    # Header with metadata, footer with legend
    n_ctd = len(centroids)
    fig.suptitle(
        f"{series_id}   source={meta.get('source_format')}   "
        f"shape={msk_pir.shape}   spacing={tuple(round(s, 3) for s in spacing_pir)}   "
        f"centroids={n_ctd}",
        fontsize=12, y=0.995,
    )
    fig.text(0.5, 0.02,
             "Red + : TUM-published centroid (labeled by vertebra number).  "
             "Yellow ● : actual mask center-of-mass.  "
             "Mask colors: per-label (random).",
             ha="center", fontsize=9, color="#444441")

    plt.tight_layout(rect=[0, 0.04, 1, 0.97])
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(str(out_path), dpi=dpi, bbox_inches="tight", facecolor="white")
    plt.close(fig)

    return {
        "series_id":    series_id,
        "source_format": meta.get("source_format"),
        "n_centroids": n_ctd,
        "pir_shape":   list(msk_pir.shape),
        "out_path":    str(out_path),
    }


# =============================================================================
# parallel orchestration
# =============================================================================

def _render_one(args: tuple[str, str, int]) -> dict[str, Any]:
    """Worker entry point — takes string paths for pickle-friendliness."""
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

    n_ok = sum(1 for r in results if "error" not in r)
    n_err = len(results) - n_ok
    log.info("Render done: %d ok, %d failed", n_ok, n_err)
    return results


# =============================================================================
# CLI
# =============================================================================

def _scans_to_render(args: argparse.Namespace) -> list[Path]:
    """Determine which scan-dirs to render based on CLI args."""
    all_scans = sorted(d for d in args.unified_dir.iterdir()
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
        description="Render per-scan QC PNGs for the unified VerSe corpus.",
    )
    p.add_argument("--unified_dir", type=Path, required=True,
                   help="Path to data/unified/ (contains scan-* subdirs).")
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
                   help="PNG dpi (default 80; bump to 150 for paper figures).")
    p.add_argument("--log_level", default="INFO",
                   choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    if not args.unified_dir.is_dir():
        log.error("unified_dir not found: %s", args.unified_dir)
        return 1

    scans = _scans_to_render(args)
    if not scans:
        log.error("No scans selected. Check --scans / --flagged_from / --unified_dir.")
        return 1

    results = render_many(scans, args.out_dir,
                          workers=args.workers, dpi=args.dpi)

    # Write a small index manifest alongside the PNGs
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
