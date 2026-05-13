"""
render_gallery.py — build an HTML index of QC renders for browsing.

Combines:
  - The renders directory (filled by visualize.py) containing per-scan PNGs
  - data/qc/qc_manifest.json (per-scan QC status from qc.py)
  - data/unified/unify_manifest.json (per-scan metadata)

Outputs a single self-contained HTML page at out_path with:
  - Status filter: PASS / WARN / FAIL / all
  - Source filter: miccai / bids / all
  - Search box: filter by series_id
  - Sort by series_id (default) or by overall status
  - Each entry shows: thumbnail image, series_id, source, status, failing checks

Usage:
    python -m verse_pipeline.render_gallery \
        --renders_dir   data/qc/renders \
        --qc_manifest   data/qc/qc_manifest.json \
        --unify_manifest data/unified/unify_manifest.json \
        --out_path      data/qc/renders/index.html
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

log = logging.getLogger("verse.gallery")


HTML_TEMPLATE = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>VerSeFusion QC renders</title>
<style>
  body {
    margin: 0; padding: 24px;
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", system-ui, sans-serif;
    background: #fafaf8; color: #2c2c2a;
  }
  h1 { margin: 0 0 4px; font-size: 22px; font-weight: 500; }
  .subtitle { color: #5f5e5a; font-size: 14px; margin-bottom: 24px; }
  .controls {
    position: sticky; top: 0; background: #fafaf8; padding: 12px 0; z-index: 10;
    border-bottom: 1px solid #d3d1c7; margin-bottom: 20px;
    display: flex; flex-wrap: wrap; gap: 12px; align-items: center;
  }
  .controls label { font-size: 13px; color: #5f5e5a; }
  .controls select, .controls input {
    padding: 6px 10px; border: 1px solid #b4b2a9; border-radius: 6px;
    background: white; font-size: 13px; font-family: inherit;
  }
  .controls input[type="text"] { width: 180px; }
  .summary {
    display: flex; gap: 16px; padding: 12px 16px; background: white;
    border: 1px solid #d3d1c7; border-radius: 8px; margin-bottom: 20px;
    font-size: 13px;
  }
  .summary .pill { padding: 2px 10px; border-radius: 12px; font-weight: 500; }
  .pill.pass { background: #EAF3DE; color: #3B6D11; }
  .pill.warn { background: #FAEEDA; color: #854F0B; }
  .pill.fail { background: #FCEBEB; color: #791F1F; }
  .pill.skip { background: #F1EFE8; color: #5F5E5A; }

  .grid {
    display: grid; grid-template-columns: repeat(auto-fill, minmax(360px, 1fr));
    gap: 16px;
  }
  .card {
    background: white; border: 1px solid #d3d1c7; border-radius: 10px;
    overflow: hidden; transition: box-shadow .15s;
  }
  .card:hover { box-shadow: 0 2px 8px rgba(0,0,0,0.08); }
  .card-header {
    display: flex; align-items: center; justify-content: space-between;
    padding: 10px 14px; border-bottom: 1px solid #f1efe8;
  }
  .card-title { font-size: 14px; font-weight: 500; }
  .card-source {
    font-size: 11px; color: #5f5e5a; font-family: ui-monospace, monospace;
    text-transform: uppercase;
  }
  .card-img-wrap { background: white; display: block; }
  .card-img-wrap img { display: block; width: 100%; height: auto; }
  .card-meta {
    padding: 10px 14px; font-size: 12px; color: #5f5e5a;
    border-top: 1px solid #f1efe8;
  }
  .card-meta .failing-checks {
    color: #791F1F; font-family: ui-monospace, monospace;
    font-size: 11px; margin-top: 4px;
  }
  .empty {
    text-align: center; padding: 60px 20px; color: #5f5e5a;
  }
</style>
</head>
<body>
<h1>VerSeFusion QC renders</h1>
<div class="subtitle">Per-scan visualization of CT + mask + mask-derived centroid markers. __N_RENDERS__ renders, sourced from __N_SCANS__ unified scans.</div>

<div class="summary">
  <span>Status:</span>
  <span class="pill pass">PASS · __N_PASS__</span>
  <span class="pill warn">WARN · __N_WARN__</span>
  <span class="pill fail">FAIL · __N_FAIL__</span>
  <span class="pill skip">SKIP · __N_SKIP__</span>
</div>

<div class="controls">
  <label>Status: <select id="filter-status">
    <option value="all">all</option>
    <option value="PASS">PASS</option>
    <option value="WARN">WARN</option>
    <option value="FAIL">FAIL</option>
    <option value="SKIP">SKIP</option>
  </select></label>
  <label>Source: <select id="filter-source">
    <option value="all">all</option>
    <option value="miccai">miccai</option>
    <option value="bids">bids</option>
  </select></label>
  <label>Search: <input type="text" id="filter-search" placeholder="series_id..."></label>
  <label>Sort: <select id="sort-by">
    <option value="series_id">series_id</option>
    <option value="status">status (FAIL first)</option>
  </select></label>
  <span id="count" style="margin-left:auto;font-size:13px;color:#5f5e5a"></span>
</div>

<div class="grid" id="grid"></div>
<div class="empty" id="empty" style="display:none">No renders match the current filters.</div>

<script>
const ENTRIES = __ENTRIES_JSON__;

const STATUS_ORDER = {FAIL: 0, WARN: 1, SKIP: 2, PASS: 3};

function render() {
  const filterStatus = document.getElementById("filter-status").value;
  const filterSource = document.getElementById("filter-source").value;
  const search = document.getElementById("filter-search").value.toLowerCase();
  const sortBy = document.getElementById("sort-by").value;

  let filtered = ENTRIES.filter(e => {
    if (filterStatus !== "all" && e.overall !== filterStatus) return false;
    if (filterSource !== "all" && e.source_format !== filterSource) return false;
    if (search && !e.series_id.toLowerCase().includes(search)) return false;
    return true;
  });

  if (sortBy === "status") {
    filtered.sort((a, b) => {
      const da = (STATUS_ORDER[a.overall] ?? 99) - (STATUS_ORDER[b.overall] ?? 99);
      return da !== 0 ? da : a.series_id.localeCompare(b.series_id);
    });
  } else {
    filtered.sort((a, b) => a.series_id.localeCompare(b.series_id));
  }

  document.getElementById("count").textContent =
    `${filtered.length} of ${ENTRIES.length} scans`;

  const grid = document.getElementById("grid");
  const empty = document.getElementById("empty");
  if (filtered.length === 0) {
    grid.innerHTML = "";
    empty.style.display = "block";
    return;
  }
  empty.style.display = "none";

  grid.innerHTML = filtered.map(e => `
    <div class="card">
      <div class="card-header">
        <span class="card-title">${e.series_id}</span>
        <span>
          <span class="pill ${(e.overall || 'skip').toLowerCase()}">${e.overall || '–'}</span>
        </span>
      </div>
      <a class="card-img-wrap" href="${e.png}" target="_blank">
        <img loading="lazy" src="${e.png}" alt="${e.series_id}">
      </a>
      <div class="card-meta">
        source: <code>${e.source_format || '–'}</code> ·
        labels: ${e.n_labels ?? '?'}
        ${e.failing_checks && e.failing_checks.length
          ? `<div class="failing-checks">flagged: ${e.failing_checks.join(", ")}</div>`
          : ''}
      </div>
    </div>
  `).join('');
}

document.getElementById("filter-status").addEventListener("change", render);
document.getElementById("filter-source").addEventListener("change", render);
document.getElementById("filter-search").addEventListener("input", render);
document.getElementById("sort-by").addEventListener("change", render);

render();
</script>
</body>
</html>"""


def build_gallery(renders_dir: Path, qc_manifest_path: Path | None,
                  unify_manifest_path: Path | None, out_path: Path) -> Path:
    renders_dir = renders_dir.resolve()
    out_path = out_path.resolve()

    # Collect rendered PNGs
    renders_idx = renders_dir / "renders_manifest.json"
    if renders_idx.exists():
        renders = json.loads(renders_idx.read_text()).get("renders", [])
    else:
        # Fall back to filesystem listing
        renders = [{"series_id":     p.stem,
                    "out_path":      str(p),
                    "source_format": None,
                    "n_labels":      None}
                   for p in sorted(renders_dir.glob("*.png"))]
    by_sid = {r["series_id"]: r for r in renders}

    # Cross-reference with QC manifest
    qc_lookup: dict[str, dict] = {}
    if qc_manifest_path and qc_manifest_path.exists():
        qc = json.loads(qc_manifest_path.read_text())
        for scan in qc.get("scans", []):
            qc_lookup[scan["series_id"]] = scan

    # Cross-reference with unify manifest
    unify_lookup: dict[str, dict] = {}
    if unify_manifest_path and unify_manifest_path.exists():
        u = json.loads(unify_manifest_path.read_text())
        for scan in u.get("scans", []):
            unify_lookup[scan["series_id"]] = scan

    # Build entries
    entries = []
    for sid, render in sorted(by_sid.items()):
        qc_entry = qc_lookup.get(sid, {})
        unify_entry = unify_lookup.get(sid, {})
        failing = []
        for cname, c in qc_entry.get("checks", {}).items():
            if c.get("status") in ("WARN", "FAIL"):
                failing.append(cname)

        png_rel = Path(render["out_path"]).name   # gallery is in same dir as PNGs

        entries.append({
            "series_id":      sid,
            "png":            png_rel,
            "source_format":  render.get("source_format") or unify_entry.get("source_format"),
            "n_labels":       render.get("n_labels"),
            "overall":        qc_entry.get("overall"),
            "failing_checks": failing,
        })

    # Status summary
    n_pass = sum(1 for e in entries if e["overall"] == "PASS")
    n_warn = sum(1 for e in entries if e["overall"] == "WARN")
    n_fail = sum(1 for e in entries if e["overall"] == "FAIL")
    n_skip = sum(1 for e in entries if e["overall"] not in ("PASS", "WARN", "FAIL"))

    html = (HTML_TEMPLATE
            .replace("__N_RENDERS__", str(len(entries)))
            .replace("__N_SCANS__",   str(len(unify_lookup) or len(entries)))
            .replace("__N_PASS__",    str(n_pass))
            .replace("__N_WARN__",    str(n_warn))
            .replace("__N_FAIL__",    str(n_fail))
            .replace("__N_SKIP__",    str(n_skip))
            .replace("__ENTRIES_JSON__", json.dumps(entries)))

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(html)
    log.info("Wrote gallery: %s  (%d entries)", out_path, len(entries))
    return out_path


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="verse-render-gallery",
        description="Build an HTML index of QC renders.",
    )
    p.add_argument("--renders_dir", type=Path, required=True,
                   help="Directory of *.png files from visualize.py")
    p.add_argument("--qc_manifest", type=Path,
                   help="Optional data/qc/qc_manifest.json for status info")
    p.add_argument("--unify_manifest", type=Path,
                   help="Optional data/unified/unify_manifest.json for metadata")
    p.add_argument("--out_path", type=Path, required=True,
                   help="Where to write the HTML index")
    p.add_argument("--log_level", default="INFO",
                   choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )
    if not args.renders_dir.is_dir():
        log.error("renders_dir not found: %s", args.renders_dir)
        return 1
    build_gallery(args.renders_dir, args.qc_manifest, args.unify_manifest,
                  args.out_path)
    return 0


if __name__ == "__main__":
    sys.exit(main())
