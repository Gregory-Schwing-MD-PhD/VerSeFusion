"""
dataset_interface.py — Runtime interface for the VerSeFusion HF dataset.

Two classes:
  VerSeFusion           dict-style dataset wrapping an HF export directory.
                        No torch dependency.  Use this for
                        benchmarking / visualization / cohort analysis.
  VerSeFusionDataset    PyTorch Dataset adapter on top of VerSeFusion.

Expected layout (produced by stages 10a+10b+11 of the pipeline):
  <root>/
    scans/<series_id>/ct.nii.gz
    scans/<series_id>/mask.nii.gz
    manifest.json                 schema_version: 1
    manifest.csv                  same data, flat tabular form
    manifest_summary.json         cross-tabs by split × lstv_class
    splits_5fold.json             schema_version: 1, patient-level
                                  CV folds with test held out
    corrections/veridah_manifest.json
    orientation_audit.json
    LICENSE
    README.md

Quickstart (analysis / viz — no torch needed):
  >>> from dataset_interface import VerSeFusion
  >>> ds = VerSeFusion("data/hf_staging")
  >>> print(ds.stats())
  >>> t13_cases = ds.filter(lstv_class="t13_supernumerary")

Quickstart (HF Hub — lazy NIfTI fetch on first access):
  >>> ds = VerSeFusion.from_hub("gregoryschwingmdphd/VerseFusion")
  >>> ct_arr, affine = ds.cases[0].load_ct()   # downloads on first call

Quickstart (training):
  >>> from dataset_interface import VerSeFusionDataset
  >>> ds_tr = VerSeFusionDataset("data/hf_staging", split=("fold", 0, "train"))
  >>> ds_va = VerSeFusionDataset("data/hf_staging", split=("fold", 0, "val"))
  >>> ds_te = VerSeFusionDataset("data/hf_staging", split="test")

PATIENT-LEVEL SPLITS
====================
Both the test holdout and the 5-fold CV are stratified at the patient
level.  Paired patients (where a single patient has multiple scans) keep
all their scans in the same fold to prevent leakage.

LSTV CLASSES
============
The dataset is stratified on a 4-way `lstv_class` derived from the LSTV
audit flags during manifest construction:
    t13_supernumerary   has_T13 = True  (~18 cases)
    lumbarization       has_L6  = True  (~44 cases)
    truncated           lacks_T12_TLJ_in_FOV  (~6 cases)
    normal              otherwise  (~290 cases)

The per-patient class is the WORST-CASE across that patient's scans
(t13 > lumb > trunc > normal).  See verse_pipeline/splits_builder.py.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple


# VerSeFusion mask label scheme (28-class):
#   0       background
#   1-7     C1-C7
#   8-19    T1-T12
#   20-25   L1-L6
#   26      sacrum
#   27      coccyx
#   28      T13 (supernumerary, after VERIDAH t13_shift)
LABEL_NAMES = (
    "background",
    "C1", "C2", "C3", "C4", "C5", "C6", "C7",
    "T1", "T2", "T3", "T4", "T5", "T6", "T7", "T8", "T9", "T10", "T11", "T12",
    "L1", "L2", "L3", "L4", "L5", "L6",
    "sacrum", "coccyx",
    "T13",
)
NUM_CLASSES = len(LABEL_NAMES)


# ============================================================================
# Case record
# ============================================================================

@dataclass
class Case:
    """One scan (CT + mask) with metadata.

    For HF-backed datasets, ct_path / mask_path may not exist on disk yet —
    files are fetched lazily on first call to load_ct() / load_mask() via
    the back-reference to the parent dataset.  For local roots the
    back-reference is None and load_* just opens the file directly.
    """
    series_id:          str
    patient_id:         Optional[str]
    ct_path:            Path
    mask_path:          Path
    split:              str = "unknown"     # training/validation/test
    source_dataset:     Optional[str] = None
    source_format:      Optional[str] = None

    # geometry
    shape:              Optional[Tuple[int, int, int]] = None
    spacing_mm:         Optional[Tuple[float, float, float]] = None

    # demographics (often missing)
    age:                Optional[float] = None
    sex:                Optional[str] = None
    patient_pos:        Optional[str] = None

    # corrections
    veridah_applied:    bool = False
    veridah_action:     Optional[str] = None
    veridah_kind:       Optional[str] = None

    # LSTV
    n_labels:           int = 0
    labels_present:     List[int] = field(default_factory=list)
    has_T13:            bool = False
    has_L6:             bool = False
    lacks_T12_TLJ_in_FOV: bool = False
    lstv_class:         str = "normal"

    # Manifest-relative paths (used by lazy fetch)
    ct_file_rel:        str = ""
    mask_file_rel:      str = ""

    # Back-ref to parent VerSeFusion instance for HF lazy fetch.
    # Marked compare=False so equality / repr stay sane.
    _parent: object = field(default=None, repr=False, compare=False)

    def exists(self) -> bool:
        """True iff both files are present on disk RIGHT NOW.  Returns
        False for HF-backed cases that haven't been fetched yet."""
        return self.ct_path.exists() and self.mask_path.exists()

    def has_label(self, label: int) -> bool:
        return int(label) in self.labels_present

    def _ensure_local(self) -> None:
        """Download from HF if needed.  No-op for local datasets."""
        if self._parent is None:
            return
        fetcher = getattr(self._parent, "_hf_fetch", None)
        if fetcher is None:
            return
        if not self.ct_path.exists():
            new_ct = fetcher(self.ct_file_rel)
            if new_ct is not None:
                self.ct_path = Path(new_ct)
        if not self.mask_path.exists():
            new_msk = fetcher(self.mask_file_rel)
            if new_msk is not None:
                self.mask_path = Path(new_msk)

    def load_ct(self):
        """Returns (ct_array float32 in PIR, affine 4x4)."""
        import nibabel as nib
        import numpy as np
        self._ensure_local()
        img = nib.load(str(self.ct_path))
        return np.asarray(img.dataobj, dtype=np.float32), img.affine

    def load_mask(self):
        """Returns (mask_array int16 in PIR, affine 4x4)."""
        import nibabel as nib
        import numpy as np
        self._ensure_local()
        img = nib.load(str(self.mask_path))
        return np.asarray(img.dataobj, dtype=np.int16), img.affine

    # Backwards-compat alias for code expecting CTSpinoPelvic1K's load_label
    def load_label(self):
        return self.load_mask()


# ============================================================================
# coercion / path resolution helpers (mirror CTSpinoPelvic1K conventions)
# ============================================================================

def _coerce_optional_str(v) -> Optional[str]:
    if v is None:
        return None
    try:
        import pandas as _pd
        if _pd.isna(v):
            return None
    except Exception:
        pass
    s = str(v)
    return s if s and s.lower() != "nan" else None


def _coerce_optional_float(v) -> Optional[float]:
    if v is None or v == "":
        return None
    try:
        import pandas as _pd
        if _pd.isna(v):
            return None
    except Exception:
        pass
    try:
        f = float(v)
    except (TypeError, ValueError):
        return None
    return None if f != f else f


def _coerce_optional_int(v) -> Optional[int]:
    f = _coerce_optional_float(v)
    return int(f) if f is not None else None


def _coerce_bool(v) -> bool:
    if isinstance(v, bool):
        return v
    if v is None:
        return False
    if isinstance(v, str):
        return v.strip().lower() in ("true", "1", "yes")
    try:
        return bool(int(v))
    except (TypeError, ValueError):
        return bool(v)


def _coerce_labels_list(v) -> List[int]:
    """labels_present may be a JSON string (from CSV) or already a list."""
    if v is None or v == "":
        return []
    if isinstance(v, list):
        return [int(x) for x in v]
    if isinstance(v, str):
        try:
            parsed = json.loads(v)
            if isinstance(parsed, list):
                return [int(x) for x in parsed]
        except (TypeError, ValueError, json.JSONDecodeError):
            pass
    return []


def _resolve_file(root: Path, rel: str) -> Path:
    """Resolve a manifest-declared relative path against the root.

    Always tries `root/rel` first.  For HF-backed datasets where the file
    hasn't been fetched yet, the result won't exist — that's fine, the
    lazy-fetch path in Case._ensure_local() handles it.
    """
    if not rel:
        return root
    return root / rel


# ============================================================================
# main dataset class (no torch dep)
# ============================================================================

class VerSeFusion:
    """Directory-backed dataset with rich per-scan metadata.

    For HF-backed instances (via from_hub), only metadata files are
    downloaded eagerly (manifest, splits, README — kilobytes).  CT and
    mask NIfTIs are fetched lazily on first call to Case.load_ct() /
    load_mask() via _hf_fetch(), and cached for future calls under the
    huggingface_hub cache.
    """

    # Splits schema recorded after _resolve_splits so callers can introspect
    splits_schema_version: Optional[int] = None
    splits_scheme:         Optional[str] = None

    # HF lazy-fetch state.  None for purely local datasets.
    _hf_repo_id:    Optional[str] = None
    _hf_token:      Optional[str] = None
    _hf_cache_dir:  Optional[str] = None

    def __init__(self, root):
        self.root = Path(os.path.expanduser(str(root)))
        if not self.root.exists():
            raise FileNotFoundError(f"Dataset root not found: {self.root}")
        self._load()

    # ── HF lazy-fetch ────────────────────────────────────────────────────
    def _hf_fetch(self, rel_path: str) -> Optional[str]:
        """Ensure the file at rel_path exists locally.  Returns local path
        as a string, or None if this dataset isn't HF-backed.

        Race-safe across processes (huggingface_hub uses file locks).
        Network errors propagate.
        """
        if not self._hf_repo_id or not rel_path:
            return None
        try:
            from huggingface_hub import hf_hub_download
        except ImportError as e:
            raise RuntimeError(
                "huggingface_hub not installed.  pip install huggingface_hub"
            ) from e
        return hf_hub_download(
            repo_id   = self._hf_repo_id,
            repo_type = "dataset",
            filename  = rel_path,
            token     = self._hf_token,
            cache_dir = self._hf_cache_dir,
        )

    # ── splits resolution ────────────────────────────────────────────────
    def _resolve_splits(self) -> Tuple[Dict[str, str], Optional[Dict]]:
        """Read splits_5fold.json.  Returns (series_id_to_split, cv_doc).

        series_id_to_split maps to "test" or "trainval".  cv_doc is the
        full splits document for fold() lookups, or None if missing.

        Falls back to the manifest's native `split` column for the
        "split" attribute when splits_5fold.json is absent — but in
        that case fold() will raise.
        """
        series_to_split: Dict[str, str] = {}
        cv_doc: Optional[Dict] = None

        splits_path = self.root / "splits_5fold.json"
        if splits_path.exists():
            try:
                doc = json.loads(splits_path.read_text())
                self.splits_schema_version = int(doc.get("schema_version", 0) or 0)
                self.splits_scheme = doc.get("strata_scheme")
                for sid in doc.get("test_series_ids", []) or []:
                    series_to_split[str(sid)] = "test"
                if "folds" in doc:
                    cv_doc = doc
                return series_to_split, cv_doc
            except (OSError, ValueError, TypeError) as e:
                import warnings as _w
                _w.warn(
                    f"Could not read {splits_path}: {e}.  fold() will fail.",
                    stacklevel=3,
                )
        return series_to_split, cv_doc

    def _load_manifest_records(self) -> List[Dict[str, Any]]:
        """Read manifest.json (preferred) or manifest.csv as records."""
        json_path = self.root / "manifest.json"
        if json_path.exists():
            doc = json.loads(json_path.read_text())
            if isinstance(doc, dict):
                return list(doc.get("subjects", []))
            if isinstance(doc, list):
                return list(doc)
        csv_path = self.root / "manifest.csv"
        if csv_path.exists():
            import pandas as pd
            return pd.read_csv(csv_path).to_dict(orient="records")
        raise FileNotFoundError(
            f"No manifest found under {self.root}.  Looked for "
            f"manifest.json and manifest.csv.  Did you run "
            f"`make manifest-slurm`?"
        )

    def _load(self) -> None:
        records = self._load_manifest_records()
        series_to_split, self.cv = self._resolve_splits()

        self.cases: List[Case] = []
        for r in records:
            sid = str(r.get("series_id", ""))
            if not sid:
                continue

            ct_rel  = r.get("ct_relative_path")   or f"scans/{sid}/ct.nii.gz"
            msk_rel = r.get("mask_relative_path") or f"scans/{sid}/mask.nii.gz"

            # Determine split: splits_5fold.json wins; else manifest's native
            # split column.  Native splits are training/validation/test
            # (per VerSe).  splits_5fold.json collapses non-test to
            # "trainval" so fold() can do the rest.
            split = series_to_split.get(sid) or _coerce_optional_str(r.get("split")) or "unknown"

            shape = (
                _coerce_optional_int(r.get("shape_p")),
                _coerce_optional_int(r.get("shape_i")),
                _coerce_optional_int(r.get("shape_r")),
            )
            spacing = (
                _coerce_optional_float(r.get("spacing_p_mm")),
                _coerce_optional_float(r.get("spacing_i_mm")),
                _coerce_optional_float(r.get("spacing_r_mm")),
            )

            self.cases.append(Case(
                series_id            = sid,
                patient_id           = _coerce_optional_str(r.get("patient_id")),
                ct_path              = _resolve_file(self.root, ct_rel),
                mask_path            = _resolve_file(self.root, msk_rel),
                split                = split,
                source_dataset       = _coerce_optional_str(r.get("source_dataset")),
                source_format        = _coerce_optional_str(r.get("source_format")),
                shape                = shape if all(v is not None for v in shape) else None,
                spacing_mm           = spacing if all(v is not None for v in spacing) else None,
                age                  = _coerce_optional_float(r.get("age")),
                sex                  = _coerce_optional_str(r.get("sex")),
                patient_pos          = _coerce_optional_str(r.get("patient_pos")),
                veridah_applied      = _coerce_bool(r.get("veridah_applied", False)),
                veridah_action       = _coerce_optional_str(r.get("veridah_action")),
                veridah_kind         = _coerce_optional_str(r.get("veridah_kind")),
                n_labels             = _coerce_optional_int(r.get("n_labels")) or 0,
                labels_present       = _coerce_labels_list(r.get("labels_present")),
                has_T13              = _coerce_bool(r.get("has_T13", False)),
                has_L6               = _coerce_bool(r.get("has_L6", False)),
                lacks_T12_TLJ_in_FOV = _coerce_bool(r.get("lacks_T12_TLJ_in_FOV", False)),
                lstv_class           = _coerce_optional_str(r.get("lstv_class")) or "normal",
                ct_file_rel          = ct_rel,
                mask_file_rel        = msk_rel,
                _parent              = self,
            ))

        self._by_series: Dict[str, Case] = {c.series_id: c for c in self.cases}

    # ── construction from the Hub ────────────────────────────────────────
    @classmethod
    def from_hub(cls,
                  repo_id:   str,
                  token:     Optional[str] = None,
                  cache_dir: Optional[str] = None) -> "VerSeFusion":
        """Construct a dataset backed by a HuggingFace dataset repo.

        Eagerly downloads only metadata files (manifest, splits, README,
        small auxiliary JSONs).  NIfTIs are fetched lazily on first
        Case.load_ct() / load_mask() call.
        """
        try:
            from huggingface_hub import snapshot_download
        except ImportError as e:
            raise RuntimeError(
                "huggingface_hub not installed.  pip install huggingface_hub"
            ) from e
        local_dir = snapshot_download(
            repo_id   = repo_id,
            repo_type = "dataset",
            token     = token,
            cache_dir = str(Path(os.path.expanduser(cache_dir))) if cache_dir else None,
            allow_patterns = [
                "manifest.json",
                "manifest.csv",
                "manifest_summary.json",
                "splits_5fold.json",
                "splits.csv",
                "corrections/**",
                "orientation_audit.json",
                "sample_selection.json",
                "README.md",
                "LICENSE",
                "LICENSE.txt",
                "dataset_interface.py",
            ],
        )
        inst = cls(local_dir)
        inst._hf_repo_id   = repo_id
        inst._hf_token     = token
        inst._hf_cache_dir = (
            str(Path(os.path.expanduser(cache_dir))) if cache_dir else None
        )
        return inst

    # ── filtering ────────────────────────────────────────────────────────
    def filter(self,
                split:           Optional[str | Sequence[str]] = None,
                lstv_class:      Optional[str | Sequence[str]] = None,
                source_dataset:  Optional[str | Sequence[str]] = None,
                veridah_applied: Optional[bool] = None,
                has_label:       Optional[int] = None,
                present_only:    bool = False) -> List[Case]:
        """Filter cases by metadata attributes.

        Each filter accepts a single value or a list of values to match
        against.  `present_only=True` means present-on-disk RIGHT NOW —
        for HF-backed datasets that haven't fetched the data yet this
        will return an empty list.
        """
        def _as_list(x):
            if x is None: return None
            return [x] if isinstance(x, str) else list(x)

        sp  = _as_list(split)
        lc  = _as_list(lstv_class)
        sd  = _as_list(source_dataset)

        out = list(self.cases)
        if sp:  out = [c for c in out if c.split in sp]
        if lc:  out = [c for c in out if c.lstv_class in lc]
        if sd:  out = [c for c in out if c.source_dataset in sd]
        if veridah_applied is not None:
            out = [c for c in out if bool(c.veridah_applied) == bool(veridah_applied)]
        if has_label is not None:
            out = [c for c in out if c.has_label(int(has_label))]
        if present_only:
            out = [c for c in out if c.exists()]
        return out

    # ── split accessors ──────────────────────────────────────────────────
    def test_set(self) -> List[Case]:
        """Fixed test holdout (patient-level), per splits_5fold.json or
        the manifest's native `split` column."""
        return [c for c in self.cases if c.split == "test"]

    def trainval(self) -> List[Case]:
        """Train+val pool — everything not in the test holdout.

        Native VerSe splits are training/validation; the splits_5fold.json
        path collapses both into "trainval".  We accept all three labels
        here so the same code works whichever splits source is in play.
        """
        keep = {"training", "validation", "trainval"}
        return [c for c in self.cases if c.split in keep]

    def fold(self, i: int) -> Tuple[List[Case], List[Case]]:
        """Return (train_cases, val_cases) for fold i.

        Lookup is by series_id against splits_5fold.json fold[i].
        Raises RuntimeError if no CV folds are available.
        """
        if self.cv is None:
            raise RuntimeError(
                f"No 5-fold CV found at {self.root}/splits_5fold.json.  "
                f"Run `python -m verse_pipeline.splits_builder` "
                f"or `make splits-slurm` to produce one."
            )
        folds = self.cv.get("folds", [])
        if not 0 <= i < len(folds):
            raise IndexError(f"fold {i} out of range [0, {len(folds)})")
        train_set = set(folds[i].get("train_series_ids", []))
        val_set   = set(folds[i].get("val_series_ids",   []))
        train = [c for c in self.cases if c.series_id in train_set]
        val   = [c for c in self.cases if c.series_id in val_set]
        return train, val

    @property
    def n_folds(self) -> int:
        if not self.cv:
            return 0
        return len(self.cv.get("folds", []))

    def splits(self) -> Tuple[List[Case], List[Case], List[Case]]:
        """Backward-compatible 3-tuple (train, val, test) — train is the
        full train+val pool, val is empty.  Use fold(i) for real splits."""
        return self.trainval(), [], self.test_set()

    # ── lookup ───────────────────────────────────────────────────────────
    def get(self, series_id: str) -> Optional[Case]:
        return self._by_series.get(str(series_id))

    def __len__(self) -> int:
        return len(self.cases)

    def __iter__(self):
        return iter(self.cases)

    # ── stats ────────────────────────────────────────────────────────────
    def stats(self) -> str:
        from collections import Counter
        sp   = Counter(c.split for c in self.cases)
        lst  = Counter(c.lstv_class for c in self.cases)
        sd   = Counter(c.source_dataset or "?" for c in self.cases)
        fmt  = Counter(c.source_format or "?"  for c in self.cases)
        n_present = sum(1 for c in self.cases if c.exists())
        n_t13     = sum(1 for c in self.cases if c.has_T13)
        n_l6      = sum(1 for c in self.cases if c.has_L6)
        n_trunc   = sum(1 for c in self.cases if c.lacks_T12_TLJ_in_FOV)
        n_ver     = sum(1 for c in self.cases if c.veridah_applied)
        n_pats    = len({c.patient_id for c in self.cases if c.patient_id})

        lines = [
            "VerSeFusion",
            f"  root:            {self.root}",
            f"  scans:           {len(self.cases)}  (present on disk: {n_present})",
            f"  unique patients: {n_pats}",
            f"  splits:          {dict(sp)}",
            f"  lstv_class:      {dict(lst)}",
            f"  source_dataset:  {dict(sd)}",
            f"  source_format:   {dict(fmt)}",
            f"  flags:           has_T13={n_t13}  has_L6={n_l6}  truncated={n_trunc}",
            f"  veridah_applied: {n_ver}",
            f"  cv folds:        {self.n_folds}",
        ]
        if self.splits_schema_version:
            lines.append(f"  splits source:   schema_v{self.splits_schema_version}  "
                         f"scheme={self.splits_scheme or '-'}")
        else:
            lines.append("  splits source:   (manifest native splits; no CV)")
        if self._hf_repo_id:
            lines.append(
                f"  hf-backed:       {self._hf_repo_id}  "
                f"(NIfTIs fetched lazily; cache_dir={self._hf_cache_dir or 'default'})"
            )
        return "\n".join(lines)

    def __repr__(self) -> str:
        return f"VerSeFusion(root={self.root!s}, n_scans={len(self)}, n_folds={self.n_folds})"


# ============================================================================
# PyTorch Dataset adapter
# ============================================================================

try:
    import torch
    from torch.utils.data import Dataset
    _HAS_TORCH = True
except ImportError:
    _HAS_TORCH = False
    Dataset = object  # type: ignore


class VerSeFusionDataset(Dataset):
    """PyTorch Dataset yielding per-case tensors from NIfTI files.

    Split selection:
        split="trainval"           — whole train+val pool
        split="test"               — fixed test holdout
        split=("fold", 0, "train") — fold 0 train side of 5-fold CV
        split=("fold", 0, "val")   — fold 0 val side
        split="all"                — every scan

    HF-backed roots fetch NIfTIs lazily on first __getitem__.  With
    num_workers>0, multiple workers may race to fetch the same case —
    huggingface_hub uses file locks to make this safe.
    """

    def __init__(self,
                 root,
                 split=("fold", 0, "train"),
                 lstv_class: Optional[str | Sequence[str]] = None,
                 transform=None,
                 cache_dir: Optional[str] = None):
        if not _HAS_TORCH:
            raise RuntimeError("torch is required for VerSeFusionDataset")

        # Auto-detect HF vs local
        root_path = Path(os.path.expanduser(str(root)))
        if root_path.exists() and (root_path / "manifest.json").exists():
            self._ds = VerSeFusion(root_path)
        else:
            self._ds = VerSeFusion.from_hub(repo_id=str(root), cache_dir=cache_dir)

        self.split     = split
        self.transform = transform

        if isinstance(split, tuple) and len(split) == 3 and split[0] == "fold":
            _, fold_i, side = split
            tr, va = self._ds.fold(int(fold_i))
            cases = tr if side == "train" else va
        elif split == "test":
            cases = self._ds.test_set()
        elif split == "trainval":
            cases = self._ds.trainval()
        elif split == "all":
            cases = list(self._ds.cases)
        else:
            raise ValueError(f"Unknown split spec: {split!r}")

        # Optional further filter
        if lstv_class is not None:
            lc = [lstv_class] if isinstance(lstv_class, str) else list(lstv_class)
            cases = [c for c in cases if c.lstv_class in lc]

        # For HF-backed: don't filter on present_only (files arrive lazily)
        if self._ds._hf_repo_id:
            self.cases: List[Case] = list(cases)
        else:
            self.cases = [c for c in cases if c.exists()]

    def __len__(self) -> int:
        return len(self.cases)

    def __getitem__(self, idx: int) -> dict:
        c = self.cases[idx]
        ct_np,  affine = c.load_ct()
        msk_np, _      = c.load_mask()
        return self._collate(c, ct_np, msk_np, affine)

    def _collate(self, c: Case, ct_np, msk_np, affine) -> dict:
        ct   = torch.from_numpy(ct_np.astype("float32")).unsqueeze(0)   # (1, P, I, R)
        mask = torch.from_numpy(msk_np.astype("int64"))                  # (P, I, R)
        item = {
            "ct":         ct,
            "mask":       mask,
            "affine":     torch.from_numpy(affine.astype("float32")),
            "series_id":  c.series_id,
            "patient_id": c.patient_id or "",
            "split":      c.split,
            "meta": {
                "source_dataset":       c.source_dataset,
                "source_format":        c.source_format,
                "spacing_mm":           c.spacing_mm,
                "shape":                c.shape,
                "age":                  c.age,
                "sex":                  c.sex,
                "veridah_applied":      c.veridah_applied,
                "veridah_action":       c.veridah_action,
                "lstv_class":           c.lstv_class,
                "has_T13":              c.has_T13,
                "has_L6":               c.has_L6,
                "lacks_T12_TLJ_in_FOV": c.lacks_T12_TLJ_in_FOV,
                "n_labels":             c.n_labels,
                "labels_present":       list(c.labels_present),
            },
        }
        if self.transform is not None:
            item = self.transform(item)
        return item


# ============================================================================
# CLI smoke test
# ============================================================================

if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(description="Smoke test: load + print stats.")
    ap.add_argument("--root", required=True,
                    help="Local dataset dir OR HF repo_id (e.g. user/repo)")
    ap.add_argument("--cache_dir", default=None)
    args = ap.parse_args()

    root_path = Path(os.path.expanduser(args.root))
    if root_path.exists():
        ds = VerSeFusion(root_path)
    else:
        ds = VerSeFusion.from_hub(args.root, cache_dir=args.cache_dir)
    print(ds.stats())

    print(f"\ntest / trainval: {len(ds.test_set())} / {len(ds.trainval())}")
    if ds.n_folds > 0:
        tr, va = ds.fold(0)
        print(f"fold 0 train/val: {len(tr)} / {len(va)}")

    sample = ds.trainval() or list(ds.cases)
    if sample:
        c = sample[0]
        print(f"\nfirst case:")
        print(f"  series_id:   {c.series_id}")
        print(f"  patient_id:  {c.patient_id}")
        print(f"  split:       {c.split}")
        print(f"  lstv_class:  {c.lstv_class}")
        print(f"  ct_path:     {c.ct_path}  (exists={c.ct_path.exists()})")
        print(f"  mask_path:   {c.mask_path}  (exists={c.mask_path.exists()})")
        print(f"  spacing_mm:  {c.spacing_mm}")
        print(f"  shape:       {c.shape}")
        print(f"  veridah:     applied={c.veridah_applied}  action={c.veridah_action}")
        print(f"  n_labels:    {c.n_labels}")
