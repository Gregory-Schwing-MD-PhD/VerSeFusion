"""
verse_pipeline.dataset — VerSeFusionDataset, a manifest-aware dataset interface.

Designed to be consumed three ways:
  1. Pure-Python iteration / pandas filtering (no torch dep)
  2. PyTorch DataLoader (just wraps; __len__ + __getitem__ are present)
  3. nnU-Net conversion (use the manifest's split / cv_fold columns directly)

Examples
--------

    # From a local staging dir (e.g. data/hf_staging/)
    from verse_pipeline.dataset import VerSeFusionDataset

    ds = VerSeFusionDataset("data/hf_staging")              # all 374
    ds = VerSeFusionDataset("data/hf_staging", split="training")
    ds = VerSeFusionDataset("data/hf_staging",
                            cv_fold=0, cv_role="train")
    ds = VerSeFusionDataset("data/hf_staging",
                            cv_fold=0, cv_role="val")
    ds = VerSeFusionDataset("data/hf_staging",
                            lstv_class="t13_supernumerary")
    ds = VerSeFusionDataset("data/hf_staging",
                            lstv_class=["lumbarization", "t13_supernumerary"])

    # From HuggingFace
    ds = VerSeFusionDataset.from_hf("gregoryschwingmdphd/VerseFusion",
                                     split="test")

    # Iteration — paths only
    for item in ds:
        # item is a dict with: series_id, patient_id, ct_path, mask_path,
        # spacing_mm, metadata (full manifest row)
        print(item["series_id"], item["ct_path"])

    # Eager loading — fills item["ct"], item["mask"], item["affine"]
    ds_eager = VerSeFusionDataset("data/hf_staging", load_data=True)

    # PyTorch
    from torch.utils.data import DataLoader
    loader = DataLoader(ds, batch_size=1, collate_fn=lambda batch: batch)
"""

from __future__ import annotations

import json
import logging
from collections.abc import Callable, Iterable, Sequence
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

log = logging.getLogger("verse.dataset")


_VALID_SPLITS    = ("training", "validation", "test")
_VALID_CV_ROLES  = ("train", "val")


def _coerce_str_list(x: str | Sequence[str] | None) -> list[str] | None:
    if x is None:
        return None
    if isinstance(x, str):
        return [x]
    return list(x)


class VerSeFusionDataset:
    """Manifest-aware view over a staged VerSeFusion dataset directory.

    Parameters
    ----------
    root:
        Path to a dataset root that contains `manifest.csv` and a `scans/`
        directory with per-subject `ct.nii.gz` and `mask.nii.gz`.  Works for
        the local staging dir produced by `verse_pipeline.hf_export` and
        for a downloaded HuggingFace snapshot.
    split:
        Filter on the `split` column.  Accepts a single string or a list.
        One of: "training", "validation", "test".
    cv_fold:
        Filter on `cv_fold`.  If provided, also requires `cv_role`.
    cv_role:
        Required when `cv_fold` is set.  "train" returns subjects in
        train+val whose cv_fold ≠ the chosen fold; "val" returns subjects
        whose cv_fold == the chosen fold.  Test subjects are excluded
        regardless of cv_role.
    lstv_class:
        Filter on `lstv_class`.  Accepts a single class string or a list.
    veridah_applied:
        If True, keep only VERIDAH-corrected scans.  If False, keep only
        uncorrected.  None = no filter.
    load_data:
        If True, `__getitem__` calls nibabel and fills `ct`, `mask`,
        `affine`.  Default False — paths-only mode is much faster.
    transform:
        Optional callable applied to each item before return.
    """

    def __init__(
        self,
        root:            str | Path,
        split:           str | Sequence[str] | None = None,
        cv_fold:         int | None = None,
        cv_role:         str | None = None,
        lstv_class:      str | Sequence[str] | None = None,
        veridah_applied: bool | None = None,
        load_data:       bool = False,
        transform:       Callable[[dict[str, Any]], dict[str, Any]] | None = None,
    ):
        self.root = Path(root)
        self.load_data = load_data
        self.transform = transform

        manifest_csv = self.root / "manifest.csv"
        if not manifest_csv.exists():
            raise FileNotFoundError(
                f"manifest.csv not found at {manifest_csv}.  Did you run "
                f"`make manifest-slurm` (stage 10) before staging?"
            )
        self.manifest = pd.read_csv(manifest_csv)

        # Apply filters
        df = self.manifest
        df = self._filter_split(df, _coerce_str_list(split))
        df = self._filter_cv(df, cv_fold, cv_role)
        df = self._filter_lstv(df, _coerce_str_list(lstv_class))
        df = self._filter_veridah(df, veridah_applied)
        self.df = df.reset_index(drop=True)

        log.info("VerSeFusionDataset: %d / %d subjects after filtering",
                 len(self.df), len(self.manifest))

    # ----- filters --------------------------------------------------------

    @staticmethod
    def _filter_split(df: pd.DataFrame, splits: list[str] | None) -> pd.DataFrame:
        if not splits:
            return df
        bad = [s for s in splits if s not in _VALID_SPLITS]
        if bad:
            raise ValueError(f"Invalid split(s) {bad}; valid: {_VALID_SPLITS}")
        return df[df["split"].isin(splits)]

    @staticmethod
    def _filter_cv(df: pd.DataFrame,
                    cv_fold: int | None,
                    cv_role: str | None) -> pd.DataFrame:
        if cv_fold is None and cv_role is None:
            return df
        if cv_fold is None or cv_role is None:
            raise ValueError("cv_fold and cv_role must be set together")
        if cv_role not in _VALID_CV_ROLES:
            raise ValueError(f"cv_role must be one of {_VALID_CV_ROLES}, got {cv_role!r}")

        trainval = df["split"].isin(["training", "validation"])
        if cv_role == "train":
            return df[trainval & (df["cv_fold"] != cv_fold) & (df["cv_fold"] >= 0)]
        else:  # "val"
            return df[trainval & (df["cv_fold"] == cv_fold)]

    @staticmethod
    def _filter_lstv(df: pd.DataFrame, classes: list[str] | None) -> pd.DataFrame:
        if not classes:
            return df
        return df[df["lstv_class"].isin(classes)]

    @staticmethod
    def _filter_veridah(df: pd.DataFrame, flag: bool | None) -> pd.DataFrame:
        if flag is None:
            return df
        return df[df["veridah_applied"].astype(bool) == bool(flag)]

    # ----- access ---------------------------------------------------------

    def __len__(self) -> int:
        return len(self.df)

    def __iter__(self) -> Iterable[dict[str, Any]]:
        for i in range(len(self)):
            yield self[i]

    def __getitem__(self, idx: int) -> dict[str, Any]:
        row = self.df.iloc[idx]
        ct_path   = self.root / str(row["ct_relative_path"])
        mask_path = self.root / str(row["mask_relative_path"])

        # Parse labels_present back into a list of ints
        labels_present = []
        if pd.notna(row.get("labels_present")):
            try:
                labels_present = json.loads(row["labels_present"])
            except (TypeError, json.JSONDecodeError):
                labels_present = []

        item: dict[str, Any] = {
            "series_id":       str(row["series_id"]),
            "patient_id":      None if pd.isna(row.get("patient_id")) else str(row["patient_id"]),
            "split":           str(row["split"]),
            "cv_fold":         int(row["cv_fold"]),
            "ct_path":         ct_path,
            "mask_path":       mask_path,
            "spacing_mm":      (
                float(row["spacing_p_mm"]) if pd.notna(row.get("spacing_p_mm")) else None,
                float(row["spacing_i_mm"]) if pd.notna(row.get("spacing_i_mm")) else None,
                float(row["spacing_r_mm"]) if pd.notna(row.get("spacing_r_mm")) else None,
            ),
            "lstv_class":      str(row["lstv_class"]),
            "labels_present":  labels_present,
            "veridah_applied": bool(row["veridah_applied"]),
            "metadata":        row.to_dict(),
        }

        if self.load_data:
            import nibabel as nib   # only import when needed
            ct_img  = nib.load(str(ct_path))
            msk_img = nib.load(str(mask_path))
            item["ct"]     = np.asarray(ct_img.get_fdata(),  dtype=np.float32)
            item["mask"]   = np.asarray(msk_img.dataobj,     dtype=np.int32)
            item["affine"] = np.asarray(ct_img.affine,       dtype=np.float64)

        if self.transform is not None:
            item = self.transform(item)
        return item

    # ----- HuggingFace ----------------------------------------------------

    @classmethod
    def from_hf(cls,
                 repo_id:    str,
                 cache_dir:  str | Path | None = None,
                 token:      str | None = None,
                 **kwargs) -> "VerSeFusionDataset":
        """Download the dataset snapshot and open it.

        Extra kwargs are forwarded to the constructor (split, cv_fold,
        cv_role, lstv_class, veridah_applied, load_data, transform).
        """
        try:
            from huggingface_hub import snapshot_download
        except ImportError as e:
            raise ImportError(
                "huggingface_hub is required for VerSeFusionDataset.from_hf().  "
                "Install with: pip install huggingface_hub"
            ) from e
        local = snapshot_download(
            repo_id=repo_id,
            repo_type="dataset",
            cache_dir=str(cache_dir) if cache_dir else None,
            token=token,
        )
        return cls(local, **kwargs)

    # ----- handy summaries -------------------------------------------------

    def summary(self) -> dict[str, Any]:
        """Quick stats on the *filtered* view."""
        return {
            "n_subjects":     int(len(self.df)),
            "n_patients":     int(self.df["patient_id"].nunique()),
            "splits":         dict(self.df["split"].value_counts()),
            "lstv_class":     dict(self.df["lstv_class"].value_counts()),
            "cv_folds":       dict(self.df["cv_fold"].value_counts()),
            "veridah_applied": int(self.df["veridah_applied"].sum()),
        }

    def __repr__(self) -> str:
        return (f"VerSeFusionDataset(root={self.root!s}, "
                f"n={len(self)}, "
                f"splits={sorted(self.df['split'].unique())})")
