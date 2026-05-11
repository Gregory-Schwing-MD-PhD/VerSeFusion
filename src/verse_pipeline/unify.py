"""Unify the VerSe 2019 and VerSe 2020 releases into one subject-keyed tree.

The two releases share ~105 image series (re-released with refreshed
annotations in VerSe20).  This module:

1.  Walks each extracted release tree.
2.  Resolves the subject-level overlap by **subject id** (e.g.
    ``sub-verse014`` exists in both; VerSe20's copy wins by default).
3.  Symlinks (or copies) the surviving files into a flat layout::

        <out_dir>/sub-verseNNN/
            sub-verseNNN_ct.nii.gz
            sub-verseNNN_msk.nii.gz
            sub-verseNNN_ctd.json
            sub-verseNNN_snp.png             (optional)

4.  Records provenance in ``<out_dir>/unify_manifest.json``.

The dedup policy is configurable via ``--prefer {verse20, verse19}``; the
default (VerSe20-wins) matches the upstream maintainers' recommendation that
the newer annotations supersede VerSe19's.

The output filenames intentionally drop the BIDS ``dir-*`` entity — every
subject now carries a *single* canonical CT/mask/centroid, so the
disambiguating entity is no longer needed.  Splits and orientation tags are
recorded in the manifest rather than the filename.
"""

from __future__ import annotations

import argparse
import json
import logging
import shutil
import sys
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Literal

from verse_pipeline.utils.bids import SubjectFiles, discover_subjects

Prefer = Literal["verse19", "verse20"]

log = logging.getLogger("verse.unify")


# =============================================================================
# layout discovery
# =============================================================================

@dataclass(frozen=True)
class ReleaseLayout:
    release:    str          # "verse19" | "verse20"
    splits:     tuple[Path, ...]   # one path per split (training/validation/test)


def discover_release_layout(raw_dir: Path, release: str) -> ReleaseLayout:
    """Find the per-split roots inside a release's raw_dir/<release>/ tree."""
    rel_root = raw_dir / release
    if not rel_root.is_dir():
        raise SystemExit(f"Release directory missing: {rel_root}")

    splits: list[Path] = []
    for sub in sorted(rel_root.iterdir()):
        if not sub.is_dir():
            continue
        if sub.name in {"downloads"} or sub.name.startswith("."):
            continue
        # Heuristic: a split root has rawdata/ AND derivatives/ siblings.
        if (sub / "rawdata").is_dir() or (sub / "derivatives").is_dir():
            splits.append(sub)

    if not splits:
        raise SystemExit(
            f"No split roots (rawdata/+derivatives/) found under {rel_root}.  "
            f"Did the extraction step finish?"
        )
    return ReleaseLayout(release=release, splits=tuple(splits))


# =============================================================================
# dedup
# =============================================================================

@dataclass
class UnifiedSubject:
    subject:        str
    chosen_release: str
    chosen_split:   str
    other_releases: list[str] = field(default_factory=list)
    src_paths:      dict[str, str] = field(default_factory=dict)
    out_paths:      dict[str, str] = field(default_factory=dict)


def collect_per_release(
    raw_dir: Path,
    release: str,
) -> dict[str, tuple[SubjectFiles, str]]:
    """Return ``{subject: (SubjectFiles, split_name)}`` for one release."""
    layout = discover_release_layout(raw_dir, release)
    accum: dict[str, tuple[SubjectFiles, str]] = {}
    for split_root in layout.splits:
        found = discover_subjects(split_root)
        for sub, files in found.items():
            if sub in accum:
                log.warning(
                    "Subject %s appears in multiple splits of %s — keeping first (%s)",
                    sub, release, accum[sub][1],
                )
                continue
            accum[sub] = (files, split_root.name)
    log.info("%s: %d subjects across %d split(s)", release, len(accum), len(layout.splits))
    return accum


def dedup(
    v19: dict[str, tuple[SubjectFiles, str]],
    v20: dict[str, tuple[SubjectFiles, str]],
    prefer: Prefer,
) -> dict[str, UnifiedSubject]:
    """Apply the dedup policy and return a flat ``{subject: UnifiedSubject}`` map."""
    all_subjects = set(v19) | set(v20)
    out: dict[str, UnifiedSubject] = {}

    for sub in sorted(all_subjects):
        in19 = sub in v19
        in20 = sub in v20

        if in19 and in20:
            chosen = "verse20" if prefer == "verse20" else "verse19"
            other = ["verse19"] if chosen == "verse20" else ["verse20"]
        elif in20:
            chosen, other = "verse20", []
        else:
            chosen, other = "verse19", []

        files, split = (v20[sub] if chosen == "verse20" else v19[sub])
        out[sub] = UnifiedSubject(
            subject=sub,
            chosen_release=chosen,
            chosen_split=split,
            other_releases=other,
            src_paths={
                k: str(getattr(files, k))
                for k in ("ct", "msk", "ctd", "snp")
                if getattr(files, k) is not None
            },
        )
    return out


# =============================================================================
# placement
# =============================================================================

def place_subjects(
    unified: dict[str, UnifiedSubject],
    out_dir: Path,
    *,
    mode: Literal["symlink", "copy"],
) -> None:
    """Stage every unified subject into ``out_dir/sub-verseNNN/<kind>.<ext>``.

    Filenames are canonicalised to ``sub-<subject>_<kind>.<ext>`` (no BIDS
    entities), e.g. ``sub-verse014_ct.nii.gz``.
    """
    out_dir.mkdir(parents=True, exist_ok=True)

    suffix_for_kind: dict[str, str] = {
        "ct":  "ct.nii.gz",
        "msk": "msk.nii.gz",
        "ctd": "ctd.json",
        "snp": "snp.png",
    }

    for sub, u in unified.items():
        subj_dir = out_dir / f"sub-{sub}"
        subj_dir.mkdir(exist_ok=True)

        for kind, src in u.src_paths.items():
            dst = subj_dir / f"sub-{sub}_{suffix_for_kind[kind]}"
            # remove stale destination so re-runs are idempotent
            if dst.is_symlink() or dst.exists():
                dst.unlink()

            if mode == "symlink":
                dst.symlink_to(Path(src).resolve())
            else:
                shutil.copy2(src, dst)

            u.out_paths[kind] = str(dst)


# =============================================================================
# CLI
# =============================================================================

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="verse-unify",
        description="Merge VerSe19 and VerSe20 into one dedup'd subject-keyed tree.",
    )
    p.add_argument("--raw_dir",   type=Path, required=True, help="Output of `verse-download`.")
    p.add_argument("--out_dir",   type=Path, required=True, help="Where to stage the unified tree.")
    p.add_argument(
        "--prefer",
        choices=["verse20", "verse19"],
        default="verse20",
        help="On subject overlap, prefer this release.  Default verse20.",
    )
    p.add_argument(
        "--mode",
        choices=["symlink", "copy"],
        default="symlink",
        help="Stage files via symlink (fast) or copy.  Default symlink.",
    )
    p.add_argument("--log_level", default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    v19 = collect_per_release(args.raw_dir, "verse19")
    v20 = collect_per_release(args.raw_dir, "verse20")
    unified = dedup(v19, v20, prefer=args.prefer)

    log.info(
        "Unified: %d subjects (verse19=%d, verse20=%d, overlap=%d, prefer=%s)",
        len(unified),
        len(v19),
        len(v20),
        len(set(v19) & set(v20)),
        args.prefer,
    )

    place_subjects(unified, args.out_dir, mode=args.mode)

    manifest = {
        "n_subjects": len(unified),
        "prefer":      args.prefer,
        "mode":        args.mode,
        "subjects":    [asdict(u) for u in unified.values()],
    }
    (args.out_dir / "unify_manifest.json").write_text(json.dumps(manifest, indent=2))
    log.info("Wrote %s", args.out_dir / "unify_manifest.json")
    return 0


if __name__ == "__main__":
    sys.exit(main())
