"""Unit tests for verse_pipeline.utils.bids."""

from __future__ import annotations

import pytest

from verse_pipeline.utils.bids import BIDSName, parse_bids_name


@pytest.mark.parametrize(
    "name, expected_subject, expected_kind, expected_ext, expected_entities",
    [
        # Canonical CT
        (
            "sub-verse000_dir-orient_ct.nii.gz",
            "verse000", "ct", "nii.gz", {"dir": "orient"},
        ),
        # Canonical mask
        (
            "sub-verse014_dir-orient_seg-vert_msk.nii.gz",
            "verse014", "msk", "nii.gz", {"dir": "orient", "seg": "vert"},
        ),
        # Centroid JSON with seg-subreg
        (
            "sub-verse014_dir-orient_seg-subreg_ctd.json",
            "verse014", "ctd", "json", {"dir": "orient", "seg": "subreg"},
        ),
        # Preview PNG
        (
            "sub-verse033_dir-orient_seg-vert_snp.png",
            "verse033", "snp", "png", {"dir": "orient", "seg": "vert"},
        ),
        # With extra split-NN entity
        (
            "sub-verse500_split-01_dir-ax_ct.nii.gz",
            "verse500", "ct", "nii.gz", {"split": "01", "dir": "ax"},
        ),
    ],
)
def test_parse_canonical_names(name, expected_subject, expected_kind, expected_ext, expected_entities):
    parsed = parse_bids_name(name)
    assert parsed.subject == expected_subject
    assert parsed.kind == expected_kind
    assert parsed.extension == expected_ext
    assert parsed.entities == expected_entities
    assert parsed.original == name


def test_parse_rejects_missing_subject():
    with pytest.raises(ValueError, match="sub-"):
        parse_bids_name("verse000_dir-orient_ct.nii.gz")


def test_parse_rejects_unknown_suffix():
    with pytest.raises(ValueError, match="suffix"):
        parse_bids_name("sub-verse000_dir-orient_foo.txt")


def test_parse_rejects_malformed_entity():
    with pytest.raises(ValueError, match="entity"):
        parse_bids_name("sub-verse000_dir_ct.nii.gz")


def test_bidsname_accessors():
    n = parse_bids_name("sub-verse014_dir-orient_seg-vert_msk.nii.gz")
    assert n.sub_id == "sub-verse014"
    assert n.direction == "orient"
    assert n.seg == "vert"
    assert n.split is None
    assert n.is_mask is True
    assert n.is_image is False
    assert n.is_centroid_json is False
