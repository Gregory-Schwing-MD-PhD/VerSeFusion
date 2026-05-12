"""Read TUM's published VerSe demographics CSV.

The CSV (``configs/verse_demographics.csv``) is the authoritative source for
patient identity in VerSe.  Schema (column-by-column):

    subject                 canonical patient ID ("verse014", "verse400", "gl003")
    split                   empty for single-series patients;
                            for multi-series patients, the original MICCAI
                            series ID for this row (e.g. "verse090", "verse155"
                            for the two scans of patient "verse400")
    CT_image_series         position indicator ("1 of 1", "1 of 2", ...)
    verse_2019              1 if subject appears in VerSe 2019 release, else 0
    verse_2020              1 if subject appears in VerSe 2020 release, else 0
    sex (0= f, 1= m)        0 = female, 1 = male, blank = unknown
    age                     integer years (may be blank for sibling-scan rows)

Total rows: 374 image series across 355 unique patients.

The last row of the CSV is a numeric summary line (e.g. "374, ..., 58.7")
which we detect and skip.

This module exposes one ``DemographicRow`` per image series with:
  - ``series_id``  set to the bare MICCAI filename stem (e.g. ``verse090``)
  - ``patient_id`` set to the canonical grouping key (e.g. ``verse400``)
"""

from __future__ import annotations

import csv
import logging
from dataclasses import dataclass
from pathlib import Path

log = logging.getLogger("verse.demographics")


@dataclass(frozen=True)
class DemographicRow:
    """One row of TUM's demographic table, normalised."""
    series_id:  str          # MICCAI filename stem (e.g. "verse090", "gl003")
    patient_id: str          # canonical patient group ID (e.g. "verse400")
    position:   str          # "1 of 1", "1 of 2", "2 of 3", etc.
    in_v19:     bool
    in_v20:     bool
    sex:        str          # "F" | "M" | "?"
    age:        int | None


# Header tokens we expect at the top of the CSV.  We don't pin every column —
# only enough to detect that the user pointed us at the right file.
_EXPECTED_HEADER_PREFIX = ("subject", "split", "CT_image_series",
                          "verse_2019", "verse_2020")


def _is_subject_id(value: str) -> bool:
    """True iff value looks like a VerSe subject ID (verseNNN or glNNN)."""
    return value.startswith(("verse", "gl"))


def load_demographics(csv_path: Path) -> list[DemographicRow]:
    """Parse the demographics CSV; return one ``DemographicRow`` per image series.

    Rows whose ``subject`` column does not start with ``verse`` or ``gl``
    (e.g. the summary row at the bottom) are silently skipped.

    Raises ``ValueError`` if the header doesn't match the expected schema —
    catches the case where the user pointed at the wrong CSV.
    """
    with csv_path.open(newline="", encoding="utf-8") as f:
        reader = csv.reader(f)
        try:
            header = next(reader)
        except StopIteration:
            raise ValueError(f"{csv_path}: empty CSV")

        normalised = tuple(h.strip() for h in header[:len(_EXPECTED_HEADER_PREFIX)])
        if normalised != _EXPECTED_HEADER_PREFIX:
            raise ValueError(
                f"{csv_path}: unexpected header {normalised!r}; "
                f"expected to start with {_EXPECTED_HEADER_PREFIX!r}"
            )

        rows: list[DemographicRow] = []
        skipped = 0

        for raw_row in reader:
            # Pad short rows (in case of trailing-comma omissions)
            while len(raw_row) < 7:
                raw_row.append("")

            canon    = raw_row[0].strip()
            col_b    = raw_row[1].strip()
            position = raw_row[2].strip()
            in_v19   = raw_row[3].strip() == "1"
            in_v20   = raw_row[4].strip() == "1"
            sex_raw  = raw_row[5].strip()
            age_raw  = raw_row[6].strip()

            # Skip the summary row and any other non-subject lines.
            if not _is_subject_id(canon):
                skipped += 1
                continue

            # Disambiguate series_id vs patient_id by column B's content.
            if _is_subject_id(col_b):
                series_id  = col_b   # multi-series row: B holds this scan's MICCAI id
                patient_id = canon   # A holds the canonical group key
            else:
                series_id  = canon
                patient_id = canon

            if sex_raw == "0":
                sex = "F"
            elif sex_raw == "1":
                sex = "M"
            else:
                sex = "?"

            try:
                age: int | None = int(age_raw) if age_raw else None
            except ValueError:
                # Multi-series sibling rows sometimes leave age blank; tolerate.
                age = None

            rows.append(DemographicRow(
                series_id=series_id, patient_id=patient_id, position=position,
                in_v19=in_v19, in_v20=in_v20, sex=sex, age=age,
            ))

    if skipped:
        log.debug("Skipped %d non-subject row(s) from %s", skipped, csv_path)
    log.info("Loaded %d demographic rows from %s", len(rows), csv_path)
    return rows


def index_by_series(rows: list[DemographicRow]) -> dict[str, DemographicRow]:
    """Return a mapping from series_id to its single ``DemographicRow``.

    Raises ``ValueError`` if duplicate series IDs are encountered, indicating
    either a corrupted CSV or a parser bug.
    """
    out: dict[str, DemographicRow] = {}
    for r in rows:
        if r.series_id in out:
            raise ValueError(
                f"Duplicate series_id {r.series_id!r} in demographics; "
                "the CSV should have one row per image series."
            )
        out[r.series_id] = r
    return out


def patients(rows: list[DemographicRow]) -> dict[str, list[DemographicRow]]:
    """Group rows by ``patient_id``; useful for multi-series accounting."""
    out: dict[str, list[DemographicRow]] = {}
    for r in rows:
        out.setdefault(r.patient_id, []).append(r)
    return out
