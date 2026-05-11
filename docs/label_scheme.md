# VerSe label scheme

VerSe uses a single 1-indexed 28-class labelling that combines anatomical
position (C1 ... T12 / L1 ... L5) with two explicit anomaly classes.

The full scheme is mirrored in `configs/label_scheme.yaml` and consumed
by `verse_pipeline.label_crosswalk`.

## Canonical 28-class scheme

| value | region    | name    | annotated in VerSe? | notes                                  |
|------:|-----------|---------|:-------------------:|----------------------------------------|
|  0    | —         | background | n/a              | implicit                               |
|  1    | cervical  | C1      | yes                 |                                        |
|  2    | cervical  | C2      | yes                 |                                        |
|  3    | cervical  | C3      | yes                 |                                        |
|  4    | cervical  | C4      | yes                 |                                        |
|  5    | cervical  | C5      | yes                 |                                        |
|  6    | cervical  | C6      | yes                 |                                        |
|  7    | cervical  | C7      | yes                 |                                        |
|  8    | thoracic  | T1      | yes                 |                                        |
|  9    | thoracic  | T2      | yes                 |                                        |
| 10    | thoracic  | T3      | yes                 |                                        |
| 11    | thoracic  | T4      | yes                 |                                        |
| 12    | thoracic  | T5      | yes                 |                                        |
| 13    | thoracic  | T6      | yes                 |                                        |
| 14    | thoracic  | T7      | yes                 |                                        |
| 15    | thoracic  | T8      | yes                 |                                        |
| 16    | thoracic  | T9      | yes                 |                                        |
| 17    | thoracic  | T10     | yes                 |                                        |
| 18    | thoracic  | T11     | yes                 |                                        |
| 19    | thoracic  | T12     | yes                 |                                        |
| 20    | lumbar    | L1      | yes                 |                                        |
| 21    | lumbar    | L2      | yes                 |                                        |
| 22    | lumbar    | L3      | yes                 |                                        |
| 23    | lumbar    | L4      | yes                 |                                        |
| 24    | lumbar    | L5      | yes                 |                                        |
| 25    | lumbar    | L6      | yes (when present)  | **lumbarized LSTV**                    |
| 26    | sacrum    | sacrum  | **no**              | scheme placeholder; never labelled     |
| 27    | coccyx    | coccyx  | **no**              | scheme placeholder; never labelled     |
| 28    | thoracic  | T13     | yes (when present)  | **supernumerary cranial vertebra**     |

## Anomaly semantics

* **Label 25 (L6) → LSTV.**  Present iff the subject has a lumbarized
  transitional vertebra at the lumbosacral junction.  Empirically the
  failure mode most segmentation models hit is *not* mis-segmenting the
  voxels but mis-counting them: predicting `last_lumbar` on the L4
  vertebra of an L6 anatomy.  Explicit label 25 lets a downstream
  sequence-prediction step (VERIDAH-style) train to discriminate L5
  from L6 from sacrum.

* **Label 28 (T13) → cranial transitional.**  A 13th thoracic vertebra
  bearing rudimentary ribs.  Rarer than LSTV (≈3–5% in VerSe vs ≈10–15%
  for LSTV).  Frequently co-occurs with L6 (see `verse-lstv --audit`).

* **Labels 26 (sacrum) and 27 (coccyx)** appear in the scheme so that
  *if a future release* annotates them they fit without re-numbering.
  No current VerSe scan has voxels at these values; downstream tooling
  should treat them as undefined.

## Practical notes

* `centroid_json.parse_centroid_json()` does **not** enforce label
  validity; out-of-scheme integers (>28 or 0) propagate through as-is.
  This is intentional — buggy upstream files should be diagnosed in QA,
  not silently dropped.

* `vertebra_count(file, region="thoracic")` counts label 28 (T13) as
  thoracic, matching the anatomical region but **not** the numeric
  range.

* When the same scan carries both L6 and T13 (some elderly cohorts and
  cervical-fusion patients), the anomaly category in
  `placed_manifest.json` is `"both"`.
