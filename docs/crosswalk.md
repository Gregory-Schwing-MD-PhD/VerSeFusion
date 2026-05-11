# VerSe ↔ CTSpinoPelvic1K crosswalk

## Why a crosswalk is needed

VerSeFusion and CTSpinoPelvic1K cover **overlapping but non-identical**
anatomy:

| structure                     | VerSe        | CTSpinoPelvic1K |
|-------------------------------|:------------:|:---------------:|
| cervical vertebrae (C1–C7)    |      ✅      |       —         |
| thoracic vertebrae (T1–T12)   |      ✅      |       —         |
| T13 (supernumerary thoracic)  |      ✅      |       —         |
| L1–L5                         |      ✅      |      ✅         |
| L6 (LSTV)                     |      ✅      |      ✅         |
| sacrum                        |  scheme only |      ✅         |
| coccyx                        |  scheme only |       —         |
| left / right hip              |      —       |      ✅         |

So the mapping is intrinsically lossy in *both* directions:

  * VerSe → CTSpinoPelvic1K drops everything cranial to L1.
  * CTSpinoPelvic1K → VerSe has no hip labels to write into.

## Forward direction (VerSe → CTSpinoPelvic1K)

This is the operational direction.  It exists so that **models trained
on CTSpinoPelvic1K can be externally validated on VerSeFusion**: the
ground-truth VerSe mask is remapped into the CTSpinoPelvic1K 10-class
scheme, then standard DSC / surface-distance metrics are computed
against the model's prediction.

```
VerSe label  →  CTSpinoPelvic1K label
-----------     ----------------------
20  (L1)     →  1
21  (L2)     →  2
22  (L3)     →  3
23  (L4)     →  4
24  (L5)     →  5
25  (L6)     →  6   (LSTV)
26  (sacrum) →  7   (only relevant if upstream ever annotates it)
1–19, 27, 28 →  0   (background)
```

The forward crosswalk is applied in `verse_pipeline.label_crosswalk`
via a 1-D look-up table over the source-mask `max + 1` range:

```python
from verse_pipeline.label_crosswalk import load_crosswalk, apply_mapping
forward, _ = load_crosswalk()
new_mask = apply_mapping(verse_mask, forward)   # uint8
```

### What this enables (and what it doesn't)

Enables:

  * Cross-dataset DSC for L1–L6 + sacrum.
  * Stratified DSC by anomaly category, since the LSTV label survives the
    crosswalk.

Does **not** enable:

  * Hip evaluation (no hip labels in VerSe).
  * Whole-spine evaluation (cervical/thoracic dropped).
  * Direct comparison to the original VerSe leaderboard, which uses the
    full 28-class scheme.

## Reverse direction (CTSpinoPelvic1K → VerSe)

Defined for completeness but rarely useful:

```
CTSpinoPelvic1K label  →  VerSe label
---------------------     ------------
1  (L1)                →  20
2  (L2)                →  21
3  (L3)                →  22
4  (L4)                →  23
5  (L5)                →  24
6  (L6 / LSTV)         →  25
7  (sacrum)            →  26
8  (left hip)          →  —   (dropped; no VerSe equivalent)
9  (right hip)         →  —
```

Reverse-mapping a CTSpinoPelvic1K prediction onto VerSe coordinates
loses the hip predictions entirely.  If you need to evaluate a
CTSpinoPelvic1K model's *hip* DSC, you must do so on CTSpinoPelvic1K
itself — VerSe cannot serve as an external test set for hips.

## Configuration

Both directions are stored in `configs/label_scheme.yaml`:

```yaml
crosswalk:
  verse_to_ctspinopelvic1k:
    20: 1
    21: 2
    ...
  ctspinopelvic1k_to_verse:
    1: 20
    2: 21
    ...
```

Adding a new target scheme is a matter of appending a new top-level key
to that file and a matching loader to `label_crosswalk.py`.
