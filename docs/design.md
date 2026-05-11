# VerSeFusion — design notes

## Why fuse VerSe19 and VerSe20?

VerSe was released in two iterations, each at a separate MICCAI challenge:

| Iteration | MICCAI | Subjects | Vertebrae | Notes                                              |
|-----------|--------|---------:|----------:|----------------------------------------------------|
| VerSe19   | 2019   |      141 |     1 725 | Initial release.                                    |
| VerSe20   | 2020   |      300 |     4 142 | Enriched for anatomical variants + fracture grades. |

These two releases share **~105 image series**: the VerSe20 maintainers
re-released a subset of VerSe19 cases with refreshed annotations (the
fracture-grading layer was added between the two iterations).  The OSF
mirrors keep both annotation versions live; the GitHub-restructured
BIDS-style distribution on S3 ships them as separate trees.

For most downstream uses, **a single dedup'd corpus is what you actually
want**.  That's what this repo produces.

## Why the GitHub-restructured format (not OSF)?

The upstream README is explicit about this:

> The annotation format of the complete VerSe data is **NOT** identical
> to the one used for the MICCAI challenges.  The OSF repositories also
> point to the MICCAI version of the data and annotations.  Nonetheless,
> we recommend usage of the restructured data and annotations.

So we download the six S3 zips and ignore OSF.

## Why dedup with "VerSe20 wins" as the default?

Two reasons:

  1.  VerSe20's annotations include fracture grading; VerSe19's don't.
      Even if we don't use fracture labels today, keeping the richer
      annotation layer leaves the door open.
  2.  Some VerSe19 cases had labelling errors that VerSe20 corrected
      (per the Liebl/Schinz paper).  Choosing VerSe20 inherits the fixes.

`--prefer verse19` is available for users who specifically want to
reproduce VerSe19-era experiments.

## Why reorient everything to PIR?

The CTSpinoPelvic1K pipeline standardised on **PIR** (Posterior /
Inferior / Right) — the axcode triple in nibabel — as the canonical
orientation throughout, because it matches the orientation the nnU-Net v2
trainer expects after its preprocessing pass.  Re-using that convention
here means:

  * a model trained on CTSpinoPelvic1K can be evaluated on VerSeFusion
    without an inference-time reorientation step;
  * downstream training scripts (`tools/nnunet_wandb_variant.py`) work
    on either dataset without modification.

The reorientation is **lossless** — it's a permutation/flip of voxel
axes, not a resample.  Affines and centroid coordinates are transformed
in lockstep so geometric correspondence is preserved exactly.

## Why explicit anomaly flags?

VerSe encodes enumeration anomalies directly in the segmentation labels:

  * label **25** → L6 (a *lumbarized* lumbosacral transitional vertebra)
  * label **28** → T13 (a *supernumerary* thoracic vertebra)

Most published VerSe benchmarks pool these cases into a single "all
subjects" score, which obscures the failure modes most relevant in
clinical practice (level-counting at the lumbosacral junction).
VerSeFusion stratifies CV folds on the anomaly axis so every fold sees
representative `normal / lstv / t13 / both` cases — and so reported DSC
can be sliced by anomaly category at evaluation time.

## Relationship to VERIDAH

The VERIDAH (Möller 2026) work demonstrates that an **independent
sequence predictor** is the necessary disambiguation component when
the segmentation model alone cannot decide whether the cranial sacral
segment is "L6" or "S1" — i.e. when the morphology of the lowest lumbar
is ambiguous.

VerSeFusion exists in part to give VERIDAH-style methods a properly
stratified evaluation set.  Because VerSe encodes the LSTV label
explicitly, ground truth for the disambiguation step is unambiguous, so
LSTV-stratified DSC and sequence-prediction accuracy can both be
reported.

## What this repo deliberately does *not* do

  * **No fracture-grading inference.**  VerSe20 ships fracture grades but
    they are not propagated into the manifest — that's a separate
    downstream pipeline.
  * **No DICOM ingestion.**  We consume the BIDS-restructured NIfTIs
    that VerSe ships; we do not re-derive them from DICOM.
  * **No registration / atlas alignment.**  Each scan stays in its own
    voxel grid — only the orientation is canonicalised.
  * **No iPhone / on-device export.**  That's a CTSpinoPelvic1K concern,
    not a VerSeFusion one.
