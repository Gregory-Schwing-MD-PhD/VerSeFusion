# Pretrained Weights

The 5-fold nnU-Net checkpoints accompanying this paper are hosted on
HuggingFace because they exceed GitHub's repository size limits (~14 GB total
across the 5 folds × 2 checkpoints).

**Hub URL**: https://huggingface.co/anonymous-neurips-ED/spinopelvic-seg-weights

## Contents

For each cross-validation fold (`fold_0` through `fold_4`):

```
nnUNetTrainerWandB_500ep_LSTVOversample__nnUNetResEncUNetPlans_100G__3d_fullres/
├── fold_0/
│   ├── checkpoint_best.pth       (~1.4 GB, best validation-Dice checkpoint)
│   ├── checkpoint_final.pth      (~1.4 GB, last epoch)
│   ├── debug.json                (trainer config)
│   ├── progress.png              (loss curves)
│   └── training_log_*.txt        (full training log)
├── fold_1/  ...
├── fold_2/  ...
├── fold_3/  ...
└── fold_4/  ...

dataset.json
plans.json
dataset_fingerprint.json
```

## Download

Install the HuggingFace CLI:

```bash
pip install huggingface-hub
```

Download all checkpoints into a local nnU-Net results tree:

```bash
huggingface-cli download anonymous-neurips-ED/spinopelvic-seg-weights \
    --local-dir nnUNet_results/Dataset803_SpineSurgCTFullMerged
```

Or just the best checkpoint from one fold:

```bash
huggingface-cli download anonymous-neurips-ED/spinopelvic-seg-weights \
    nnUNetTrainerWandB_500ep_LSTVOversample__nnUNetResEncUNetPlans_100G__3d_fullres/fold_0/checkpoint_best.pth \
    --local-dir nnUNet_results/Dataset803_SpineSurgCTFullMerged
```

## Inference

Once the weights are in place, run inference with nnU-Net v2:

```bash
export nnUNet_results=$(pwd)/nnUNet_results

nnUNetv2_predict \
    -i  path/to/input/cts \
    -o  path/to/output/predictions \
    -d  803 \
    -c  3d_fullres \
    -tr nnUNetTrainerWandB_500ep_LSTVOversample \
    -p  nnUNetResEncUNetPlans_100G \
    -f  all                 # ensemble across all 5 folds
```

For single-fold inference, replace `-f all` with e.g. `-f 0`.

## Training your own

The pipeline in this repository produces the unified scan-dirs that feed
into the nnU-Net training stage (chunk 2 of the paper — `reorient`,
`veridah`, and the nnU-Net trainer variant). Training a single fold takes
roughly 60-80 hours on an H200 GPU at 500 epochs.

See the paper's supplementary materials for the full training recipe,
hyperparameters, and ablation details.

## License

Weights are released under CC BY 4.0, the same license as the underlying
VerSe dataset.
