# Vegetation coverage correction (FLAIR mapping v4)

## Why this is necessary

The website computes vegetation coverage as the fraction of predicted class IDs 2, 3, 4, and 5. The arithmetic in `inference.py` is appropriate for an accurate vegetation mask, but **land-use labels are not necessarily pixel-level live vegetation**.

FLAIR label 12 means **plowed land** (tilled / exposed agricultural soil). Mapping all of its pixels to class 5 (Cultivated) teaches the model that exposed soil is vegetation. In v4 it is mapped to class 0 (Bare Soil).

Other mapped classes, including FLAIR label 11 (agricultural land), can still include uncovered soil. This revision is a **partial label correction, not proof of accurate coverage**. Audit these images before claiming quantitative vegetation coverage or carbon uptake.

## Changes

- `training/prepare_flair_6class.py`: map label 12 to class 0; output newly generated masks under `D:\FLAIR\flair_6class_masks_v4`.
- `training/train.py`: record `v4_plowed_land_bare_soil` inside new checkpoint metadata (without changing the model architecture or loss).
- Existing checkpoints **are not modified** and will continue producing their previous predictions until retrained.

## Rebuild masks before training

From the repository root in Windows CMD (inside the project's Python virtual environment):

```cmd
python training\prepare_flair_6class.py
python training\train.py
```

The first command regenerates FLAIR train/val/test manifest CSVs with paths to v4 masks. This overwrites the old manifest CSVs, so keep a backup if you need reproducibility of the old training run.

The second command trains a fresh DeepLabV3+ model because `RESUME_FROM_BEST = False`. The output remains `outputs_training\best_model.pth`. Move previous checkpoints and training outputs to a named backup folder before starting if you need to preserve them.

**Important:** The website loads `models\deeplabv3plus_best.pth` first, and then `models\best_model.pth`. Training to `outputs_training\best_model.pth` alone **does not update the website**. After evaluation, copy the new best checkpoint to `models\deeplabv3plus_best.pth` (back up the previous file first) and restart Flask.

## How to validate improvement

1. Save a hand-annotated six-class mask for `56.png` and a small set of other images; separately label actual exposed soil between crop rows.
2. Evaluate the **same** images with the old and new checkpoints, record per-class IoU and bare-soil-vs-cultivated confusion.
3. Compare the resulting **vegetation area error** and **coverage percentage-point error** with ground truth. Do not treat visually plausible masks as ground truth.
4. Confirm the source image GSD before interpreting square meters or annual CO2e. The website's carbon coefficient is still one shared coefficient across all vegetation classes.

The U-Net benchmark cited in the project uses different FLAIR label mappings for several classes. To claim a controlled comparison, first unify label definitions and evaluation masks.
