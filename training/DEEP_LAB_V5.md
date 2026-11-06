# DeepLabV3+ V5: vegetation coverage / six-class validation

This is an **experimental branch**, not an established accuracy improvement. V5 keeps the existing six-class DeepLabV3+ ResNet50 architecture and loss (0.30 weighted CE + 0.30 multiclass Lovasz + 0.20 weighted Dice + 0.20 morphological-gradient L1 boundary).

## Mask policy (FLAIR v5)

- Original label **12 plowed land** -> **0 Bare Soil**, inherited from v4.
- Original label **11 agricultural land** -> **255 Ignore** (cannot know actual green pixel coverage from a land-use label). This intentionally reduces supervised pixels; it does not magically relabel soil within farms.
- Original label **8 brushwood** -> **3 Shrub** (coarse provisional annotation). Inspect samples: this boundary is not necessarily a precise shrub delineation.
- Other mappings remain unchanged. Do NOT use V5 class statistics as directly comparable to v3/v4 without noting changed training labels.
- Images with actual fine vegetation vs exposed soil need manual pixel-level annotation for a trustworthy coverage ground truth.

## Safe run (Windows CMD, from project root)

```cmd
git fetch origin
git switch feature/deeplab-v5-label-audit
.venv\Scripts\activate
python training\prepare_flair_6class.py
python training\train.py
```

**Important:** The preparation step **overwrites local `flair_manifests/{train,val,test}.csv`**. Back up manifests first when preserving earlier experiments. It generates new masks in `D:\FLAIR\flair_6class_masks_v5`, keeping v4 masks intact. Check `IMAGE_ROOT` and `LABEL_ROOT` paths in preparation script for your PC.

The training script now saves in **`outputs_training_v5/`**; previous `outputs_training/best_model.pth` and the website checkpoints are untouched. The script **stops before training** if any of the six classes is absent in the *full* validation masks. Validation crops could still omit some classes, so monitor the printed per-epoch support too.

Each validation pass prints aggregate predicted/GT vegetation pixel coverage and **signed dataset-wide coverage bias in percentage points**, based only on pixels with valid GT masks. This is *not* per-image MAE: errors can cancel. Because validation labels may be coarse, the metric alone does not prove real-world coverage accuracy. Make a small, independently annotated test set before promoting weights.

## Train vs. deploy

V5 uses `RESUME_FROM_BEST = False` and therefore starts new optimization, rather than silently continuing from previous weights. After training, validate against hand-labeled RGB images like `56.png` and compare to v4. Only deploy if coverage error, class IoU and shrubs are improved. Back up the existing `models/deeplabv3plus_best.pth`, then copy the **evaluated** `outputs_training_v5/best_model.pth` to that website path and restart Flask. Never assume a new checkpoint is better.

## Research note

This revision also changes the distribution of valid pixels and can harm cultivated supervision. Report that limitation when comparing ablations, and consider relabeling agricultural scenes rather than permanently ignoring label 11.
