# DeepLabV3+ V5: vegetation coverage / six-class validation

This is an **experimental branch**, not an established accuracy improvement. V5 keeps the existing six-class DeepLabV3+ ResNet50 architecture and loss (0.30 weighted CE + 0.30 multiclass Lovasz + 0.20 weighted Dice + 0.20 morphological-gradient L1 boundary).

## Mask policy (FLAIR v5) — identical to the supplied U-Net Stage 3 FLAIR LUT

The DeepLab V5 six-class mapping now matches the U-Net reference exactly, including the questionable land-use definitions. Do not adjust these per-model during a controlled architecture comparison.

| FLAIR original labels | Mapped class |
|---|---|
| 2, 4, 12, 15 | 0 Bare Soil |
| 1, 3, 5, 13, 14, 18 | 1 Other Non-Veg |
| 10 | 2 Herbaceous |
| 8, 17 | 3 Shrub |
| 6, 7, 16 | 4 Trees |
| 9, 11 | 5 Cultivated |
| 0, 19 | 255 Ignore |

**Important:** Agricultural Land (11) may contain exposed soil. Consistent LUTs are necessary for fair model comparison, but they do not make FLAIR land-use labels an accurate pixel-level vegetation coverage ground truth. Evaluate real area estimates on separately hand-annotated masks.

The supplied U-Net project also specifies a separate CUSTOM_LUT: original labels 1 -> 4, 2 -> 2, 3 -> 1, 4 -> 1, 5 -> 0; all others -> 255. Check that existing `dataset/{train,val}/masks` were generated with exactly this LUT. This PR remaps **FLAIR** only; it does not silently rewrite existing custom annotations.

## Safe run (Windows CMD, from project root)

```cmd
git fetch origin
git switch feature/deeplab-v5-label-audit
.venv\Scripts\activate
python training\prepare_flair_6class.py
python training\train.py
```

**Important:** The preparation step **overwrites local `flair_manifests/{train,val,test}.csv`**. Back up manifests first when preserving earlier experiments. It generates new masks in `D:\FLAIR\flair_6class_masks_v5_unet_aligned`, keeping v4 masks intact. Check `IMAGE_ROOT` and `LABEL_ROOT` paths in preparation script for your PC.

The training script now saves in **`outputs_training_v5/`**; previous `outputs_training/best_model.pth` and the website checkpoints are untouched. The script **stops before training** if any of the six classes is absent in the *full* validation masks. Validation crops could still omit some classes, so monitor the printed per-epoch support too.

Each validation pass prints aggregate predicted/GT vegetation pixel coverage and **signed dataset-wide coverage bias in percentage points**, based only on pixels with valid GT masks. This is *not* per-image MAE: errors can cancel. Because validation labels may be coarse, the metric alone does not prove real-world coverage accuracy. Make a small, independently annotated test set before promoting weights.

## Train vs. deploy

V5 uses `RESUME_FROM_BEST = False` and therefore starts new optimization, rather than silently continuing from previous weights. After training, validate against hand-labeled RGB images like `56.png` and compare to v4. Only deploy if coverage error, class IoU and shrubs are improved. Back up the existing `models/deeplabv3plus_best.pth`, then copy the **evaluated** `outputs_training_v5/best_model.pth` to that website path and restart Flask. Never assume a new checkpoint is better.

## Research note

For controlled comparison, use identical LUTs, source images, train/val/test manifests, augmentation, sampling, loss and evaluation. The same U-Net label policy is restored, even though some agricultural-land labels are too coarse for reliable physical vegetation area estimation. Keep a separate manually labeled test set for the latter objective.
