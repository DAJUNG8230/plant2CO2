# models

目前網站與模型統一使用 6 類土地覆蓋分類：

```text
0 = Bare Soil
1 = Other Non-Veg
2 = Herbaceous
3 = Shrub
4 = Trees
5 = Cultivated
```

植被類別為：

```text
2 = Herbaceous
3 = Shrub
4 = Trees
5 = Cultivated
```

網站中的「植被覆蓋率、植被面積、碳匯估算」會加總 2、3、4、5 類。

## 模型檔案

```text
models/
├─ deeplabv3plus_best.pth
├─ unet_best.pth
└─ best_model_segformer/
   ├─ config.json
   ├─ model.safetensors
   └─ preprocessor_config.json
```

## DeepLabV3+ / U-Net

checkpoint 必須使用 6 個輸出類別，且 class order 必須與上述 0~5 完全一致。

建議 checkpoint metadata：

```python
{
    "model_state_dict": model.state_dict(),
    "num_classes": 6,
    "class_names": [
        "Bare Soil",
        "Other Non-Veg",
        "Herbaceous",
        "Shrub",
        "Trees",
        "Cultivated",
    ],
    "image_size": 512,
    "best_miou": best_miou,
}
```

## SegFormer

SegFormer 的 `config.json` 必須設定：

```text
num_labels = 6
```

並建議：

```json
"id2label": {
  "0": "Bare Soil",
  "1": "Other Non-Veg",
  "2": "Herbaceous",
  "3": "Shrub",
  "4": "Trees",
  "5": "Cultivated"
}
```

網站會直接使用 `SegformerImageProcessor` 與 `SegformerForSemanticSegmentation` 載入本機 Hugging Face 模型。

## 重要

舊的 3 類或 11 類模型不能直接當成新的 6 類模型使用。
必須重新訓練，或取得已按照這 6 類訓練完成的 checkpoint。
