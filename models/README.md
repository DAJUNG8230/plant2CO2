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
├─ segformer_best.pth            # 實際推論用權重
└─ best_model_segformer/         # config + processor
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

網站優先且僅使用 `models/segformer_best.pth` 作為 SegFormer 推論權重。`best_model_segformer/` 資料夾保留 `config.json` 與 `preprocessor_config.json`，僅提供網路架構設定與影像前處理設定，不再使用資料夾中的舊模型權重。\n\n`.pth` 應為 Hugging Face `SegformerForSemanticSegmentation` 的 state_dict，或含 `model_state_dict` / `state_dict` 的 checkpoint；若 checkpoint 內有 `config` 字典會優先使用。必須與六類架構相符。若 `.pth` 來自其他 SegFormer 實作，須先轉換權重，否則會顯示清楚的載入錯誤。

## 重要

舊的 3 類或 11 類模型不能直接當成新的 6 類模型使用。
必須重新訓練，或取得已按照這 6 類訓練完成的 checkpoint。
