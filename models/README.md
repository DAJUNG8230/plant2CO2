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

## SegFormer-B0 v3（組員提供的六類部署權重）

**正確檔案是 `segformer_best_v3.pth`，不要使用之前的 `model.safetensors`。**
已對上傳檔進行只讀結構檢查：`.pth` 的 classifier weight 為 `(6,256,1,1)`，bias 為 `(6,)`；`model.safetensors` 仍為 11 類且採用不同的權重鍵名稱。

從 ChatGPT 提供的部署 ZIP 解壓縮後，網站預期放在：

```text
models/
  segformer_best_v3/
    config.json
    deployment_info.json
    preprocessor_config.json
    segformer_best_v3.pth
```

`inference.py` 將優先載入新目錄，再回退到舊的 `best_model_segformer` 等路徑。新 v3 使用組員專案的 `SegformerConfig` + `SegformerForSemanticSegmentation` + `load_state_dict(strict=True)`，而不是從舊的 `model.safetensors` 直接 `from_pretrained()`。保持 RGB、長邊縮至 512、黑色置中 Padding、ImageNet Normalize，輸出後去除 Padding 並以 NEAREST 恢復原始解析度。

組員的 `requirements.txt` 指定 `transformers==5.22.0`。網站需使用與權重架構相容的 Transformers 版本（如遇大量 missing/unexpected keys 請先核對版本），**不要**改 `ignore_mismatched_sizes=True`。本分支尚未在你的 Windows 環境以實際模型完成端到端驗證。

組員 README 記載 Epoch 10、FLAIR validation mIoU 約 68.29%、Combined validation mIoU 約 69.54%；目前僅為組員報告的數字，缺少完整驗證 CSV 與測試集，論文不能直接將其與其他模型的 vegetation mIoU 進行排名。
