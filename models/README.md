# models

網站目前支援三種模型：

```text
models/
├─ deeplabv3plus_best.pth
├─ unet_best.pth
└─ segformer_best.pth
```

DeepLabV3+ 也相容舊檔名：

```text
models/best_model.pth
```

## DeepLabV3+

建議 checkpoint：

```python
{
    "model_state_dict": model.state_dict(),
    "encoder": "resnet50",
    "num_classes": 3,
    "image_size": 512,
    "best_miou": best_miou,
}
```

## U-Net

建議 checkpoint：

```python
{
    "model_state_dict": model.state_dict(),
    "encoder": "resnet34",
    "num_classes": 3,
    "image_size": 512,
    "best_miou": best_miou,
}
```

## SegFormer

網站預設以 MiT-B0 為 SegFormer 骨幹，權重檔：

```text
models/segformer_best.pth
```

建議保存：

```python
{
    "model_state_dict": model.state_dict(),
    "hf_model_name": "nvidia/segformer-b0-finetuned-ade-512-512",
    "num_classes": 3,
    "image_size": 512,
    "best_miou": best_miou,
}
```

SegFormer 需要 `transformers` 套件；執行 `pip install -r requirements.txt` 即可安裝。

若你的 SegFormer 是用其他 MiT-B1/B2/B3/B4/B5、不同 image processor，或 checkpoint 格式不同，請依實際訓練程式調整 `hf_model_name` 與前處理設定。

模型檔預設由 `.gitignore` 排除，不會直接提交 GitHub。
