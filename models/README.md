# models

網站目前支援兩種模型：

```text
models/
├─ deeplabv3plus_best.pth
└─ unet_best.pth
```

DeepLabV3+ 也相容舊檔名：

```text
models/best_model.pth
```

## 建議 checkpoint 格式

```python
{
    "model_state_dict": model.state_dict(),
    "encoder": "resnet34",
    "num_classes": 3,
    "class_names": ["Background", "Grassland", "Barren"],
    "image_size": 512,
    "best_miou": best_miou,
}
```

U-Net 預設使用 ResNet34；DeepLabV3+ 預設使用 ResNet50。若 checkpoint 內包含 encoder、image_size、num_classes 等 metadata，網站會優先使用 checkpoint 內的設定。

模型檔預設由 .gitignore 排除，不直接提交 GitHub。
