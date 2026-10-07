# models

網站目前支援三種模型：

```text
models/
├─ deeplabv3plus_best.pth
├─ unet_best.pth
└─ best_model_segformer/
   ├─ config.json
   ├─ model.safetensors
   └─ preprocessor_config.json
```

## DeepLabV3+

建議檔名：

```text
models/deeplabv3plus_best.pth
```

舊檔名 `models/best_model.pth` 仍可使用。

## U-Net

建議檔名：

```text
models/unet_best.pth
```

## SegFormer

SegFormer 目前直接支援 Hugging Face `save_pretrained()` 輸出格式。

請把組員給你的整個資料夾放到：

```text
models/best_model_segformer/
```

資料夾至少要包含：

```text
config.json
model.safetensors
preprocessor_config.json
```

網站會使用：

```python
SegformerForSemanticSegmentation.from_pretrained(...)
SegformerImageProcessor.from_pretrained(...)
```

直接讀取本機模型與原本的前處理設定，不需要將 `model.safetensors` 轉成 `.pth`。

### SegFormer 類別順序

目前網站統計與顏色設定仍假設：

```text
0 = Background
1 = Grassland
2 = Barren
```

請確認 `config.json` 的 `id2label` / `label2id` 與這個順序一致。

如果組員的類別順序不同，請先修改網站 mapping，不要直接用錯誤順序計算面積與碳匯。

## 套件

SegFormer 需要：

```bat
python -m pip install -r requirements.txt
```

其中 `requirements.txt` 已包含 `transformers`。

模型權重與模型資料夾建議不要提交 GitHub；放在本機 `models/` 即可。
