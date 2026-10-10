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

## DeepLabV3+ / U-Net

DeepLabV3+ 與 U-Net 目前維持 3 類：

```text
0 = Background
1 = Grassland
2 = Barren
```

## SegFormer

SegFormer 現在改成 **11 類**。

網站會直接從：

```text
models/best_model_segformer/config.json
```

讀取：

```text
num_labels
id2label
label2id
```

並使用：

```python
SegformerForSemanticSegmentation.from_pretrained(...)
SegformerImageProcessor.from_pretrained(...)
```

載入：

```text
config.json
model.safetensors
preprocessor_config.json
```

網站會驗證：

```text
num_labels = 11
```

如果模型不是 11 類會直接提示錯誤。

### 網頁顯示

選擇 SegFormer 時，土地覆蓋分析會改為動態顯示 11 個類別的：

- Class ID
- 類別名稱
- Pixel 比例
- 對應顏色

類別名稱直接使用 SegFormer 的 `config.json -> id2label`，因此不需要在網站程式中手動寫死 11 個名稱。

### 碳匯

目前碳匯仍以 Grassland 類別計算。網站會從 `id2label` 自動尋找名稱包含：

```text
Grassland
Grass
草地
草坪
```

的類別。

因此若你的 11 類使用其他名稱，需再指定哪些 Class ID 應納入「植被面積」與碳匯計算。

模型資料夾與大型權重不建議提交 GitHub。
