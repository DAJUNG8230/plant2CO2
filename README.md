# plant2CO2

AI 植被辨識與碳匯估算系統。  
目前包含 **Flask 展示網站、Label Studio 標註轉換流程、DeepLabV3+ 訓練骨架**。

## 系統流程

```text
空拍 RGB 影像
   ↓
DeepLabV3+ / Demo segmentation
   ↓
Background / Grassland / Barren
   ↓
土地覆蓋率
   ↓
GSD 面積換算
   ↓
碳匯估算
   ↓
Web Dashboard
```

## 專案結構

```text
plant2CO2/
├─ app.py
├─ inference.py
├─ requirements.txt
├─ templates/
│  └─ index.html
├─ models/
│  └─ README.md
├─ uploads/
│  └─ .gitkeep
├─ outputs/
│  └─ .gitkeep
└─ training/
   ├─ check_json.py
   ├─ prepare_dataset.py
   ├─ split_dataset.py
   └─ train.py
```

訓練資料、Label Studio JSON、虛擬環境與大型模型權重均由 `.gitignore` 排除。

## 1. 建立 Python 環境

Windows CMD：

```bat
python -m venv .venv
.venv\Scripts\activate.bat
python -m pip install --upgrade pip
pip install -r requirements.txt
```

## 2. 啟動網站

```bat
python app.py
```

瀏覽器開啟：

```text
http://127.0.0.1:5000
```

網站可以：

- 上傳 JPG / PNG 空拍影像
- 顯示 Original / Mask / Overlay
- 顯示 Grassland / Barren / Background 比例
- 根據 GSD 計算植被面積
- 根據碳匯係數估算 CO₂e
- 匯出分析結果 JSON

## 3. 目前推論模式

現在 `inference.py` 可以在沒有模型權重時使用 **Demo RGB segmentation**，因此網站 clone 下來後即可操作。

這只是介面與資料流測試，不代表正式模型精度。

正式模型完成後，將：

```text
best_model.pth
```

放到：

```text
models/best_model.pth
```

並完成 `inference.py` 中的：

```python
predict_with_model()
```

即可將 Dashboard 切換成真正的 DeepLabV3+ 結果。

## 4. Label Studio → Dataset

將 Label Studio 匯出的 JSON 放在專案根目錄：

```text
export.json
```

將對應原始影像放到：

```text
original_images/
```

目前類別：

```text
0 = Background
1 = 草地 (Grassland)
2 = 裸地 (Barren)
```

執行：

```bat
python training\check_json.py
python training\prepare_dataset.py
python training\split_dataset.py
```

最後會產生：

```text
dataset/
├─ train/
│  ├─ images/
│  └─ masks/
├─ val/
│  ├─ images/
│  └─ masks/
└─ test/
   ├─ images/
   └─ masks/
```

## 5. 訓練 DeepLabV3+

```bat
python training\train.py
```

目前預設：

- Model: DeepLabV3+
- Encoder: ResNet50
- Input: 512 × 512
- Classes: 3
- Loss: Cross Entropy + Dice
- Optimizer: AdamW
- Metric: IoU / foreground mIoU

最佳模型會輸出到：

```text
outputs_training/best_model.pth
```

確認模型後再自行複製到：

```text
models/best_model.pth
```

## 注意

- `export.json`、原始資料集與模型 checkpoint 不直接提交到 GitHub。
- GSD 與碳匯係數目前由 Dashboard 輸入；正式研究版應使用專題確認過的參數與文獻依據。
- 如果影像是從同一張大型空拍影像切割而來，Train / Val / Test 建議按照場景分組，避免相鄰 patch 洩漏造成 mIoU 高估。
