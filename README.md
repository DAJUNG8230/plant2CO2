# plant2CO2

AI 植被辨識與碳匯估算系統。  
目前包含 **Flask 網站、Label Studio 標註轉換、DeepLabV3+ 訓練與正式模型推論流程**。

## 系統流程

```text
空拍 RGB 影像
   ↓
DeepLabV3+ semantic segmentation
   ↓
Background / Grassland / Barren
   ↓
土地覆蓋率
   ↓
GSD 面積換算
   ↓
碳匯估算
   ↓
Mask / Overlay / Web Dashboard
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
│  ├─ README.md
│  └─ best_model.pth        # 本機訓練後放這裡，不提交 GitHub
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

瀏覽器：

```text
http://127.0.0.1:5000
```

模型/裝置狀態：

```text
http://127.0.0.1:5000/api/status
```

## 3. 推論模式

### 尚未有模型

如果：

```text
models/best_model.pth
```

不存在，網站會使用 **Demo RGB segmentation**，方便先測試完整 UI 與 Flask 資料流。

### 已有正式模型

`inference.py` 已經完成 DeepLabV3+ 載入與推論。

只要將 `training/train.py` 產生的 checkpoint 放到：

```text
models/best_model.pth
```

下一次分析時會自動：

1. 載入 checkpoint
2. 建立 DeepLabV3+ + ResNet50
3. Resize / ImageNet Normalize
4. GPU 可用時自動使用 CUDA
5. 執行 semantic segmentation
6. 將 logits resize 回原始影像尺寸
7. 產生 class-index mask
8. 產生彩色 Mask / Overlay
9. 計算 Grassland / Barren / Background 百分比
10. 計算植被面積與碳匯

如果你在 Flask 執行期間替換 `best_model.pth`，程式也會根據檔案修改時間重新載入模型。

## 4. Label Studio → Dataset

將 Label Studio JSON 放在：

```text
export.json
```

原始影像放在：

```text
original_images/
```

類別：

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

輸出：

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

預設：

- Model: DeepLabV3+
- Encoder: ResNet50
- Input: 512 × 512
- Classes: 3
- Loss: Cross Entropy + Dice
- Optimizer: AdamW
- Metric: IoU / foreground mIoU

最佳 checkpoint：

```text
outputs_training/best_model.pth
```

訓練完成後複製：

```bat
copy outputs_training\best_model.pth models\best_model.pth
```

然後重新進入網站或直接再次按「開始 AI 分析」即可使用正式模型。

## 6. 確認 GPU

```bat
python -c "import torch; print('CUDA:', torch.cuda.is_available()); print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU')"
```

如果輸出：

```text
CUDA: True
```

則訓練與推論會使用 NVIDIA GPU。

## 注意

- `best_model.pth` 預設不提交 GitHub。
- Demo segmentation 僅用來測試網站流程，不能拿來代表正式模型準確率。
- GSD 與碳匯係數應使用專題最後確認的參數與研究依據。
- 若資料為同一大型空拍影像切割出的相鄰 patch，正式評估建議按場景/田區分組後再切 Train / Val / Test，避免資料洩漏。
