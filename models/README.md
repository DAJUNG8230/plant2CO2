# models

將訓練完成的模型權重放在此資料夾：

```text
models/best_model.pth
```

模型檔預設被 `.gitignore` 排除，避免將大型 checkpoint 直接提交到 GitHub。

目前 `inference.py` 若沒有可用的正式模型，會自動使用展示用 RGB segmentation，讓網站仍可完整操作。
