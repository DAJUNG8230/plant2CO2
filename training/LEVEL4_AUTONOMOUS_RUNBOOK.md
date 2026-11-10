# Level 4 — Codex 自主迭代訓練（前景程序）

從 Level 3 的一次或兩次 pilot，升級為連續「分析 → 選方案 → 跑配對對照組與實驗組 → 比較 → 保存更佳權重 → 下一輪」的自主搜尋。**這不是 ChatGPT 雲端背景工作**；PowerShell 和電腦必須保持開啟。所有 GPU 訓練、Codex CLI 呼叫都在你的 Windows 本機執行。

## 用法

1. 確認舊的 Level 3 程序完全停止；避免兩份訓練程序共用 GPU。
2. 在 PowerShell 執行：

```powershell
cd "C:\Users\wxes1\OneDrive\桌面\plant2CO2"
git fetch origin
git switch feature/deeplab-v5-level4-autonomous-loop
git pull --ff-only
.\.venv\Scripts\python.exe -m unittest discover -s tests -p test_level4_loop.py
.\.venv\Scripts\python.exe -m unittest discover -s tests -p test_level3_runtime_recovery.py
```

3. **預覽（不會動 GPU）**：

```powershell
.\.venv\Scripts\python.exe training\level4_loop.py --assume-baseline-stopped --reasoning-effort high
```

4. 確認 Codex 輸出的計畫後，**授權本機自動執行**：

```powershell
.\.venv\Scripts\python.exe training\level4_loop.py --assume-baseline-stopped --execute --max-rounds 12 --max-hours 24 --max-stall 5 --epochs 8 --target-miou 0.80 --reasoning-effort high
```

若想讓 Codex 在每輪判斷時花更多思考 Token，可將 `--reasoning-effort high` 改為 `--reasoning-effort xhigh`（需目前 Codex 模型支援）。每輪 Codex 可看到原始最佳驗證歷史、當前勝出者的混淆矩陣、前 6 組實驗的逐 Epoch 紀錄，以及前 12 輪的選擇與得失。提高推理等級會增加耗時與 Codex 額度使用，並不保證指標提升。

## 自動修正項目與安全限制

- **只允許預先定義的安全超參數候選**：Shrub/Cultivated/Herbaceous/Trees 的類別 CrossEntropy 權重微調，以及 Fine-tuning LR = `1e-5` 或 `5e-6`。不允許 Codex 自行編輯 Python、資料集、LUT、模型架構，也不能執行其輸出的 shell 指令。
- 候選由當前最佳權重對應的最佳 Epoch 混淆矩陣自動生成，判斷 GT/Predicted 方向；對降低／提高權重保留多個有限強度。LR 候選不必有類別面積偏差。
- 每輪訓練固定跑「未改參數對照組 + Codex 選定實驗組」，使用**相同當前最佳權重起點**、相同隨機種子 `42`、相同資料集與最多 `8` 個 Epoch。兩個輸出路徑各自獨立。
- 模型勝出條件：對比目前保存的最佳權重，驗證植被四類 mIoU **至少提升 0.2 百分點**，且 Binary vegetation IoU **最多下降 0.5 百分點**。對照組與實驗組**兩者都可能成為最佳模型**；驗證植被 mIoU 必須超越前一最佳至少 0.2 百分點、Binary vegetation IoU 最多下降 0.5 百分點；若已有可比較的逐張覆蓋率 MAE，則新模型的 MAE 不可惡化超過 0.5 百分點。只有達標才更新下一輪的最佳權重起點；原始 Level 2 `best_model.pth` 永不覆蓋。
- 預設最多 **12 輪，每輪 2 個 GPU 訓練，最長 24 小時**（時間上限在每輪與對照組結束後檢查，不會強制中斷正在運行的單一訓練），連續 5 輪未改善即停止；也會因達成 80% 植被 mIoU、磁碟可用空間低於 15 GB、GPU 溫度已達 80°C、可用候選耗盡、Codex 判斷 skip 或訓練異常而停止。
- 正常執行完每一輪會記錄 `outputs_training_v5_experiments/monitor/level4_state.json`，每組的 `level4_comparison.json` 也獨立儲存。所有輸出受 `.gitignore` 保護。

## 繼續／中斷

預覽會建立或更新 Level 4 狀態檔，但不啟動模型；緊接著使用 `--execute` 即可。若曾正常停止，且要增加時間或輪數，再執行原指令加 `--resume-search`。如果程序被 Ctrl+C 或異常中斷，請先確定所有 `train.py` 程序都已停止，再使用 `--resume-search --skip-incomplete` 明確宣告**跳過未完成的一輪**（不會刪除或覆寫中斷的權重）。

```powershell
.\.venv\Scripts\python.exe training\level4_loop.py --assume-baseline-stopped --execute --resume-search --skip-incomplete --reasoning-effort high
```

若上次是被強制關閉而留下 `outputs_training_v5_experiments/level4_loop.lock`，**不要直接刪除鎖**。先確定訓練程序與 Level 4 控制器都已停止，人工檢查狀態後再處理。

## 指標與研究限制

系統每輪報告與**未改動參數的配對對照組**差異，並區分「對照組有所改善」和「真的超越目前最佳權重」。原始最佳 63.26% 是保護基準。模型仍可能過擬合驗證集（多次自主嘗試會增加選擇偏差），正式專題成果必須在獨立、未參與方案搜尋的固定測試集驗證。新 Level 4 訓練會直接利用現有驗證批次的 GT Mask 與模型預測，記錄**每張 512×512 裁切影像**在有效像素上的植被覆蓋率絕對誤差平均（`coverage_mae_pp`，百分點），同時排除 ignore=255。這是有真實 GT 支持的驗證裁切影像 MAE，不代表原始整張空拍圖的 GSD 面積或碳匯估算準確率。原本 Level 2 舊的 history.csv 不添加欄位、不能直接與 Level 4 MAE 比較，必須在同一固定資料集獨立評估才能正式比較。

## Dashboard

原本的獨立審核中心 http://127.0.0.1:8765/ 新增「Level 4：Codex 自主迭代訓練」區塊，顯示輪數、目前最佳 mIoU、最近一次決策理由及選出的最佳權重路徑。`app.py` 不變。

## 驗證聲明

已修改 GitHub 程式並新增離線單元測試，但未在使用者 Windows / CUDA / 完整 50GB 資料上完成端到端測試。應先確認全部單元測試通過，再預覽並明確授權 `--execute`。

**注意**：前 12 輪選擇紀錄會提供給 Codex，最近 6 輪的對照組／實驗組完整逐 Epoch 訓練曲線與混淆矩陣也會放進 Codex 的分析內容。實際 Token 消耗受 Codex 訂閱、模型和上下文限制影響，並非固定額度。
