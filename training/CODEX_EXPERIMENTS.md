# DeepLab V5 — Codex 自主新實驗規劃器

這是 `feature/deeplab-v5-autonomous-controller` 的延伸，不會更動原本的 `outputs_training_v5_level2/best_model.pth` 或網站 `app.py`。所有新試驗的輸出都放在 `outputs_training_v5_experiments/`，並被 `.gitignore` 排除。

## 能自動做什麼

Codex 讀取 `history.csv`、最佳輪與最後輪六類混淆矩陣及可用固定影像指標，從**已在程式碼中定義的有限候選**選一個或跳過。第一版候選只有兩組，不會接受 Codex 自行撰寫的訓練指令或新的 Loss。

| ID | 唯一變因 | 預期觀察 |
|---|---|---|
| `shrub_weight` | Shrub 的 CrossEntropy 權重 1.45 → 1.65 | Shrub IoU、其他類 IoU、植被四類 mIoU |
| `cultivated_weight` | Cultivated 的 CrossEntropy 權重 0.90 → 1.10 | Cultivated IoU、其他類 IoU、植被四類 mIoU |

其他五個類別權重、資料集、LUT、網路架構、Batch Size、模型初始化架構、Loss 組合均不變。每項訓練從**既有 `best_model.pth`** 讀取模型權重，建立新的 Optimizer/Scheduler/AMP 狀態，不假裝從上一輪的完整 optimizer 狀態續訓。這是新實驗，不是原本 Epoch 42 接到 Epoch 43。

## Windows：先預覽再自主執行

請先確認目前沒有其他模型訓練占用相同 GPU，且舊版 Level 2 已結束。使用有 Codex CLI 的 PowerShell：

```powershell
cd "C:\Users\wxes1\OneDrive\桌面\plant2CO2"
git fetch origin
git switch feature/deeplab-v5-autonomous-experiments
git pull --ff-only
.\.venv\Scripts\python.exe -m unittest discover -s tests -p test_experiment_planner.py
```

舊版已結束但缺少 `training_finished.json` 時，可顯式聲明確認訓練已停止（此參數不是程序偵測）：

```powershell
.\.venv\Scripts\python.exe training\experiment_planner.py --assume-baseline-stopped
```

Codex 顯示候選方案與理由；預覽不啟動 GPU。確認流程通過、你已授權自主啟動後：

```powershell
.\.venv\Scripts\python.exe training\experiment_planner.py --assume-baseline-stopped --execute
```

預設執行**最多 12 Epoch** 的 pilot；可設定 `--pilot-epochs 16`，限制為 1–20。即使原本模型的限制是 50 Epoch，也不能用此功能直接建立超過 20 Epoch 的探索性訓練。只選一個候選，或跳過。子程序會同步在 PowerShell 顯示訓練進度；關閉終端就不會繼續執行。這不是 ChatGPT 雲端背景工作。

新實驗會輸出到：

```text
outputs_training_v5_experiments/
  monitor/experiment_decision.json
  shrub_weight_pilot/      （Codex 選 shrub 才會出現）
  cultivated_weight_pilot/（Codex 選 cultivated 才會出現）
```

在原本的 `http://127.0.0.1:8765/` 審核中心會顯示「Codex 自主新實驗」的選擇、理由、風險和狀態；實驗網頁仍只讀取紀錄，不執行命令。

**保障與限制**：沒有可核對的混淆矩陣時跳過；只選有 class IoU 且低於 55% 的類別；新輸出資料夾已存在時拒絕覆寫；每個試驗最多 20 Epoch。原本最佳權重是只讀輸入，不得覆蓋。所有模型比較仍須在一致的驗證集／固定測試集上進行，沒有 GT 的影像不會被誤當成 MAE。Codex 選擇是研究假設而非已證實的最佳方案。此版本尚未在你的 Windows RTX GPU 完成實際端到端測試。

### 注意

已存在某個 `_pilot` 目錄時，程式會拒絕再次使用該實驗名稱，而不是清空結果。若要第二次試驗，應以程式版本與明確的新實驗 ID 建立另一個隔離目錄。執行前請確認可用 SSD 空間與 GPU 訓練環境。
