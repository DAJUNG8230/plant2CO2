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

## 科學比較的重要限制

每個 pilot 的設定相較於對應的**同起始權重、同訓練輪數的控制組**只改動一種類別權重；但目前程式沒有自動跑這個額外的對照組。因此 `comparison.json` 中「原始 baseline 最佳」對「pilot 最佳」的差異**不能歸因於類別權重改動本身**，因為 pilot 還額外進行了最多 12 輪訓練、重新初始化 Optimizer，且隨機資料順序可能不同。這份比較只能用於初步篩選，不能作為論文的單一變因因果結論。正式比較應另跑一個相同 warm-start、optimizer 初始化、epoch 數、種子與評估流程的無改權重控制組，並且以固定 GT 測試集獨立報告覆蓋率誤差。

## Level 3：混淆方向與有界搜尋（新）

現在的候選不再一律增加類別權重。使用**最佳 Epoch** 的六類混淆矩陣；當 Shrub 或 Cultivated IoU < 55%、GT 像素至少 100 且預測像素相比 GT 有至少 10% 的偏差時，產生相應候選：

| 情況 | 方向 | 改動 |
|---|---|---|
| Shrub 預測比 GT 多 >= 10% | 減少假陽性假設 | Shrub CE 權重 1.45 → 1.25 |
| Shrub 預測比 GT 少 >= 10% | 增加召回假設 | Shrub CE 權重 1.45 → 1.65 |
| Cultivated 預測比 GT 多 >= 10% | 減少假陽性假設 | Cultivated CE 權重 0.90 → 0.75 |
| Cultivated 預測比 GT 少 >= 10% | 增加召回假設 | Cultivated CE 權重 0.90 → 1.10 |

判斷的是類別預測面積偏差方向，並不是權重調整的因果證明。Codex 還是可以選 skip。預設每個 pilot **8 Epoch**，同一批次會建立「不更動類別權重的對照組」與「只改一個類別權重的試驗組」，兩者均從現有最佳 checkpoint 初始化新的 optimizer，使用相同 seed 42、資料集、架構、Batch 4、訓練/驗證流程。GPU 非確定性仍可能造成變異；若要研究發表，還應增加重複種子與固定測試集。

在已結束舊版基準訓練、沒有其他相同 GPU 訓練時，先預覽：

```powershell
.\.venv\Scripts\python.exe training\experiment_planner.py --assume-baseline-stopped
```

一次授權最多搜尋兩組（**每組包含對照與實驗兩次訓練**，最壞共 4 次 8 Epoch 試驗，耗時較多）：

```powershell
.\.venv\Scripts\python.exe training\experiment_planner.py --assume-baseline-stopped --execute --max-trials 2
```

若第一組相對配對控制組的最佳植被 mIoU 與目標類別 IoU **沒有同時正向改善**，搜尋立即停止，不自動進入第二組。已建立的實驗目錄會保留，不能自動覆蓋。結束後可以在 `outputs_training_v5_experiments/monitor/search_summary.json` 查看選擇及結果，個別試驗有 `comparison.json`。本程式不會自行把任何 pilot 模型部署到網站；正式測試仍須核對相同 GSD/GT 的覆蓋率 MAE。

**舊版保護：** baseline 的 `best_model.pth` 是只讀來源，Level 2 的 `history.csv`、`latest_state.pth` 不會被新實驗寫入。主網站 `app.py` 未更動，審核頁維持 `http://127.0.0.1:8765/`。新增程式尚需在 Windows 本機跑單元測試與 GPU 實驗驗證。
