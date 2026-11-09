# DeepLab V5 Codex 自主方案選擇與執行（Level 2 Auto Controller）

此分支在現有 Level 2 上新增 **Codex 自主篩選 → 終端與網頁顯示決策／理由／風險 → 符合安全條件時自行續訓**。它不是開放任意程式碼修改權限；只允許既有、內容經 SHA256 驗證、符合當前 Epoch 且經範圍限制的訓練參數 JSON。Codex 可選 `execute` 或 `skip`；如證據不足，理應選 `skip`。

## Windows 開始前

先在 PowerShell 中確認 `codex --version` 可以使用，並保留 `.venv` 的 PyTorch 套件。此功能與 `app.py` 的 Flask 連接埠無關，審核網頁繼續使用 `127.0.0.1:8765`。

```powershell
cd "C:\Users\wxes1\OneDrive\桌面\plant2CO2"
git fetch origin
git switch feature/deeplab-v5-autonomous-controller
git pull --ff-only
.\.venv\Scripts\python.exe -m unittest discover -s tests -p test_auto_controller.py
```

**現有 Epoch 38 訓練已經結束**，舊版訓練不會有 `training_finished.json`，只能在確認訓練程序已完全停止之後，手動使用一次性舊版模式。先只看 Codex 決策、不真正啟動：

```powershell
.\.venv\Scripts\python.exe training\auto_controller.py --once --assume-stopped
```

若 Codex 認為有方案值得採用、你已希望授權它執行，可以使用：

```powershell
.\.venv\Scripts\python.exe training\auto_controller.py --once --assume-stopped --execute
```

這個指令是你授權**不需逐案核准**的自主模式。Codex 只能從當前有效的「訓練參數提案」選取；`review_task` 離線評估提案被排除。若選 `skip`，系統會告知理由並**不啟動訓練**。若選 `execute`，會先寫入 `outputs_training_v5_level2/monitor/auto_decision.json`，列出理由、風險和參數，並備份 `best_model.pth`、`latest_state.pth`、`history.csv` 到 `auto_backups/`。最後在 8 秒倒數後透過同一個 PowerShell 啟動安全續訓；可按 Ctrl+C 取消。

**新一輪 Level 2 訓練**在正常退出時會保存 `training_finished.json`。之後可以在第二個 PowerShell 以 `--watch --execute` 自動監看完成標記，等監控器產生提案後讓 Codex 篩選並可能自行續訓。此選項不能與 `--assume-stopped` 同時使用。

```powershell
.\.venv\Scripts\python.exe training\auto_controller.py --watch --execute --interval 120
```

訓練與提案生成是兩個不同程序：一個 PowerShell 執行 `training/train.py`，第二個執行 `training/monitor_v5.py --watch --interval 120 --codex`，第三個才執行 `training/auto_controller.py --watch --execute`。本機審核網站另用獨立 CMD 或 PowerShell 執行 `training/level2_dashboard.py --port 8765`。

## 安全邊界

- 維持累計 **50 Epoch 上限**；只在訓練完成時使用完整 `latest_state.pth`，不把 `best_model.pth` 冒充完整續訓狀態。
- Codex 不能修改資料集、LUT、模型架構、Loss、Batch Size 或程式碼；只選既有參數方案。CLI 執行使用唯讀 sandbox。
- 選中的提案須通過原始 JSON SHA256、Epoch、一致性和允許參數範圍檢查；不接受自由文字生成的可執行指令。
- **自主授權**使用 `autonomous-codex-guarded` 獨立標記，不冒充人工核准，只有 trainer 明確收到 `--allow-auto-approved` 才能使用。
- 控制器使用 lock 防止自身重複啟動；每個 Epoch + 提案 ID 有獨立備份目錄並拒絕重複執行。
- 若缺少 Codex CLI、無合法 JSON、缺少 best/full checkpoint、紀錄 Epoch 不一致、訓練未完成、或 Codex 選 `skip`，都不得開始續訓。
- 目前的 `scheduler_patience=4` 僅是探索性候選。當最後植被 mIoU 42.80% 遠低於最佳 63.26%，Codex 應評估是否直接跳過而非無條件延長訓練。

## 審核網頁

在 `http://127.0.0.1:8765/` 的「Codex 自主實驗決策」區塊可查看選中提案 ID、參數、理由、風險與狀態。它只讀取本機 `auto_decision.json`，不控制 GPU，與 plant2CO2 主網站完全分離。

## 注意事項

這是自主執行高成本 GPU 實驗的功能，可能消耗大量時間與 Codex 額度；且 Codex 的推薦不是經對照實驗驗證的最優解。請先執行不帶 `--execute` 的預覽測試。現有舊版 V5 訓練已停止時，務必**確認不存在其他同時寫入 outputs_training_v5_level2 的程序**，再使用 `--assume-stopped`。本功能目前未在你的本機完整資料集做過端到端驗證。
