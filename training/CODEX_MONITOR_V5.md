# Codex CLI 半自動訓練監控（DeepLabV3+ V5）

本系統讀取本機 \`outputs_training_v5/history.csv\`、透過 \`nvidia-smi\` 擷取 NVIDIA GPU 即時快照、產生規則式異常報告。可選擇使用 Codex CLI 對最近一次驗證結果提供**唯讀**研究建議。

**這不是遠端連線/雲端監控。** CMD 視窗關閉後監控結束；訓練程式仍由原本的 Python 程序管理。監控器不會更改訓練程式、Learning Rate、Checkpoint、標籤或資料。

## 安裝與準備（Windows CMD）

從專案根目錄執行：

\`\`\`cmd
git fetch origin
git switch feature/deeplab-v5-codex-monitor
.venv\Scripts\activate
python -m unittest discover -s tests -p test_monitor_v5.py
\`\`\`

如尚未安裝 Codex CLI，依照官方安裝步驟安裝並登入，然後確認 \`codex --version\` 可執行。Codex 是**選用**的；沒有 Codex 一樣可以輸出規則式報告。需要 Git、Python 3.10+；GPU 數據需要 NVIDIA 驅動及 \`nvidia-smi\`。對於不同電腦路徑，先將 CMD 切換到正確專案目錄。

## 先執行一次診斷

\`\`\`cmd
python training\monitor_v5.py --once
\`\`\`

輸出檔：

- \`outputs_training_v5/monitor/status.json\`：機器可讀的指標和警告。
- \`outputs_training_v5/monitor/report.md\`：繁體中文規則式報告。
- \`outputs_training_v5/monitor/codex_review.md\`：選用 Codex 分析時才會產生。

## 持續監控（另一個 CMD 視窗）

\`\`\`cmd
python training\monitor_v5.py --watch --interval 120
\`\`\`

若希望**每次有新的驗證 Epoch**時，自動要求 Codex 提出建議：

\`\`\`cmd
python training\monitor_v5.py --watch --interval 120 --codex
\`\`\`

只執行一次 Codex 建議：

\`\`\`cmd
python training\monitor_v5.py --once --codex
\`\`\`

監控模式以 Ctrl+C 結束。Codex 每次分析會把目前狀態摘要送交所設定的模型提供者；如需完全離線，請勿加 \`--codex\`。請勿在報告中放入私密資料或帳號資訊。

## 異常與審核政策

| 分析 | 判斷 | 動作 |
|---|---|---|
| 缺少 history.csv | 檔案不存在、缺欄 | 提示，**不建立假數據** |
| 忽略未完成驗證 | val_loss NaN | 等下一次驗證 |
| 六類 IoU | 缺失/NaN 或低於 0.5 | 警告、檢查標籤 |
| 過擬合徵兆 | 最近四次驗證 Train Loss 下降 5% 且 Val Loss 上升 5% | 提出下一輪實驗建議 |
| mIoU 停滯 | 最近五次驗證差距 < 0.005 | 建議檢查 LR |
| GPU 溫度 | 當下溫度 >= 80°C | 警告，**不自行關閉訓練** |
| GPU 利用率/VRAM/功耗 | nvidia-smi 單次快照 | 報告；不根據瞬時利用率亂調 Batch |

以上都是**診斷門檻**，不是經實驗確認的最佳超參數。

在本專題六類比較中，請先人工確認 LUT 和測試集一致。不能把植被四類 mIoU 或整體 IoU 當作真實植被覆蓋率誤差。需要單獨人工標記的固定測試集。

## 人工確認流程（沒有自動修改指令）

1. 開啟 \`report.md\` 和 \`codex_review.md\`。
2. 人工核對建議，決定是否改動下一輪實驗。
3. 以**新的 Git 分支和獨立輸出目錄**實作改動，避免覆蓋 V5 最佳模型。
4. 比較相同 Test Set 的 mIoU、六類 IoU、覆蓋率誤差、面積誤差與推論時間。
5. 未通過驗證時，不更新網站權重。

Codex CLI 呼叫採 \`codex exec --sandbox read-only --ephemeral --output-last-message ... -\`，以標準輸入提供摘要，沒有自動套用建議或 Git commit 的邏輯。參數在 CLI 版本中可能改變，若呼叫失敗可先用不含 \`--codex\` 的模式。

## 與現有 V5 訓練配合

在第一個 CMD 中執行 \`python training\train.py\`，第二個 CMD 中啟動監控。當前 \`train.py\` 每兩輪驗證一次（另含首尾），監控器會跳過無驗證資料的 Epoch。它不會要求訓練停止或重新啟動，也不會覆蓋 \`outputs_training_v5/best_model.pth\`。

輸出資料夾 \`outputs_training_v5/\` 已加入 \`.gitignore\`，避免意外上傳 checkpoint、記錄與建議報告。
