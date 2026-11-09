# DeepLabV3+ V5 Level 2 — Codex 建議、人工核准、完整狀態續訓

此分支以 **feature/deeplab-v5-codex-monitor** + 已統一 U-Net FLAIR LUT 的 V5 為基底，新增有界超參數建議與手動核准機制。**不會自動停止訓練、不會修改正在執行的 optimizer、不會自動啟動下一次訓練。**

## 安全隔離

- 舊 V5：`outputs_training_v5/`，完全不更動。
- 新 Level 2：`outputs_training_v5_level2/`。
- `latest_state.pth`：每個**完整** Epoch 後以暫存檔原子替換，保存模型 / optimizer / scheduler / AMP scaler / Python / NumPy / PyTorch CPU 與 CUDA 隨機狀態、已完成 Epoch、best mIoU、early stop 計數、資料集 manifest 指紋與設定。
- `best_model.pth`：最佳模型權重；每次更新之前把上次最佳保存成 `best_model_previous.pth`。兩者均不會被監控器更動。
- `history.csv`：手動續訓時採追加模式。若最後 Epoch 不符合 checkpoint，**拒絕恢復**以免紀錄被覆蓋或混用。
- 輸出目錄列入 `.gitignore`。資料原始標註、LUT、六類 ID 與訓練/驗證切分不在 Codex 可批准的修改清單。

## Windows CMD 執行

```cmd
cd /d "C:\Users\wxes1\OneDrive\桌面\plant2CO2"
git fetch origin
git switch feature/deeplab-v5-level2-approval-resume
git pull --ff-only
.venv\Scripts\activate
python -m unittest discover -s tests -p test_level2.py
python -m unittest discover -s tests -p test_monitor_v5.py
```

**如果舊 V5 還在跑，不要停止它**。你可以繼續用舊版本監控：

```cmd
python training\monitor_v5.py --watch --interval 120 --codex --history outputs_training_v5\history.csv --output outputs_training_v5\monitor
```

目前正在執行的舊版 V5 並沒有保存 optimizer / scheduler / scaler 的完整狀態，所以**無法**直接用舊 `best_model.pth` 進行精確續訓。Level 2 的完整續訓功能只適用於以 Level 2 新訓練程式產生的 `latest_state.pth`，不能把舊權重假裝成完整 checkpoint。

在完成目前的 V5 實驗後，開始**新的** Level 2 實驗：

```cmd
python training\train.py
```

請先確認已用 V5 U-Net-aligned LUT 產生正確的 FLAIR masks 與 manifests。新輸出寫到獨立 `outputs_training_v5_level2/`。

在**第二個** CMD 視窗啟動監控：

```cmd
python training\monitor_v5.py --watch --interval 120 --codex
```

監控器每 120 秒讀取新資料；每次有新的有效驗證 Epoch 時向 Codex 請求**文字分析**，保存至 `outputs_training_v5_level2/monitor/codex_review.md`。若規則式分析認定有過擬合或 mIoU 停滯，額外寫出一份 **pending** 參數方案：

```text
outputs_training_v5_level2\approvals\proposal_<ID>.json
```

只有明確安全門檻觸發時才產生方案；出現 Shrub NaN 或類別 IoU 偏低時先要求核對資料，不會擅自調整 Loss 或 LUT。自動建議目前只允許調低下一階段 Learning Rate，單次減半（最低 1e-7）；框架還允許人工核准 Weight Decay（0..0.02）、Scheduler Patience（1..12），但自動分析不會擅自建議它們。這些只是受限範圍，**不是已驗證最佳值**。

## 手動批准流程

先查看 `report.md`、`codex_review.md`，再打開提案：

```cmd
type outputs_training_v5_level2\approvals\proposal_<ID>.json
```

如果同意這份確切方案，使用畫面上的 ID（請自行取代示例）：

```cmd
python training\level2_review.py approve --id <ID> --confirm APPROVE
```

產生 `approved_<ID>.json` 並**不會立刻套用**。人工確認後，在本輪訓練已正常結束或你決定安全停止之後，執行：

```cmd
python training\train.py --resume-full outputs_training_v5_level2\latest_state.pth --approved-config outputs_training_v5_level2\approvals\approved_<ID>.json --max-epochs 50
```

若只想延續、不改參數：

```cmd
python training\train.py --resume-full outputs_training_v5_level2\latest_state.pth --max-epochs 50
```

`--max-epochs` 是**總 Epoch 上限**，不是額外訓練輪數，且不允許超過 50。Level 2 總訓練 Epoch 上限固定為 **50**。已訓練到 Epoch 50 的模型不能藉由續訓超出上限；若提前停止，只能在未滿 50 的前提下續訓，仍受到 early stopping 與指標目標限制。

要在未來新實驗中手動提出非預設參數改動，可先設計明確、獨立的實驗方案，經人員審核後再擴充提案生成器；不要直接修改已核准的 JSON，因為會破壞審核紀錄。

## 已知限制與必要驗證

1. 完整 checkpoint 是以 **Epoch 邊界**為單位。GPU 過熱、程式崩潰或 Ctrl+C 發生於 Epoch 中間時，最多回到前一個已完成 Epoch。
2. Windows DataLoader worker / prefetch 狀態未逐個保存，因此續訓不保證和完全不中斷的執行逐位元相同；但 optimizer、scheduler、scaler 與主要 RNG 狀態可恢復。
3. 儲存的 manifest SHA256 只能偵測 manifest 變更；若原始 mask 位元內容被原地替換而路徑未改，仍需人工檢查資料版本。
4. Codex CLI 必須可從啟動監控的 Shell 找到；若 PATH 未更新，請用新開的 PowerShell 重新啟動或加入 Codex 安裝目錄。
5. Codex `exec --sandbox read-only` 沒有修改原始程式的授權；**不要把 Codex 文字建議直接當成核准的參數設定**。
6. 目前監控只提供驗證 IoU 和整體覆蓋率診斷。改善真實碳匯/覆蓋率精度必須額外測試人工標註的 Ground Truth。
7. **尚未在你的本地 GPU / FLAIR 完整資料集上完成端到端續訓驗證**。請先跑單元測試與小型試驗再進行完整訓練。

## 獨立本機審核網頁（不使用 app.py）

在專案根目錄執行：

```cmd
python training\level2_dashboard.py --port 8765
```

會自動開啟 http://127.0.0.1:8765/ 。程式只監聽 127.0.0.1，不共用 Flask 或 app.py 連接埠；主網站可照常運行。若連接埠被占用，改用 --port 8766。按 Ctrl+C 停止審核網頁，不會中斷模型訓練。

網頁每 10 秒更新待核准提案、最新 Epoch、最佳植被 mIoU、警告、Codex Token 使用紀錄與可能的短期/長期額度百分比。勾選一筆或多筆提案，再按「確認核准勾選提案」，經二次確認後只會寫入 approved_<ID>.json，不會自動修改訓練、不會自動執行續訓。多筆不同提案可能包含衝突的超參數；請只選擇彼此一致的提案，手動續訓時一次指定一份 approved JSON。

Token 資料僅來自本機 CODEX_HOME/sessions （預設使用者資料夾 .codex/sessions）下的 token_count JSONL 紀錄；不讀取 auth.json，不傳送憑證，不連線至第三方額度 API。只有 Codex 日誌明確包含 rate_limits.used_percent 才顯示 100 - used_percent 的剩餘百分比；否則顯示「無資料」。這是最近可用快照，**不是保證即時同步的帳戶總額度**。欲核對官方額度，請在 Codex 執行 /status 或查看官方用量頁。Token 數與剩餘百分比並非同一種單位。

目前 Level 2 實驗最多訓練到累計 Epoch 50。審核介面不能繞過此限制，也不能把舊版 V5 權重當成完整 checkpoint。

離線測試：

```cmd
python -m unittest discover -s tests -p test_level2_dashboard.py
```

**安全提醒**：不要把本機審核服務修改成 0.0.0.0 或用 ngrok 對外公開，因為它可以核准訓練參數。

## 為什麼 Early Stopping 後沒有待核准提案？（2026-11 修正）

舊版只有「最近四次驗證可能過擬合」或「最近五次 mIoU 波動極小」才會建立 pending JSON。Shrub/Cultivated 低 IoU 本身只會警告；Codex 生成的 `codex_review.md` 也不會直接轉成已批准參數。現在已新增依據 `history.csv` 推算「驗證無改善次數 >= 12，且 Epoch 尚未到 50」的 `early_stop_patience_exceeded` 規則。會提出探索性候選：`scheduler_patience: 4`，**不改極低的現有 LR、不改 LUT**，並清楚說明並無改善保證。此警告是根據 CSV 推算，不表示監控程式能判定訓練進程是否真的已結束。

在 **Level 2 分支** 更新程式並重跑單次監控：

```cmd
git pull
python -m unittest discover -s tests -p test_level2.py
python training\monitor_v5.py --once --codex
```

網頁仍開著時按「重新整理」，檢查 `outputs_training_v5_level2/approvals/proposal_<ID>.json` 是否出現。若未觸發，開啟 `monitor/status.json` 確認是否有 `early_stop_patience_exceeded`，並檢查監控器是否指向正確 history 路徑。網頁上顯示的 300 / 10080 分鐘剩餘百分比是本機 Codex 上次回報的快照，請看旁邊的更新時間；**頁面刷新不等於 Codex 額度剛更新**。

人工核准只建立 JSON；要在 Epoch 38 的完整狀態上繼續訓練，仍需先核准再**手動**以 `--resume-full`、`--approved-config` 啟動，總 Epoch 上限 50。

## 進階證據分析：完整 History、混淆矩陣與固定圖片

已新增 `training/level2_evidence.py`。它只讀取既有訓練產物，不重新訓練、不更動 LUT 或最佳模型。

監控器現在會輸出 `outputs_training_v5_level2/monitor/evidence.json`，包含完整 `history.csv` 所有 Epoch 的指標、最佳/最新驗證 Epoch、`confusion_matrix_epoch_XXX.csv` 的六類 IoU 與每個 GT 類別最常誤判成哪些 Pred 類別、`best_model.pth` 是否存在及檔案大小。若需要額外讀取**可信本地檔案**的最佳權重 metadata，可使用 `--inspect-checkpoint`（PyTorch `torch.load(..., weights_only=True, mmap=True)`）；監控器不會用它進行推論。

```cmd
python training\monitor_v5.py --once --codex --inspect-checkpoint
```

**固定測試圖片：** 如果已有同一組圖的人工 GT 和兩個 checkpoint 的預測 mask，可在 `outputs_training_v5_level2/evaluation/fixed_predictions.csv` 填寫以下欄位，並在相同資料夾中保存相對路徑所指向的遮罩：

```csv
image_id,gt_mask,pred_best_mask,pred_latest_mask
scene001,gt/scene001.png,best/scene001.png,latest/scene001.png
scene002,gt/scene002.png,best/scene002.png,latest/scene002.png
```

其中每張 mask 必須已對齊、尺寸相同、為單通道 **0–5 類別索引**；GT 可以有 255 ignore。不可把 RGB 彩色渲染遮罩當作類別索引，亦不可把不同尺度的圖片直接比較。本系統不會自動產生這些預測，使用者須先用固定推論條件產生 best / latest 各一份，並使用**相同的人工 GT**。前一輪若沒有保留 Epoch 38 的權重，不能把 `latest_state.pth` 直接當作預測 Mask；須透過相同推論環境讀取該權重，另行生成離線預測。現有 `best_model.pth` 不能用來還原最後輪的權重。

固定圖片有資料時，系統以每張有效 GT 像素計算 GT 與預測植被覆蓋率（類別 2,3,4,5），得到逐張絕對誤差 MAE（百分點）、有號偏差及絕對誤差 P90。這是**像素占比差異**，不等於真實地面面積誤差或碳匯誤差；要估算面積仍需 GSD 或幾何校正。

### 審核網頁的新內容

進入 `http://127.0.0.1:8765/`，可以看到最佳 Epoch、已讀取的混淆矩陣數量、固定圖片 MAE（若有），以及兩種不同的提案：

- **離線評估審核：** 例如檢查 Shrub 最常被誤判成哪類、補齊固定圖片預測、比較最佳與最後輪的混淆矩陣。勾選後只留下人工核准紀錄，**不會**套用於 train.py、也不會自動修改 Loss 或類別權重。
- **訓練參數提案：** 仍受原本有界的 Learning Rate / Scheduler Patience / Weight Decay 規則限制。此類提案需核准並**手動**指定 `--approved-config` 才會套用。

Codex 會收到完整 JSON 證據，而不是只有最後 Epoch 的摘要，並輸出 `codex_review.md` 研究建議；它產生的自由文字不會直接成為可執行超參數調整。

如需指定另外一組固定測試清單：

```cmd
python training\monitor_v5.py --once --codex --fixed-manifest outputs_training_v5_level2\evaluation\fixed_predictions.csv
```

離線測試：

```cmd
python -m unittest discover -s tests -p test_level2_evidence.py
python -m unittest discover -s tests -p test_level2.py
python -m unittest discover -s tests -p test_level2_dashboard.py
```

**重要：** 舊訓練權重完全保留，Level 2 累計 Epoch 仍最多 50；審核中心繼續只監聽 127.0.0.1:8765，不占用 app.py 連接埠。
