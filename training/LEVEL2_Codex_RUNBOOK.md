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
