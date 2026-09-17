# Repository Guidelines

## 必須遵守的回覆與開發規則

1. **每次回覆的開頭都要稱呼使用者 `USER`。** 進度回報與最終答覆也適用，例如「USER，已完成……」。
2. **本專案由多人協作，並在 Windows 與 macOS 上開發及執行。** 修改前先檢查工作樹，保留其他人的未提交變更；程式路徑使用 `pathlib.Path`，避免寫死磁碟代號、路徑分隔符、攝影機索引或僅某一系統可用的命令。涉及攝影機、模型裝置及啟動方式時，分別考慮 Windows、macOS；CUDA、MPS 不可用時要有 CPU 路徑。
3. **永遠不允許使用虛擬環境。** 不建立、不啟用，也不要求協作者使用 `venv`、`.venv`、`virtualenv` 或 Conda 隔離環境。不得在專案文件、範例指令或自動化流程新增虛擬環境步驟。
4. **專案固定使用 Python 3.12.x。** 不得升至 3.13，也不得降至 3.11 或其他版本；每次執行程式、安裝套件、跑測試與工具腳本，都必須明確呼叫 Python 3.12.x：macOS 用 `python3.12`，Windows 用 `py -3.12`，並先確認 `--version` 輸出為 3.12.x。不要使用未指定版本的 `python`、`python3` 或 `py`；若 3.12.x 不存在，先回報問題，不得改用其他版本。
5. **每次完成修改後，都必須同步檢查並更新 `README.md` 與 `AGENTS.md`，才可交付。** 依實際程式與檔案狀態修正兩份文件的功能範圍、執行方式、驗證結果及未完成目標；不得沿用失效說明或把規劃寫成已完成。即使修改只涉及其中一份文件，也要核對另一份是否仍為最新狀態，並在交付說明中交代兩份文件的更新情況。

## 專案定位與目前狀態

本專案是學校專題比賽作品，現階段以自行錄製的道路影片回測。預想使用者包含一般民眾、社區出入口管理者，以及路口事件的初步篩查人員。目標是讓交通違規偵測與場景設定更容易客製化，未來可與主管機關資料比對；現有輸出僅供輔助判斷，須由人員複核。

- `python/main.py` 提供單一影像來源的 FastAPI／MJPEG 服務。第一幀供操作員在 `frontend/main.html` 校正停止線與至少三條 Gate，開始後才逐幀分析。
- YOLO11（目前預設 `yolo11x.pt`，可指定其他權重）與 ByteTrack 偵測、追蹤車輛；現有號誌顏色、停止線越線、`IN`／`OUT` 計數與 `ILLEGAL` 提示。`MOVING`／`STOPPED` 是影像位移分類，並非實際速度。
- `python/road_marking.py` 使用團隊訓練的 Stage-5 SegFormer checkpoint。按下 **Reload LaneNet** 後，服務才對下一張影格推論一次，釋放模型並快取標線結果供後續畫面顯示；目前不是逐幀分割。API 為 `POST /api/reload_lanenet`、`GET /api/road_markings`。
- `cord_data/points_data.json` 是操作員產生的執行時校正資料。Gate 的 `lane` 是車道數，前端目前以 `Gate 序號 × 10` 填入 `station_m`；此數字未經現地測量，主程式尚未用 Gate 測速。
- 違規變道（跨越實線、未打方向燈）、Gate 測速及雲端多鏡頭監看仍屬規劃，不能寫成已完成或已驗證的功能。

## 專案結構

- `python/`：`main.py` 為服務與分析流程；`Car.py`、`trafficLight.py` 為車輛與號誌邏輯；`road_marking.py` 為標線分割；`video_source.py` 為影片、串流與攝影機來源。
- `frontend/main.html`：目前服務使用的操作介面。`frontend/main2.html`、`frontend/toweb.py` 為其他／舊版介面或輔助程式。
- `cord_data/points_data.json`：按影像來源校正的停止線與 Gate；不得把某支影片的座標當成其他場景的通用設定。
- `inputVideos/`、`models/road_marking/`、`result/lanenet/`：分別放本機輸入影片、Stage-5 權重、精選的定性驗證圖。`result/yolo/` 預留 YOLO 成果。
- `tests/`：已有影像來源、道路標線座標及 Reload LaneNet 相關測試。`yaml/my_track.yaml` 為設定範例；主程式目前指定 Ultralytics 的 `bytetrack.yaml`。

## 執行與驗證

先確認所用的執行器確實是 Python 3.12.x。依 `requirements.txt` 準備相應套件與模型權重；若系統限制套件安裝，先釐清該平台的安裝方式，不得改用虛擬環境或其他 Python 版本。

macOS：

```bash
python3.12 --version
python3.12 -m pip install -r requirements.txt
python3.12 python/main.py --source /path/to/video.mp4
python3.12 -m compileall -q python
python3.12 -m unittest discover -s tests -v
```

Windows PowerShell：

```powershell
py -3.12 --version
py -3.12 -m pip install -r requirements.txt
py -3.12 python\main.py --source C:\path\to\video.mp4
py -3.12 -m compileall -q python
py -3.12 -m unittest discover -s tests -v
```

服務啟動後開啟 `http://127.0.0.1:8000`。需要 CPU 推論時用 `--device cpu`；切換 YOLO 權重用 `--model`。功能變更時，除相關單元測試外，還要以適用的影片或攝影機檢查受影響的 API／網頁流程。跨平台行為變更應在 Windows、macOS 各驗證一次，無法驗證的平台要在交付說明中標明。

## 未完成目標與實現辦法

1. **違規變道與方向燈判定。** 將 Stage-5 偵測的實線折線轉為場景中的車道邊界，把 ByteTrack 車輛 ID 的連續軌跡與車道關聯；利用車輛底部接地點跨越邊界的時間序列產生跨實線候選事件。再對同一車輛左右方向燈 ROI 做多幀閃爍辨識，於變道前後的時間窗判斷有無打燈。跨實線與未打燈應保留為可分別設定的規則，並加入遮擋、追蹤 ID 中斷與標線快取過期時的處理及回測案例。
2. **Gate 實際測速。** 先讓使用者標示已確認尺寸的道路標線週期與 Gate，將目前按序號推定的 `station_m` 改成有來源的實際道路距離。透過透視校正把影像點映射到道路平面；彎道需分車道建立行駛路徑並計算弧長，不能只憑兩條 Gate 在影像中的夾角推算距離。依同一追蹤車輛穿越 Gate 的影格時間戳計算 `速度 = 距離 ÷ 時間差`，處理跳幀、追蹤遺失、低幀率與測量誤差，再與參考速度比對。先確認現地標線規格，不能把預填的 10 公尺直接當成量測結果。
3. **雲端即時多鏡頭。** 先定義手機、攝影機到伺服器的串流協定與連線需求，再將目前單一來源的擷取、模型處理、校正設定及狀態拆成每個鏡頭各自獨立的工作單元。增加串流重連、資源限制、身分驗證與鏡頭管理 API，網頁再呈現多來源畫面及事件；先以兩路以上的測試串流驗證隔離與延遲，之後才進行遠端部署。
4. **可靠性與量化驗證。** 補齊 Stage-5 的資料來源清單、標籤映射、訓練設定及可重現的分割指標；為車流計數、紅燈提示、變道和未來的測速建立人工標註真值與評估方式。使用白天、夜間、雨天、遮擋及鏡頭晃動影像檢查誤報、漏報與延遲，並記錄 Windows、macOS 及 CPU／CUDA／MPS 的實際測試結果。`result/lanenet/` 目前只有精選視覺對照圖，不能據此宣稱量化準確率。
5. **執行入口符合版本規範。** 現有 `start.command`、`start.bat` 仍會尋找虛擬環境。後續需移除啟動腳本的環境搜尋，改為檢查並呼叫 Python 3.12.x，再於 Windows、macOS 各實際啟動與驗證。完成前應依本文件及 `README.md` 明確指定 3.12.x 的命令執行，不使用上述舊啟動腳本。

## 程式風格與協作交付

Python 採四個空白縮排，標準函式庫匯入置於第三方套件之前；函式、變數用 `snake_case`，類別用 `PascalCase`，常數用 `UPPER_SNAKE_CASE`。註解與介面文字沿用繁體中文。修改要聚焦，修正行為缺陷時在 `tests/test_<feature>.py` 加入可重現的回歸測試，優先使用小型測試資料，避免依賴大型影片或模型下載。

提交前檢查差異與測試結果，排除無關產物、其他人的未提交變更及個人校正資料。PR 說明行為變更、驗證平台與命令、模型／影片／校正需求；介面或標註畫面變更附上截圖。流程或命令變更時同步更新 `README.md`，並保持其中的功能狀態與本文件一致。
