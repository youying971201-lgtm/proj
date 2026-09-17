# 交通影像監測系統

本專案是學校專題比賽作品，目標是讓交通影像的違規偵測、車流觀察與場景校正更容易設定，也更容易依使用情境調整。現階段以團隊自行錄製的實際道路影片回測功能；長期目標是讓手機或其他有鏡頭的裝置將即時影像送到雲端伺服器，讓使用者在同一處查看不同地點的交通狀況。

目前主程式以影片、攝影機或 OpenCV 可讀取的影像串流為輸入，分析車輛、交通號誌、停止線越線事件及道路標線。操作員在瀏覽器的第一幀畫面完成校正後，FastAPI 背景工作會逐幀處理影像，並以 MJPEG 串流顯示標註結果。

> **目前功能範圍：** 每台被偵測到的車都顯示速度欄位；累積足夠軌跡後，以相鄰 Gate 的局部道路平面透視映射顯示 `EST LIVE ... km/h`，穿越兩條相鄰 Gate 的當幀另顯示 `EST AVG ... km/h`。前端預填的 10／4／6 公尺尚未做現地測量，彎道內外車道差仍以像素路徑近似，因此目前數字只供測試，不可視為精確或執法測速。`ILLEGAL` 是依程式的號誌與越線規則產生的提示，應由人員複核。

## 背景、目標與應用

團隊關注臺灣道路車流混雜、路口衝突及人工監看的時間與範圍限制，希望透過影像辨識及可調整的規則，及早發現高風險行為、提高監看效率，並累積可供交通管理參考的車流資訊。系統定位為民間可使用的輔助工具，期望未來與公路局等主管機關合作，比對雙方資料；目前尚未接入公部門資料，也不將模型輸出視為正式裁罰依據。

預想情境包含一般民眾觀察自家車庫前的交通、社區出入口管理，以及政府部門在路口進行初步事件篩查。專題規劃的議題是闖紅燈、違規變道、車流量，以及利用車道標線作為測速基準；各功能的實作狀態如下。

| 功能 | 現階段狀態 |
| --- | --- |
| 紅燈越線提示 | 已依單一號誌顏色與停止線跨越事件實作；仍需人工確認。 |
| 車流量觀察 | 已顯示 `IN`／`OUT` 越線計數與本次處理摘要，尚未提供長期統計。 |
| 車輛行駛／停止狀態 | 已依影像位移分類；不是實際車速。 |
| 道路標線辨識 | 按 Reload LaneNet 後以 Stage-5 對下一幀分割一次，快取遮罩及座標供後續畫面使用。 |
| 違規變道判定 | 規劃以跨越實線、變道時未打方向燈為候選條件；尚未實作事件判定。 |
| Gate 測速 | 每台車顯示待估或即時預估時速；跨相鄰 Gate 的當幀顯示區間均速。局部透視映射處理近大遠小與 Gate 角度，彎道內外車道差以像素路徑近似；尚未經現地距離與實際車速驗證。 |
| 雲端多鏡頭監看 | 長期目標；目前為單一來源的本機 FastAPI 服務。 |

## 使用流程

```mermaid
flowchart LR
    A[影片／攝影機] --> B[第一幀校正]
    B --> C[儲存停止線與 Gate]
    C --> D[逐幀分析]
    D --> E[YOLO11 + ByteTrack]
    D -. 按 Reload LaneNet .-> F[Stage-5 對下一幀分割一次]
    E --> G[號誌、越線、行駛狀態]
    E --> J[接地點局部透視估速與 Gate 區間均速]
    F --> H[遮罩與座標 API]
    G --> I[MJPEG 網頁畫面]
    J --> I
    H --> I
```

1. **準備來源。** 未指定時，程式讀取 `python/main.py` 中 `DEFAULT_VIDEO_NAME` 指定的 `inputVideos/` 影片；也可用 `--source` 指定影片／串流，或用 `--camera` 選擇實體、虛擬攝影機。
2. **校正畫面。** 伺服器先顯示第一幀，此時不執行 YOLO。於網頁上點兩下設定停止線端點、逐點標出道路左右邊界並按 Enter 完成，再點選虛線頂端建立至少五條一般 Gate，並設定整條路的車道數。
3. **啟動判斷。** 點選「開始判斷」後，網頁先呼叫 `/save_points` 寫入校正檔，再呼叫 `/start_processing` 啟動背景工作。影片會從指定的起始影格重新讀取。
4. **查看結果。** 網頁顯示車輛 ID、偵測框、`MOVING`／`STOPPED` 狀態、號誌顏色及 `IN`／`OUT` 越線計數；每個車框上方先顯示 `EST --`，連續觀測後顯示 `EST LIVE ... km/h`，完成兩條相鄰 Gate 穿越的當幀顯示 `EST AVG ... km/h`。開始判斷後按「Reload LaneNet」才會偵測道路標線並顯示快取結果，座標可由 API 讀取；終端機在處理結束時輸出本次統計。
5. **重新處理或結束。**「重新判斷」會中止目前工作並以網頁上的校正重新開始；「關閉系統」會停止服務。目前只持久儲存校正資料，沒有匯出標註影片、逐筆事件或統計報表。

系統將「物件偵測與追蹤」及「逐像素標線分割」分為兩條處理路徑：前者保留同一車輛的時間資訊，後者提供道路幾何形狀。停止線與 Gate 由人員在當前場景校正，以適應不同鏡頭位置；正規化座標讓網頁縮放時仍能對應回原始影像。多幀投票、越線防抖區與移動狀態遲滯門檻用來降低單幀偵測抖動造成的反覆切換。

## 安裝與執行

本專題固定使用 **Python 3.12.x**，每次執行都須明確指定此版本。另需 `requirements.txt` 中的套件、可讀取的影像來源及模型權重。PyTorch／TorchVision 的安裝方式可能因作業系統與 CUDA 版本而異。

macOS：

```bash
python3.12 --version
python3.12 -m pip install -r requirements.txt
python3.12 python/main.py --source /path/to/video.mp4
```

Windows PowerShell：

```powershell
py -3.12 --version
py -3.12 -m pip install -r requirements.txt
py -3.12 python\main.py --source C:\path\to\video.mp4
```

啟動後開啟 <http://127.0.0.1:8000>。若要使用預設來源，先將 `DEFAULT_VIDEO_NAME` 指定的影片放進 `inputVideos/`，再省略 `--source`。專題會依測試需求使用 YOLO11s 或 YOLO11x；目前程式預設 `yolo11x.pt`，第一次使用該官方模型名稱時，Ultralytics 會嘗試下載。要改用 11s 或其他權重，先備妥本機 `.pt` 檔，再以 `--model` 指定。道路標線模型預設讀取團隊自行訓練的 `models/road_marking/stage5_roadmark_best.pth`；缺少該模型時可用 `--no-road-markings`。

常用選項（macOS；Windows 將 `python3.12` 改為 `py -3.12`，並依系統調整路徑）：

```bash
# 指定裝置及自訂 YOLO 權重
python3.12 python/main.py --source /path/to/video.mp4 --device cpu --model /path/to/model.pt

# 使用攝影機或 OBS Virtual Camera；索引依電腦而異
python3.12 python/main.py --camera 1 --camera-width 1280 --camera-height 720 --camera-fps 30
python3.12 python/main.py --list-cameras --camera-backend auto

# 調整 Reload LaneNet 時的道路標線推論解析度，或停用道路標線
python3.12 python/main.py --road-marking-max-side 960
python3.12 python/main.py --no-road-markings
```

`--device auto` 依序選擇 CUDA、MPS、CPU；明確指定 CUDA 或 MPS 但當前不可用時會改用 CPU。YOLO 輸入長邊目前於 MPS 使用 640，CUDA／CPU 使用 512；道路標線模型預設使用長邊 1280。Stage-5 在 YOLO 使用 MPS 時改用 CPU，使用 CUDA 時仍用 CUDA。Windows 攝影機無法讀取時可嘗試 `--camera-backend dshow` 或 `msmf`。其餘選項請在 macOS 執行 `python3.12 python/main.py --help`，或在 Windows 執行 `py -3.12 python\main.py --help`。

Gate 與效能更改已在 macOS 使用 Python 3.12 執行 `python3.12 -m unittest discover -s tests -v`（33 項通過，含合成影格的雙車速度文字、Gate 角度、停止線與局部透視測試）及 `python3.12 -m compileall -q python`。前端幾何與 JSON 流程以 JavaScriptCore 模擬測試通過，`start.command` 已通過 zsh 語法檢查及 `./start.command --help` 啟動檢查。專案目前沒有可用測試影片，尚未做真實車速對照、完整 MJPEG 畫面驗證或 Windows 實測。

`start.command` 與 `start.bat` 現在會確認 Python 3.12.x 並直接呼叫該版本，不搜尋其他執行環境；macOS 腳本遇到已占用的埠號會報錯，不會強制結束其他服務。`start-virtual-camera.bat` 使用 `--device auto`，有 CUDA 時會選 CUDA，否則走 CPU。腳本仍需在各自系統以實際影片或攝影機完成啟動驗證；若要使用 Windows 的 OBS Virtual Camera，先啟動它，再執行 `start-virtual-camera.bat` 或明確指定 `py -3.12 python\main.py --camera 1 --camera-width 1280 --camera-height 720 --camera-fps 30 --device auto`。若只想驗證道路標線模型，macOS 可執行 `python3.12 python/road_marking_demo.py /path/to/image.jpg`，產生單張影像的疊圖與 JSON 座標。

## 停止線與 Gate 校正

在第一幀畫面先點一下**停止線**的 **L** 錨點，移動滑鼠可即時預覽線段，再點一下設定 **R**；停止線應跨越所有要校正的車道。接著從停止線 L 端沿道路逐點點擊左邊界，移動滑鼠時會預覽目前點到滑鼠位置的線段，按 **Enter** 完成；再從 R 端逐點點擊右邊界並按 Enter 完成。邊界可用多點貼合彎道，也可按畫面上的完成按鈕。畫面中的校正線段寬度為 1 px。設定整條路的預設車道數後，逐次點選白色虛線頂端建立 Gate；點選 Gate 清單項目，再拖曳與錨點分開的圓形把手調整角度。前端將經過錨點的直線與兩側邊界求交，交點就是有限 Gate 的兩端。Gate 無法與兩側邊界相交時會顯示警示，不能開始判斷。工具列可暫停繪製、復原、重做及更換 Gate 顏色。

道路車道數固定時，調整「預設車道數」會套用至所有 Gate；它不能單獨推算遠處的畫面寬度，因此仍須標出道路邊界。停止線提供邊界起點與 Gate 初始角度，之後每條 Gate 可自行調整。這只是影像上的人工幾何校正，並未保證 Gate 在真實道路平面與車道方向成 90 度。

前端至少需要五條一般 Gate；在一般 Gate 後可選擇下一條為 `Gate X.4`，表示同週期後 4 公尺的選配位置。前端暫以 `Gate 1 = 0`、`Gate 1.4 = 4`、`Gate 2 = 10` 公尺依序填入 `station_m`，**尚未以現地尺寸驗證**。測速程式會用 Gate 端點做局部透視映射，但不會因此驗證預填的物理距離；正式評估前務必先確認每兩條 Gate 的真實里程。`lane` 代表車道數而非車道編號。按「開始判斷」時，前端把 Gate 換算成正規化 L/R 線段，連同錨點、角度與道路邊界傳給 `/save_points`；後端寫入 `cord_data/points_data.json` 的版本 3 格式，例如：

停止線兩端的順序會影響跨線側別與 `IN`／`OUT` 的方向；不同影片可在 `python/main.py` 的 `VIDEO_CONFIGS` 設定 `in_transition`。未列入的來源預設以上側到下側為 `IN`，校正時應依畫面中的行進方向確認。

```json
{
  "schema_version": 3,
  "video": {"name": "example.mp4", "coordinate_system": "normalized_0_to_1", "width": 1000, "height": 650},
  "lane_count": 2,
  "stop_line": {"line": {"L": {"x": 0.15, "y": 0.60}, "R": {"x": 0.85, "y": 0.60}}},
  "road_boundaries": {
    "L": [{"x": 0.15, "y": 0.60}, {"x": 0.25, "y": 0.10}],
    "R": [{"x": 0.85, "y": 0.60}, {"x": 0.75, "y": 0.10}]
  },
  "gates": [
    {"gate": "Gate 1", "lane": 2, "station_m": 0, "anchor": {"x": 0.50, "y": 0.55}, "angle_deg": 0, "line": {"L": {"x": 0.16, "y": 0.55}, "R": {"x": 0.84, "y": 0.55}}},
    {"gate": "Gate 2", "lane": 2, "station_m": 10, "anchor": {"x": 0.50, "y": 0.45}, "angle_deg": 0, "line": {"L": {"x": 0.18, "y": 0.45}, "R": {"x": 0.82, "y": 0.45}}},
    {"gate": "Gate 3", "lane": 2, "station_m": 20, "anchor": {"x": 0.50, "y": 0.35}, "angle_deg": 0, "line": {"L": {"x": 0.20, "y": 0.35}, "R": {"x": 0.80, "y": 0.35}}},
    {"gate": "Gate 4", "lane": 2, "station_m": 30, "anchor": {"x": 0.50, "y": 0.25}, "angle_deg": 0, "line": {"L": {"x": 0.22, "y": 0.25}, "R": {"x": 0.78, "y": 0.25}}},
    {"gate": "Gate 5", "lane": 2, "station_m": 40, "anchor": {"x": 0.50, "y": 0.15}, "angle_deg": 0, "line": {"L": {"x": 0.24, "y": 0.15}, "R": {"x": 0.76, "y": 0.15}}}
  ]
}
```

後端驗證至少三條 Gate、非零線段、`lane ≥ 1`、座標介於 0～1，且 `station_m` 依順序遞增；前端另要求五條一般 Gate。該 JSON 屬執行時校正資料，請勿將不同影像來源的座標直接共用。

「匯出校正 JSON」會另外下載**版本 3** 的 `gate_geometry.json`，儲存停止線、原始影像尺寸、車道數、左右邊界，以及各 Gate 的錨點、影像像素角度與 `station_m`；Gate 的 L/R 由前端重新計算，不存於此匯出檔。可用「匯入校正 JSON」載回同尺寸影片的校正，也可手動匯入新版 `points_data.json`；仍須人工確認是同一個場景。後端儲存檔同時保留計算出的 L/R，供目前的越線與測速流程使用。網頁不會自動回填既有 Gate。

## 技術原理

### 車輛與交通號誌

YOLO11 偵測 COCO 類別中的汽車、機車、公車、卡車與交通號誌，Ultralytics 的 ByteTrack 為車輛維持追蹤 ID。程式從號誌框中選出「信心分數 × 面積」最高者，將其 ROI 轉為 HSV 色彩空間，以紅、綠像素比例判斷顏色，再以近期有效結果投票取得穩定狀態。號誌辨識不依賴車輛追蹤 ID；該幀只有號誌偵測、沒有追蹤 ID 時仍會更新狀態。無法辨識的一幀不會立即清除前一個狀態。目前沒有黃燈判斷，也沒有將號誌分別對應到不同車道。

### 越線事件與行駛狀態

程式取車輛偵測框的**底部中心點**作為位置，用向量叉積判定其位於停止線哪一側。接地點移動線段確實與**有限的停止線**相交、且跨過 12 像素防抖區時，依方向記為 `IN` 或 `OUT`；垂直與傾斜停止線使用同一幾何判定。同一追蹤狀態下，每個方向只計一次。若當下穩定號誌為 `RED`，車輛被標記為 `ILLEGAL`。目前兩個越線方向都可能觸發紅燈標記。ByteTrack 暫時沒有 ID 的車框會用短期位置關聯供測速，但不參與越線違規與正式車流計數。

行駛狀態先以背景特徵點光流及 RANSAC 估計相鄰影格的局部仿射變換，再扣除攝影機移動對車輛位置的影響，對剩餘的每影格位移做指數平滑。低於 0.5 像素／影格或高於 1.5 像素／影格連續五幀，分別判定 `STOPPED` 或 `MOVING`。這是影像位移分類，**不是實際道路速度**。

### 道路標線

開始判斷後按 **Reload LaneNet**，Stage-5 SegFormer 才會載入並對當前影格進行一次逐像素語意分割；模型隨後釋放，偵測結果在後續影格持續疊圖，直到下一次 Reload。未按 Reload 時不執行 Stage-5 推論；重新判斷會等待上一輪工作執行緒完全結束，並沿用本次服務記憶體中的標線，服務重啟則需重新偵測。標線疊圖的類別色彩遮罩與折線像素座標也會隨結果快取，避免每幀重建。合成 1280×720 影像、20 次繪製的本機測試中，每幀約由 7.5 ms 降至 4.6 ms；這不是完整 YOLO 處理速度。目前輸出 RLMD 類別 1～10、16，包括槽化區、斑馬線、停止線、實線、虛線及導流線。程式透過連通元件擷取輪廓、中心、外接框及中心線；線類擬合為折線。虛線的 `components` 保留各小段，`polylines` 則將可能屬於同一車道的小段串接。API 同時提供原圖像素 `x/y` 與正規化座標 `nx/ny`；`lane_id` 是單次結果內的折線序號，不是跨影格追蹤 ID。

### 違規變道構想（尚未實作）

團隊規劃以道路標線分割辨認實線邊界，再將車輛追蹤軌跡與邊界對照，判定是否跨越實線；另一個候選條件是變道時未打方向燈。後續仍須實作車輛與車道的關聯、方向燈的時序辨識，以及跨線前後的判定時間窗。現有程式只提供標線座標和車輛追蹤資料，尚未將兩者結合成變道事件。

### Gate 即時與區間預估時速

團隊讓操作員標示白色虛線及相鄰週期的起訖點，將道路標線當作距離基準。[《道路交通標誌標線號誌設置規則》第 182 條](https://www.mvdis.gov.tw/webMvdisLaw/LawArticle.aspx?LawID=A0011025)對一般車道線規定：白虛線線段長 4 公尺、間距 6 公尺，因此一個完整週期可作為 **10 公尺的設計參考**。實際測速前仍須確認該路段標線類型、現地尺寸，以及影像中 Gate 與標線的對應關係；前端自動填入的里程不是現地量測結果。

目前 `python/gate_speed.py` 以 YOLO 框的**底部中心點**作為接地點。每對相鄰 Gate 的四個端點經 [OpenCV 四點透視轉換](https://docs.opencv.org/doc/doxygen/html/d9/ded/group__geometry__shape.html)映到該段道路的橫向位置與里程；因此水平、垂直、傾斜 Gate 都能以其實際方向估算車輛的**沿道路行進速度**，並降低近大遠小的影像透視影響。停止線若位於 Gate 區域外，其里程由最近的 Gate 區間外推，讓接近、進入停止線的車也能取得 `EST LIVE` 預估值；Gate 範圍外的外推較不可靠。每台車在取得至少兩個相隔足夠時間的連續接地點前顯示 `EST --`。ByteTrack 暫無 ID 時，程式以鄰近偵測框建立短期 `TEMP` 身分，能連續觀測後同樣顯示即時估計。

同一追蹤 ID 在相鄰影格的接地點形成移動線段；線段穿越有限 Gate 時，在兩個影格時間之間插值通過時間。影片優先使用 OpenCV 提供且單調遞增的時間戳，不可用時以影格序號／FPS 回退；攝影機使用擷取後的單調時鐘。跨越相鄰兩條 Gate 後，另計算 `預估區間均速 = Gate 距離 ÷ 通過時間差 × 3.6`，當幀顯示 `EST AVG`，其他已具備連續軌跡的影格顯示 `EST LIVE`。追蹤中斷會重置跨 Gate 計時；車道切換不產生跨 Gate 均速，但仍可顯示即時估計。異常時間或高於 250 km/h 的結果會略過。

彎道的 Lane Differential 暫以左右道路邊界在兩 Gate 間的**像素弧長**建立內外側路徑，再依車輛通過 Gate 的橫向位置內插；該路徑長度與道路中央像素路徑長度之比，用於修正預設距離與即時里程速度。若邊界資料不可用，則退回兩 Gate 端點之間的直線像素比例。透視可能使這個比例嚴重失真；超出 0.75～1.25 或無法計算時，改用未經車道差修正的 Gate 預設距離，車框文字會標示 `base`。四點映射只是在**局部平坦道路**上近似消除透視，並非完成相機高度、鏡頭參數及道路坡度的三維標定；目前也沒有實測車道寬度或路面高程，無法得到上坡、下坡及橫向變道的完整三維車速。更精準的下一步是量測道路平面與各車道的實際幾何，以多組對應點校正全場景，再用參考測速設備驗證；[ICCV 2021 的單眼測速研究](https://openaccess.thecvf.com/content/ICCV2021/papers/Revaud_Robust_Automatic_Monocular_Vehicle_Speed_Estimation_for_Traffic_Surveillance_ICCV_2021_paper.pdf)也將道路平面校正作為核心。鏡頭晃動、接地點偏差及低 FPS 仍會造成誤差。請勿用 `EST` 數字判定違規。

## 模型與資料來源

團隊自行訓練 Stage-5 道路標線 checkpoint，並提供下列資料與方法來源。這些連結的用途不同：SegFormer 是模型架構與參考實作，不能當作一份獨立的標註資料集。資料集的實際合併比例、標籤映射與訓練配置仍需由團隊補上，才能完整重現訓練。

| 來源 | 在本專題中的參考角色 |
| --- | --- |
| [RLMD](https://github.com/stu9113611/RLMD) | 道路標線語意分割資料集；目前推論類別採其 25 類標籤中的部分類別。 |
| [RMSeg-UDA](https://github.com/stu9113611/RMSeg-UDA) | 惡劣天候道路標線分割的訓練框架與 RLMD-AC 資料集說明。 |
| [Road-Marking-Segmentation](https://github.com/namemzy/Road-Marking-Segmentation) | 另一份道路標線資料集與標註參考。 |
| [SegFormer](https://github.com/NVlabs/SegFormer) | 語意分割模型架構與公開實作。 |
| [RLMD-AC 雲端資料夾](https://drive.google.com/drive/folders/1ZVxIanHEPq5gPjDbKlmZV3K99SsM1NT-?usp=sharing) | 團隊提供的晴天、夜間、雨天資料來源；標註範圍依資料集分割而異。 |

### Stage-5 視覺對照

下圖展示白天、夜間與路口三種影像，並排比較來源影像、原始 28k checkpoint 與 Stage-5 best（圖面標示 iter 2000）的分割疊圖。這是**定性對照**，不代表已取得 mIoU、precision 或 recall 等量化指標。其餘三張詳細遮罩對照見 [result/lanenet 說明](result/lanenet/README.md)。

![Stage-5 三種路況的道路標線分割對照](result/lanenet/three_images_28k_vs_stage5/three_images_overlay_comparison.jpg)

## 輸出與 API

| 路徑 | 用途 |
| --- | --- |
| `GET /` | 載入 `frontend/main.html` 操作介面。 |
| `GET /first_frame.jpg` | 校正用的第一幀。 |
| `POST /save_points` | 驗證並儲存停止線與 Gate。 |
| `POST /start_processing` | 啟動背景影像分析。 |
| `GET /video_feed` | 標註畫面的 MJPEG 串流。 |
| `GET /api/status` | 執行狀態、影像尺寸及校正資料。 |
| `GET /api/road_markings` | 最新道路標線結果及模型狀態。 |
| `POST /api/reload_lanenet` | 排程對下一張影格重新載入 Stage-5 並偵測一次。 |
| `POST /api/reset` | 中止並重置本次分析。 |
| `POST /shutdown` | 停止服務。 |

道路標線結果包含 `frame_index`、`inference_ms`、`model_input_size` 與按類別整理的 `objects`。`frame_index` 是最近一次按 Reload 後的偵測影格。網頁顯示畫面與終端機摘要；目前沒有資料庫或報表輸出。

## 專案結構

| 路徑 | 角色 |
| --- | --- |
| `python/main.py` | CLI、FastAPI、分析主迴圈及串流。 |
| `python/Car.py` | 車輛狀態、移動判斷與越線標記。 |
| `python/trafficLight.py` | 號誌 HSV 辨識及多幀投票。 |
| `python/road_marking.py` | SegFormer 推論、疊圖及座標擷取。 |
| `python/gate_speed.py` | Gate 局部透視映射、即時估速、穿越插值、車道像素路徑差與預估區間均速。 |
| `python/video_source.py` | 影像來源與攝影機探測。 |
| `frontend/main.html` | 目前服務的網頁介面。 |
| `cord_data/points_data.json` | 操作員校正產生的執行時資料。 |
| `models/road_marking/` | Stage-5 checkpoint。 |
| `result/lanenet/` | 精選的 Stage-5 定性視覺與模型對照圖。 |
| `result/yolo/` | 預留 YOLO 驗證成果目錄。 |
| `tests/` | 影像來源、道路標線座標、Reload LaneNet、Gate 校正與測速的單元測試。 |

`frontend/main2.html` 與 `frontend/toweb.py` 是其他／舊版介面及串流輔助程式，目前首頁不使用。`yaml/my_track.yaml` 是追蹤器設定範例，主程式目前指定 Ultralytics 的 `bytetrack.yaml`。

## 驗證與限制

macOS：

```bash
python3.12 -m compileall -q python
python3.12 -m unittest discover -s tests -v
```

Windows PowerShell：

```powershell
py -3.12 -m compileall -q python
py -3.12 -m unittest discover -s tests -v
```

現有測試涵蓋道路標線座標、Reload LaneNet 及影像來源輔助函式；團隊也以自行錄製的多種實際路況影片，在 Windows、macOS 的筆電與桌機上進行回測。Stage-5 的四張精選結果圖已放在 `result/lanenet/`，但目前沒有可重現的評估設定或量化指標，因此不能由視覺對照推論偵測準確率。

完整流程需要模型、影片或攝影機及可用的運算環境。單一號誌選取、追蹤 ID 中斷、遮蔽、鏡頭晃動與透視變化都可能影響結果。現階段沒有現場執法或物理測速驗證，也沒有逐筆事件持久化。

多人協作時，每次完成修改都要依實際程式狀態同步檢查並更新本文件與 `AGENTS.md`；目前功能、操作命令與未完成目標應在兩份文件中保持一致。
