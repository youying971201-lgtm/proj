# 交通影像判斷

主程式會在背景執行 YOLO 車輛追蹤、紅綠燈判斷與越線計數，並將已標註的畫面以網頁串流顯示。

## 啟動

請先將模型放在專案根目錄的 `my_model.pt`，影片放在 `inputVideos/`（名稱需與 `python/main.py` 的 `VIDEO_NAME` 相符）。macOS 的 Homebrew Python 不允許直接把套件安裝到系統 Python，因此請先建立專案專用的虛擬環境，再安裝與啟動：

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python python/main.py
```

開啟 `http://127.0.0.1:8000` 即可觀看執行畫面。可在畫面上點選兩點來設定計數線；新設定會立即套用，並儲存為 `cord_data/points_data.json`，下次啟動會自動載入。

## 測速 Gate 校正流程

啟動後只會顯示影片第一幀，不會立刻跑 YOLO。請依序標示停止線、設定 Gate 名稱／`lane`／`station_m`，並為每個 Gate 點選左端 L、右端 R。至少三個 Gate 後按「開始判斷」。`lane` 必須從 1 連續遞增，`station_m` 是沿道路的真實里程（正常路段可填 10、20、30；急彎的 Gate 2.4 可填 24）。資料會以原始影片像素座標存入 `cord_data/points_data.json`：

```json
{
  "stop_line": {"line": {"L": {"x": 100, "y": 200}, "R": {"x": 500, "y": 200}}},
  "gates": [{
    "gate": "Gate 1",
    "lane": 1,
    "station_m": 10,
    "line": {"L": {"x": 110, "y": 220}, "R": {"x": 510, "y": 220}}
  }]
}
```

如果自訂模型不在專案根目錄，可在啟動時指定它的完整路徑，例如：

```bash
python python/main.py --device cpu --model /完整路徑/my_model.pt
```

日後重新開啟終端機時，只需要先執行 `source .venv/bin/activate`，再執行最後一行啟動程式。離開虛擬環境可輸入 `deactivate`。

此專案預設使用 YOLO11s，第一次執行會自動下載 `yolo11s.pt`。在 MacBook Air M2 上，程式會自動優先使用 MPS（Apple GPU）；若 PyTorch 回報 MPS 不可用，則會安全地改用 CPU。也可手動指定：

```bash
python python/main.py --device mps
```

YOLO11s 使用 COCO 類別，已設定為偵測汽車、機車、公車、卡車與交通號誌。MPS 可用時推論影像尺寸為 960；若回退 CPU，會自動改為 640，以降低 M2 Air 長時間運行的熱降頻與延遲。

## Stage-5 道路標線整合

專案已內建 Stage-5 SegFormer checkpoint：

```text
models/road_marking/stage5_roadmark_best.pth
```

直接執行 `start.bat` 時，畫面會同時顯示 YOLO 車輛／號誌結果及 Stage-5
道路標線遮罩。Windows 上如果專案本身沒有虛擬環境，啟動器會使用
`C:\vs\lanenet_taiwan\.venv`，該環境已安裝 CUDA Torch、Ultralytics、
Transformers、FastAPI 與 Uvicorn。

目前只輸出需要的 RLMD 類別 1～10、16：槽化區、斑馬線、停止線、各種
單／雙實線、單虛線及導流線。最新座標可由下列 API 讀取：

```text
http://127.0.0.1:8000/api/road_markings
```

每個座標同時含原圖像素 `x/y` 與 0～1 的 `nx/ny`。虛線的
`components` 是各小段座標，`polylines` 則是把同一車道的小段串接後
擬合出的整條線，並附有 `lane_id`。

RTX 4050 預設以長邊 1280、每幀推論。若需要較高 FPS，可改成每 3 幀推論
一次，或將長邊降成 960：

```powershell
C:\vs\lanenet_taiwan\.venv\Scripts\python.exe python\main.py --device cuda --road-marking-every 3 --road-marking-max-side 960
```

只測試一張圖片並輸出疊圖與 JSON 座標：

```powershell
C:\vs\lanenet_taiwan\.venv\Scripts\python.exe python\road_marking_demo.py "C:\path\to\image.jpg"
```

若暫時只想執行原本的 YOLO，可加上 `--no-road-markings`。
