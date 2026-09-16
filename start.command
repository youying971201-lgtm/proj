#!/bin/zsh

# 1. 自動切換到 .command 所在的專案根目錄 (確保相對路徑絕對正確)
cd "$(dirname "$0")"

echo "========================================"
echo "  YOLO 車流監控系統 - 啟動程序 (Mac)"
echo "========================================"

# 2. 強制清空舊的 8000 埠號 (防呆機制，解決 Errno 48)
lsof -ti :8000 | xargs kill -9 2>/dev/null

# 3. 自動偵測 Python 環境 (最高相容性核心)
# 腳本會依序尋找常見的虛擬環境，找不到則退回系統全域 python3
PYTHON_CMD="python3"
if [ -f "./.venv/bin/python" ]; then
    PYTHON_CMD="./.venv/bin/python"
    echo "[INFO] 偵測到 .venv 虛擬環境"
elif [ -f "./venv/bin/python" ]; then
    PYTHON_CMD="./venv/bin/python"
    echo "[INFO] 偵測到 venv 虛擬環境"
elif [ -f "./env/bin/python" ]; then
    PYTHON_CMD="./env/bin/python"
    echo "[INFO] 偵測到 env 虛擬環境"
else
    echo "[INFO] 未偵測到本地虛擬環境，將使用系統全域 Python3"
fi

# 4. 將「輪詢檢查與開網頁」丟到背景執行
(
    echo "[INFO] 正在等待 FastAPI 伺服器啟動..."
    while ! curl -s -o /dev/null -w "%{http_code}" http://127.0.0.1:8000 | grep "200" > /dev/null; do
        sleep 0.5
    done
    echo "[INFO] ✅ 伺服器已就緒，正在開啟網頁..."
    open http://127.0.0.1:8000
) &

# 5. 在前景啟動 Python 後端主程式 (終端機畫面會留著顯示 Log)
echo "[INFO] 正在啟動後端主程式..."
echo "----------------------------------------"
$PYTHON_CMD python/main.py