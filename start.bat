@echo off
chcp 65001 >nul
title YOLO 車流監控系統啟動器

:: 1. 自動切換到 .bat 所在的專案根目錄 (確保相對路徑絕對正確)
cd /d "%~dp0"

echo ========================================
echo   YOLO 車流監控系統 - 啟動程序
echo ========================================

:: 2. 強制清空舊的 8000 埠號 (防呆機制，解決 Errno 48)
FOR /F "tokens=5" %%T IN ('netstat -a -n -o ^| findstr "0.0.0.0:8000"') DO (
    taskkill /F /PID %%T >nul 2>&1
)

:: 3. 自動偵測 Python 環境 (最高相容性核心)
:: 腳本會依序尋找常見的虛擬環境資料夾，如果都找不到，就退回使用系統安裝的全域 Python
SET PYTHON_CMD=python
IF EXIST ".\.venv\Scripts\python.exe" (
    SET PYTHON_CMD=".\.venv\Scripts\python.exe"
    echo [INFO] 偵測到 .venv 虛擬環境
) ELSE IF EXIST ".\venv\Scripts\python.exe" (
    SET PYTHON_CMD=".\venv\Scripts\python.exe"
    echo [INFO] 偵測到 venv 虛擬環境
) ELSE IF EXIST ".\env\Scripts\python.exe" (
    SET PYTHON_CMD=".\env\Scripts\python.exe"
    echo [INFO] 偵測到 env 虛擬環境
) ELSE (
    echo [INFO] 未偵測到本地虛擬環境，將使用系統全域 Python
)

:: 4. 將「輪詢檢查與開網頁」丟到背景執行
start /b cmd /c "echo [INFO] 正在等待 FastAPI 伺服器啟動... & :loop & curl -s -o nul -w "%%{http_code}" http://127.0.0.1:8000 | find "200" >nul || (timeout /t 1 >nul & goto loop) & echo [INFO] ✅ 伺服器已就緒，正在開啟網頁... & start http://127.0.0.1:8000"

:: 5. 在前景啟動 Python 後端主程式
echo [INFO] 正在啟動後端主程式...
echo ----------------------------------------
%PYTHON_CMD% python\main.py

echo.
echo [系統] 程式已結束運行。
pause