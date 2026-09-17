@echo off
setlocal
cd /d "%~dp0"
set "YOLO_CONFIG_DIR=%CD%\.ultralytics"
if not exist "%YOLO_CONFIG_DIR%\Ultralytics" mkdir "%YOLO_CONFIG_DIR%\Ultralytics"

py -3.12 --version 2>&1 | findstr /B /C:"Python 3.12." >nul
if errorlevel 1 goto missing_python

echo ========================================
echo YOLO and Stage-5 traffic monitor
echo Python: py -3.12
echo URL: http://127.0.0.1:8000
echo ========================================

py -3.12 -u "python\main.py" %*
set "EXIT_CODE=%ERRORLEVEL%"

if not "%EXIT_CODE%"=="0" (
    echo.
    echo ERROR: main.py exited with code %EXIT_CODE%.
    pause
)
exit /b %EXIT_CODE%

:missing_python
echo ERROR: 找不到 Python 3.12.x。請先安裝，並確認 py -3.12 --version。
pause
exit /b 1
