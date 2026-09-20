@echo off
setlocal
cd /d "%~dp0"
set "YOLO_CONFIG_DIR=%CD%\.ultralytics"
set "SYSTEM_PYTHON=%LOCALAPPDATA%\Programs\Python\Python312\python.exe"
if not exist "%YOLO_CONFIG_DIR%\Ultralytics" mkdir "%YOLO_CONFIG_DIR%\Ultralytics"

if not exist "%SYSTEM_PYTHON%" goto missing_python
"%SYSTEM_PYTHON%" --version 2>&1 | findstr /B /C:"Python 3.12." >nul
if errorlevel 1 goto missing_python

echo ========================================
echo YOLO and Stage-5 traffic monitor
echo Python: %SYSTEM_PYTHON%
echo URL: http://127.0.0.1:8000
echo ========================================

"%SYSTEM_PYTHON%" -u "python\main.py" %*
set "EXIT_CODE=%ERRORLEVEL%"

if not "%EXIT_CODE%"=="0" (
    echo.
    echo ERROR: main.py exited with code %EXIT_CODE%.
    pause
)
exit /b %EXIT_CODE%

:missing_python
echo ERROR: 找不到系統 Python 3.12.x：%SYSTEM_PYTHON%
pause
exit /b 1
