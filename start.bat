@echo off
setlocal
cd /d "%~dp0"
set "YOLO_CONFIG_DIR=%CD%\.ultralytics"
if not exist "%YOLO_CONFIG_DIR%\Ultralytics" mkdir "%YOLO_CONFIG_DIR%\Ultralytics"

set "PYTHON_EXE="
if exist ".venv\Scripts\python.exe" set "PYTHON_EXE=%CD%\.venv\Scripts\python.exe"
if not defined PYTHON_EXE if exist "venv\Scripts\python.exe" set "PYTHON_EXE=%CD%\venv\Scripts\python.exe"
if not defined PYTHON_EXE if exist "env\Scripts\python.exe" set "PYTHON_EXE=%CD%\env\Scripts\python.exe"
if not defined PYTHON_EXE if exist "C:\vs\lanenet_taiwan\.venv\Scripts\python.exe" set "PYTHON_EXE=C:\vs\lanenet_taiwan\.venv\Scripts\python.exe"

if not defined PYTHON_EXE goto missing_python

echo ========================================
echo YOLO and Stage-5 traffic monitor
echo Python: %PYTHON_EXE%
echo URL: http://127.0.0.1:8000
echo ========================================

"%PYTHON_EXE%" -u "python\main.py" %*
set "EXIT_CODE=%ERRORLEVEL%"

if not "%EXIT_CODE%"=="0" (
    echo.
    echo ERROR: main.py exited with code %EXIT_CODE%.
    pause
)
exit /b %EXIT_CODE%

:missing_python
echo ERROR: Compatible Python environment was not found.
echo Expected: C:\vs\lanenet_taiwan\.venv\Scripts\python.exe
pause
exit /b 1
