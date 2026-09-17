@echo off
setlocal
cd /d "%~dp0"

set "CAMERA_INDEX=%~1"
if not defined CAMERA_INDEX set "CAMERA_INDEX=1"

echo Starting virtual camera index %CAMERA_INDEX% at 1280x720 30 FPS.
call "%~dp0start.bat" --camera "%CAMERA_INDEX%" --camera-backend auto --camera-width 1280 --camera-height 720 --camera-fps 30 --device auto
exit /b %ERRORLEVEL%
