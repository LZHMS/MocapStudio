@echo off
chcp 65001 >nul
powershell.exe -NoLogo -NoProfile -ExecutionPolicy Bypass -File "%~dp0启动MocapStudio.ps1" %*
set "MOCAP_EXIT=%ERRORLEVEL%"
if not "%MOCAP_EXIT%"=="0" pause
exit /b %MOCAP_EXIT%
