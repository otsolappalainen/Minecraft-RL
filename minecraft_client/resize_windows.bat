@echo off
echo Waiting for Minecraft windows to open...
timeout /t 3 /nobreak >nul

powershell.exe -ExecutionPolicy Bypass -NoProfile -File "resize_windows.ps1"

echo Resizing completed.
pause