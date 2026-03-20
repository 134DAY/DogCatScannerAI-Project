@echo off
title DoggyCatScaner Launcher
color 0A

echo.
echo  ==========================================
echo    DoggyCatScaner - Starting...
echo  ==========================================
echo.

:: Start Flask in background
echo  [1/2] Starting Flask server...
start "Flask Server" cmd /k "python app.py"

:: Wait for Flask to start
timeout /t 3 /nobreak >nul

:: Start ngrok
echo  [2/2] Starting ngrok tunnel...
echo.
echo  ==========================================
echo   Copy the "Forwarding" URL below
echo   and share it with anyone!
echo  ==========================================
echo.
ngrok http 5000
