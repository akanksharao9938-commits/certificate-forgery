@echo off
title CertificateGuard AI

cd /d "D:\certificate-forgery-backend(2)"

echo ==========================================
echo       CERTIFICATEGUARD AI
echo ==========================================
echo.
echo Starting AI server...
echo Please wait for YOLO11 and ResNet50...
echo.

start "CertificateGuard Server" cmd /k "cd /d D:\certificate-forgery-backend(2) && D:\certificate-forgery-backend(2)\venv\Scripts\python.exe -m uvicorn app.main:app"

echo Waiting for server...

:CHECK
timeout /t 3 /nobreak >nul

powershell -NoProfile -Command "try { Invoke-WebRequest -Uri 'http://127.0.0.1:8000/health' -UseBasicParsing -TimeoutSec 2 | Out-Null; exit 0 } catch { exit 1 }"

if errorlevel 1 (
    echo Server is still starting...
    goto CHECK
)

echo.
echo ==========================================
echo Server is ready!
echo Opening CertificateGuard AI...
echo ==========================================
echo.

start "" "http://127.0.0.1:8000"

exit