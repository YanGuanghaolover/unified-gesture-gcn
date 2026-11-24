@echo off
cd /d "%~dp0"
set "PYTHONPATH=src"

rem --- pick a usable interpreter: local venv -> shared venv -> system python ---
set "PY=%~dp0.venv\Scripts\python.exe"
if not exist "%PY%" set "PY=%~dp0..\hand-gesture-recognition\.venv\Scripts\python.exe"
if not exist "%PY%" set "PY=python"

echo.
echo   ============================================================
echo    Unified Gesture GCN - Web Studio
echo    Interpreter: %PY%
echo.
echo    IMPORTANT: startup takes 1-2 minutes because it first trains
echo    a small base model. Wait until you see "Uvicorn running on
echo    http://127.0.0.1:8000", THEN open the browser.
echo.
echo    Browser: http://localhost:8000
echo    Allow camera access when the browser asks.
echo    Press Ctrl+C in this window to stop.
echo   ============================================================
echo.

"%PY%" scripts\run_web.py --host 127.0.0.1 --port 8000 %*

echo.
echo [service stopped] See the log above if it exited unexpectedly.
echo Common causes:
echo   - port 8000 already in use  (try: start_web.bat --port 8010)
echo   - missing deps              (pip install fastapi uvicorn pydantic pyyaml)
echo.
pause >nul
