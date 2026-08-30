@echo off
rem ──────────────────────────────────────────────────────────────────────────
rem  FoodDetector-2 V3 — Streamlit launcher
rem  - Uses THIS project's .venv python explicitly (avoids PATH picking the
rem    sibling FoodDetector-2 venv).
rem  - The app runs headless (see .streamlit/config.toml); the browser is
rem    opened separately after the server has had time to start.
rem ──────────────────────────────────────────────────────────────────────────
cd /d "%~dp0"

echo Dang khoi dong FoodDetector — trinh duyet se mo http://localhost:8501 sau ~15 giay...
start "" /min cmd /c "timeout /t 15 /nobreak >nul & start "" http://localhost:8501"

".venv\Scripts\python.exe" -m streamlit run main.py

pause
