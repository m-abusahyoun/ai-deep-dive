@echo off
cd /d "%~dp0"
where py >nul 2>nul
if errorlevel 1 (
  python start_ai_game.py
) else (
  py -3 start_ai_game.py
)
pause
