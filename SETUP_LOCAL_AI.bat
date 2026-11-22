@echo off
cd /d "%~dp0"
where ollama >nul 2>nul
if errorlevel 1 (
  echo Install Ollama from https://ollama.com/download and open it first.
  echo Then run this setup again.
  pause
  exit /b 1
)
echo One-time download of the local Qwen3 4B model. Keep Ollama open.
ollama pull qwen3:4b
if errorlevel 1 (
  echo Download failed. Open Ollama and check your internet connection, then retry.
  pause
  exit /b 1
)
echo Setup complete. Use START_GAME.bat to play.
pause
