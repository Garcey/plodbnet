@echo off
REM One-click: create the Desktop + Start Menu shortcut for CFR Solver.
cd /d "%~dp0"
if exist ".venv\Scripts\python.exe" (
  ".venv\Scripts\python.exe" scripts\install_cfr_desktop_shortcut.py
) else (
  echo Missing .venv — run from the plodbnet repo after creating the venv.
  pause
  exit /b 1
)
echo.
pause
