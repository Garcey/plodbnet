@echo off
REM One-click: create the Desktop + Start Menu shortcut for CFR Solver.
cd /d "%~dp0"
if exist ".venv\Scripts\python.exe" (
  ".venv\Scripts\python.exe" scripts\install_cfr_desktop_shortcut.py
) else (
  REM Keep round brackets out of this block: a closing one ends the else-block early.
  echo Missing .venv. From this folder run - see README-DESKTOP.txt step 3:
  echo   py -3.12 -m venv .venv
  echo   .venv\Scripts\pip install -e ".[ui,dev]"
  echo   .venv\Scripts\maturin develop --release
  echo then double-click this file again.
  pause
  exit /b 1
)
echo.
pause
