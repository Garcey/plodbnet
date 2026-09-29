@echo off
REM One-click: set up the Python environment if needed, then create the Desktop +
REM Start Menu shortcuts for CFR Solver. Full notes: python\plo5bp\cfr_app\README.md
REM (and SETUP.md, "A Windows PC").
setlocal
cd /d "%~dp0"
if exist ".venv\Scripts\python.exe" goto shortcut

echo CFR Solver needs a Python environment in this folder (.venv). None was found.
where py >nul 2>nul
if errorlevel 1 goto nopython
where cargo >nul 2>nul
if errorlevel 1 goto norust
choice /c YN /m "Set it up now? This downloads packages and builds the engine - about 5-10 minutes"
if errorlevel 2 goto manual

echo.
echo [1/4] Creating .venv ...
py -3 -m venv .venv
if errorlevel 1 goto failed
echo [2/4] Updating pip ...
".venv\Scripts\python.exe" -m pip install --upgrade pip
if errorlevel 1 goto failed
echo [3/4] Installing packages - the app, its window and the build tools ...
".venv\Scripts\pip.exe" install -e ".[dev,desktop]"
if errorlevel 1 goto failed
echo [4/4] Building the Rust engine - the first build takes a few minutes ...
".venv\Scripts\maturin.exe" develop --release
if errorlevel 1 goto failed
echo Environment ready.
echo.

:shortcut
".venv\Scripts\python.exe" scripts\install_cfr_desktop_shortcut.py
echo.
pause
exit /b 0

:nopython
echo.
echo Python 3.11 or newer is not installed - get it from https://www.python.org/downloads/
echo (tick "Add python.exe to PATH"), then double-click this file again.
goto manual_end

:norust
echo.
echo The Rust toolchain is not installed - get it from https://rustup.rs/ (the default
echo install), open a NEW window, then double-click this file again.
goto manual_end

:failed
echo.
echo Setup stopped at the step above. Fix what it says, then double-click this file again;
echo or run the steps by hand:

:manual
echo.
echo From this folder:
echo   py -3 -m venv .venv
echo   .venv\Scripts\pip install -e ".[dev,desktop]"
echo   .venv\Scripts\maturin develop --release
echo then double-click this file again.

:manual_end
pause
exit /b 1
