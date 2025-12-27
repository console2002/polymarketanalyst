@echo off
setlocal EnableExtensions EnableDelayedExpansion

REM ------------------------------------------------------------------
REM install.bat - Windows environment bootstrap for polymarketanalyst
REM Creates .venv and installs requirements.txt
REM ------------------------------------------------------------------

REM Always run from the folder this .bat file is in
cd /d "%~dp0"

echo ==============================================================
echo Polymarket Analyst - Windows Setup
echo ==============================================================
echo.

REM --- Pick a Python launcher (prefer py -3, fallback to python) ---
set "PY_CMD="
where py >nul 2>nul
if %errorlevel%==0 (
    set "PY_CMD=py -3"
) else (
    where python >nul 2>nul
    if %errorlevel%==0 (
        set "PY_CMD=python"
    )
)

if "%PY_CMD%"=="" (
    echo [ERROR] Python not found. Install Python 3, then re-run this script.
    echo         https://www.python.org/downloads/windows/
    exit /b 1
)

echo Using Python: %PY_CMD%
echo.

REM --- Create venv if missing ---
if not exist ".venv\Scripts\python.exe" (
    echo Creating virtual environment: .venv
    %PY_CMD% -m venv .venv
    if errorlevel 1 (
        echo [ERROR] Failed to create virtual environment.
        exit /b 1
    )
) else (
    echo Virtual environment already exists: .venv
)

REM --- Activate venv ---
call ".venv\Scripts\activate.bat"
if errorlevel 1 (
    echo [ERROR] Failed to activate virtual environment.
    exit /b 1
)

echo.
echo Upgrading pip/setuptools/wheel...
python -m pip install --upgrade pip setuptools wheel
if errorlevel 1 (
    echo [ERROR] Failed to upgrade pip tooling.
    exit /b 1
)

REM --- Install dependencies ---
if not exist "requirements.txt" (
    echo [ERROR] requirements.txt not found in: %cd%
    exit /b 1
)

echo.
echo Installing dependencies from requirements.txt...
python -m pip install -r requirements.txt
if errorlevel 1 (
    echo [ERROR] pip install failed.
    exit /b 1
)

echo.
echo ==============================================================
echo Setup complete.
echo ==============================================================
echo Next steps:
echo   1) Run the data logger (with the GUI stream enabled):
echo        .\.venv\Scripts\python.exe data_logger.py --ui-stream
echo   2) In a new terminal, start the GUI:
echo        .\.venv\Scripts\python.exe -m streamlit run logger_gui.py
echo   3) Re-run install.bat any time you need to refresh dependencies.
echo   NOTE: Always use the .\.venv\Scripts\python.exe interpreter (not system Python),
echo         or dependencies like websockets will be missing.
echo.

pause
endlocal
