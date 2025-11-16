@echo off
REM Request elevation if not already elevated
net session >nul 2>&1
if %errorlevel% neq 0 (
    echo Requesting administrative privileges...
    powershell -NoProfile -Command "Start-Process -FilePath '%~f0' -Verb RunAs"
    exit /b
)

REM Switch to the script directory
pushd "%~dp0"

REM Activate virtualenv and run main.py.
REM Prefer the batch activator if present (keeps env in this process).
if exist "venv\Scripts\activate.bat" (
    call "venv\Scripts\activate.bat"
    python "%~dp0main.py"
) else (
    REM Fall back to running the PowerShell Activate.ps1 and then python in the same PowerShell process.
    powershell -NoProfile -ExecutionPolicy Bypass -Command "& '%~dp0venv\Scripts\Activate.ps1'; python '%~dp0main.py'"
)

REM Return to original folder and pause so you can see output
popd
pause
```:: filepath: c:\Users\darre\VSCode Projects\gestureModel\run_as_admin.bat
@echo off
REM Request elevation if not already elevated
net session >nul 2>&1
if %errorlevel% neq 0 (
    echo Requesting administrative privileges...
    powershell -NoProfile -Command "Start-Process -FilePath '%~f0' -Verb RunAs"
    exit /b
)

REM Switch to the script directory
pushd "%~dp0"

REM Activate virtualenv and run main.py.
REM Prefer the batch activator if present (keeps env in this process).
if exist "venv\Scripts\activate.bat" (
    call "venv\Scripts\activate.bat"
    python "%~dp0main.py"
) else (
    REM Fall back to running the PowerShell Activate.ps1 and then python in the same PowerShell process.
    powershell -NoProfile -ExecutionPolicy Bypass -Command "& '%~dp0venv\Scripts\Activate.ps1'; python '%~dp0main.py'"
)

REM Return to original folder and pause so you can see output
popd
pause