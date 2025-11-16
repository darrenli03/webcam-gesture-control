// ...existing code...
# webcam-gesture-control

## Quick setup (PowerShell on Windows)

Run these commands in PowerShell to enable script execution for the session, activate the virtual environment, upgrade packaging tools, and install dependencies:

```powershell
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope Process -Force

& "C:/Users/darre/VSCode Projects/gestureModel/venv/Scripts/Activate.ps1"

python -m pip install --upgrade pip setuptools wheel

pip install opencv-python
pip install numpy
```