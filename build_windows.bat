@echo off
echo 🪟 Building Garmin Analyzer (v2.3 Flat Structure)...

REM 1. Clean previous builds
rmdir /s /q build dist

REM 2. Build the Exe
REM Logic: We build 'gui.py' directly from the root.
pyinstaller --noconfirm --onedir --windowed --clean ^
    --name "GarminAnalyzer" ^
    --icon="runner.ico" ^
    --collect-all customtkinter ^
    --hidden-import="PIL" ^
    gui.py

echo.
echo ✅ DONE! The app is in dist\GarminAnalyzer\GarminAnalyzer.exe
pause