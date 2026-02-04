# 🛠️ Build Instructions for Garmin Analyzer

These are the commands to package the application for release.

## 🍎 macOS Build
**Run in Terminal:**
```bash
# 1. Clean previous artifacts
rm -rf build dist

# 2. Build the App
# Note: We build 'gui.py' directly now. No src folder needed.
pyinstaller --noconfirm --onedir --windowed --clean \
    --name "GarminAnalyzer" \
    --icon="runner.icns" \
    --collect-all customtkinter \
    --hidden-import="PIL" \
    gui.py

# 3. Create DMG Installer
hdiutil create dist/GarminAnalyzer.dmg -volname "GarminAnalyzer" -srcfolder dist/GarminAnalyzer.app -ov


##  Windows Build
**Run in Command Prompt / Powershell:**
REM 1. Clean previous artifacts
rmdir /s /q build dist

REM 2. Build the Exe
pyinstaller --noconfirm --onedir --windowed --clean ^
    --name "GarminAnalyzer" ^
    --icon="runner.ico" ^
    --collect-all customtkinter ^
    --hidden-import="PIL" ^
    gui.py