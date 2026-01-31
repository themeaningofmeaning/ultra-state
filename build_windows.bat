@echo off
REM Build script for Windows
REM Usage: Double-click build_windows.bat or run from command prompt

echo 🪟 Building Garmin Analyzer for Windows...

REM Install dependencies first
echo 📦 Installing dependencies...
python -m pip install fitparse pandas numpy pyinstaller

REM Build the executable with hidden imports
echo 🔨 Building executable...
python -m PyInstaller --onefile --windowed --name GarminAnalyzer --hidden-import fitparse --hidden-import pandas --hidden-import numpy --hidden-import pandas._libs.tslibs.np_datetime src/garmin_analyzer/gui.py

echo.
echo ✅ Build complete!
echo 📦 Output: dist\GarminAnalyzer.exe
echo.
echo To reduce file size, consider using UPX:
echo   https://upx.github.io/
