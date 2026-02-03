@echo off
echo 🪟 Building Garmin Analyzer for Windows...

REM Install dependencies
python -m pip install fitparse pandas numpy customtkinter pyinstaller matplotlib

REM Get CustomTkinter path for bundling
for /f "delims=" %%i in ('python -c "import customtkinter; import os; print(os.path.dirname(customtkinter.__file__))"') do set CTK_PATH=%%i

REM Build the application
REM Note: Ensure 'runner.ico' is in your project folder
python -m PyInstaller --onedir --windowed --name GarminAnalyzer ^
    --icon="runner.ico" ^
    --add-data "%CTK_PATH%;customtkinter/" ^
    --hidden-import fitparse --hidden-import pandas --hidden-import numpy --hidden-import matplotlib ^
    src/garmin_analyzer/gui.py

echo.
echo ✅ Build complete! 📦 Output: dist\GarminAnalyzer