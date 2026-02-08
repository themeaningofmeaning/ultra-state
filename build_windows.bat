@echo off
echo 🪟 Building Garmin Analyzer Pro (NiceGUI)...

REM 1. Get NiceGUI path dynamically
for /f "delims=" %%i in ('python -c "import nicegui; import os; print(os.path.dirname(nicegui.__file__))"') do set NICEGUI_PATH=%%i
echo 📦 Found NiceGUI at: %NICEGUI_PATH%

REM 2. Clean previous builds
if exist build rmdir /s /q build
if exist dist rmdir /s /q dist

REM 3. Build the Exe
echo 🔨 Running PyInstaller...
pyinstaller --noconfirm --onefile --windowed --clean ^
    --name "GarminAnalyzerPro" ^
    --icon="runner.ico" ^
    --add-data "%NICEGUI_PATH%;nicegui" ^
    --hidden-import="nicegui" ^
    --hidden-import="analyzer" ^
    --hidden-import="pandas" ^
    --hidden-import="plotly" ^
    --hidden-import="scipy" ^
    --hidden-import="sqlite3" ^
    app.py

echo.
echo ✅ Build Complete!
echo 📂 Executable: dist\GarminAnalyzerPro.exe
echo.
pause