@echo off
echo Building Ultra State (NiceGUI)...

REM 1. Get NiceGUI path dynamically
for /f "delims=" %%i in ('python -c "import nicegui; import os; print(os.path.dirname(nicegui.__file__))"') do set NICEGUI_PATH=%%i
echo Found NiceGUI at: %NICEGUI_PATH%

REM 2. Clean previous builds
if exist build rmdir /s /q build
if exist dist rmdir /s /q dist

REM 3. Build the Exe
echo Running PyInstaller...
python -m PyInstaller --noconfirm --onedir --windowed --clean ^
    --name "UltraState" ^
    --icon="runner.ico" ^
    --splash="splash.png" ^
    --add-data "%NICEGUI_PATH%;nicegui" ^
    --add-data "assets;assets" ^
    --hidden-import="nicegui" ^
    --hidden-import="analyzer" ^
    --hidden-import="pandas" ^
    --hidden-import="plotly" ^
    --hidden-import="scipy" ^
    --hidden-import="sqlite3" ^
    --hidden-import="PIL" ^
    --hidden-import="kaleido" ^
    --hidden-import="fitparse" ^
    --hidden-import="requests" ^
    --hidden-import="tzlocal" ^
    --hidden-import="pywebview" ^
    --exclude-module="matplotlib" ^
    app.py

echo.
echo ✅ Build Complete!
echo 📂 Executable: dist\UltraState.exe
echo.
pause