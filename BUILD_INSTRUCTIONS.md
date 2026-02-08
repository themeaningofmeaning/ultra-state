# 🛠️ Build Instructions for Garmin Analyzer Pro

These instructions will help you package the NiceGUI-based application into standalone executables.

---

## 📋 Prerequisites

Before building, ensure you have:

1. **Python 3.8+** installed
2. **PyInstaller** installed:
   ```bash
   pip install pyinstaller
   ```
3. **All dependencies** installed:
   ```bash
   pip install -r requirements.txt
   ```

---

## 🍎 macOS Build

### Option 1: Use the Build Script (Recommended)
```bash
./build_mac.sh
```

### Option 2: Manual Build
```bash
# Get NiceGUI path
NICEGUI_PATH=$(python3 -c "import nicegui; import os; print(os.path.dirname(nicegui.__file__))")

# Clean previous builds
rm -rf build dist

# Build the app
pyinstaller --noconfirm --onefile --windowed --clean \
    --name "GarminAnalyzerPro" \
    --icon="runner.icns" \
    --add-data "$NICEGUI_PATH:nicegui" \
    --hidden-import="nicegui" \
    --hidden-import="analyzer" \
    --hidden-import="pandas" \
    --hidden-import="plotly" \
    --hidden-import="scipy" \
    --hidden-import="sqlite3" \
    app.py
```

### Create DMG Installer (Optional)
```bash
hdiutil create dist/GarminAnalyzerPro.dmg -volname "Garmin Analyzer Pro" -srcfolder dist/GarminAnalyzerPro.app -ov
```

**Output:** `dist/GarminAnalyzerPro` or `dist/GarminAnalyzerPro.app`

---

## 🪟 Windows Build

### Option 1: Use the Build Script (Recommended)
```cmd
build_windows.bat
```

### Option 2: Manual Build
```cmd
REM Get NiceGUI path
for /f "delims=" %%i in ('python -c "import nicegui; import os; print(os.path.dirname(nicegui.__file__))"') do set NICEGUI_PATH=%%i

REM Clean previous builds
if exist build rmdir /s /q build
if exist dist rmdir /s /q dist

REM Build the exe
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
```

**Output:** `dist\GarminAnalyzerPro.exe`

---

## 🐛 Troubleshooting

### "Module not found" errors
- Ensure all dependencies are installed: `pip install -r requirements.txt`
- Add missing modules with `--hidden-import="module_name"`

### NiceGUI assets not loading
- The `--add-data` flag bundles NiceGUI's web assets
- Verify the path is correct by running:
  ```bash
  python3 -c "import nicegui; import os; print(os.path.dirname(nicegui.__file__))"
  ```

### App crashes on startup
- Run the executable from terminal/command prompt to see error messages
- Check that `analyzer.py` is in the same directory as `app.py`

### Large file size
- The `--onefile` option creates a single executable but is larger
- Use `--onedir` instead for a folder-based distribution (smaller startup time)

---

## 📦 Distribution

After building:

1. **Test the executable** on a clean machine without Python installed
2. **Compress for distribution**:
   - macOS: Create a DMG or ZIP file
   - Windows: Create a ZIP file or installer with Inno Setup
3. **Include README** with usage instructions

---

## 🔧 Build Flags Explained

- `--onefile`: Creates a single executable (slower startup, easier distribution)
- `--windowed`: Hides the console window (GUI apps only)
- `--clean`: Removes temporary build files before building
- `--add-data`: Bundles additional files/folders (NiceGUI assets)
- `--hidden-import`: Forces PyInstaller to include modules it might miss
- `--icon`: Sets the application icon

---

## 📝 Notes

- **First build** may take 2-5 minutes
- **Subsequent builds** are faster due to caching
- The `build/` folder contains temporary files (can be deleted)
- The `dist/` folder contains the final executable