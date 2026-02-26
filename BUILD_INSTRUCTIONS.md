# 🛠️ Build Instructions for Ultra State

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

# Build the app (Note: --onedir is required for macOS to avoid compression crashes)
pyinstaller --noconfirm --onedir --windowed --clean \
    --name "UltraState" \
    --icon="runner.icns" \
    --add-data "$NICEGUI_PATH:nicegui" \
    --add-data "assets:assets" \
    --hidden-import="nicegui" \
    --hidden-import="analyzer" \
    --hidden-import="pandas" \
    --hidden-import="plotly" \
    --hidden-import="scipy" \
    --hidden-import="sqlite3" \
    --hidden-import="PIL" \
    --hidden-import="kaleido" \
    --hidden-import="fitparse" \
    --hidden-import="requests" \
    --hidden-import="tzlocal" \
    --exclude-module="matplotlib" \
    app.py

# Fix macOS PyInstaller pathing for NiceGUI templates and assets
ln -sf ../Resources/nicegui dist/UltraState.app/Contents/MacOS/nicegui
ln -sf ../Resources/assets dist/UltraState.app/Contents/MacOS/assets
```

### Create DMG Installer (Requires Option 1 or Option 2 with --onedir)
```bash
hdiutil create dist/UltraState.dmg -volname "Ultra State" -srcfolder dist/UltraState.app -ov
```

**Output:** `dist/UltraState.app` and `dist/UltraState.dmg`

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
    --hidden-import="pywebview" ^
    --hidden-import="tzlocal" ^
    --exclude-module="matplotlib" ^
    app.py
```

**Output:** `dist\UltraState\` directory.
**Distribution:** Windows users expect "portable" apps to be distributed as standard `.zip` files. Right-click the `dist\UltraState` folder and select **Compress to Zip file**, naming it `UltraState-Windows.zip`. When they download and unzip it, they just run the `UltraState.exe` file inside.

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
- Use `--onedir` instead of `--onefile` on macOS. The massive compression required for `--onefile` bundled with the NiceGUI implicit routing causes 404 parsing errors.
- Run the executable from terminal/command prompt to see error messages:
  - Mac: `./dist/UltraState.app/Contents/MacOS/UltraState`
  - Windows: `.\dist\UltraState.exe`

### Database Errors (OperationalError)
- Ultra State uses SQLite. To ensure the database persists between app updates and avoids permission errors in macOS root `/` environments, the database is stored at `~/.ultra_state/ultra_state.db`.

### 500 Error / Native Blank Screen
- Make sure `app.py` UI bindings are rooted inside a `@ui.page('/')` function decorator. Implied routing breaks in native mode under PyInstaller.

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