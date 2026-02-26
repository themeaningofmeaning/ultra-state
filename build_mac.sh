#!/bin/bash
set -e

echo "🍎 Building Ultra State (NiceGUI)..."

# 1. Get NiceGUI path dynamically
NICEGUI_PATH=$(python3 -c "import nicegui; import os; print(os.path.dirname(nicegui.__file__))")
echo "📦 Found NiceGUI at: $NICEGUI_PATH"

# 2. Clean previous builds
rm -rf build dist

# 3. Build the App
echo "🔨 Running PyInstaller..."
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

echo "🔧 Fixing macOS PyInstaller 6+ pathing for NiceGUI..."
# PyInstaller puts data in Resources, but sys._MEIPASS points to MacOS. 
# We symlink them so NiceGUI can find its static assets and templates.
ln -sf ../Resources/nicegui dist/UltraState.app/Contents/MacOS/nicegui
ln -sf ../Resources/assets dist/UltraState.app/Contents/MacOS/assets

echo ""
echo "✅ Build Complete!"
echo "📂 Executable (.app bundle): dist/UltraState.app"
echo ""
echo "To create a DMG installer, run:"
echo "  hdiutil create dist/UltraState.dmg -volname 'Ultra State' -srcfolder dist/UltraState.app -ov"