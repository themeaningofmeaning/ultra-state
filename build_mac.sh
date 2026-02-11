#!/bin/bash
set -e

echo "🍎 Building Garmin Analyzer Pro (NiceGUI)..."

# 1. Get NiceGUI path dynamically
NICEGUI_PATH=$(python3 -c "import nicegui; import os; print(os.path.dirname(nicegui.__file__))")
echo "📦 Found NiceGUI at: $NICEGUI_PATH"

# 2. Clean previous builds
rm -rf build dist

# 3. Build the App
echo "🔨 Running PyInstaller..."
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
    --hidden-import="PIL" \
    --hidden-import="kaleido" \
    app.py

echo ""
echo "✅ Build Complete!"
echo "📂 Executable: dist/GarminAnalyzerPro"
echo ""
echo "To create a DMG installer, run:"
echo "  hdiutil create dist/GarminAnalyzerPro.dmg -volname 'Garmin Analyzer Pro' -srcfolder dist/GarminAnalyzerPro.app -ov"