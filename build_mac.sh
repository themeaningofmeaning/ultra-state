#!/bin/bash
set -e

echo "🍎 Building Garmin Analyzer (v2.3 Flat Structure)..."

# 1. Clean previous builds
rm -rf build dist

# 2. Build the App
# Logic: We build 'gui.py' directly from the root.
pyinstaller --noconfirm --onedir --windowed --clean \
    --name "GarminAnalyzer" \
    --icon="runner.icns" \
    --collect-all customtkinter \
    --hidden-import="PIL" \
    gui.py

# 3. Create DMG Installer
echo "💿 Packaging into .dmg..."
hdiutil create dist/GarminAnalyzer.dmg -volname "GarminAnalyzer" -srcfolder dist/GarminAnalyzer.app -ov

echo "✅ DONE! App is in dist/GarminAnalyzer.app"
echo "✅ DMG is in dist/GarminAnalyzer.dmg"