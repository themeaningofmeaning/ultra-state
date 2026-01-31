#!/bin/bash
# Build script for macOS using PyInstaller
# Usage: chmod +x build_mac.sh && ./build_mac.sh

set -e

echo "🍎 Building Garmin Analyzer for macOS..."

# Install PyInstaller
pip3 install pyinstaller

# Clean dist folder completely
rm -rf dist/GarminAnalyzer dist/GarminAnalyzer.app

# Build the app with hidden imports
pyinstaller --windowed --name "GarminAnalyzer" \
    --hidden-import fitparse \
    --hidden-import pandas \
    --hidden-import numpy \
    --hidden-import pandas._libs.tslibs.np_datetime \
    --clean \
    src/garmin_analyzer/gui.py

echo "✅ Build complete!"
echo "📦 Output: dist/GarminAnalyzer.app"
echo ""
echo "To create a DMG for distribution:"
echo "  hdiutil create -volname 'GarminAnalyzer' -srcfolder dist/GarminAnalyzer.app -ovfile dist/GarminAnalyzer.dmg"
