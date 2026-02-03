#!/bin/bash
# Build script for macOS using PyInstaller
set -e

echo "🍎 Building Garmin Analyzer for macOS..."

# 1. Install ALL dependencies
pip3 install pyinstaller customtkinter matplotlib

# Get the location of the customtkinter library for bundling
CTK_PATH=$(python3 -c "import customtkinter; import os; print(os.path.dirname(customtkinter.__file__))")

# Clean dist folder
rm -rf dist/GarminAnalyzer dist/GarminAnalyzer.app dist/GarminAnalyzer.dmg

# 2. Build with Icon and Dependencies
# Note: Ensure 'runner.icns' is in your project folder
pyinstaller --windowed --name "GarminAnalyzer" \
    --icon="runner.icns" \
    --add-data "$CTK_PATH:customtkinter/" \
    --hidden-import fitparse \
    --hidden-import pandas \
    --hidden-import numpy \
    --hidden-import matplotlib \
    --clean \
    src/garmin_analyzer/gui.py

# 3. Create DMG Installer
echo "💿 Packaging into .dmg..."
# FIXED: Removed invalid "-ovfile" flag. Used "-ov" and placed filename at the end.
hdiutil create -volname "GarminAnalyzer" -srcfolder dist/GarminAnalyzer.app -ov -format UDZO dist/GarminAnalyzer.dmg

echo "✅ Build complete!"
echo "📦 App: dist/GarminAnalyzer.app"
echo "💿 Installer: dist/GarminAnalyzer.dmg"