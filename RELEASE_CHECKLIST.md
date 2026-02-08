# 🚀 Release Checklist for Garmin Analyzer Pro

Use this checklist before creating a release build.

---

## ✅ Pre-Build Checklist

### 1. Code Quality
- [ ] All features tested and working
- [ ] No console errors or warnings
- [ ] Database operations tested (import, export, delete)
- [ ] All tabs functional (Trends, Report, Activities)
- [ ] Chart interactions working (zoom, hover, legend toggle)

### 2. Dependencies
- [ ] All dependencies in `requirements.txt`
- [ ] PyInstaller installed: `pip install pyinstaller`
- [ ] Test imports work:
  ```bash
  python3 -c "import nicegui, plotly, pandas, scipy, analyzer"
  ```

### 3. Assets
- [ ] Icon files present: `runner.icns` (macOS), `runner.ico` (Windows)
- [ ] `analyzer.py` in root directory
- [ ] Database schema up to date

### 4. Documentation
- [ ] README.md updated with current features
- [ ] BUILD_INSTRUCTIONS.md reviewed
- [ ] Version number updated (if applicable)

---

## 🔨 Build Process

### macOS
```bash
./build_mac.sh
```

**Expected Output:**
- `dist/GarminAnalyzerPro` (executable)
- Build time: 2-5 minutes

### Windows
```cmd
build_windows.bat
```

**Expected Output:**
- `dist\GarminAnalyzerPro.exe`
- Build time: 2-5 minutes

---

## 🧪 Post-Build Testing

### Test on Clean Machine
- [ ] Run executable on machine WITHOUT Python installed
- [ ] Test folder import functionality
- [ ] Test database persistence
- [ ] Test all tabs and features
- [ ] Test chart interactions
- [ ] Test export CSV
- [ ] Test copy to clipboard

### Smoke Tests
- [ ] App launches without errors
- [ ] UI renders correctly (no missing assets)
- [ ] Import folder dialog opens
- [ ] Activities display in all tabs
- [ ] Chart renders with correct colors
- [ ] Filters work (Last 30 Days, etc.)
- [ ] Delete activity works
- [ ] Export CSV creates valid file

---

## 📦 Distribution Preparation

### macOS
1. **Test the app:**
   ```bash
   open dist/GarminAnalyzerPro.app
   ```

2. **Create DMG (optional):**
   ```bash
   hdiutil create dist/GarminAnalyzerPro.dmg \
     -volname "Garmin Analyzer Pro" \
     -srcfolder dist/GarminAnalyzerPro.app \
     -ov
   ```

3. **Compress for distribution:**
   ```bash
   cd dist
   zip -r GarminAnalyzerPro-macOS.zip GarminAnalyzerPro.app
   ```

### Windows
1. **Test the exe:**
   ```cmd
   dist\GarminAnalyzerPro.exe
   ```

2. **Compress for distribution:**
   ```cmd
   cd dist
   tar -a -c -f GarminAnalyzerPro-Windows.zip GarminAnalyzerPro.exe
   ```

---

## 🐛 Common Build Issues

### Issue: "Module not found" error
**Solution:** Add to build script:
```bash
--hidden-import="missing_module_name"
```

### Issue: NiceGUI assets not loading
**Solution:** Verify NiceGUI path:
```bash
python3 -c "import nicegui; import os; print(os.path.dirname(nicegui.__file__))"
```

### Issue: App crashes on startup
**Solution:** Run from terminal to see errors:
```bash
./dist/GarminAnalyzerPro.app/Contents/MacOS/GarminAnalyzerPro
```

### Issue: Large file size (>100MB)
**Solution:** This is normal for `--onefile` builds with NiceGUI. Consider:
- Using `--onedir` instead (faster startup)
- Excluding unnecessary packages
- Using UPX compression (advanced)

---

## 📝 Release Notes Template

```markdown
# Garmin Analyzer Pro v1.0

## 🎉 Features
- Modern web-based UI with dark theme
- Interactive training trends chart with zoom
- Performance categorization (Race Ready, Base Maintenance, etc.)
- Detailed run reports with HR Recovery analysis
- Activity management with inline delete
- CSV export for data analysis
- LLM-ready data formatting

## 🔧 Technical
- Built with NiceGUI and Plotly
- SQLite database for activity storage
- Supports Garmin .FIT files
- Cross-platform (macOS, Windows)

## 📥 Installation
1. Download the appropriate version for your OS
2. Extract the archive
3. Run the executable
4. Import your Garmin .FIT files

## 🐛 Known Issues
- First launch may take 10-15 seconds (NiceGUI initialization)
- Large datasets (>100 runs) may slow down chart rendering

## 🙏 Credits
- FIT file parsing: fitparse
- Data analysis: pandas, scipy
- Visualization: Plotly
- UI framework: NiceGUI
```

---

## ✅ Final Checklist

Before releasing:
- [ ] Build tested on clean machine
- [ ] All features working
- [ ] Documentation complete
- [ ] Release notes written
- [ ] Version tagged in git (if using version control)
- [ ] Distribution files created and tested
- [ ] File sizes reasonable (<150MB)
- [ ] No sensitive data in build

---

## 🎯 Quick Commands

**Clean everything:**
```bash
rm -rf build dist __pycache__ *.spec
```

**Rebuild from scratch:**
```bash
rm -rf build dist && ./build_mac.sh
```

**Check dependencies:**
```bash
pip list | grep -E "nicegui|plotly|pandas|scipy|pyinstaller"
```
