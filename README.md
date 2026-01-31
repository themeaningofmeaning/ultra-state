# Garmin FIT Analyzer

A cross-platform desktop application for analyzing Garmin running fit files. Features a graphical interface for selecting folders and processing multiple files at once.

![Platform](https://img.shields.io/badge/platform-Windows%20%7C%20macOS-blue)
![Python](https://img.shields.io/badge/python-3.8+-green)

## Features

- 📁 **Folder Selection** - Pick any folder containing `.fit` files
- 🏃 **Running Metrics**:
  - Grade Adjusted Pace (GAP)
  - Heart Rate Decoupling
  - Form Analysis (Ground Contact Time)
  - Cadence & Power
  - Elevation Gain
- 📊 **CSV Export** - Export all metrics to spreadsheet-compatible CSV
- 📋 **Copy to Clipboard** - One-click copy for pasting into Claude/GPT/Gemini
- 🔄 **Cross-Platform** - Works on Windows and macOS
- 📦 **Standalone Executables** - No Python installation required for end users

## Installation

### Option 1: Run from Source (Requires Python)

```bash
# Clone the repository
git clone https://github.com/YOUR_USERNAME/garmin-fit-analyzer.git
cd garmin-fit-analyzer

# Create virtual environment (recommended)
python -m venv venv
source venv/bin/activate  # On macOS/Linux
# or: venv\Scripts\activate  # On Windows

# Install dependencies
pip install -r requirements.txt

# Run the application
python -m garmin_analyzer
```

### Option 2: Download Executable

Pre-built executables for Windows and macOS are available on the [Releases page](https://github.com/YOUR_USERNAME/garmin-fit-analyzer/releases).

- **Windows**: Download `GarminAnalyzer.exe`
- **macOS**: Download `GarminAnalyzer.dmg` or `GarminAnalyzer.app.zip`

## Building Executables

### For macOS

```bash
# Install py2app
pip install py2app

# Build the application
python setup.py py2app

# Output will be in dist/GarminAnalyzer.app
```

### For Windows

```bash
# Install PyInstaller
pip install pyinstaller

# Build executable
pyinstaller --onefile --windowed --name GarminAnalyzer src/garmin_analyzer/gui.py

# Output will be in dist/GarminAnalyzer.exe
```

## What the Metrics Mean

| Metric | Description | Good Range |
|--------|-------------|------------|
| **GAP** | Grade Adjusted Pace - accounts for elevation | Lower = harder effort |
| **Decoupling** | HR/Speed efficiency drift | < 5% = Excellent |
| **Ground Contact Time** | Time foot stays on ground | 200-300ms typical |
| **Cadence** | Steps per minute | 170-180 spm optimal |

## Project Structure

```
garmin-fit-analyzer/
├── src/
│   └── garmin_analyzer/
│       ├── __init__.py
│       ├── __main__.py
│       ├── analyzer.py      # Core analysis engine
│       ├── cli.py           # Command-line interface
│       └── gui.py           # Tkinter GUI
├── pyproject.toml           # Package configuration
├── requirements.txt         # Dependencies
├── build_mac.sh             # macOS build script
├── build_windows.bat        # Windows build script
└── README.md
```

## LLM Integration

The output is designed to be easily copy-pasted into LLMs like Claude, ChatGPT, or Gemini for deeper analysis. Each run report includes:

- **Stats**: Distance, pace, GAP
- **Effort**: Moving/rest time, work/rest ratio
- **Climb**: Elevation gain
- **Metrics**: HR, cadence, power
- **Engine**: HR decoupling percentage with interpretation
- **Form**: Ground contact time change

## Dependencies

- `fitparse` - FIT file parsing
- `pandas` - Data manipulation
- `numpy` - Numerical computations

## License

MIT License - see LICENSE file for details.

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Submit a pull request

## Acknowledgments

- Based on the Minetti (2002) energy cost equation for grade adjustments
- Inspired by Strava's GAP methodology
