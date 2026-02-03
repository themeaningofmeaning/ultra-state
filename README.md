# Garmin FIT Analyzer 🏃‍♂️💨

Garmin FIT Analyzer is a cross-platform desktop application designed to unlock advanced running metrics hidden by standard platforms. It features a modern GUI for batch processing files and generating data specifically optimized for analysis by your LLM of choice (ChatGPT, Claude, Gemini, whatever).

This project is built for serious runners who want deeper insights into efficiency and form.  It's not garbage data - if you feed this into your homerolled LLM running coach, it will help you can tailor your form (cadence), avoid or heal from injuries, and maximize a training plan based on your own body's every-changing capabilities (do hill spints on tuesday, rest tomorrow to lock in gains, do a pace run on Friday, etc.).  Like our training, this tool is a work in progress so contributions, suggestions, and positive vibes will help us all.

![Platform](https://img.shields.io/badge/platform-Windows%20%7C%20macOS-blue)
![Python](https://img.shields.io/badge/python-3.8+-green)

## ✨ Features

- 📁 **Folder Selection** - Pick any folder containing `.fit` files
- 📉 **Trend Analysis** - Visualizes Aerobic Decoupling and Form metrics over time.
- 🏃 **Running Metrics**:
  - **Grade Adjusted Pace (GAP):** Normalizes your effort on hills.
  - **Aerobic Decoupling:** Measures cardiac drift / fatigue.
  - Form Analysis (Ground Contact Time)
  - Cadence & Power
  - Elevation Gain
  - **Efficiency Factor (EF):** Tracks your engine size (Speed ÷ Heart Rate).
- 📋 **Copy to Clipboard** - One-click copy for pasting into Claude/GPT/Gemini
- 📊 **CSV Export** - [optional] Export report as a CSV file for Excel/Sheets analysis.
- 📋 **LLM Ready** - One-click copy for pasting a text report optimized for ChatGPT/Claude/Gemini.
- 📦 **Standalone Executables** - No Python installation required for end users.
- 🔄 **Cross-Platform** - Works on Windows and macOS

## 📸 Screenshots

### The Dashboard
![Dashboard View](assets/dashboard.png)

### Trend Analysis (Matplotlib Integration)
![Graph View](assets/graph.png)

### 🔍 How to Read the Diagnostic Graph

The trend graph uses a unique 4-color system to grade the **Quality** of every run:

- 🟢 **Green (High Quality):** High Efficiency + Low Decoupling. You were fast and stable. (Race Ready).
- 🟡 **Yellow (Maintenance Quality for Base/Recovery):** Low Efficiency + Low Decoupling. A slow, easy aerobic maintenance run.
- 🔴 **Red ("Expensive" Workout):** High Efficiency + High Decoupling. You ran fast, but your heart rate drifted significantly (>5%).
- ⚫ **Dark (Slow & Struggling):** Low Efficiency + High Decoupling. You were slow *and* struggling. Rest needed.

## 🚀 Installation

### Option 1: Download the App (Easiest)

Don't want to mess with Python code? No problem.

1. **[Click here to go to the Releases page](https://github.com/themeaningofmeaning/garmin-fit-analyzer/releases).**
2. **Mac Users:** Download `GarminAnalyzer.dmg`.
3. **Windows Users:** Download `GarminAnalyzer.zip`, unzip it, and run the app inside.

### Option 2: Run from Source (For Developers)

If you want to modify the code or contribute:

```bash
# Clone the repository
git clone [https://github.com/themeaningofmeaning/garmin-fit-analyzer.git](https://github.com/themeaningofmeaning/garmin-fit-analyzer.git)
cd garmin-fit-analyzer

# Create virtual environment (recommended)
python -m venv venv
source venv/bin/activate  # On macOS/Linux
# or: venv\Scripts\activate  # On Windows

# Install dependencies
pip install -r requirements.txt

# Run the application
python -m src.garmin_analyzer.gui
```

## Building Executables

### For macOS
Run the build script to create a .dmg installer:
```bash
./build_mac.sh
# Output: dist/GarminAnalyzer.dmg
```

### For Windows
Run the batch file to build the application:
```bash
build_windows.bat
# Output: dist/GarminAnalyzer
# Note: You must manually zip this folder to share it.
```

## What the Metrics Mean

| Metric | Description | Good Range |
|--------|-------------|------------|
| **GAP** | Grade Adjusted Pace - accounts for elevation | Lower = harder effort |
| **Decoupling** | HR/Speed efficiency drift | < 5% = Excellent |
| **Ground Contact Time** | Time foot stays on ground | 200-300ms typical |
| **Cadence** | Steps per minute | 170-180 spm optimal |

## LLM Integration

The output is designed to be easily copy-pasted into LLMs like Claude, ChatGPT, or Gemini for deeper analysis. Each run report includes:

- **Stats**: Distance, pace, GAP
- **Effort**: Moving/rest time, work/rest ratio
- **Climb**: Elevation gain
- **Metrics**: HR, cadence, power
- **Engine**: HR decoupling percentage with interpretation
- **Form**: Ground contact time change

### 🧠 Pro Tip: Getting the most out of your Garmin data w/ this app

For the most powerful insights, combine these three data sources. This allows the AI to distinguish between a "bad day" and a "bad month."

**1. The Macro Trends (Scope: All History)**
* **File:** `Activities.csv`
* **Source:** Garmin Connect Web $\rightarrow$ *Activities List* $\rightarrow$ Export CSV.
* **Why:** Provides your 90-day baseline. It tells the LLM if your fitness is trending up or down over time.

**2. This App's Report (Scope: This Run Only)**
* **File:** Clipboard Text
* **Source:** **Garmin Analyzer App** $\rightarrow$ "Copy for LLM".
* **Why:** Provides the deep-dive mechanics (Decoupling, Form Efficiency) that Garmin Connect hides.

**3. The Splits (Scope: This Run Only)**
* **File:** `activity_1234.csv`
* **Source:** Garmin Connect Web $\rightarrow$ *Specific Activity* $\rightarrow$ Export Splits to CSV.
* **Why:** Shows pacing strategy. It tells the LLM exactly *where* in the run you started to struggle.

### 📋 Recommended Prompt Strategy
> "I am providing three pieces of data:
> 1. My **Activities Overview** (past 3 months of training).
> 2. My **Splits** for today's run.
> 3. The **Analyzer Report** for today's run (efficiency & decoupling).
>
> **Task:** Analyze today's performance in the context of my recent training load. Was the high cardiac drift caused by poor pacing (see Splits csv, attached), or accumulated fatigue from the last 2 weeks (see Activities csv, attached)?"

## Project Structure

```
garmin-fit-analyzer/
├── src/
│   └── garmin_analyzer/
│       ├── __init__.py
│       ├── analyzer.py      # Core analysis logic
│       └── gui.py           # GUI & Matplotlib charts
├── assets/                  # README screenshots
├── build_mac.sh             # Mac build script
├── build_windows.bat        # Windows build script
├── runner.icns              # Mac App Icon
├── runner.ico               # Windows App Icon
├── requirements.txt         # Dependencies
└── README.md
```

## 🛠️ Built With

* **[CustomTkinter](https://github.com/TomSchimansky/CustomTkinter)** - Modern, dark-mode UI library.
* **[Matplotlib](https://matplotlib.org/)** - Graphing and data visualization.
* **[FitParse](https://github.com/dtcooper/python-fitparse)** - Low-level FIT file parsing.
* **[Pandas](https://pandas.pydata.org/)** & **[NumPy](https://numpy.org/)** - Data manipulation and vector math.

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

---
*Built with love for 🏃‍♂️ and 🍵 by Dylan Goldfus*