"""
Garmin FIT Analyzer - Command Line Interface
Run analysis from terminal without GUI
"""

import sys
import os
from garmin_analyzer.analyzer import FitAnalyzer


def main():
    """CLI entry point."""
    if len(sys.argv) < 2:
        print("Usage: python -m garmin_analyzer.cli <folder_path>")
        print("       python -m garmin_analyzer.cli .")
        sys.exit(1)
    
    folder_path = sys.argv[1]
    
    if not os.path.isdir(folder_path):
        print(f"❌ Error: {folder_path} is not a valid directory")
        sys.exit(1)
    
    print(f"📂 Analyzing FIT files in: {folder_path}")
    print("=" * 60)
    
    analyzer = FitAnalyzer()
    results = analyzer.analyze_folder(folder_path)
    
    print(f"\n✅ Processed {len(results)} file(s)")


if __name__ == "__main__":
    main()
