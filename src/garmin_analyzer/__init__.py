"""
Garmin FIT Analyzer
A cross-platform desktop application for analyzing Garmin running fit files.
"""

__version__ = "1.0.0"
__author__ = ""

from garmin_analyzer.analyzer import FitAnalyzer
from garmin_analyzer.gui import main

__all__ = ["FitAnalyzer", "main"]
