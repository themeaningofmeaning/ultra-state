"""
Setup script for py2app (macOS build)
Run: python setup.py py2app
"""

from setuptools import setup

APP = ['src/garmin_analyzer/gui.py']
DATA_FILES = []
OPTIONS = {
    'argv_emulation': True,
    'packages': ['fitparse', 'pandas', 'numpy'],
    'strip': True,
    'compressed': True,
}

setup(
    name='GarminAnalyzer',
    app=APP,
    data_files=DATA_FILES,
    options={'py2app': OPTIONS},
    setup_requires=['py2app'],
)
