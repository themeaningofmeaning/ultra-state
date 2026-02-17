"""Shared heart-rate zone thresholds, labels, and colors."""

from __future__ import annotations

from typing import Dict, Tuple

DEFAULT_MAX_HR = 185.0

HR_ZONE_ORDER: Tuple[str, ...] = (
    'Zone 1',
    'Zone 2',
    'Zone 3',
    'Zone 4',
    'Zone 5',
)

HR_ZONE_COLORS: Dict[str, str] = {
    'Zone 1': '#60a5fa',  # Blue
    'Zone 2': '#34d399',  # Emerald
    'Zone 3': '#64748b',  # Slate
    'Zone 4': '#f97316',  # Orange
    'Zone 5': '#ef4444',  # Red
}

HR_ZONE_RANGE_LABELS: Dict[str, str] = {
    'Zone 1': 'Zone 1 (<60%)',
    'Zone 2': 'Zone 2 (60-70%)',
    'Zone 3': 'Zone 3 (70-80%)',
    'Zone 4': 'Zone 4 (80-90%)',
    'Zone 5': 'Zone 5 (>90%)',
}

HR_ZONE_DESCRIPTIONS: Dict[str, str] = {
    'Zone 1': 'Easy (<60% max HR)',
    'Zone 2': 'Aerobic (60-70% max HR)',
    'Zone 3': 'The Grey Zone (70-80% max HR)',
    'Zone 4': 'Hard (80-90% max HR)',
    'Zone 5': 'Max effort (>90% max HR)',
}

HR_ZONE_MAP_LEGEND_GRADIENT = (
    'linear-gradient(to right, '
    '#60a5fa 0%, '
    '#34d399 30%, '
    '#64748b 40%, '
    '#64748b 60%, '
    '#f97316 70%, '
    '#ef4444 100%)'
)


def normalize_max_hr(max_hr, default: float = DEFAULT_MAX_HR) -> float:
    """Return a valid max-HR value."""
    try:
        value = float(max_hr or 0)
    except (TypeError, ValueError):
        value = 0.0
    if value <= 0:
        return float(default)
    return value


def get_zone_thresholds(max_hr) -> Dict[str, float]:
    """Return upper bounds for Zones 1-4 and lower bound for Zone 5."""
    max_hr_value = normalize_max_hr(max_hr)
    return {
        'Zone 1': max_hr_value * 0.60,
        'Zone 2': max_hr_value * 0.70,
        'Zone 3': max_hr_value * 0.80,
        'Zone 4': max_hr_value * 0.90,
        'Zone 5': max_hr_value * 0.90,  # Zone 5 lower bound
    }


def classify_hr_zone_by_ratio(ratio: float) -> str:
    """Classify by HR/max-HR ratio using 5-zone Garmin-style boundaries."""
    try:
        ratio = float(ratio)
    except (TypeError, ValueError):
        ratio = 0.0
    if ratio < 0.60:
        return 'Zone 1'
    if ratio < 0.70:
        return 'Zone 2'
    if ratio < 0.80:
        return 'Zone 3'
    if ratio < 0.90:
        return 'Zone 4'
    return 'Zone 5'


def classify_hr_zone(hr_value, max_hr) -> str:
    """Classify a heart-rate value into one of 5 zones."""
    try:
        hr = float(hr_value or 0)
    except (TypeError, ValueError):
        hr = 0.0
    if hr <= 0:
        return 'Zone 1'
    ratio = hr / normalize_max_hr(max_hr)
    return classify_hr_zone_by_ratio(ratio)


def hr_zone_color(zone: str) -> str:
    """Return the canonical color for a zone key."""
    return HR_ZONE_COLORS.get(zone, HR_ZONE_COLORS['Zone 1'])


def hr_color_for_value(hr_value, max_hr) -> str:
    """Return the canonical HR map color for the provided heart-rate value."""
    return hr_zone_color(classify_hr_zone(hr_value, max_hr))
