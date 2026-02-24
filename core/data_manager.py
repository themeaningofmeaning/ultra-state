"""Core data lifecycle and filtering logic for Ultra State."""

from __future__ import annotations

import os
from datetime import datetime

import pandas as pd


class DataManager:
    """Owns activity data loading, filtering, classification, and CSV export."""

    def __init__(self, db, state):
        self.db = db
        self.state = state
        self.df = None
        self.activities_data = []

    def _rebuild_dataframe(self) -> None:
        if self.activities_data:
            self.df = pd.DataFrame(self.activities_data)
            if 'date' in self.df.columns:
                self.df['date_obj'] = pd.to_datetime(self.df['date'])
        else:
            self.df = None

    def load_data(self):
        """Load activities from DB based on current app state and rebuild DataFrame."""
        sort_order = 'desc' if self.state.sort_desc else 'asc'
        self.activities_data = self.db.get_activities(
            self.state.timeframe,
            self.state.session_id,
            sort_by=self.state.sort_by,
            sort_order=sort_order,
        )
        self._rebuild_dataframe()
        return self.df, self.activities_data

    def set_activities_data(self, activities_data):
        """Replace in-memory activity payloads and rebuild DataFrame cache."""
        self.activities_data = activities_data or []
        self._rebuild_dataframe()
        return self.df, self.activities_data

    def _get_long_run_threshold(self) -> float:
        if (
            self.df is not None
            and len(self.df) >= 5
            and 'distance_mi' in self.df.columns
        ):
            return self.df['distance_mi'].quantile(0.8)
        return 10.0

    @staticmethod
    def _clean_context_tag(tag: str) -> str:
        return (
            tag.replace('⛰️', '')
            .replace('🔥', '')
            .replace('🦅', '')
            .replace('⚡', '')
            .replace('🏃', '')
            .replace('🧘', '')
            .replace('🔷', '')
            .strip()
        )

    def classify_run_type(self, run_data, long_run_threshold):
        """
        Stackable classification system.
        Uses burst_count to distinguish true workouts from speed outliers.
        """
        tags = []

        distance_mi = run_data.get('distance_mi', 0)
        avg_hr = run_data.get('avg_hr', 0)
        max_hr = run_data.get('max_hr', 185)
        elevation_ft = run_data.get('elevation_ft', 0)
        avg_temp = run_data.get('avg_temp', 0)
        burst_count = run_data.get('burst_count', 0)

        hr_ratio = avg_hr / max_hr if max_hr > 0 else 0
        grade = (elevation_ft / distance_mi) if distance_mi > 0 else 0

        if distance_mi > long_run_threshold:
            primary = "🦅 Long Run"
        elif distance_mi < 4.0 and hr_ratio < 0.75:
            primary = "🧘 Recovery"
        elif hr_ratio > 0.82:
            primary = "🔥 Tempo"
        elif hr_ratio > 0.75:
            primary = "🔷 Steady"
        else:
            primary = "🟡 Base"
        tags.append(primary)

        if grade > 75:
            tags.append("⛰️ Hilly")

        if burst_count >= 4:
            tags.append("⚡ Intervals")
        elif burst_count >= 2:
            tags.append("💨 Strides")

        if avg_temp and avg_temp > 25:
            tags.append("🥵 Hot")
        elif avg_temp and avg_temp < 5:
            tags.append("🥶 Cold")

        return " | ".join(tags)

    def get_filtered_data(self):
        """Apply active distance/tag filters to in-memory activities."""
        if not self.activities_data:
            return []

        filtered_data = []
        long_run_threshold = self._get_long_run_threshold()
        active_filters = self.state.active_filters or set()

        for activity in self.activities_data:
            distance_mi = activity.get('distance_mi', 0)

            run_tags_set = set()
            classified_tags = self.classify_run_type(activity, long_run_threshold)
            if classified_tags:
                for tag in classified_tags.split(' | '):
                    cleaned = self._clean_context_tag(tag)
                    if cleaned:
                        run_tags_set.add(cleaned)

            te_label = activity.get('te_label')
            if te_label and str(te_label) != "None":
                run_tags_set.add(str(te_label).strip())

            include = True
            if active_filters:
                active_filter_set = set(active_filters)
                distance_keys = {'short', 'med', 'long_dist', 'all'}
                active_tag_filters = {f for f in active_filter_set if f not in distance_keys}

                if 'short' in active_filter_set and not (distance_mi < 5):
                    include = False
                if 'med' in active_filter_set and not (5 <= distance_mi <= 10):
                    include = False
                if 'long_dist' in active_filter_set and not (distance_mi > 10):
                    include = False
                if active_tag_filters and not active_tag_filters.issubset(run_tags_set):
                    include = False

            if include:
                filtered_data.append(activity)

        return filtered_data

    def _generate_csv_content(self, df=None) -> str:
        """Generate CSV body + data dictionary for AI analysis exports."""
        export_df = self.df if df is None else df
        if export_df is None or export_df.empty:
            raise ValueError("No data to export")

        export_columns = [
            'date',
            'filename',
            'distance_mi',
            'pace',
            'gap_pace',
            'moving_time_min',
            'rest_time_min',
            'avg_hr',
            'max_hr',
            'avg_power',
            'avg_cadence',
            'efficiency_factor',
            'decoupling',
            'load_score',
            'zone1_mins',
            'zone2_mins',
            'zone3_mins',
            'zone4_mins',
            'zone5_mins',
            'hrr_list',
            'v_ratio',
            'gct_balance',
            'gct_change',
            'elevation_ft',
            'avg_resp',
            'avg_temp',
        ]

        prepared_df = export_df.copy()
        text_columns = {'date', 'filename', 'pace', 'gap_pace', 'hrr_list'}
        for col in export_columns:
            if col not in prepared_df.columns:
                prepared_df[col] = '' if col in text_columns else 0

        csv_content = prepared_df[export_columns].to_csv(index=False)
        data_dictionary = """

=== DATA DICTIONARY FOR AI ANALYSIS ===

EFFICIENCY FACTOR (EF):
- Normalized graded speed (m/min) divided by heart rate
- Higher is better - indicates aerobic efficiency
- Typical range: 0.8-1.5 for recreational runners, 1.5-2.5+ for elite

AEROBIC DECOUPLING:
- Percentage loss of efficiency from first half to second half of run
- Lower is better - indicates aerobic durability
- < 5%: Excellent aerobic fitness
- 5-10%: Moderate drift, acceptable for long runs
- > 10%: High fatigue, may indicate overtraining or insufficient base

HEART RATE RECOVERY (HRR):
- BPM drop 60 seconds after peak efforts
- Higher is better - indicates cardiovascular fitness
- > 30 bpm: Excellent recovery
- 20-30 bpm: Good recovery
- < 20 bpm: Poor recovery, may indicate fatigue

FORM METRICS:
- Cadence: Steps per minute (target: 170-180 spm)
- Vertical Ratio: Vertical oscillation / stride length (lower is better)
- GCT Balance: Ground contact time balance left/right (target: 50/50)
- GCT Change: Ground contact time change over run (lower is better)

PACE METRICS:
- Pace: Actual pace (min/mile)
- GAP Pace: Grade-adjusted pace accounting for elevation changes

TRAINING ZONES:
- Green (Peak Efficiency): High EF + Low Decoupling = Your output (Speed) was high relative to your input (Heart Rate)
- Yellow (Base Maintenance): Low EF + Low Decoupling = Building aerobic base
- Orange (Expensive Speed): High EF + High Decoupling = Fast but unsustainable
- Red (Struggling): Low EF + High Decoupling = Fatigue or overtraining

TRAINING LOAD (load_score):
- TRIMP-style internal load score (duration x HR intensity)
- < 75: Recovery / Easy
- 75-150: Base / Aerobic
- 150-300: Overload / Hard
- > 300: Overreaching / Extreme

TIME IN ZONES (zone1_mins ... zone5_mins):
- Minutes spent in each heart-rate zone
- Zone 1: <60% max HR (Recovery)
- Zone 2: 60-70% max HR (Endurance)
- Zone 3: 70-80% max HR (Steady)
- Zone 4: 80-90% max HR (Threshold)
- Zone 5: >90% max HR (VO2/Max Effort)
"""
        return csv_content + data_dictionary

    def export_csv(self, destination_dir=None) -> str:
        """Write CSV export to disk and return the saved file path."""
        full_content = self._generate_csv_content(self.df)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"garmin_analysis_{timestamp}.csv"
        output_dir = destination_dir or os.path.expanduser("~/Downloads")
        os.makedirs(output_dir, exist_ok=True)
        file_path = os.path.join(output_dir, filename)

        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(full_content)

        return file_path

