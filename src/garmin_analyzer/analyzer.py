"""
Garmin FIT File Analyzer - Core Analysis Engine
Refactored from analyze_run.py with improved structure
"""

import fitparse
import pandas as pd
import numpy as np
from datetime import datetime
from typing import Optional, Dict, Any, List
import os


def get_best_value(record, legacy_key, enhanced_key):
    """Get value from either enhanced or legacy key."""
    val = record.get(enhanced_key) or record.get(legacy_key)
    return val


def minetti_cost_of_running(grade):
    """
    Grade in decimal (0.05 = 5%)
    Minetti (2002) Energy Cost Equation (J/kg/m)
    This aligns with Strava's GAP logic roughly
    """
    grade = np.clip(grade, -0.45, 0.45)  # Cap at extreme grades
    cost = 155.4 * (grade**5) - 30.4 * (grade**4) - 43.3 * (grade**3) + 46.3 * (grade**2) + 19.5 * grade + 3.6
    return cost


class FitAnalyzer:
    """Analyzes Garmin FIT files and extracts running metrics."""
    
    def __init__(self, output_callback=None):
        """
        Initialize analyzer.
        
        Args:
            output_callback: Optional function to call with output lines
        """
        self.output_callback = output_callback or self._default_output
    
    def _default_output(self, text: str):
        """Default output handler - prints to console."""
        print(text)
    
    def _emit(self, text: str):
        """Emit output through callback."""
        self.output_callback(text)
    
    def analyze_file(self, filename: str) -> Optional[Dict[str, Any]]:
        """
        Analyze a single FIT file.
        
        Args:
            filename: Path to FIT file
            
        Returns:
            Dictionary with analysis results or None if invalid
        """
        try:
            fitfile = fitparse.FitFile(filename)
        except Exception as e:
            self._emit(f"❌ Error opening {filename}: {e}")
            return None

        data = []
        start_time = None

        # Parsing the file
        for record in fitfile.get_messages("record"):
            r = record.get_values()
            
            if start_time is None and r.get('timestamp'):
                start_time = r.get('timestamp')

            point = {
                'timestamp': r.get('timestamp'),
                'hr': r.get('heart_rate'),
                'cadence': r.get('cadence'), 
                'speed': get_best_value(r, 'speed', 'enhanced_speed'),
                'dist': get_best_value(r, 'distance', 'enhanced_distance'),
                'alt': get_best_value(r, 'altitude', 'enhanced_altitude'),
                'power': r.get('power'),
                'gct': get_best_value(r, 'stance_time', 'ground_contact_time'),
            }
            data.append(point)

        df = pd.DataFrame(data)

        # Quality control
        if df.empty or 'speed' not in df.columns:
            return None
        
        # Fill N/A for calculation safety
        df['alt'] = df['alt'].interpolate(method='linear')
        
        return self._compute_metrics(df, filename, start_time)
    
    def _compute_metrics(self, df: pd.DataFrame, filename: str, start_time) -> Dict[str, Any]:
        """Compute all metrics from parsed data."""
        
        # --- METRIC 1: WORK/REST ANALYSIS ---
        # Total Elapsed Time
        total_duration_sec = (df['timestamp'].iloc[-1] - df['timestamp'].iloc[0]).total_seconds()
        
        # Filter for Moving (Active) vs Standing
        df_active = df[df['speed'] > 1.0].copy()  # Moving threshold > 1.0 m/s
        
        if len(df_active) < 60:
            return None  # Skip tiny files

        moving_time_sec = len(df_active) * 1.0  # Approx 1s per record usually
        rest_time_min = (total_duration_sec - moving_time_sec) / 60
        moving_time_min = moving_time_sec / 60
        
        # --- METRIC 2: ELEVATION & GAP ---
        total_climb_ft = 0
        avg_gap_pace_str = "--:--"
        
        if 'alt' in df.columns and df['alt'].notnull().any():
            # 1. Calculate Elevation Gain (Simple sum of positive deltas)
            df['alt_diff'] = df['alt'].diff()
            # Filter tiny noise (< 0.2m jumps)
            climb_meters = df[df['alt_diff'] > 0.2]['alt_diff'].sum()
            total_climb_ft = climb_meters * 3.28084
            
            # 2. Calculate GAP
            df_active['dist_diff'] = df_active['dist'].diff()
            df_active['alt_diff'] = df_active['alt'].diff()
            
            # Rolling sum for 10s window to get stable grade
            window = 10
            df_active['roll_dist'] = df_active['dist_diff'].rolling(window).sum()
            df_active['roll_alt'] = df_active['alt_diff'].rolling(window).sum()
            
            # Calculate Grade
            df_active['grade'] = df_active['roll_alt'] / df_active['roll_dist']
            df_active['grade'] = df_active['grade'].fillna(0).replace([np.inf, -np.inf], 0)
            
            # Calculate Energy Cost Ratio (Cost / Flat_Cost)
            flat_cost = 3.6
            df_active['energy_cost'] = df_active['grade'].apply(minetti_cost_of_running)
            df_active['gap_factor'] = df_active['energy_cost'] / flat_cost
            
            # GAP Speed = Speed * Factor
            df_active['gap_speed'] = df_active['speed'] * df_active['gap_factor']
            
            # Avg GAP
            avg_gap_speed = df_active['gap_speed'].mean()
            if avg_gap_speed > 0:
                gap_pace_min = 26.8224 / avg_gap_speed
                avg_gap_pace_str = f"{int(gap_pace_min)}:{int((gap_pace_min % 1) * 60):02d}"

        # --- METRIC 3: BASICS ---
        avg_hr = df_active['hr'].mean()
        
        # Cadence Fix (Doubling if single-sided)
        raw_cadence = df_active['cadence'].mean()
        avg_cadence = raw_cadence * 2 if raw_cadence < 130 else raw_cadence
        
        # Power
        avg_power = df_active['power'].mean() if 'power' in df_active.columns else 0
        
        # Pace
        avg_speed = df_active['speed'].mean()
        pace_min = 26.8224 / avg_speed
        pace_str = f"{int(pace_min)}:{int((pace_min % 1) * 60):02d}"
        
        total_dist_mi = df['dist'].max() * 0.000621371

        # --- METRIC 4: ENGINE CHECK (Decoupling) ---
        df_active['ef'] = df_active['speed'] / df_active['hr']
        mid = len(df_active) // 2
        ef1 = df_active.iloc[:mid]['ef'].mean()
        ef2 = df_active.iloc[mid:]['ef'].mean()
        decoupling = ((ef1 - ef2) / ef1) * 100

        # --- METRIC 5: FORM CHECK ---
        gct_change = 0
        if 'gct' in df_active.columns and df_active['gct'].notna().sum() > 10:
            gct1 = df_active.iloc[:mid]['gct'].mean()
            gct2 = df_active.iloc[mid:]['gct'].mean()
            gct_change = gct2 - gct1

        # Generate Report
        self._emit(f"\n🏃 REPORT: {start_time.strftime('%Y-%m-%d %H:%M')} ({filename})")
        self._emit("-" * 50)
        self._emit(f"Stats:    {total_dist_mi:.1f} mi  @  {pace_str}/mi  (GAP: {avg_gap_pace_str}/mi)")
        self._emit(f"Effort:   Moving: {int(moving_time_min)}m  |  Rest: {int(rest_time_min)}m  (Work:Rest Ratio {moving_time_min/rest_time_min:.1f}:1)")
        self._emit(f"Climb:    {total_climb_ft:.0f} ft Gain")
        
        pwr_txt = f"{avg_power:.0f} W" if avg_power > 0 else "N/A"
        self._emit(f"Metrics:  HR: {avg_hr:.0f} bpm | Cad: {avg_cadence:.0f} spm | Pwr: {pwr_txt}")
        
        drift_msg = "✅ Excellent" if decoupling < 5 else ("⚠️ Minor Drift" if decoupling < 8 else "❌ OVERHEAT")
        self._emit(f"Engine:   {decoupling:5.1f}% Decoupling -> {drift_msg}")
        
        form_msg = "✅ Stable" if gct_change < 10 else "⚠️ BREAKDOWN"
        self._emit(f"Form:     {gct_change:+.1f} ms change    -> {form_msg}")

        return {
            'filename': os.path.basename(filename),
            'date': start_time,
            'distance_mi': total_dist_mi,
            'pace': pace_str,
            'gap_pace': avg_gap_pace_str,
            'moving_time_min': moving_time_min,
            'rest_time_min': rest_time_min,
            'elevation_ft': total_climb_ft,
            'avg_hr': round(avg_hr, 1) if not pd.isna(avg_hr) else 0,
            'avg_cadence': round(avg_cadence, 1) if not pd.isna(avg_cadence) else 0,
            'avg_power': round(avg_power, 1) if not pd.isna(avg_power) else 0,
            'decoupling': round(decoupling, 2),
            'gct_change': round(gct_change, 1),
        }
    
    def analyze_folder(self, folder_path: str) -> List[Dict[str, Any]]:
        """
        Analyze all FIT files in a folder.
        
        Args:
            folder_path: Path to folder containing FIT files
            
        Returns:
            List of analysis results
        """
        import os
        files = sorted([f for f in os.listdir(folder_path) if f.endswith('.fit')])
        
        if not files:
            self._emit(f"⚠️ No .fit files found in {folder_path}")
            return []
        
        self._emit(f"\n📁 Found {len(files)} FIT file(s) in {folder_path}")
        self._emit("=" * 60)
        
        results = []
        for f in files:
            filepath = os.path.join(folder_path, f)
            result = self.analyze_file(filepath)
            if result:
                results.append(result)
        
        self._emit(f"\n✅ Analysis complete! Processed {len(results)} file(s).")
        return results
