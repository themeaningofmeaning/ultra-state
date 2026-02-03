"""
Garmin FIT File Analyzer - Core Analysis Engine
Upgraded for Modern Metrics: Vertical Ratio, GCT Balance, and Minetti Downhill Logic.
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
    Minetti (2002) Energy Cost Equation (J/kg/m).
    Optimized for Ultra Running: Corrects for the "J-curve" where steep 
    downhills (> -10%) actually increase energy cost due to braking.
    """
    grade = np.clip(grade, -0.45, 0.45)
    # The original polynomial naturally handles the upturn at steep negative grades
    cost = 155.4*(grade**5) - 30.4*(grade**4) - 43.3*(grade**3) + 46.3*(grade**2) + 19.5*grade + 3.6
    return cost

class FitAnalyzer:
    """Analyzes Garmin FIT files and extracts advanced running dynamics."""
    
    def __init__(self, output_callback=None):
        self.output_callback = output_callback or self._default_output
    
    def _default_output(self, text: str):
        print(text)
    
    def _emit(self, text: str):
        self.output_callback(text)
    
    def analyze_file(self, filename: str) -> Optional[Dict[str, Any]]:
        try:
            fitfile = fitparse.FitFile(filename)
        except Exception as e:
            self._emit(f"❌ Error opening {filename}: {e}")
            return None

        data = []
        start_time = None

        for record in fitfile.get_messages("record"):
            r = record.get_values()
            if start_time is None and r.get('timestamp'):
                start_time = r.get('timestamp')

            # Extracting standard and advanced running dynamics
            point = {
                'timestamp': r.get('timestamp'),
                'hr': r.get('heart_rate'),
                'cadence': r.get('cadence'), 
                'speed': get_best_value(r, 'speed', 'enhanced_speed'),
                'dist': get_best_value(r, 'distance', 'enhanced_distance'),
                'alt': get_best_value(r, 'altitude', 'enhanced_altitude'),
                'power': r.get('power'),
                # Advanced Dynamics
                'gct': get_best_value(r, 'stance_time', 'ground_contact_time'),
                'v_osc': r.get('vertical_oscillation'), # in mm
                'gct_bal': r.get('left_right_balance') or r.get('stance_time_balance'),
                'stride_len': r.get('stride_length'), # in mm
            }
            data.append(point)

        df = pd.DataFrame(data)
        if df.empty or 'speed' not in df.columns:
            return None
        
        df['alt'] = df['alt'].interpolate(method='linear')
        return self._compute_metrics(df, filename, start_time)
    
    def _compute_metrics(self, df: pd.DataFrame, filename: str, start_time) -> Dict[str, Any]:
        # --- 1. WORK/REST ---
        total_duration_sec = (df['timestamp'].iloc[-1] - df['timestamp'].iloc[0]).total_seconds()
        df_active = df[df['speed'] > 1.0].copy()
        if len(df_active) < 60: return None

        moving_time_min = len(df_active) / 60
        rest_time_min = (total_duration_sec - len(df_active)) / 60
        
        # --- 2. GAP & CLIMB ---
        total_climb_ft = 0
        avg_gap_pace_str = "--:--"
        if 'alt' in df.columns and df['alt'].notnull().any():
            df['alt_diff'] = df['alt'].diff()
            total_climb_ft = df[df['alt_diff'] > 0.2]['alt_diff'].sum() * 3.28084
            
            df_active['dist_diff'] = df_active['dist'].diff()
            df_active['alt_diff'] = df_active['alt'].diff()
            window = 10
            df_active['grade'] = df_active['alt_diff'].rolling(window).sum() / df_active['dist_diff'].rolling(window).sum()
            df_active['grade'] = df_active['grade'].fillna(0).replace([np.inf, -np.inf], 0)
            
            flat_cost = 3.6
            df_active['gap_speed'] = df_active['speed'] * (df_active['grade'].apply(minetti_cost_of_running) / flat_cost)
            avg_gap_speed = df_active['gap_speed'].mean()
            if avg_gap_speed > 0:
                gap_pace_min = 26.8224 / avg_gap_speed
                avg_gap_pace_str = f"{int(gap_pace_min)}:{int((gap_pace_min % 1) * 60):02d}"

        # --- 3. RUNNING DYNAMICS (NEW) ---
        # Vertical Ratio = (Vertical Oscillation / Stride Length) * 100
        v_ratio = 0
        if 'v_osc' in df_active.columns and 'stride_len' in df_active.columns:
            # Garmin stores v_osc in mm and stride_len in mm
            valid_dynamics = df_active[(df_active['v_osc'] > 0) & (df_active['stride_len'] > 0)]
            if not valid_dynamics.empty:
                v_ratio = (valid_dynamics['v_osc'].mean() / valid_dynamics['stride_len'].mean()) * 100

        # GCT Balance (L/R Symmetry)
        # Garmin often encodes this as (Value - 32768) / 100 or a direct %
        gct_bal_val = df_active['gct_bal'].mean() if 'gct_bal' in df_active.columns else 50.0

        # --- 4. ENGINE & FORM ---
        df_active['ef'] = df_active['speed'] / df_active['hr']
        mid = len(df_active) // 2
        ef1, ef2 = df_active.iloc[:mid]['ef'].mean(), df_active.iloc[mid:]['ef'].mean()
        decoupling = ((ef1 - ef2) / ef1) * 100
        
        gct_change = (df_active.iloc[mid:]['gct'].mean() - df_active.iloc[:mid]['gct'].mean()) if 'gct' in df_active.columns else 0

        # --- 5. REPORTING ---
        pace_min = 26.8224 / df_active['speed'].mean()
        pace_str = f"{int(pace_min)}:{int((pace_min % 1) * 60):02d}"
        
        self._emit(f"\n🏃 REPORT: {start_time.strftime('%Y-%m-%d %H:%M')} ({os.path.basename(filename)})")
        self._emit("-" * 50)
        self._emit(f"Stats:    {df['dist'].max()*0.00062:.1f} mi @ {pace_str}/mi (GAP: {avg_gap_pace_str})")
        self._emit(f"Climb:    {total_climb_ft:.0f} ft Gain")
        self._emit(f"Engine:   {decoupling:.1f}% Decoupling -> {'✅ Excellent' if decoupling < 5 else '❌ DRIFT'}")
        
        # New Form Insights
        form_status = "✅ Efficient" if v_ratio < 7.0 else "⚠️ High Bounce"
        self._emit(f"Form:     V-Ratio: {v_ratio:.1f}% | GCT Bal: {gct_bal_val:.1f}% L -> {form_status}")
        self._emit(f"Fatigue:  GCT Drift: {gct_change:+.1f} ms -> {'✅ Stable' if gct_change < 10 else '❌ BREAKDOWN'}")

        return {'filename': filename, 'date': start_time, 'v_ratio': v_ratio, 'decoupling': decoupling}

    def analyze_folder(self, folder_path: str):
        files = sorted([f for f in os.listdir(folder_path) if f.endswith('.fit')])
        results = []
        for f in files:
            res = self.analyze_file(os.path.join(folder_path, f))
            if res: results.append(res)
        return results