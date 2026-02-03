"""
Garmin FIT File Analyzer - Core Analysis Engine
Upgraded: Added Real-Time Progress Tracking via callback.
"""

import fitparse
import pandas as pd
import numpy as np
from datetime import datetime
from typing import Optional, Dict, Any, List
import os

def get_best_value(record, legacy_key, enhanced_key):
    val = record.get(enhanced_key) or record.get(legacy_key)
    return val

def minetti_cost_of_running(grade):
    grade = np.clip(grade, -0.45, 0.45)
    cost = 155.4*(grade**5) - 30.4*(grade**4) - 43.3*(grade**3) + 46.3*(grade**2) + 19.5*grade + 3.6
    return cost

class FitAnalyzer:
    """Analyzes Garmin FIT files and extracts advanced running dynamics."""
    
    def __init__(self, output_callback=None, progress_callback=None):
        self.output_callback = output_callback or self._default_output
        self.progress_callback = progress_callback # Hook for GUI progress bar
    
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

            # FIX: Garmin stores running cadence as RPM (one foot). We want SPM (steps).
            raw_cadence = r.get('cadence')
            cadence_spm = raw_cadence * 2 if raw_cadence else None

            point = {
                'timestamp': r.get('timestamp'),
                'hr': r.get('heart_rate'),
                'cadence': cadence_spm,
                'speed': get_best_value(r, 'speed', 'enhanced_speed'),
                'dist': get_best_value(r, 'distance', 'enhanced_distance'),
                'alt': get_best_value(r, 'altitude', 'enhanced_altitude'),
                'power': r.get('power'),
                'gct': get_best_value(r, 'stance_time', 'ground_contact_time'),
                'v_osc': r.get('vertical_oscillation'),
                'gct_bal': r.get('left_right_balance') or r.get('stance_time_balance'),
                'stride_len': r.get('stride_length'),
                'temp': r.get('temperature'),
                'resp': r.get('respiration_rate')
            }
            data.append(point)

        df = pd.DataFrame(data)
        if df.empty or 'speed' not in df.columns:
            return None
        
        # Ensure numeric types
        cols_to_numeric = ['hr', 'cadence', 'speed', 'dist', 'alt', 'power', 'temp', 'resp']
        for col in cols_to_numeric:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')

        df['alt'] = df['alt'].interpolate(method='linear')
        return self._compute_metrics(df, filename, start_time)
    
    def _compute_metrics(self, df: pd.DataFrame, filename: str, start_time) -> Dict[str, Any]:
        # --- 1. WORK/REST ---
        total_duration_sec = (df['timestamp'].iloc[-1] - df['timestamp'].iloc[0]).total_seconds()
        df_active = df[df['speed'] > 1.0].copy()
        
        if len(df_active) < 60: return None

        moving_time_min = len(df_active) / 60
        rest_time_min = (total_duration_sec - len(df_active)) / 60
        
        # --- 2. GAP & EF ---
        total_climb_ft = 0
        avg_gap_pace_str = "--:--"
        avg_gap_speed_m_min = 0
        
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
                avg_gap_speed_m_min = avg_gap_speed * 60

        # --- 3. DYNAMICS ---
        v_ratio = 0
        if 'v_osc' in df_active.columns and 'stride_len' in df_active.columns:
            valid_dynamics = df_active[(df_active['v_osc'] > 0) & (df_active['stride_len'] > 0)]
            if not valid_dynamics.empty:
                v_ratio = (valid_dynamics['v_osc'].mean() / valid_dynamics['stride_len'].mean()) * 100

        gct_bal_val = df_active['gct_bal'].mean() if 'gct_bal' in df_active.columns else 0

        # --- 4. ENGINE ---
        df_active['ef_metric'] = df_active['speed'] / df_active['hr']
        mid = len(df_active) // 2
        ef1, ef2 = df_active.iloc[:mid]['ef_metric'].mean(), df_active.iloc[mid:]['ef_metric'].mean()
        decoupling = ((ef1 - ef2) / ef1) * 100
        
        efficiency_factor = 0.0
        avg_hr = df_active['hr'].mean()
        if avg_gap_speed_m_min > 0 and avg_hr > 0:
            efficiency_factor = avg_gap_speed_m_min / avg_hr

        gct_change = (df_active.iloc[mid:]['gct'].mean() - df_active.iloc[:mid]['gct'].mean()) if 'gct' in df_active.columns else 0

        # --- 5. AVERAGES ---
        avg_temp = df_active['temp'].mean() if 'temp' in df_active.columns else 0
        avg_resp = df_active['resp'].mean() if 'resp' in df_active.columns else 0
        avg_power = df_active['power'].mean() if 'power' in df_active.columns else 0
        avg_cadence = df_active['cadence'].mean() if 'cadence' in df_active.columns else 0

        pace_min = 26.8224 / df_active['speed'].mean()
        pace_str = f"{int(pace_min)}:{int((pace_min % 1) * 60):02d}"

        self._emit(f"Processing: {os.path.basename(filename)}... Done.")
        
        return {
            'filename': os.path.basename(filename),
            'date': start_time.strftime('%Y-%m-%d %H:%M'),
            'distance_mi': round(df['dist'].max() * 0.000621371, 2),
            'pace': pace_str,
            'gap_pace': avg_gap_pace_str,
            'avg_hr': int(avg_hr) if not pd.isna(avg_hr) else 0,
            'avg_power': int(avg_power) if not pd.isna(avg_power) else 0,
            'avg_cadence': int(avg_cadence) if not pd.isna(avg_cadence) else 0,
            'efficiency_factor': round(efficiency_factor, 2),
            'decoupling': round(decoupling, 2),
            'avg_temp': round(avg_temp, 1) if avg_temp else 0,
            'avg_resp': round(avg_resp, 1) if avg_resp else 0,
            'elevation_ft': int(total_climb_ft),
            'moving_time_min': round(moving_time_min, 1),
            'rest_time_min': round(rest_time_min, 1),
            'gct_change': round(gct_change, 1),
            'v_ratio': round(v_ratio, 2),
            'gct_balance': round(gct_bal_val, 1)
        }

    def analyze_folder(self, folder_path: str):
        files = sorted([f for f in os.listdir(folder_path) if f.endswith('.fit')])
        total_files = len(files)
        results = []
        
        for i, f in enumerate(files):
            # Report Progress (Current, Total)
            if self.progress_callback:
                self.progress_callback(i + 1, total_files)
                
            res = self.analyze_file(os.path.join(folder_path, f))
            if res: results.append(res)
            
        return results