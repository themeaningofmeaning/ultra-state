"""
Garmin FIT File Analyzer - Core Analysis Engine
Upgraded: Fixed 'NoneType' crash by sanitizing all inputs to 0 if missing.
"""

import fitparse
import pandas as pd
import numpy as np
from scipy.signal import find_peaks
from datetime import datetime, timezone
from typing import Optional, Dict, Any, List
import os

def get_best_value(record, legacy_key, enhanced_key):
    val = record.get(enhanced_key) or record.get(legacy_key)
    return val

def analyze_form(cadence, gct=None, stride=None, bounce=None):
    """
    Analyze running form and return verdict, color, icon, and prescription.
    """
    # Default Result
    res = {
        'verdict': 'ANALYZING',
        'color': 'text-zinc-500',
        'bg': 'border-zinc-700',
        'icon': 'help_outline',
        'prescription': 'Not enough data.'
    }
    
    # 1. Safely cast and validate inputs
    try:
        cadence = float(cadence or 0)
        gct = float(gct or 0)
        stride = float(stride or 0)
        bounce = float(bounce or 0)
    except (ValueError, TypeError):
        return res
    
    if cadence == 0:
        return res

    # 2. Normalize Units (Target: mm for both)
    if stride < 10: stride_mm = stride * 1000
    else: stride_mm = stride

    if bounce < 1: bounce_mm = bounce * 1000
    elif bounce < 20: bounce_mm = bounce * 10
    else: bounce_mm = bounce

    # --- DIAGNOSIS TREE ---
    if cadence >= 170:
        res.update({'verdict': 'ELITE FORM', 'color': 'text-emerald-400', 'bg': 'border-emerald-500/30', 'icon': 'verified', 'prescription': 'Pro-level mechanics. Excellent turnover.'})
    elif cadence >= 160:
        res.update({'verdict': 'GOOD FORM', 'color': 'text-blue-400', 'bg': 'border-blue-500/30', 'icon': 'check_circle', 'prescription': 'Balanced mechanics. Solid turnover.'})
    elif cadence < 135:
        res.update({'verdict': 'HIKING / REST', 'color': 'text-blue-400', 'bg': 'border-blue-500/30', 'icon': 'hiking', 'prescription': 'Power hiking or recovery interval.'})
    elif cadence < 155:
        res.update({'verdict': 'HEAVY FEET', 'color': 'text-orange-400', 'bg': 'border-orange-500/30', 'icon': 'warning', 'prescription': 'Cadence is low. Focus on quick turnover.'})
    else:
        res.update({'verdict': 'PLODDING', 'color': 'text-yellow-400', 'bg': 'border-yellow-500/30', 'icon': 'do_not_step', 'prescription': 'Turnover is sluggish. Pick up your feet.'})
    
    return res

def classify_split(cadence, hr, max_hr, grade):
    """
    Classify a single split (mile/lap) into 3 Buckets.
    """
    cadence = cadence or 0
    hr = hr or 0
    max_hr = max_hr or 185
    grade = grade or 0
    z2_limit = max_hr * 0.78
    
    if grade > 8 or cadence < 140: return 'STRUCTURAL'
    if hr > 0 and hr <= z2_limit: return 'STRUCTURAL'
    if cadence >= 160: return 'HIGH QUALITY'
    else: return 'BROKEN'

def minetti_cost_of_running(grade):
    grade = np.clip(grade, -0.45, 0.45)
    cost = 155.4*(grade**5) - 30.4*(grade**4) - 43.3*(grade**3) + 46.3*(grade**2) + 19.5*grade + 3.6
    return cost

class FitAnalyzer:
    def __init__(self, output_callback=None, progress_callback=None):
        self.output_callback = output_callback or self._default_output
        self.progress_callback = progress_callback
    
    def _default_output(self, text: str):
        print(text)
    
    def _emit(self, text: str):
        self.output_callback(text)
    
    def get_training_label(self, aerobic, anaerobic):
        """
        Selective Adaptation Filter.
        Returns custom (Label, Color_Class) based on Training Effect.
        COACHING LOGIC: Filters out 'accidental' anaerobic noise.
        """
        # Ensure inputs are floats (Handle None/Missing data safely)
        try:
            aer = float(aerobic or 0)
            ana = float(anaerobic or 0)
        except (ValueError, TypeError):
            return None, None

        # 1. MAX POWER (The Purple Zone)
        # 3.5+ is undeniable sprint work.
        if ana >= 3.5:
            return "🚀 MAX POWER", "text-purple-400"

        # 2. ANAEROBIC CAPACITY (The Orange Zone)
        # The Threshold: 2.5 (Catches your Flat Sprints).
        # The Filter: Must be within 1.0 of Aerobic score (Ignores Long Runs with hills).
        if ana >= 2.5 and ana > (aer - 1.0):
            return "🔋 ANAEROBIC", "text-orange-400"
            
        # 3. VO2 MAX (The Red Zone)
        if aer >= 4.2:
            return "🫀 VO2 MAX", "text-red-400"

        # 4. THRESHOLD (The Emerald Zone)
        if aer >= 3.5:
            return "📈 THRESHOLD", "text-emerald-400"

        # 5. BASE (Clean UI)
        return None, None
    
    def analyze_file(self, filename: str) -> Optional[Dict[str, Any]]:
        try:
            fitfile = fitparse.FitFile(filename)
        except Exception as e:
            self._emit(f"❌ Error opening {filename}: {e}")
            return None

        data = []
        start_time = None
        
        # --- NEW: Extract Session & Profile Metadata (Safe Mode) ---
        session_max_speed = 0.0
        session_ascent = 0
        user_max_hr = 0
        
        # Initialize defaults for new metrics
        session_metrics = {
            'total_calories': 0,
            'total_training_effect': 0.0,
            'total_anaerobic_training_effect': 0.0,
            'recovery_time': 0,
            'avg_vertical_oscillation': 0.0,
            'avg_stance_time': 0.0,
            'avg_step_length': 0.0,
            'avg_respiration_rate': 0.0,
            'avg_temperature': 0
        }
        
        # 1. Check Session Messages (Official Summary)
        for msg in fitfile.get_messages("session"):
            vals = msg.get_values()
            
            # --- TIMEZONE FIX: Capture and Convert Start Time ---
            if vals.get('start_time'):
                # Convert Naive UTC -> Local System Time
                start_time = vals.get('start_time').replace(tzinfo=timezone.utc).astimezone()
            # ----------------------------------------------------

            # Basic Speed/Ascent/HR (Existing)
            if 'enhanced_max_speed' in vals: 
                session_max_speed = vals.get('enhanced_max_speed') or 0.0
            elif 'max_speed' in vals: 
                session_max_speed = vals.get('max_speed') or 0.0
            
            if 'total_ascent' in vals: 
                session_ascent = vals.get('total_ascent') or 0
            
            if 'max_heart_rate' in vals: 
                user_max_hr = vals.get('max_heart_rate') or 0
            
            # --- FIX: Explicitly extract Temperature & Calories from Summary ---
            if 'avg_temperature' in vals:
                session_metrics['avg_temperature'] = vals.get('avg_temperature') or 0
            if 'total_calories' in vals:
                session_metrics['total_calories'] = vals.get('total_calories') or 0
            
            # --- NEW: PHYSIO & METABOLIC ---
            if 'total_training_effect' in vals:
                session_metrics['total_training_effect'] = vals.get('total_training_effect') or 0.0
            if 'total_anaerobic_training_effect' in vals:
                session_metrics['total_anaerobic_training_effect'] = vals.get('total_anaerobic_training_effect') or 0.0
            
            # --- NEW: RUNNING DYNAMICS (Session Averages) ---
            if 'avg_vertical_oscillation' in vals:
                session_metrics['avg_vertical_oscillation'] = vals.get('avg_vertical_oscillation') or 0.0
            if 'avg_stance_time' in vals:
                session_metrics['avg_stance_time'] = vals.get('avg_stance_time') or 0.0
            if 'avg_step_length' in vals:
                session_metrics['avg_step_length'] = vals.get('avg_step_length') or 0.0
            if 'avg_respiration_rate' in vals:
                session_metrics['avg_respiration_rate'] = vals.get('avg_respiration_rate') or 0.0

        # 2. Check User Profile (Static Setting - Preferred for HR Zones)
        for msg in fitfile.get_messages("user_profile"):
            vals = msg.get_values()
            if 'max_heart_rate' in vals:
                # Only override if we get a valid positive integer
                profile_hr = vals.get('max_heart_rate') or 0
                if profile_hr > 0:
                    user_max_hr = profile_hr

        # --- Parse Records (Time Series) ---
        for record in fitfile.get_messages("record"):
            r = record.get_values()
            if start_time is None and r.get('timestamp'):
                start_time = r.get('timestamp')

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
        
        cols_to_numeric = ['hr', 'cadence', 'speed', 'dist', 'alt', 'power', 'temp', 'resp']
        for col in cols_to_numeric:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')

        df['alt'] = df['alt'].interpolate(method='linear')
        
        # Pass the extracted session metadata to compute metrics
        metadata = {
            'session_max_speed': session_max_speed,
            'session_ascent': session_ascent,
            'user_max_hr': user_max_hr,
            **session_metrics  # Merge in the new metrics
        }
        
        return self._compute_metrics(df, filename, start_time, metadata)
    
    def _compute_metrics(self, df: pd.DataFrame, filename: str, start_time, metadata: Dict) -> Dict[str, Any]:
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
        
        # Use session ascent if available, else calculate from DF
        if metadata['session_ascent'] and metadata['session_ascent'] > 0:
            total_climb_ft = metadata['session_ascent'] * 3.28084
        elif 'alt' in df.columns and df['alt'].notnull().any():
            df['alt_diff'] = df['alt'].diff()
            total_climb_ft = df[df['alt_diff'] > 0.2]['alt_diff'].sum() * 3.28084
            
        if 'alt' in df.columns and df['alt'].notnull().any():
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

        # --- 3. MAX SPEED & HR LOGIC ---
        # Speed: m/s -> mph
        max_speed_mph = 0.0
        if metadata['session_max_speed'] and metadata['session_max_speed'] > 0:
            max_speed_mph = metadata['session_max_speed'] * 2.23694
        elif df['speed'].max() > 0:
            max_speed_mph = df['speed'].max() * 2.23694

        # Max HR: User Profile > Session Max > Observed Max > Default (185)
        max_hr = 185
        if metadata['user_max_hr'] and metadata['user_max_hr'] > 0:
            max_hr = metadata['user_max_hr']
        elif df['hr'].max() > 0:
            max_hr = df['hr'].max()
            
        # --- 4. HRR Analysis ---
        hrr_list = self._calculate_hrr(df)

        # --- 5. DYNAMICS ---
        v_ratio = 0
        if 'v_osc' in df_active.columns and 'stride_len' in df_active.columns:
            valid_dynamics = df_active[(df_active['v_osc'] > 0) & (df_active['stride_len'] > 0)]
            if not valid_dynamics.empty:
                v_ratio = (valid_dynamics['v_osc'].mean() / valid_dynamics['stride_len'].mean()) * 100

        gct_bal_val = df_active['gct_bal'].mean() if 'gct_bal' in df_active.columns else 0

        # --- 6. ENGINE ---
        df_active['ef_metric'] = df_active['speed'] / df_active['hr']
        mid = len(df_active) // 2
        ef1, ef2 = df_active.iloc[:mid]['ef_metric'].mean(), df_active.iloc[mid:]['ef_metric'].mean()
        decoupling = ((ef1 - ef2) / ef1) * 100
        
        efficiency_factor = 0.0
        avg_hr = df_active['hr'].mean()
        if avg_gap_speed_m_min > 0 and avg_hr > 0:
            efficiency_factor = avg_gap_speed_m_min / avg_hr

        gct_change = (df_active.iloc[mid:]['gct'].mean() - df_active.iloc[:mid]['gct'].mean()) if 'gct' in df_active.columns else 0

        # --- 7. AVERAGES ---
        # Use session temperature if available, otherwise fall back to record-level average
        avg_temp = metadata.get('avg_temperature', 0)
        if not avg_temp or avg_temp == 0:
            if 'temp' in df_active.columns:
                temp_mean = df_active['temp'].mean()
                avg_temp = temp_mean if pd.notna(temp_mean) else 0
            else:
                avg_temp = 0
        avg_resp = df_active['resp'].mean() if 'resp' in df_active.columns else 0
        avg_power = df_active['power'].mean() if 'power' in df_active.columns else 0
        avg_cadence = df_active['cadence'].mean() if 'cadence' in df_active.columns else 0

        pace_min = 26.8224 / df_active['speed'].mean()
        pace_str = f"{int(pace_min)}:{int((pace_min % 1) * 60):02d}"

        self._emit(f"Processing: {os.path.basename(filename)}... Done.")
        
        # Get Training Label
        te_label, te_color = self.get_training_label(
            metadata.get('total_training_effect', 0), 
            metadata.get('total_anaerobic_training_effect', 0)
        )

        # --- 8.. BURST DETECTION (Section 3.1) ---
        # 1. Define 'High Speed' as 30% faster than average moving speed
        # Use gap_speed if available (to capture uphill sprints and ignore downhill bombing)
        # Fallback to raw speed if altitude data is missing.
        speed_col = 'gap_speed' if 'gap_speed' in df_active.columns else 'speed'
        
        avg_val = df_active[speed_col].mean()
        burst_threshold = avg_val * 1.3 # 30% intensity jump

        # 2. Count sustained bursts (8+ seconds of continuous high speed)
        is_burst = df_active['speed'] > burst_threshold
        burst_count = 0
        current_streak = 0
        
        for above_limit in is_burst:
            if above_limit:
                current_streak += 1
            else:
                if current_streak >= 8: # 8-second sprint window
                    burst_count += 1
                current_streak = 0
        
        # Catch a burst finishing at the end of the file
        if current_streak >= 8:
            burst_count += 1
        
        return {
            'filename': os.path.basename(filename),
            'filename': os.path.basename(filename),
            'timestamp_utc': int(start_time.timestamp()),
            'date': start_time.strftime('%Y-%m-%d %H:%M'),
            'distance_mi': round(df['dist'].max() * 0.000621371, 2),
            'pace': pace_str,
            'avg_speed_mph': round(df_active['speed'].mean() * 2.23694, 2),
            'burst_count': burst_count,
            'gap_pace': avg_gap_pace_str,
            'avg_hr': int(avg_hr) if not pd.isna(avg_hr) else 0,
            'max_hr': int(max_hr),
            'avg_power': int(avg_power) if not pd.isna(avg_power) else 0,
            'avg_cadence': int(avg_cadence) if not pd.isna(avg_cadence) else 0,
            'efficiency_factor': round(efficiency_factor, 2),
            'decoupling': round(decoupling, 2),
            'avg_temp': round(avg_temp, 1) if (avg_temp and pd.notna(avg_temp) and avg_temp > 0) else 0,
            'avg_resp': round(avg_resp, 1) if avg_resp else 0,
            'hrr_list': hrr_list,
            'elevation_ft': int(total_climb_ft),
            'max_speed_mph': round(max_speed_mph, 2),
            'moving_time_min': round(moving_time_min, 1),
            'rest_time_min': round(rest_time_min, 1),
            'gct_change': round(gct_change, 1),
            'v_ratio': round(v_ratio, 2),
            'gct_balance': round(gct_bal_val, 1),
            # --- NEW FIELDS ---
            'calories': int(metadata.get('total_calories', 0)),
            'training_effect': round(metadata.get('total_training_effect', 0), 1),
            'anaerobic_te': round(metadata.get('total_anaerobic_training_effect', 0), 1),
            'te_label': te_label,
            'te_label_color': te_color,
            # Dynamics (Session averages)
            'avg_vertical_oscillation': round(metadata.get('avg_vertical_oscillation', 0) / 10, 2) if metadata.get('avg_vertical_oscillation') else 0,  # Convert mm to cm
            'avg_stance_time': int(metadata.get('avg_stance_time', 0)),
            'avg_step_length': round(metadata.get('avg_step_length', 0) / 1000, 2) if metadata.get('avg_step_length') else 0,  # Convert mm to m
            'avg_respiration_rate': int(metadata.get('avg_respiration_rate', 0)),
            'recovery_time': int(metadata.get('recovery_time', 0))
        }

    def _calculate_hrr(self, df: pd.DataFrame) -> List[int]:
        if 'hr' not in df.columns or df['hr'].isnull().all():
            return []
        hr_smooth = df['hr'].rolling(window=10, center=True).mean().fillna(df['hr'])
        peaks, _ = find_peaks(hr_smooth, height=140, distance=180)
        hrr_values = []
        for p in peaks:
            if p + 60 < len(df):
                peak_hr = df['hr'].iloc[p]
                recovery_hr = df['hr'].iloc[p + 60]
                delta = peak_hr - recovery_hr
                if delta > 10:
                    hrr_values.append(int(delta))
        return hrr_values

    def analyze_folder(self, folder_path: str):
        files = sorted([f for f in os.listdir(folder_path) if f.endswith('.fit')])
        total_files = len(files)
        results = []
        for i, f in enumerate(files):
            if self.progress_callback:
                self.progress_callback(i + 1, total_files)
            res = self.analyze_file(os.path.join(folder_path, f))
            if res: results.append(res)
        return results