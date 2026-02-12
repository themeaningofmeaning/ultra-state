"""
Garmin FIT Analyzer
"""

# Standard library imports
import os
import time
import json
import hashlib
import sqlite3
import asyncio
from datetime import datetime, timezone

# Third-party imports
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from nicegui import ui, run

# Local imports
from analyzer import FitAnalyzer


# --- DATABASE MANAGER (Preserved from gui.py) ---
class DatabaseManager:
    def __init__(self, db_path='runner_stats.db'):
        self.db_path = db_path
        self.create_tables()

    def get_connection(self):
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        return conn

    def create_tables(self):
        with self.get_connection() as conn:
            conn.execute('''
                CREATE TABLE IF NOT EXISTS activities (
                    hash TEXT PRIMARY KEY,
                    filename TEXT,
                    date TEXT,
                    json_data TEXT,
                    session_id INTEGER,
                    file_path TEXT
                )
            ''')

            # Migration: Add file_path column if it doesn't exist
            try:
                conn.execute("SELECT file_path FROM activities LIMIT 1")
            except sqlite3.OperationalError:
                # Column doesn't exist, add it
                print("Migrating database: Adding file_path column...")
                conn.execute("ALTER TABLE activities ADD COLUMN file_path TEXT")
                print("Migration complete!")



    def activity_exists(self, file_hash):
        with self.get_connection() as conn:
            cursor = conn.execute("SELECT 1 FROM activities WHERE hash = ?", (file_hash,))
            return cursor.fetchone() is not None

    def insert_activity(self, activity_data, file_hash, session_id, file_path=None):
        json_str = json.dumps(activity_data, default=str)
        with self.get_connection() as conn:
            conn.execute(
                "INSERT OR REPLACE INTO activities (hash, filename, date, json_data, session_id, file_path) VALUES (?, ?, ?, ?, ?, ?)",
                (file_hash, activity_data.get('filename'), activity_data.get('date'), json_str, session_id, file_path)
            )

            
    def delete_activity(self, file_hash):
        with self.get_connection() as conn:
            conn.execute("DELETE FROM activities WHERE hash = ?", (file_hash,))

    def get_count(self):
        with self.get_connection() as conn:
            return conn.execute("SELECT COUNT(*) FROM activities").fetchone()[0]

    def get_activities(self, timeframe="All Time", current_session_id=None):
        query = "SELECT json_data, hash FROM activities"
        params = []
        
        if timeframe == "Last Import" and current_session_id:
            query += " WHERE session_id = ?"
            params.append(current_session_id)
        elif timeframe == "Last 30 Days":
            date_Limit = (datetime.now() - pd.Timedelta(days=30)).strftime("%Y-%m-%d")
            query += " WHERE date >= ?"
            params.append(date_Limit)
        elif timeframe == "Last 90 Days":
            date_Limit = (datetime.now() - pd.Timedelta(days=90)).strftime("%Y-%m-%d")
            query += " WHERE date >= ?"
            params.append(date_Limit)
        elif timeframe == "This Year":
            current_year = datetime.now().year
            query += " WHERE date >= ?"
            params.append(f"{current_year}-01-01")
            
        query += " ORDER BY date DESC"
        
        with self.get_connection() as conn:
            rows = conn.execute(query, params).fetchall()
            results = []
            for row in rows:
                d = json.loads(row[0])
                d['db_hash'] = row[1]
                results.append(d)
            return results
    def get_activity_by_hash(self, file_hash):
        """Get activity data by hash."""
        with self.get_connection() as conn:
            row = conn.execute(
                "SELECT json_data, file_path FROM activities WHERE hash = ?",
                (file_hash,)
            ).fetchone()
            if row:
                activity = json.loads(row[0])
                activity['file_path'] = row[1]
                activity['hash'] = file_hash
                return activity
            return None



# --- HELPER: FILE HASHING (Preserved from gui.py) ---
def calculate_file_hash(filepath):
    """Calculate SHA-256 hash of a file for deduplication."""
    hasher = hashlib.sha256()
    with open(filepath, 'rb') as f:
        buf = f.read(65536)
        while len(buf) > 0:
            hasher.update(buf)
            buf = f.read(65536)
    return hasher.hexdigest()

# --- HELPER: FORM ANALYSIS (Centralized Logic) ---
def analyze_form(cadence, gct=None, stride=None, bounce=None):
    """
    Analyze running form and return verdict, color, icon, and prescription.
    Used for single-point diagnosis (Tooltips, Run Summary).
    UPDATED: Better thresholds for mixed run/walk activities.
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
    
    # If cadence is missing, we can't analyze
    if cadence == 0:
        return res

    # 2. Normalize Units (Target: mm for both)
    # Stride: If small (< 10), assume meters. If large (> 10), assume mm.
    if stride < 10: 
        stride_mm = stride * 1000
    else:
        stride_mm = stride

    # Bounce: If small (< 1), assume meters. If moderate (< 20), assume cm. Else mm.
    if bounce < 1:
        bounce_mm = bounce * 1000
    elif bounce < 20:
        bounce_mm = bounce * 10
    else:
        bounce_mm = bounce

    # 3. Calculate Vertical Ratio (%)
    ratio = (bounce_mm / stride_mm) * 100 if stride_mm > 0 else 0
    
    # --- DIAGNOSIS TREE (Simple & Robust) ---
    
    # Elite/Good
    if cadence >= 170:
        res.update({
            'verdict': 'ELITE FORM',
            'color': 'text-emerald-400',
            'bg': 'border-emerald-500/30',
            'icon': 'verified',
            'prescription': 'Pro-level mechanics. Excellent turnover.'
        })
    elif cadence >= 160:
        res.update({
            'verdict': 'GOOD FORM',
            'color': 'text-blue-400',
            'bg': 'border-blue-500/30',
            'icon': 'check_circle',
            'prescription': 'Balanced mechanics. Solid turnover.'
        })
    # The "Structural" Zone (Widened to 135 to catch recovery jogs/walks)
    elif cadence < 135:
        res.update({
            'verdict': 'HIKING / REST',
            'color': 'text-blue-400',
            'bg': 'border-blue-500/30',
            'icon': 'hiking',
            'prescription': 'Power hiking or recovery interval.'
        })
    # The "Grey Area" (Renamed from Overstriding to Heavy Feet)
    # Captures the 135-154 range which is typical for tired jogging
    elif cadence < 155:
        res.update({
            'verdict': 'HEAVY FEET', 
            'color': 'text-orange-400',
            'bg': 'border-orange-500/30',
            'icon': 'warning',
            'prescription': 'Cadence is low. Focus on quick turnover.'
        })
    # The "Meh" Zone (155-159)
    else:
        res.update({
            'verdict': 'PLODDING',
            'color': 'text-yellow-400',
            'bg': 'border-yellow-500/30',
            'icon': 'do_not_step',
            'prescription': 'Turnover is sluggish. Pick up your feet.'
        })
    
    return res

def classify_split(cadence, hr, max_hr, grade):
    """
    Classify a single split (mile/lap) into 3 Buckets.
    UPDATED: Smarter handling of Recovery/Walking intervals.
    """
    # Safety defaults
    cadence = cadence or 0
    hr = hr or 0
    max_hr = max_hr or 185
    grade = grade or 0
    
    z2_limit = max_hr * 0.78
    
    # GATE 1: TERRAIN & RECOVERY (The Base)
    # If steep grade (>8%) OR low cadence (<140), it's Structural/Recovery.
    # <140 catches walking breaks and slow recovery jogs.
    if grade > 8 or cadence < 140:
        return 'STRUCTURAL'
        
    # GATE 2: METABOLIC (The Easy Miles)
    if hr > 0 and hr <= z2_limit:
        return 'STRUCTURAL'
        
    # GATE 3: PERFORMANCE (The Work)
    # If you are running hard (High HR, Flat Ground)...
    if cadence >= 160:
        return 'HIGH QUALITY'  # Good mechanics
    else:
        return 'BROKEN'   # High HR + Low Cadence (140-159) = Mechanical Fade

# --- MAIN APPLICATION CLASS ---
class GarminAnalyzerApp:
    """Main application class for the Garmin FIT Analyzer."""
    
    def __init__(self):
        """Initialize the application with database and state."""
        self.db = DatabaseManager()
        self.current_session_id = None
        self.current_timeframe = "Last 30 Days"
        self.activities_data = []
        self.df = None
        self.import_in_progress = False

        # Initialize the volume data container
        self.weekly_volume_data = None
        
        # Build UI
        self.build_ui()
        
        # Schedule initial data load using NiceGUI's timer (runs after event loop starts)
        ui.timer(0.1, self.refresh_data_view, once=True)
        
        # Show Save Chart button initially with fade (Trends tab is default)
        ui.timer(0.05, lambda: self.save_chart_btn.style('opacity: 1; pointer-events: auto;'), once=True)
    def _locate_fit_file(self, activity):
        """
        Locate FIT file on disk for an activity.

        Args:
            activity: Activity dictionary with file_path and filename

        Returns:
            Full path to FIT file if found, None otherwise
        """
        # First, check if file_path exists in database and file exists at that path
        if activity.get('file_path') and os.path.exists(activity['file_path']):
            return activity['file_path']

        # If missing or invalid, we can't locate it without additional context
        # In a future enhancement, we could search in last import directory
        return None
    
    def _calculate_distance_from_speed(self, speed_stream, timestamps):
        """
        Calculate cumulative distance from speed and timestamps.
        
        Args:
            speed_stream: List[float] - speed in m/s
            timestamps: List[datetime] - timestamp for each record
        
        Returns:
            List[float] - cumulative distance in meters
        """
        distance_stream = []
        cumulative_distance = 0.0
        
        for i in range(len(speed_stream)):
            if i == 0:
                distance_stream.append(0.0)
                continue
            
            # Get time delta in seconds
            if timestamps[i] and timestamps[i-1]:
                time_delta = (timestamps[i] - timestamps[i-1]).total_seconds()
            else:
                time_delta = 1.0  # Default to 1 second if timestamp missing
            
            # Get speed (use previous valid speed if current is None)
            speed = speed_stream[i] if speed_stream[i] is not None else (speed_stream[i-1] if i > 0 and speed_stream[i-1] is not None else 0.0)
            
            # Calculate distance increment
            distance_increment = speed * time_delta
            cumulative_distance += distance_increment
            distance_stream.append(cumulative_distance)
        
        return distance_stream
    
    async def get_activity_detail(self, activity_hash):
        """
        Parse FIT file to extract detailed stream data for modal display.

        Args:
            activity_hash: SHA-256 hash identifying the activity

        Returns:
            Dictionary containing:
            - hr_stream: List[int] - heart rate values
            - elevation_stream: List[float] - elevation values in meters
            - speed_stream: List[float] - speed values in m/s
            - lap_data: List[Dict] - lap splits with metrics
            - max_hr: int - maximum heart rate for zone calculations
            - activity_metadata: Dict - basic activity info from database
            - timestamps: List[datetime] - timestamps for each data point
        """
        # 1. Query database for activity metadata
        activity = self.db.get_activity_by_hash(activity_hash)
        if not activity:
            return None

        # 2. Locate FIT file on disk
        fit_file_path = self._locate_fit_file(activity)
        if not fit_file_path:
            return {'error': 'file_not_found', 'activity': activity}

        # 3. Parse FIT file using fitparse (run in thread to avoid blocking)
        try:
            def parse_fit():
                import fitparse
                fitfile = fitparse.FitFile(fit_file_path)

                # 4. Extract stream data
                hr_stream = []
                elevation_stream = []
                speed_stream = []
                cadence_stream = []
                distance_stream = []
                timestamps = []

                for record in fitfile.get_messages("record"):
                    vals = record.get_values()
                    
                    # --- UTC TO LOCAL FIX ---
                    ts = vals.get('timestamp')
                    if ts:
                        # Ensure we have the timezone module imported at top of file!
                        ts = ts.replace(tzinfo=timezone.utc).astimezone()
                    timestamps.append(ts)
                    # ------------------------

                    hr_stream.append(vals.get('heart_rate'))
                    elevation_stream.append(
                        vals.get('enhanced_altitude') or vals.get('altitude')
                    )
                    speed_stream.append(
                        vals.get('enhanced_speed') or vals.get('speed')
                    )
                    cadence_stream.append(vals.get('cadence'))
                    distance_stream.append(vals.get('distance'))

                # 5. Extract lap data
                lap_data = []
                for lap_msg in fitfile.get_messages("lap"):
                    vals = lap_msg.get_values()
                    
                    # --- TIMEZONE FIX ---
                    if vals.get('start_time'):
                        vals['start_time'] = vals.get('start_time').replace(tzinfo=timezone.utc).astimezone()
                    
                    if vals.get('timestamp'):
                        vals['timestamp'] = vals.get('timestamp').replace(tzinfo=timezone.utc).astimezone()
                    # --------------------

                    # Calculate speed directly if enhanced_avg_speed is missing
                    total_distance = vals.get('total_distance')
                    total_timer_time = vals.get('total_timer_time')
                    
                    if total_distance and total_timer_time and total_timer_time > 0:
                        avg_speed = total_distance / total_timer_time
                    else:
                        avg_speed = vals.get('enhanced_avg_speed') or vals.get('avg_speed')
                    
                    lap_data.append({
                        'lap_number': len(lap_data) + 1,
                        'distance': total_distance,
                        'avg_speed': avg_speed,
                        'avg_hr': vals.get('avg_heart_rate'),
                        'avg_cadence': vals.get('avg_cadence'),
                        'total_ascent': vals.get('total_ascent'),
                        'total_descent': vals.get('total_descent'),
                        'start_time': vals.get('start_time'),
                        'timestamp': vals.get('timestamp'), # Added for GAP calculation
                        'total_elapsed_time': vals.get('total_elapsed_time')
                    })
                
                # 6. Extract session-level metrics (physiology, environment, form)
                session_data = {}
                for session_msg in fitfile.get_messages("session"):
                    vals = session_msg.get_values()
                    session_data = {
                        # Environment
                        'total_calories': vals.get('total_calories'),
                        'avg_temperature': vals.get('avg_temperature'),
                        # Physiology
                        'total_training_effect': vals.get('total_training_effect'),
                        'total_anaerobic_training_effect': vals.get('total_anaerobic_training_effect'),
                        'avg_respiration_rate': vals.get('avg_respiration_rate'),
                        # Running Dynamics (Form)
                        'avg_vertical_oscillation': vals.get('avg_vertical_oscillation'),
                        'avg_stance_time': vals.get('avg_stance_time'),
                        'avg_step_length': vals.get('avg_step_length'),
                    }
                    break  # Only need first session

                # Fallback: Calculate distance from speed if not available
                if all(d is None for d in distance_stream):
                    distance_stream = self._calculate_distance_from_speed(speed_stream, timestamps)

                return {
                    'hr_stream': hr_stream,
                    'elevation_stream': elevation_stream,
                    'speed_stream': speed_stream,
                    'cadence_stream': cadence_stream,
                    'distance_stream': distance_stream,
                    'lap_data': lap_data,
                    'timestamps': timestamps,
                    'session_data': session_data
                }

            # Run parsing in background thread
            result = await run.io_bound(parse_fit)
            
            # Check if parsing failed
            if result is None:
                return {'error': 'parse_error', 'activity': activity, 'message': 'Failed to parse FIT file'}

            # 6. Get max HR (priority: user profile > session max > observed max)
            max_hr = activity.get('max_hr')
            if not max_hr and result.get('hr_stream'):
                # Use session max as fallback
                valid_hrs = [hr for hr in result['hr_stream'] if hr is not None]
                max_hr = max(valid_hrs) if valid_hrs else 185
                result['max_hr_fallback'] = True
            else:
                max_hr = max_hr or 185
                result['max_hr_fallback'] = False

            result['max_hr'] = max_hr
            result['activity_metadata'] = activity

            return result

        except Exception as e:
            print(f"Error parsing FIT file: {e}")
            import traceback
            traceback.print_exc()
            return {'error': 'parse_error', 'activity': activity, 'message': str(e)}
    
    def get_training_label(self, aerobic_te, anaerobic_te):
        """
        Determine training type label based on Training Effect values.
        
        Args:
            aerobic_te: Aerobic training effect (0.0 - 5.0)
            anaerobic_te: Anaerobic training effect (0.0 - 5.0)
            
        Returns:
            Dictionary with label, color, and bg_color
        """
        if anaerobic_te and anaerobic_te > 2.5:
            return {
                'label': 'SPEED / POWER',
                'color': 'text-purple-400',
                'bg_color': 'bg-purple-500/10'
            }
        elif aerobic_te and aerobic_te > 3.5:
            return {
                'label': 'TEMPO / THRESHOLD',
                'color': 'text-orange-400',
                'bg_color': 'bg-orange-500/10'
            }
        elif aerobic_te and aerobic_te < 2.5:
            return {
                'label': 'RECOVERY / BASE',
                'color': 'text-emerald-400',
                'bg_color': 'bg-emerald-500/10'
            }
        else:
            return {
                'label': 'AEROBIC',
                'color': 'text-blue-400',
                'bg_color': 'bg-blue-500/10'
            }
    
    def calculate_hr_zones(self, hr_stream, max_hr):
        """
        Calculate time spent in each of 5 heart rate zones.

        Args:
            hr_stream: List of heart rate values (one per second)
            max_hr: Maximum heart rate for zone calculations

        Returns:
            Dictionary with zone names as keys and time in minutes as values
        """
        # Filter out None values
        hr_stream = [hr for hr in hr_stream if hr is not None]

        if not hr_stream:
            return {
                'Zone 1 (<60%)': 0,
                'Zone 2 (60-70%)': 0,
                'Zone 3 (70-80%)': 0,
                'Zone 4 (80-90%)': 0,
                'Zone 5 (>90%)': 0
            }

        # Calculate zone boundaries
        z1_max = max_hr * 0.60
        z2_max = max_hr * 0.70
        z3_max = max_hr * 0.80
        z4_max = max_hr * 0.90

        # Count seconds in each zone
        zone_counts = {
            'Zone 1 (<60%)': 0,
            'Zone 2 (60-70%)': 0,
            'Zone 3 (70-80%)': 0,
            'Zone 4 (80-90%)': 0,
            'Zone 5 (>90%)': 0
        }

        for hr in hr_stream:
            if hr < z1_max:
                zone_counts['Zone 1 (<60%)'] += 1
            elif hr < z2_max:
                zone_counts['Zone 2 (60-70%)'] += 1
            elif hr < z3_max:
                zone_counts['Zone 3 (70-80%)'] += 1
            elif hr < z4_max:
                zone_counts['Zone 4 (80-90%)'] += 1
            else:
                zone_counts['Zone 5 (>90%)'] += 1

        # Convert seconds to minutes
        zone_times = {
            zone: count / 60.0
            for zone, count in zone_counts.items()
        }

        return zone_times
    
    def calculate_gap_for_laps(self, lap_data, elevation_stream, timestamps, cadence_stream=None, max_hr=185):
        """
        Calculate GAP, Average Cadence, and QUALITY VERDICT for each lap.
        UPDATED: Now classifies each split as High Quality / Structural / Broken.
        """
        from analyzer import minetti_cost_of_running

        enhanced_laps = []

        for lap in lap_data:
            # Skip if lap has no valid data
            if not lap.get('start_time') or not lap.get('total_elapsed_time'):
                enhanced_laps.append({
                    **lap,
                    'gap_pace': '--:--', 'actual_pace': '--:--',
                    'is_steep': False, 'avg_gradient': 0, 'avg_cadence': None,
                    'split_verdict': 'STRUCTURAL' # Default
                })
                continue
            
            # Calculate avg_speed if not available
            avg_speed = lap.get('avg_speed')
            if avg_speed is None or avg_speed == 0:
                distance = lap.get('distance', 0)
                elapsed_time = lap.get('total_elapsed_time', 0)
                if distance > 0 and elapsed_time > 0:
                    avg_speed = distance / elapsed_time
                else:
                    avg_speed = 0

            # Calculate average gradient & cadence for this lap
            lap_start = lap['start_time']
            lap_end = lap_start + pd.Timedelta(seconds=lap['total_elapsed_time'])

            lap_elevations = []
            lap_cadences = []

            for i, ts in enumerate(timestamps):
                if ts is None: continue
                if lap_start <= ts <= lap_end:
                    # Elevation
                    if elevation_stream[i] is not None:
                        if i > 0 and elevation_stream[i-1] is not None:
                            elev_diff = elevation_stream[i] - elevation_stream[i-1]
                            dist_diff = avg_speed 
                            if dist_diff is not None and dist_diff > 0:
                                gradient = elev_diff / dist_diff
                                lap_elevations.append(gradient)
                    # Cadence
                    if cadence_stream and i < len(cadence_stream) and cadence_stream[i] is not None:
                        lap_cadences.append(cadence_stream[i])

            # Averages
            avg_gradient = sum(lap_elevations) / len(lap_elevations) if lap_elevations else 0
            
            # Double cadence (Garmin 1-foot to 2-foot)
            if lap_cadences:
                avg_lap_cadence = (sum(lap_cadences) / len(lap_cadences)) * 2
            else:
                avg_lap_cadence = 0

            # --- THE NEW CLASSIFICATION LOGIC ---
            # We classify this specific mile/lap based on the same logic as the graph
            lap_hr = lap.get('avg_hr', 0)
            split_verdict = classify_split(avg_lap_cadence, lap_hr, max_hr, avg_gradient * 100)
            # ------------------------------------

            # Minetti / GAP Logic
            flat_cost = 3.6
            terrain_cost = minetti_cost_of_running(avg_gradient)
            cost_multiplier = terrain_cost / flat_cost

            if avg_speed and avg_speed > 0:
                gap_speed = avg_speed * cost_multiplier
                gap_pace_min = 26.8224 / gap_speed
                gap_pace_str = f"{int(gap_pace_min)}:{int((gap_pace_min % 1) * 60):02d}"

                actual_pace_min = 26.8224 / avg_speed
                actual_pace_str = f"{int(actual_pace_min)}:{int((actual_pace_min % 1) * 60):02d}"
                
                pace_diff_seconds = abs(gap_pace_min - actual_pace_min) * 60
                is_steep = pace_diff_seconds > 15
            else:
                gap_pace_str = "--:--"
                actual_pace_str = "--:--"
                is_steep = False

            enhanced_laps.append({
                **lap,
                'gap_pace': gap_pace_str,
                'actual_pace': actual_pace_str,
                'is_steep': is_steep,
                'avg_gradient': avg_gradient,
                'avg_cadence': avg_lap_cadence,
                'split_verdict': split_verdict # <--- STORED HERE
            })

        return enhanced_laps
    
    def copy_splits_to_clipboard(self, lap_data):
        """
        Copy lap splits data to clipboard in CSV format.
        
        Args:
            lap_data: List of lap dictionaries with metrics
        """
        try:
            # Build CSV string
            csv_lines = ['Lap,Distance,Pace,Cadence,GAP,HR,Elev']
            
            for lap in lap_data:
                lap_num = lap.get('lap_number', 0)
                distance_mi = lap.get('distance', 0) * 0.000621371
                pace = lap.get('actual_pace', '--:--')
                cadence = int(lap['avg_cadence']) if lap.get('avg_cadence') else '--'
                gap = lap.get('gap_pace', '--:--')
                hr = int(lap['avg_hr']) if lap.get('avg_hr') else '--'
                
                # Calculate net elevation change (ascent - descent)
                ascent = lap.get('total_ascent', 0) or 0
                descent = lap.get('total_descent', 0) or 0
                elev_change_m = ascent - descent
                elev_change_ft = elev_change_m * 3.28084
                
                # Format with explicit sign
                if elev_change_ft > 0:
                    elev_str = f"+{int(elev_change_ft)}"
                elif elev_change_ft < 0:
                    elev_str = f"{int(elev_change_ft)}"
                else:
                    elev_str = "0"
                
                csv_lines.append(f"{lap_num},{distance_mi:.2f},{pace},{cadence},{gap},{hr},{elev_str}")
            
            csv_string = '\n'.join(csv_lines)
            
            # Use pyperclip for reliable clipboard access
            try:
                import pyperclip
                pyperclip.copy(csv_string)
                ui.notify(
                    'Splits copied to clipboard!',
                    color='#18181b',        # Zinc-900 (Dark background to match the theme)
                    text_color='white',     # White text for readability
                    icon='check_circle',    # A distinct icon to confirm success
                    icon_color='green',     # The ONLY green element (subtle accent)
                    timeout=2000,           # Quick fade (2 seconds)
                    position='bottom'       # Bottom center (in user's focus area)
                )
            except ImportError:
                ui.notify('Please install pyperclip: pip3 install pyperclip', type='negative')
            
        except Exception as e:
            ui.notify(f'Error copying splits: {str(e)}', type='negative')
            print(f"Error in copy_splits_to_clipboard: {e}")
    
    def calculate_aerobic_decoupling(self, hr_stream, speed_stream):
        """
        Calculate aerobic decoupling (Pa:HR) by comparing first and second half efficiency.

        Args:
            hr_stream: List of heart rate values
            speed_stream: List of speed values (m/s)

        Returns:
            Dictionary containing:
            - decoupling_pct: float - percentage decoupling
            - ef_first_half: float - efficiency factor for first half
            - ef_second_half: float - efficiency factor for second half
            - status: str - 'Solid', 'Drift Detected', or 'Significant Drift'
            - color: str - hex color code for display
        """
        # Filter out None values and create paired data
        # Use speed > 1.0 to match analyzer.py's df_active filter
        paired_data = [
            (hr, speed)
            for hr, speed in zip(hr_stream, speed_stream)
            if hr is not None and speed is not None and speed > 1.0
        ]

        if len(paired_data) < 60:  # Need at least 1 minute of data
            return {
                'decoupling_pct': 0,
                'ef_first_half': 0,
                'ef_second_half': 0,
                'status': 'Insufficient Data',
                'color': '#888888'
            }

        # Split into halves
        mid_point = len(paired_data) // 2
        first_half = paired_data[:mid_point]
        second_half = paired_data[mid_point:]

        # Calculate efficiency factor for each half (speed / HR)
        ef_first = sum(speed / hr for hr, speed in first_half) / len(first_half)
        ef_second = sum(speed / hr for hr, speed in second_half) / len(second_half)

        # Calculate decoupling percentage
        # Formula: ((EF_first - EF_second) / EF_first) * 100
        # Positive value means efficiency decreased (drift occurred)
        if ef_first > 0:
            decoupling_pct = ((ef_first - ef_second) / ef_first) * 100
        else:
            decoupling_pct = 0

        # Determine status and color
        if decoupling_pct < 5:
            status = 'Solid'
            color = '#10B981'  # Green
        elif decoupling_pct <= 10:
            status = 'Drift Detected'
            color = '#ff9900'  # Orange
        else:
            status = 'Significant Drift'
            color = '#ff4d4d'  # Red

        return {
            'decoupling_pct': round(decoupling_pct, 2),
            'ef_first_half': round(ef_first, 4),
            'ef_second_half': round(ef_second, 4),
            'status': status,
            'color': color
        }
    
    def calculate_run_walk_stats(self, cadence_stream, speed_stream, hr_stream):
        """
        Calculate run/walk/stop statistics for ultra strategy analysis.
        
        Args:
            cadence_stream: List of cadence values (steps per minute)
            speed_stream: List of speed values (m/s)
            hr_stream: List of heart rate values
            
        Returns:
            Dictionary containing:
            - run_pct: float - percentage of time running
            - hike_pct: float - percentage of time hiking
            - stop_pct: float - percentage of time stopped
            - avg_run_pace: str - average running pace (min/mi)
            - avg_hike_pace: str - average hiking pace (min/mi)
            - avg_run_hr: int - average heart rate while running
            - avg_hike_hr: int - average heart rate while hiking
        """
        import numpy as np
        import pandas as pd
        
        # Convert to pandas Series for easier handling
        cadence_series = pd.Series(cadence_stream)
        speed_series = pd.Series(speed_stream)
        hr_series = pd.Series(hr_stream)
        
        # Step A: Handle Missing Data - Forward fill to handle smart recording gaps
        cadence_series = cadence_series.replace(0, np.nan).ffill()
        speed_series = speed_series.replace(0, np.nan).ffill()
        hr_series = hr_series.ffill()
        
        # Drop any remaining NaN values
        valid_mask = cadence_series.notna() & speed_series.notna() & hr_series.notna()
        cadence_clean = cadence_series[valid_mask].values
        speed_clean = speed_series[valid_mask].values
        hr_clean = hr_series[valid_mask].values
        
        if len(cadence_clean) < 60:  # Need at least 1 minute
            return {
                'run_pct': 0,
                'hike_pct': 0,
                'stop_pct': 0,
                'avg_run_pace': '--:--',
                'avg_hike_pace': '--:--',
                'avg_run_hr': 0,
                'avg_hike_hr': 0
            }
        
        # Step B: Auto-Detect Units (RPM vs SPM)
        # If 95th percentile is < 110, assume single-sided (RPM) and multiply by 2
        cadence_95th = np.percentile(cadence_clean, 95)
        if cadence_95th < 110:
            cadence_clean = cadence_clean * 2
        
        # Step C: Classify with refined thresholds
        stopped_count = 0
        hiking_count = 0
        running_count = 0
        
        run_speeds = []
        run_hrs = []
        hike_speeds = []
        hike_hrs = []
        
        for i in range(len(cadence_clean)):
            speed = speed_clean[i]
            cadence = cadence_clean[i]
            hr = hr_clean[i]
            
            if speed < 0.5:  # Idle (standing still or very slow walk)
                stopped_count += 1
            elif cadence < 135:  # Hiking (low cadence but moving)
                hiking_count += 1
                hike_speeds.append(speed)
                hike_hrs.append(hr)
            else:  # Running (cadence >= 135)
                running_count += 1
                run_speeds.append(speed)
                run_hrs.append(hr)
        
        total = len(cadence_clean)
        
        # Calculate percentages
        run_pct = (running_count / total) * 100 if total > 0 else 0
        hike_pct = (hiking_count / total) * 100 if total > 0 else 0
        stop_pct = (stopped_count / total) * 100 if total > 0 else 0
        
        # Calculate average paces
        if run_speeds:
            avg_run_speed = np.mean(run_speeds)
            avg_run_pace_min = 26.8224 / avg_run_speed  # Convert to min/mile
            avg_run_pace = f"{int(avg_run_pace_min)}:{int((avg_run_pace_min % 1) * 60):02d}"
            avg_run_hr = int(np.mean(run_hrs))
        else:
            avg_run_pace = '--:--'
            avg_run_hr = 0
        
        if hike_speeds:
            avg_hike_speed = np.mean(hike_speeds)
            avg_hike_pace_min = 26.8224 / avg_hike_speed
            avg_hike_pace = f"{int(avg_hike_pace_min)}:{int((avg_hike_pace_min % 1) * 60):02d}"
            avg_hike_hr = int(np.mean(hike_hrs))
        else:
            avg_hike_pace = '--:--'
            avg_hike_hr = 0
        
        return {
            'run_pct': round(run_pct, 1),
            'hike_pct': round(hike_pct, 1),
            'stop_pct': round(stop_pct, 1),
            'avg_run_pace': avg_run_pace,
            'avg_hike_pace': avg_hike_pace,
            'avg_run_hr': avg_run_hr,
            'avg_hike_hr': avg_hike_hr
        }
    
    def calculate_terrain_stats(self, elevation_stream, hr_stream, speed_stream, timestamps):
        """
        Calculate terrain efficiency statistics (uphill/flat/downhill).
        
        Args:
            elevation_stream: List of elevation values (meters)
            hr_stream: List of heart rate values
            speed_stream: List of speed values (m/s)
            timestamps: List of timestamps
            
        Returns:
            Dictionary containing stats for uphill, flat, and downhill:
            - time_pct: percentage of time
            - avg_hr: average heart rate
            - avg_pace: average pace (min/mi)
        """
        import numpy as np
        
        # Filter and align data
        valid_data = []
        for i in range(1, len(elevation_stream)):
            if (elevation_stream[i] is not None and 
                elevation_stream[i-1] is not None and
                hr_stream[i] is not None and 
                speed_stream[i] is not None and
                speed_stream[i] > 0.2):  # Filter out stopped periods
                
                # Calculate grade
                elev_diff = elevation_stream[i] - elevation_stream[i-1]
                speed = speed_stream[i]
                dist_diff = speed  # Approximate distance (1 second * speed)
                
                if dist_diff > 0:
                    grade = (elev_diff / dist_diff) * 100  # Convert to percentage
                    valid_data.append({
                        'grade': grade,
                        'hr': hr_stream[i],
                        'speed': speed
                    })
        
        if len(valid_data) < 60:  # Need at least 1 minute
            return {
                'uphill': {'time_pct': 0, 'avg_hr': 0, 'avg_pace': '--:--'},
                'flat': {'time_pct': 0, 'avg_hr': 0, 'avg_pace': '--:--'},
                'downhill': {'time_pct': 0, 'avg_hr': 0, 'avg_pace': '--:--'}
            }
        
        # Apply 10-second rolling average to smooth GPS noise
        grades = np.array([d['grade'] for d in valid_data])
        window_size = min(10, len(grades))
        smoothed_grades = np.convolve(grades, np.ones(window_size)/window_size, mode='same')
        
        # Update grades with smoothed values
        for i, d in enumerate(valid_data):
            d['grade'] = smoothed_grades[i]
        
        # Bucket data
        uphill_data = [d for d in valid_data if d['grade'] > 3]
        flat_data = [d for d in valid_data if -3 <= d['grade'] <= 3]
        downhill_data = [d for d in valid_data if d['grade'] < -3]
        
        total = len(valid_data)
        
        def calc_stats(data):
            if not data:
                return {'time_pct': 0, 'avg_hr': 0, 'avg_pace': '--:--'}
            
            time_pct = (len(data) / total) * 100
            avg_hr = int(np.mean([d['hr'] for d in data]))
            avg_speed = np.mean([d['speed'] for d in data])
            avg_pace_min = 26.8224 / avg_speed
            avg_pace = f"{int(avg_pace_min)}:{int((avg_pace_min % 1) * 60):02d}"
            
            return {
                'time_pct': round(time_pct, 1),
                'avg_hr': avg_hr,
                'avg_pace': avg_pace
            }
        
        return {
            'uphill': calc_stats(uphill_data),
            'flat': calc_stats(flat_data),
            'downhill': calc_stats(downhill_data)
        }
    
    def create_hr_zone_chart(self, zone_times):
        """
        Create horizontal bar chart showing time in each HR zone.

        Args:
            zone_times: Dictionary with zone names and time in minutes

        Returns:
            Plotly Figure object
        """
        import plotly.graph_objects as go

        # Define color palette (emerald green gradient)
        colors = [
            '#064e3b',  # Zone 1 - Dark green
            '#047857',  # Zone 2 - Medium green
            '#10B981',  # Zone 3 - Bright emerald
            '#ff9900',  # Zone 4 - Orange
            '#ff4d4d'   # Zone 5 - Red
        ]

        # Create horizontal bar chart
        fig = go.Figure(data=[
            go.Bar(
                y=list(zone_times.keys()),
                x=list(zone_times.values()),
                orientation='h',
                marker=dict(color=colors),
                text=[f"{t:.1f} min" for t in zone_times.values()],
                textposition='auto',
                hoverinfo='none'  # Disable hover tooltips
            )
        ])

        # Update layout
        fig.update_layout(
            title='Time in Heart Rate Zones',
            xaxis_title='Time (minutes)',
            yaxis_title='',
            template='plotly_dark',
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            height=300,
            margin=dict(l=20, r=20, t=40, b=20),
            showlegend=False
        )
        
        # Completely hide the modebar
        fig.update_layout(modebar={'remove': ['zoom', 'pan', 'select', 'lasso2d', 'zoomIn', 'zoomOut', 'autoScale', 'resetScale', 'toImage']})

        return fig
    
    def create_cadence_elevation_chart(self, distance_stream, cadence_stream, elevation_stream, use_miles=True):
        """
        Create dual-axis chart showing cadence vs elevation over distance.
        
        Args:
            distance_stream: List[float] - distance in meters
            cadence_stream: List[int] - cadence in steps per minute
            elevation_stream: List[float] - elevation in meters
            use_miles: bool - convert distance to miles (default True)
        
        Returns:
            plotly.graph_objects.Figure - configured chart
        """
        from plotly.subplots import make_subplots
        import plotly.graph_objects as go
        
        # Filter out None values while maintaining alignment
        valid_indices = [
            i for i in range(min(len(distance_stream), len(cadence_stream), len(elevation_stream)))
            if distance_stream[i] is not None and cadence_stream[i] is not None
        ]
        
        if not valid_indices:
            # Return empty figure if no valid data
            fig = go.Figure()
            fig.update_layout(
                title='No valid data available',
                template='plotly_dark',
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(0,0,0,0)'
            )
            return fig
        
        # Extract valid data
        distance_valid = [distance_stream[i] for i in valid_indices]
        # Garmin records cadence as steps per minute for ONE foot, so double it for total SPM
        cadence_valid = [cadence_stream[i] * 2 for i in valid_indices]
        elevation_valid = [elevation_stream[i] if elevation_stream[i] is not None else 0 for i in valid_indices]
        
        # Smooth cadence data with rolling average to reduce noise
        def rolling_average(data, window=30):
            """Apply rolling average to smooth noisy data."""
            smoothed = []
            for i in range(len(data)):
                start = max(0, i - window // 2)
                end = min(len(data), i + window // 2 + 1)
                smoothed.append(sum(data[start:end]) / (end - start))
            return smoothed
        
        cadence_smoothed = rolling_average(cadence_valid, window=30)
        
        # Convert units
        if use_miles:
            distance_converted = [d / 1609.34 for d in distance_valid]
            elevation_converted = [e * 3.28084 for e in elevation_valid]
            distance_unit = 'mi'
            elevation_unit = 'ft'
        else:
            distance_converted = [d / 1000 for d in distance_valid]
            elevation_converted = elevation_valid
            distance_unit = 'km'
            elevation_unit = 'm'
        
        # Create figure with secondary y-axis
        fig = make_subplots(specs=[[{"secondary_y": True}]])
        
        # Add elevation trace (background, filled area)
        fig.add_trace(
            go.Scatter(
                x=distance_converted,
                y=elevation_converted,
                name="Elevation",
                fill='tozeroy',
                fillcolor='rgba(148, 163, 184, 0.2)',
                line=dict(color='rgba(148, 163, 184, 0.4)', width=1),
                hovertemplate=f'%{{y:.0f}} {elevation_unit}<extra></extra>'
            ),
            secondary_y=True
        )
        
        # Add cadence trace (foreground, bold line)
        fig.add_trace(
            go.Scatter(
                x=distance_converted,
                y=cadence_smoothed,
                name="Cadence",
                line=dict(color='#6366f1', width=3, shape='spline'),
                hovertemplate='%{y:.0f} SPM<extra></extra>'
            ),
            secondary_y=False
        )
        
        # Configure layout
        fig.update_layout(
            title='Cadence & Elevation Profile',
            xaxis_title=f'Distance ({distance_unit})',
            template='plotly_dark',
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            height=300,
            margin=dict(l=20, r=40, t=40, b=40),
            showlegend=True,
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="right",
                x=1
            ),
            hovermode='x unified',
            # Style hover tooltip for dark theme
            hoverlabel=dict(
                bgcolor='#18181b',  # zinc-900
                font_size=13,
                font_family='monospace',
                font_color='white'
            )
        )
        
        # Configure Y-axes
        fig.update_yaxes(
            title_text="Cadence (SPM)",
            secondary_y=False,
            range=[140, 200]  # Fixed range for running cadence
        )
        
        fig.update_yaxes(
            title_text=f"Elevation ({elevation_unit})",
            secondary_y=True,
            rangemode='tozero'
        )
        
        # Hide modebar
        fig.update_layout(
            modebar={'remove': ['zoom', 'pan', 'select', 'lasso2d', 'zoomIn', 'zoomOut', 'autoScale', 'resetScale', 'toImage']}
        )
        
        return fig
    
    def create_lap_splits_table(self, lap_data):
        """
        Create table displaying lap splits with GAP and QUALITY VERDICT.
        UPDATED: Adds Units (spm, bpm) + "Guilty Metric" Highlighting.
        """
        # Define columns
        columns = [
            {'name': 'lap', 'label': '#', 'field': 'lap', 'align': 'center'},
            {'name': 'distance', 'label': 'Dist', 'field': 'distance', 'align': 'left'},
            {'name': 'quality', 'label': 'Quality', 'field': 'quality', 'align': 'left'},
            {'name': 'pace', 'label': 'Pace', 'field': 'pace', 'align': 'left'},
            {'name': 'cadence', 'label': 'Cad', 'field': 'cadence', 'align': 'center'},
            {'name': 'gap', 'label': 'GAP', 'field': 'gap', 'align': 'left'},
            {'name': 'hr', 'label': 'HR', 'field': 'hr', 'align': 'center'},
            {'name': 'elev', 'label': 'Elev', 'field': 'elev', 'align': 'left'}
        ]

        # Transform lap data into rows
        rows = []
        for lap in lap_data:
            distance_mi = lap.get('distance', 0) * 0.000621371
            
            # 1. Prepare Elevation Data
            ascent = lap.get('total_ascent', 0) or 0
            descent = lap.get('total_descent', 0) or 0
            elev_change_ft = (ascent - descent) * 3.28084
            
            if elev_change_ft > 0: elev_str = f"+{int(elev_change_ft)} ft"
            else: elev_str = f"{int(elev_change_ft)} ft"

            # 2. Prepare Formatted Strings with Units
            cad_val = int(lap['avg_cadence']) if lap.get('avg_cadence') else 0
            cad_str = f"{cad_val} spm" if cad_val > 0 else '--'
            
            hr_val = int(lap['avg_hr']) if lap.get('avg_hr') else 0
            hr_str = f"{hr_val} bpm" if hr_val > 0 else '--'
            
            # 3. Determine "The Why" (Highlight Logic)
            verdict = lap.get('split_verdict', 'STRUCTURAL')
            
            # Default styles (neutral grey)
            cad_class = 'text-zinc-400'
            elev_class = 'text-zinc-400'
            
            # Logic: Color the metric that determined the verdict
            if verdict == 'HIGH QUALITY':
                # Green Cadence = Good Mechanics
                cad_class = 'text-emerald-400 font-bold'
                
            elif verdict == 'BROKEN':
                # Red Cadence = Mechanics Failed
                cad_class = 'text-red-400 font-bold'
                
            elif verdict == 'STRUCTURAL':
                # Blue Elevation = It was steep (Grade > 8%)
                # Blue Cadence = It was a hike/walk
                dist_m = lap.get('distance', 0)
                grade = (ascent / dist_m) * 100 if dist_m > 0 else 0
                
                if grade > 8:
                    elev_class = 'text-blue-400 font-bold' # Highlight Vert
                else:
                    cad_class = 'text-blue-400 font-bold' # Highlight Hiking Cadence
            
            rows.append({
                'lap': lap.get('lap_number', 0),
                'distance': f"{distance_mi:.2f}",
                'quality': verdict,
                'pace': f"{lap.get('actual_pace', '--:--')} /mi", # Added unit
                'cadence': cad_str,
                'cad_class': cad_class, # <--- Pass the color class
                'gap': f"{lap.get('gap_pace', '--:--')} /mi", # Added unit
                'gap_highlight': lap.get('is_steep', False),
                'hr': hr_str,
                'elev': elev_str,
                'elev_class': elev_class # <--- Pass the color class
            })

        # Create table
        table = ui.table(columns=columns, rows=rows, row_key='lap').classes('w-full')

        # SLOT: Quality (Dot + Text)
        table.add_slot('body-cell-quality', '''
            <q-td :props="props">
                <div class="flex items-center gap-2">
                    <div v-if="props.value === 'HIGH QUALITY'" class="w-2 h-2 rounded-full bg-emerald-400 shadow-[0_0_8px_rgba(52,211,153,0.6)]"></div>
                    <div v-if="props.value === 'STRUCTURAL'" class="w-2 h-2 rounded-full bg-blue-400"></div>
                    <div v-if="props.value === 'BROKEN'" class="w-2 h-2 rounded-full bg-red-400 shadow-[0_0_8px_rgba(248,113,113,0.6)]"></div>
                    
                    <span class="text-xs font-bold tracking-wide"
                        :class="{
                            'text-emerald-400': props.value === 'HIGH QUALITY',
                            'text-blue-400': props.value === 'STRUCTURAL',
                            'text-red-400': props.value === 'BROKEN'
                        }">
                        {{ props.value === 'HIGH QUALITY' ? 'Quality' : props.value === 'STRUCTURAL' ? 'Base' : 'Broken' }}
                    </span>
                </div>
            </q-td>
        ''')

        # SLOT: Cadence (Dynamic Highlight)
        table.add_slot('body-cell-cadence', '''
            <q-td :props="props">
                <span :class="props.row.cad_class">{{ props.value }}</span>
            </q-td>
        ''')

        # SLOT: Elevation (Dynamic Highlight)
        table.add_slot('body-cell-elev', '''
            <q-td :props="props">
                <span :class="props.row.elev_class">{{ props.value }}</span>
            </q-td>
        ''')

        # SLOT: GAP (Steep Highlight)
        table.add_slot('body-cell-gap', '''
            <q-td :props="props">
                <span :style="props.row.gap_highlight ? 'color: #10B981; font-weight: 600;' : 'color: inherit;'">
                    {{ props.value }}
                </span>
            </q-td>
        ''')

        table.props('flat bordered dense dark')
        table.classes('bg-zinc-900 text-gray-200')
        return table

    def create_decoupling_card(self, decoupling_data):
        """
        Create card displaying aerobic decoupling metrics.

        Args:
            decoupling_data: Dictionary with decoupling metrics

        Returns:
            NiceGUI card component
        """
        with ui.card().classes('bg-zinc-900 p-4 border border-zinc-800') as card:
            card.style('border-radius: 8px; box-shadow: 0px 4px 12px rgba(0, 0, 0, 0.4);')

            # Title
            with ui.row().classes('items-baseline gap-2'):
                ui.label('AEROBIC DECOUPLING').classes('text-lg font-bold text-white')

            # Main metric with color
            with ui.row().classes('items-center gap-2'):
                ui.label(f"{decoupling_data['decoupling_pct']:.1f}%").classes('text-3xl font-bold').style(
                    f"color: {decoupling_data['color']};"
                )
                ui.label(decoupling_data['status']).classes('text-lg').style(
                    f"color: {decoupling_data['color']};"
                )

            # Efficiency details
            with ui.column().classes('mt-3 gap-1'):
                ui.label(f"1st Half Efficiency: {decoupling_data['ef_first_half']:.4f}").classes('text-sm text-gray-400')
                ui.label(f"2nd Half Efficiency: {decoupling_data['ef_second_half']:.4f}").classes('text-sm text-gray-400')

        return card
    
    def create_physiology_card(self, session_data, activity):
        """
        Create card displaying physiology metrics from session data.
        
        Args:
            session_data: Dictionary with session-level metrics
            activity: Activity dictionary with hrr_list
            
        Returns:
            NiceGUI card component
        """
        with ui.card().classes('bg-zinc-900 p-4 border border-zinc-800 h-full') as card:
            card.style('border-radius: 8px; box-shadow: 0px 4px 12px rgba(0, 0, 0, 0.4);')
            
            # Title
            ui.label('PHYSIOLOGY').classes('text-lg font-bold text-white mb-3')
            
            # Training Effect
            aerobic_te = session_data.get('total_training_effect')
            anaerobic_te = session_data.get('total_anaerobic_training_effect')
            
            with ui.column().classes('gap-3'):
                if aerobic_te is not None or anaerobic_te is not None:
                    with ui.column().classes('gap-1'):
                        ui.label('TRAINING EFFECT').classes('text-xs text-zinc-500 uppercase tracking-wider font-semibold')
                        te_str = f"{aerobic_te:.1f} Aerobic" if aerobic_te else "-- Aerobic"
                        if anaerobic_te:
                            te_str += f" / {anaerobic_te:.1f} Anaerobic"
                        ui.label(te_str).classes('text-sm text-white font-mono')
                
                # Respiration Rate
                resp_rate = session_data.get('avg_respiration_rate')
                if resp_rate:
                    with ui.column().classes('gap-1'):
                        ui.label('AVG BREATH').classes('text-xs text-zinc-500 uppercase tracking-wider font-semibold')
                        ui.label(f"{int(resp_rate)} brpm").classes('text-sm text-white font-mono')
                
                # HR Recovery (with info icon)
                hrr_list = activity.get('hrr_list')
                if hrr_list:
                    try:
                        score = hrr_list[0] if isinstance(hrr_list, list) else int(hrr_list)
                    except:
                        score = 0
                    
                    if score > 0:
                        # Determine color based on HRR score
                        if score > 30:
                            hrr_color = '#10B981'  # Green
                        elif score >= 20:
                            hrr_color = '#fbbf24'  # Yellow
                        else:
                            hrr_color = '#ff4d4d'  # Red
                        
                        with ui.column().classes('gap-1'):
                            with ui.row().classes('items-center gap-1'):
                                ui.label('HR RECOVERY (1-MIN)').classes('text-xs text-zinc-500 uppercase tracking-wider font-semibold')
                                ui.icon('help_outline').classes('text-zinc-600 hover:text-white text-xs cursor-pointer').on('click', lambda: self.show_hrr_info())
                            with ui.row().classes('items-center gap-2'):
                                ui.icon('favorite').classes('text-sm').style('color: #ff4d4d;')
                                ui.label(f"{score} bpm").classes('text-sm font-bold font-mono').style(f'color: {hrr_color};')
        
        return card
    
    def create_running_dynamics_card(self, session_data):
        """
        Create "Pro-Level" mechanics card with strict multi-factor diagnosis.
        
        Args:
            session_data: Dictionary with session-level metrics
            
        Returns:
            NiceGUI card component
        """
        # --- 1. PREPARE METRICS ---
        cadence = session_data.get('avg_cadence', 0)
        gct = session_data.get('avg_stance_time', 0)
        stride = session_data.get('avg_step_length', 0)
        v_osc = session_data.get('avg_vertical_oscillation', 0)
        
        # Convert units
        v_osc_cm = v_osc / 10 if v_osc else 0  # mm to cm
        stride_m = stride / 1000 if stride else 0  # mm to m
        bounce = v_osc_cm  # Use cm for display
        
        # --- 2. GET DIAGNOSIS (Centralized Logic) ---
        diagnosis = analyze_form(cadence, gct, stride, v_osc)
        
        # --- 3. TRAFFIC LIGHT COLORING (The Evidence) ---
        def get_metric_color(val, metric_type):
            if not val:
                return 'text-zinc-500'
            val = float(val)
            
            if metric_type == 'cadence':
                return 'text-emerald-400' if val > 170 else 'text-blue-400' if val > 160 else 'text-orange-400'
            elif metric_type == 'bounce':
                return 'text-emerald-400' if val < 8.2 else 'text-blue-400' if val < 10.0 else 'text-orange-400'
            elif metric_type == 'gct':
                return 'text-emerald-400' if val < 250 else 'text-blue-400' if val < 270 else 'text-orange-400'
            return 'text-zinc-300'
        
        # --- 4. RENDER THE CARD ---
        with ui.card().classes('bg-zinc-900 p-4 border border-zinc-800 h-full') as card:
            card.style('border-radius: 8px; box-shadow: 0px 4px 12px rgba(0, 0, 0, 0.4);')
            
            # Header
            with ui.row().classes('items-center gap-2 mb-2'):
                ui.icon('directions_run').classes('text-zinc-500 text-sm')
                ui.label('RUNNING MECHANICS').classes('text-xs font-bold text-zinc-500 tracking-wider')
            
            # Hero Section (Verdict + Prescription)
            with ui.column().classes('w-full items-center py-2 gap-1'):
                with ui.row().classes('items-center gap-2'):
                    ui.icon(diagnosis['icon']).classes(f"text-2xl {diagnosis['color']}")
                    ui.label(diagnosis['verdict']).classes(f"text-xl font-black {diagnosis['color']} tracking-tight")
                ui.label(diagnosis['prescription']).classes('text-xs text-zinc-400 italic text-center px-4')
            
            ui.separator().classes('bg-zinc-800 my-3')
            
            # Data Grid (4 Columns with Traffic Lights)
            with ui.row().classes('w-full justify-between text-center'):
                # Cadence
                with ui.column().classes('gap-0'):
                    ui.label('CADENCE').classes('text-[9px] font-bold text-zinc-600')
                    ui.label(f"{int(cadence) if cadence else '--'}").classes(f"text-sm font-bold font-mono {get_metric_color(cadence, 'cadence')}")
                    ui.label('spm').classes('text-[9px] text-zinc-600')
                
                # Bounce
                with ui.column().classes('gap-0'):
                    ui.label('BOUNCE').classes('text-[9px] font-bold text-zinc-600')
                    ui.label(f"{bounce:.1f}" if bounce else '--').classes(f"text-sm font-bold font-mono {get_metric_color(bounce, 'bounce')}")
                    ui.label('cm').classes('text-[9px] text-zinc-600')
                
                # Contact
                with ui.column().classes('gap-0'):
                    ui.label('CONTACT').classes('text-[9px] font-bold text-zinc-600')
                    ui.label(f"{int(gct) if gct else '--'}").classes(f"text-sm font-bold font-mono {get_metric_color(gct, 'gct')}")
                    ui.label('ms').classes('text-[9px] text-zinc-600')
                
                # Stride
                with ui.column().classes('gap-0'):
                    ui.label('STRIDE').classes('text-[9px] font-bold text-zinc-600')
                    ui.label(f"{stride_m:.2f}" if stride_m else '--').classes('text-sm font-bold font-mono text-zinc-400')
                    ui.label('m').classes('text-[9px] text-zinc-600')
        
        return card
    
    def create_strategy_row(self, run_walk_stats, terrain_stats):
        """
        Create the Ultra Strategy analysis row with run/walk and terrain cards.
        
        Args:
            run_walk_stats: Dictionary with run/walk statistics (or None if no cadence data)
            terrain_stats: Dictionary with terrain statistics
            
        Returns:
            NiceGUI grid component with strategy cards
        """
        with ui.grid(columns=2).classes('w-full gap-4'):
            # Left Card: Run / Walk Breakdown (Stat Blocks)
            if run_walk_stats:
                with ui.card().classes('bg-zinc-900 p-4 border border-zinc-800').style('border-radius: 8px; box-shadow: 0px 4px 12px rgba(0, 0, 0, 0.4);'):
                    ui.label('Run / Walk Breakdown').classes('text-lg font-bold text-white mb-4')
                    
                    run_pct = run_walk_stats['run_pct']
                    hike_pct = run_walk_stats['hike_pct']
                    stop_pct = run_walk_stats['stop_pct']
                    
                    # Stat Blocks (2 columns)
                    with ui.row().classes('w-full gap-4 mb-3'):
                        # Column 1: Running
                        if run_pct >= 1.0:
                            with ui.column().classes('flex-1 items-center'):
                                ui.label('RUNNING').classes('text-xs text-emerald-500 font-semibold tracking-wider mb-1')
                                ui.label(f"{run_pct:.0f}%").classes('text-3xl font-bold text-white mb-1')
                                ui.label(f"{run_walk_stats['avg_run_pace']}/mi • {run_walk_stats['avg_run_hr']} bpm").classes('text-sm text-zinc-400')
                        
                        # Column 2: Hiking
                        if hike_pct >= 1.0:
                            with ui.column().classes('flex-1 items-center'):
                                ui.label('HIKING').classes('text-xs text-blue-400 font-semibold tracking-wider mb-1')
                                ui.label(f"{hike_pct:.0f}%").classes('text-3xl font-bold text-white mb-1')
                                ui.label(f"{run_walk_stats['avg_hike_pace']}/mi • {run_walk_stats['avg_hike_hr']} bpm").classes('text-sm text-zinc-400')
                    
                    # Thin progress bar at bottom
                    with ui.element('div').classes('w-full').style('height: 6px; display: flex; border-radius: 3px; overflow: hidden; background-color: #27272a;'):
                        if run_pct >= 1.0:
                            ui.element('div').style(f'width: {run_pct}%; background-color: #10B981;')
                        if hike_pct >= 1.0:
                            ui.element('div').style(f'width: {hike_pct}%; background-color: #3B82F6;')
                        if stop_pct >= 1.0:
                            ui.element('div').style(f'width: {stop_pct}%; background-color: #6B7280;')
            
            # Right Card: Terrain Analysis (Label-Value Stacking)
            with ui.card().classes('bg-zinc-900 p-4').style('border-radius: 8px; box-shadow: 0px 4px 12px rgba(0, 0, 0, 0.4);'):
                ui.label('Terrain Analysis').classes('text-lg font-bold text-white mb-4')
                
                uphill = terrain_stats['uphill']
                flat = terrain_stats['flat']
                downhill = terrain_stats['downhill']
                
                # Check if uphill HR is significantly higher (burning matches)
                uphill_warning = False
                if uphill['avg_hr'] > 0 and flat['avg_hr'] > 0:
                    hr_diff_pct = ((uphill['avg_hr'] - flat['avg_hr']) / flat['avg_hr']) * 100
                    uphill_warning = hr_diff_pct > 10
                
                with ui.column().classes('gap-3 w-full'):
                    # Uphill row
                    if uphill['time_pct'] > 0:
                        with ui.row().classes('items-end gap-3 w-full'):
                            # Terrain name (Green for gains)
                            with ui.row().classes('items-center gap-1 w-24'):
                                ui.label('↗️').classes('text-base')
                                ui.label('UPHILL').classes('text-xs text-emerald-400 font-bold tracking-wide')
                            
                            # Stats grid
                            with ui.row().classes('flex-1 gap-4'):
                                # Time % (color-coded green)
                                with ui.column().classes('gap-0'):
                                    ui.label('TIME').classes('text-[10px] text-zinc-500 uppercase tracking-wider font-semibold')
                                    ui.label(f"{uphill['time_pct']:.0f}%").classes('text-sm text-emerald-400 font-bold font-mono')
                                
                                # HR
                                with ui.column().classes('gap-0'):
                                    ui.label('HR').classes('text-[10px] text-zinc-500 uppercase tracking-wider font-semibold')
                                    hr_label = ui.label(f"{uphill['avg_hr']}").classes('text-sm font-bold font-mono')
                                    if uphill_warning:
                                        hr_label.classes('text-orange-500')
                                    else:
                                        hr_label.classes('text-white')
                                
                                # Pace
                                with ui.column().classes('gap-0'):
                                    ui.label('PACE').classes('text-[10px] text-zinc-500 uppercase tracking-wider font-semibold')
                                    ui.label(f"{uphill['avg_pace']}").classes('text-sm text-white font-bold font-mono')
                    
                    # Flat row
                    if flat['time_pct'] > 0:
                        with ui.row().classes('items-end gap-3 w-full'):
                            # Terrain name (Grey for neutral)
                            with ui.row().classes('items-center gap-1 w-24'):
                                ui.label('➡️').classes('text-base')
                                ui.label('FLAT').classes('text-xs text-zinc-500 font-medium tracking-wide')
                            
                            # Stats grid
                            with ui.row().classes('flex-1 gap-4'):
                                # Time % (color-coded grey)
                                with ui.column().classes('gap-0'):
                                    ui.label('TIME').classes('text-[10px] text-zinc-500 uppercase tracking-wider font-semibold')
                                    ui.label(f"{flat['time_pct']:.0f}%").classes('text-sm text-zinc-500 font-bold font-mono')
                                
                                # HR
                                with ui.column().classes('gap-0'):
                                    ui.label('HR').classes('text-[10px] text-zinc-500 uppercase tracking-wider font-semibold')
                                    ui.label(f"{flat['avg_hr']}").classes('text-sm text-white font-bold font-mono')
                                
                                # Pace
                                with ui.column().classes('gap-0'):
                                    ui.label('PACE').classes('text-[10px] text-zinc-500 uppercase tracking-wider font-semibold')
                                    ui.label(f"{flat['avg_pace']}").classes('text-sm text-white font-bold font-mono')
                    
                    # Downhill row
                    if downhill['time_pct'] > 0:
                        with ui.row().classes('items-end gap-3 w-full'):
                            # Terrain name (Cyan for flow)
                            with ui.row().classes('items-center gap-1 w-24'):
                                ui.label('↘️').classes('text-base')
                                ui.label('DOWN').classes('text-xs text-cyan-400 font-bold tracking-wide')
                            
                            # Stats grid
                            with ui.row().classes('flex-1 gap-4'):
                                # Time % (color-coded cyan)
                                with ui.column().classes('gap-0'):
                                    ui.label('TIME').classes('text-[10px] text-zinc-500 uppercase tracking-wider font-semibold')
                                    ui.label(f"{downhill['time_pct']:.0f}%").classes('text-sm text-cyan-400 font-bold font-mono')
                                
                                # HR
                                with ui.column().classes('gap-0'):
                                    ui.label('HR').classes('text-[10px] text-zinc-500 uppercase tracking-wider font-semibold')
                                    ui.label(f"{downhill['avg_hr']}").classes('text-sm text-white font-bold font-mono')
                                
                                # Pace
                                with ui.column().classes('gap-0'):
                                    ui.label('PACE').classes('text-[10px] text-zinc-500 uppercase tracking-wider font-semibold')
                                    ui.label(f"{downhill['avg_pace']}").classes('text-sm text-white font-bold font-mono')
    
    async def open_activity_detail_modal(self, activity_hash, from_feed=False):
        """
        Open detailed view of an activity.
        FINAL POLISH: 
        - Badge moved to Metrics Row.
        - Splits calculation now includes max_hr for accurate verdict.
        - Uses 'Feed Style' header for everything.
        """
        # Show loading dialog
        with ui.dialog() as loading_dialog:
            with ui.card().classes('bg-zinc-900 p-6').style('min-width: 300px; box-shadow: none;'):
                with ui.column().classes('items-center gap-4'):
                    ui.spinner(size='lg', color='emerald')
                    ui.label('Loading activity details...').classes('text-lg text-white')
        loading_dialog.open()

        # Parse FIT file asynchronously
        detail_data = await self.get_activity_detail(activity_hash)
        loading_dialog.close()

        if detail_data is None or detail_data.get('error'):
            ui.notify('Error loading activity', type='negative')
            return

        # Calculate metrics
        # 1. HR Zones
        hr_zones = self.calculate_hr_zones(detail_data['hr_stream'], detail_data['max_hr'])
        
        # 2. Lap Splits (Passing max_hr for the verdict logic)
        enhanced_laps = self.calculate_gap_for_laps(
            detail_data['lap_data'],
            detail_data['elevation_stream'],
            detail_data['timestamps'],
            detail_data.get('cadence_stream'),
            max_hr=detail_data.get('max_hr', 185) # <--- CRITICAL FIX FROM PART 2
        )
        
        # 3. Decoupling
        decoupling = self.calculate_aerobic_decoupling(detail_data['hr_stream'], detail_data['speed_stream'])
        
        # 4. Run/Walk
        run_walk_stats = None
        if detail_data.get('cadence_stream'):
            run_walk_stats = self.calculate_run_walk_stats(
                detail_data['cadence_stream'], detail_data['speed_stream'], detail_data['hr_stream']
            )
        
        # 5. Terrain
        terrain_stats = self.calculate_terrain_stats(
            detail_data['elevation_stream'], detail_data['hr_stream'], 
            detail_data['speed_stream'], detail_data['timestamps']
        )

        activity = detail_data['activity_metadata']

        # --- QUALITY BADGE CALCULATION ---
        try:
            cad = activity.get('avg_cadence', 0)
            session = detail_data.get('session_data', {})
            
            form_res = analyze_form(
                cad or session.get('avg_cadence', 0),
                session.get('avg_stance_time', 0),
                session.get('avg_step_length', 0),
                session.get('avg_vertical_oscillation', 0)
            )
            
            verdict_text = form_res['verdict']
            
            # UPDATED: Full text, no brackets
            if verdict_text in ['ELITE FORM', 'GOOD FORM']:
                v_label = 'HIGH QUALITY MILES' 
                v_color = 'text-emerald-400'
                v_bg = 'bg-emerald-500/20 border-emerald-500/30'
            elif verdict_text in ['HIKING / REST', 'HIKING']:
                 v_label = 'STRUCTURAL MILES'
                 v_color = 'text-blue-400'
                 v_bg = 'bg-blue-500/20 border-blue-500/30'
            elif verdict_text in ['PLODDING', 'LOW CADENCE', 'OVERSTRIDING', 'HEAVY FEET']:
                v_label = 'BROKEN MILES'
                v_color = 'text-red-400'
                v_bg = 'bg-red-500/20 border-red-500/30'
            else:
                v_label = None
                
        except:
            v_label = None
        # -------------------------------

        # Create main modal
        with ui.dialog() as detail_dialog:
            with ui.card().classes('w-full max-w-[900px] p-0 bg-zinc-950 h-full border border-zinc-800'):
                # Close button
                with ui.row().classes('w-full justify-end p-2'):
                    close_btn = ui.button(icon='close', on_click=detail_dialog.close, color=None).props('flat round dense')
                    close_btn.style('color: #9ca3af !important;')

                # Content container
                with ui.column().classes('w-full gap-4 px-4 pb-4'):

                    # --- HEADER (Polished Layout) ---
                    with ui.column().classes('w-full px-4 gap-1'):
                        # Row 1: Date & Time (Clean Title)
                        date_str = activity.get('date', '')
                        try:
                            from datetime import datetime
                            dt = datetime.strptime(date_str, '%Y-%m-%d %H:%M')
                            formatted_date = dt.strftime('%A, %B %-d • %-I:%M %p')
                        except:
                            formatted_date = date_str
                        
                        ui.label(formatted_date).classes('text-2xl font-bold text-white tracking-tight')

                        # Row 2: Metrics + Badge (The "Story")
                        with ui.row().classes('items-center gap-3'):
                            # The Big Numbers
                            distance = activity.get('distance_mi', 0)
                            elevation = activity.get('elevation_ft', 0)
                            pace = activity.get('pace', '--:--')
                            calories = activity.get('calories', 0)
                            
                            metrics_str = f"{distance:.1f} mi • {elevation} ft • {pace} /mi"
                            if calories:
                                metrics_str += f" • {calories} cal"
                            
                            ui.label(metrics_str).classes('text-zinc-400 font-mono text-sm tracking-wide')

                            # The Badge (Injecting Context)
                            if v_label:
                                ui.label(v_label).classes(f'text-[10px] font-bold px-2 py-0.5 rounded border {v_color} {v_bg} tracking-wider')

                    # Structural divider
                    ui.separator().classes('bg-zinc-800 my-4')

                    # 1. Strategy Row
                    self.create_strategy_row(run_walk_stats, terrain_stats)

                    # 2. HR Zones Chart
                    with ui.card().classes('w-full bg-zinc-900 p-4 border border-zinc-800').style('border-radius: 8px; box-shadow: 0px 4px 12px rgba(0, 0, 0, 0.4);'):
                        if detail_data.get('max_hr_fallback'):
                            ui.label('⚠️ Zones based on Session Max HR').classes('text-xs text-yellow-500 mb-2')
                        hr_chart = self.create_hr_zone_chart(hr_zones)
                        ui.plotly(hr_chart).classes('w-full')

                    # 3. Body Response Row
                    session_data = detail_data.get('session_data', {})
                    if not session_data.get('avg_cadence'):
                        session_data['avg_cadence'] = activity.get('avg_cadence', 0)
                    
                    has_dynamics = session_data and any([session_data.get('avg_vertical_oscillation'), session_data.get('avg_stance_time')])
                    
                    with ui.row().classes('w-full gap-3'):
                        if has_dynamics:
                            with ui.column().classes('flex-1 min-w-0'):
                                self.create_running_dynamics_card(session_data)
                        with ui.column().classes('flex-1 min-w-0'):
                            self.create_decoupling_card(decoupling)
                        if session_data:
                            with ui.column().classes('flex-1 min-w-0'):
                                self.create_physiology_card(session_data, activity)

                    # 4. Cadence-Elevation Chart
                    if detail_data.get('cadence_stream') and detail_data.get('distance_stream'):
                        with ui.card().classes('w-full bg-zinc-900 p-4 border border-zinc-800').style('border-radius: 8px; box-shadow: 0px 4px 12px rgba(0, 0, 0, 0.4);'):
                            cadence_chart = self.create_cadence_elevation_chart(
                                detail_data['distance_stream'], detail_data['cadence_stream'], detail_data['elevation_stream'], use_miles=True
                            )
                            ui.plotly(cadence_chart).classes('w-full')

                    # 5. Lap Splits
                    with ui.card().classes('w-full bg-zinc-900 p-4 border border-zinc-800').style('border-radius: 8px; box-shadow: 0px 4px 12px rgba(0, 0, 0, 0.4);'):
                        with ui.row().classes('w-full justify-between items-center mb-2'):
                            ui.label('Lap Splits').classes('text-lg font-bold text-white')
                            copy_icon = ui.icon('content_copy').classes('cursor-pointer text-zinc-500 hover:text-white transition-colors duration-200 text-sm')
                            copy_icon.on('click.stop', lambda: self.copy_splits_to_clipboard(enhanced_laps))
                        self.create_lap_splits_table(enhanced_laps)

        detail_dialog.open()

 
    def build_ui(self):
        """Construct the complete UI layout with fixed sidebar."""
        # 1. CSS HACK: Hide Plotly Logo and notifications globally
        ui.add_head_html('''
        <style>
        .modebar-btn--logo { display: none !important; }
        /* Hide Plotly's "Double-click to zoom back out" notification */
        .plotly .notifier { display: none !important; }
        .js-plotly-plot .notifier { display: none !important; }
        .notifier { display: none !important; }
        
        /* Remove default page padding/margin to eliminate grey strips */
        body, .q-page, .nicegui-content {
            margin: 0 !important;
            padding: 0 !important;
        }
        
        /* Force light grey background everywhere for Apple-style contrast */
        html, body, #app, .q-page-container, .q-page {
            background-color: #F5F5F7 !important;
        }
        
        /* Prevent overscroll bounce on macOS native app */
        html, body {
            overscroll-behavior: none !important;
            overflow: hidden !important;
        }
        
        /* Allow scrolling only on main content */
        .q-page {
            overflow-y: auto !important;
        }
        
        /* Toast notification animations */
        @keyframes slideIn {
            from {
                transform: translateX(400px);
                opacity: 0;
            }
            to {
                transform: translateX(0);
                opacity: 1;
            }
        }
        
        @keyframes slideOut {
            from {
                transform: translateX(0);
                opacity: 1;
            }
            to {
                transform: translateX(400px);
                opacity: 0;
            }
        }
        }
        
        /* Smooth carousel-style transitions - keep all panels rendered */
        .q-tab-panels {
            overflow: hidden;
            position: relative;
        }
        
        .q-tab-panel {
            transition: transform 0.8s cubic-bezier(0.16, 1, 0.3, 1) !important, opacity 0.8s cubic-bezier(0.16, 1, 0.3, 1) !important;
            will-change: transform, opacity;
        }
        
        /* Force all panels to stay in DOM (prevent hide/show flash) */
        .q-tab-panel[aria-hidden="true"] {
            position: absolute;
            top: 0;
            left: 0;
            width: 100%;
            opacity: 0;
            pointer-events: none;
            transform: translateX(30px);
        }
        
        .q-tab-panel[aria-hidden="false"] {
            position: relative;
            opacity: 1;
            pointer-events: auto;
            transform: translateX(0);
        }
        
        /* Copy splits button styling - matches Activities table eye icon */
        .copy-splits-btn.q-btn {
            opacity: 0.3 !important;
            transition: all 0.15s ease !important;
        }
        .copy-splits-btn.q-btn .q-icon {
            color: #9ca3af !important;
            transition: color 0.15s ease !important;
        }
        .copy-splits-btn.q-btn:hover {
            opacity: 1 !important;
            background-color: rgba(255, 255, 255, 0.1) !important;
        }
        .copy-splits-btn.q-btn:hover .q-icon {
            color: #ffffff !important;
        }
        .copy-splits-btn.q-btn:before {
            display: none !important;
        }
        .copy-splits-btn.q-btn:not(:hover) .q-icon {
            color: #9ca3af !important;
        }
        </style>
        ''')
        # Set light grey background for entire page (Apple style) with modern emerald accent
        ui.colors(primary='#10B981', secondary='#1F1F1F', accent='#ff9900', 
                  dark='#F5F5F7', positive='#10B981', negative='#ff4d4d', 
                  info='#3b82f6', warning='#ff9900')
        ui.query('body').classes('bg-gray-100').style('background-color: #F5F5F7;')
        
        # Add custom CSS for dropdown styling
        ui.add_head_html('''
        <style>
        /* Style dropdown options to be visible */
        .q-menu .q-item {
            color: white !important;
            background-color: #1F1F1F !important;
        }
        .q-menu .q-item:hover {
            background-color: #2CC985 !important;
        }
        /* Style the select input text and icon */
        .q-field__native, .q-field__input {
            color: white !important;
        }
        /* Style the dropdown arrow icon */
        .q-field__append .q-icon {
            color: white !important;
        }
        /* Style the select border */
        .q-field--outlined .q-field__control:before {
            border-color: rgba(255, 255, 255, 0.3) !important;
        }
        .q-field--outlined .q-field__control:hover:before {
            border-color: rgba(255, 255, 255, 0.6) !important;
        }
        </style>
        ''')
        
        with ui.row().classes('w-full h-screen m-0 p-0 gap-0 no-wrap overflow-hidden'):
            # Fixed sidebar - stays in place
            self.build_sidebar()
            # Scrollable main content area
            self.build_main_content()
    
    def build_sidebar(self):
        """Create fixed left sidebar with controls."""
        with ui.column().classes('w-56 bg-zinc-900 p-4 h-screen sticky top-0 flex-shrink-0'):
            # Logo/Title
            ui.label('🏃‍♂️ Garmin\nAnalyzer Pro').classes(
                'text-2xl font-bold text-center mb-4 whitespace-pre-line text-white'
            )
            
            # Timeframe filter
            ui.label('TIMEFRAME').classes('text-xs text-gray-400 font-bold text-center mb-1')
            self.timeframe_select = ui.select(
                options=['Last Import', 'Last 30 Days', 'Last 90 Days', 
                         'This Year', 'All Time'],
                value='Last 30 Days',
                on_change=self.on_filter_change
            ).classes('w-full mb-4').style('color: white;').props('outlined dense')
            
            # Action buttons - Modern solid style with Apple aesthetics
            ui.button('Import Folder', icon='create_new_folder', on_click=self.select_folder, color=None).classes(
                'w-full mb-2 text-white hover:bg-zinc-700'
            ).style('background-color: #3a3a3c; border-radius: 10px; border: none;')
            
            self.export_btn = ui.button('Export CSV', icon='file_download',
                                         on_click=self.export_csv, color=None).classes(
                'w-full mb-2 text-white hover:bg-zinc-700'
            ).props('disable').style('background-color: #3a3a3c; border-radius: 10px; border: none;')
            
            self.copy_btn = ui.button('Copy for LLM', icon='content_copy',
                                       on_click=self.copy_to_llm, color='primary').classes(
                'w-full mb-2 text-white'
            ).props('disable').style('border-radius: 10px; box-shadow: 0 2px 8px rgba(16, 185, 129, 0.3);')
            
            # Create persistent loading dialog for copy operation
            self.copy_loading_dialog = ui.dialog().props('persistent')
            with self.copy_loading_dialog, ui.card().classes('bg-zinc-900 p-6').style('min-width: 300px; box-shadow: none;'):
                with ui.column().classes('items-center gap-4'):
                    ui.spinner(size='lg', color='emerald')
                    ui.label('Exporting data to clipboard...').classes('text-lg text-white')
            
            # Separator
            ui.separator().classes('my-4 border-zinc-700')
            
            # Stats section with modern styling
            with ui.row().classes('items-center gap-2 w-full'):
                ui.label('Runs Stored:').classes('text-sm text-gray-400')
                self.runs_count_label = ui.label(f'{self.db.get_count()}').classes('text-sm font-bold text-white')
            
            with ui.row().classes('items-center gap-1.5 mt-2 w-full'):
                ui.label('●').classes('text-xs').style('color: #10B981;')
                self.status_label = ui.label('Ready').classes('text-xs').style('color: #6B7280;')
    
    def build_main_content(self):
        """Create tabbed main content area with scrolling."""
        # Outer wrapper: No padding, flush to edges with light grey background
        with ui.column().classes('flex-1 h-screen overflow-y-auto p-0 gap-0').style('background-color: #F5F5F7;'):
            # Inner container: Adds padding for content breathing room (no bottom padding)
            with ui.column().classes('w-full min-h-full pt-6 px-6 pb-0 gap-4').style('background-color: #F5F5F7;'):
                # Create tabs row with absolute positioned Save Chart button
                with ui.row().classes('w-full items-center mb-0 relative pb-3').style('border-bottom: 1px solid #E5E5E5;'):
                    # Tabs centered, taking full width
                    with ui.tabs().classes('w-full text-gray-800 justify-center') as tabs:
                        trends_tab = ui.tab('Trends').classes('text-gray-800')
                        report_tab = ui.tab('FEED').classes('text-gray-800')
                        activities_tab = ui.tab('ACTIVITIES').classes('text-gray-800')
                    
                    # Save Chart button absolutely positioned on the right (minimal, icon-focused)
                    # Add transition for smooth fade effect
                    self.save_chart_btn = ui.button(icon='download', on_click=self.save_chart_to_downloads, color=None).classes('text-white absolute right-0 top-0 z-10').style('background-color: #3a3a3c; border-radius: 6px; border: none; padding: 8px; min-width: 40px; transition: opacity 0.3s ease-in-out, background-color 0.2s ease;').props('flat dense').tooltip('Save Chart')
                
                # Create tab panels with transparent background
                with ui.tab_panels(tabs, value=trends_tab).classes('w-full flex-1').props('transparent'):
                    # Trends tab panel
                    with ui.tab_panel(trends_tab).classes('p-0'):
                        self.build_trends_tab()
                    
                    # Report tab panel
                    with ui.tab_panel(report_tab).classes('p-0'):
                        self.build_report_tab()
                    
                    # Activities tab panel
                    with ui.tab_panel(activities_tab).classes('p-0'):
                        self.build_activities_tab()
                
                # Show/hide Save Chart button based on active tab
                tabs.on('update:model-value', lambda e: self.toggle_save_chart_button(e.args))
    
    def build_trends_tab(self):
        """Create trends tab with embedded Plotly chart."""
        # Store the plotly_container as an instance variable so it can be updated later
        self.plotly_container = ui.column().classes('w-full').style('min-height: 900px; background-color: #F5F5F7;')
        
        with self.plotly_container:
            # Show placeholder message when no data is available
            ui.label('No data available. Import activities to view trends.').classes(
                'text-center text-gray-600 mt-20'
            )
    
    def build_report_tab(self):
        """Create report tab with Card View."""
        with ui.tab_panel('report').classes('w-full p-0 bg-transparent'):
            # The container for our cards with light grey background (no top padding for alignment)
            self.report_container = ui.column().classes('w-full gap-3 px-4 pb-4').style('background-color: #F5F5F7;')

    
    def build_activities_tab(self):
        """Create activities tab with AG Grid table."""
        # Store the grid_container as an instance variable so it can be updated later
        # by update_activities_grid() method (task 9.1)
        self.grid_container = ui.column().classes('w-full').style('min-height: 800px; background-color: #F5F5F7;')
        
        # Initialize with placeholder
        with self.grid_container:
            ui.label('No activities found. Import activities to view data.').classes(
                'text-center text-gray-600 mt-20'
            )
    
    async def refresh_data_view(self):
        """
        Refresh all views with current filter.
        
        This method:
        - Queries database using DatabaseManager.get_activities()
        - Converts results to Pandas DataFrame
        - Parses dates and sorts by date
        - Updates state variables (activities_data, df)
        - Enables/disables export and copy buttons based on data availability
        
        Requirements: 6.7, 6.8, 11.1, 11.7
        """
        # Query database with current timeframe filter
        self.activities_data = self.db.get_activities(
            self.current_timeframe, 
            self.current_session_id
        )
        
        # Convert to DataFrame if we have data
        if self.activities_data:
            self.df = pd.DataFrame(self.activities_data)
            # Parse dates and sort by date
            self.df['date_obj'] = pd.to_datetime(self.df['date'])
            self.df = self.df.sort_values('date_obj')
        else:
            self.df = None
        
        # Update all tabs
        self.update_report_text()  # Task 8.1 - implemented
        self.update_activities_grid()  # Task 9.1 - implemented
        self.update_trends_chart()  # Task 12.1 - implemented
        
        # Enable/disable export and copy buttons based on data availability
        has_data = bool(self.activities_data)
        if has_data:
            self.export_btn.props(remove='disable')
            self.copy_btn.props(remove='disable')
        else:
            self.export_btn.props(add='disable')
            self.copy_btn.props(add='disable')
    
    def toggle_save_chart_button(self, tab_name):
        """
        Show/hide Save Chart button based on active tab with fade effect.
        Only show on Trends tab.
        """
        # Use opacity for smooth fade transition
        if tab_name == 'Trends':
            self.save_chart_btn.style('opacity: 1; pointer-events: auto;')
        else:
            self.save_chart_btn.style('opacity: 0; pointer-events: none;')
    
    # Placeholder methods for handlers (to be implemented in later tasks)
    async def on_filter_change(self, e):
        """
        Handle timeframe filter changes.
        
        This method:
        - Updates current_timeframe state variable
        - Calls refresh_data_view() to update all views with filtered data
        - Implements LLM safety lock for large datasets
        
        Requirements: 11.1, 11.7
        """
        # Update current timeframe state from the dropdown value
        self.current_timeframe = e.value
        
        # Refresh all data views with the new filter
        await self.refresh_data_view()
        
        # LLM Safety Lock: Disable copy button for large datasets
        if self.current_timeframe in ["All Time", "This Year"]:
            self.copy_btn.props(add='disable')
            self.copy_btn.text = 'Too much data for LLM'
        else:
            # Only enable if we have data
            if self.activities_data:
                self.copy_btn.props(remove='disable')
            self.copy_btn.text = 'Copy for LLM'
    
    async def select_folder(self):
        """
        Handle folder selection.
        - On macOS: Uses 'osascript' (AppleScript) to avoid Tkinter crashes.
        - On Windows/Linux: Uses the Tkinter subprocess fallback.
        
        Requirements: 7.1, 7.2
        """
        import sys
        import subprocess
        
        # --- MACOS NATIVE SOLUTION ---
        if sys.platform == 'darwin':
            try:
                self.timeframe_select.props(add='disable')
                
                # AppleScript command: Activate Finder, Open Dialog, Return POSIX path
                # 'User canceled' throws an error, which we catch safely.
                cmd = """osascript -e 'tell application "System Events" to activate' -e 'POSIX path of (choose folder with prompt "Select Folder with FIT Files")'"""
                
                # Run asynchronously
                process = await asyncio.create_subprocess_shell(
                    cmd,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE
                )
                
                stdout, stderr = await process.communicate()
                folder_path = stdout.decode().strip()
                
                if folder_path:
                    ui.notify(f'Selected: {folder_path}', type='positive')
                    await self.process_folder_async(folder_path)
                else:
                    # User likely hit Cancel (AppleScript outputs to stderr on cancel)
                    pass
                    
            except Exception as e:
                ui.notify(f'Error: {e}', type='negative')
            finally:
                self.timeframe_select.props(remove='disable')
            return
        
        # --- WINDOWS/LINUX FALLBACK (Tkinter Subprocess) ---
        # Define the script for Windows/Linux
        script = """
import tkinter as tk
from tkinter import filedialog
import sys

root = tk.Tk()
root.withdraw()
root.wm_attributes('-topmost', 1)

folder_path = filedialog.askdirectory(title="Select Folder with FIT Files")

if folder_path:
    print(folder_path, end='')

root.destroy()
"""
        
        try:
            self.timeframe_select.props(add='disable')
            
            process = await asyncio.create_subprocess_exec(
                sys.executable, '-c', script,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE
            )
            
            stdout, stderr = await process.communicate()
            folder_path = stdout.decode().strip()
            
            if folder_path:
                ui.notify(f'Selected: {folder_path}', type='positive')
                await self.process_folder_async(folder_path)
                
        except Exception as e:
            ui.notify(f'Error: {e}', type='negative')
        finally:
            self.timeframe_select.props(remove='disable')
            # User cancelled - just return without error
            return
    
    async def process_folder_async(self, folder_path):
        """
        Process FIT files in background with progress.
        
        This method:
        - Generates session ID from Unix timestamp
        - Shows progress dialog with progress bar
        - Gets list of .FIT files from folder
        - Loops through files with FitAnalyzer
        - Calculates file hash for each file
        - Checks if hash exists in database
        - Skips if exists, otherwise imports
        - Updates progress bar after each file
        - Closes dialog and updates UI when complete
        - Switches to "Last Import" timeframe
        - Calls refresh_data_view()
        
        Requirements: 7.3, 7.4, 7.5, 7.6, 7.7, 7.8, 7.9, 13.2, 13.3, 13.4
        """
        try:
            # Generate session ID from Unix timestamp
            self.current_session_id = int(time.time())
            
            # Get list of .FIT files from folder
            try:
                fit_files = [
                    os.path.join(folder_path, f) 
                    for f in os.listdir(folder_path) 
                    if f.lower().endswith('.fit')
                ]
            except Exception as e:
                ui.notify(f'Error reading folder: {e}', type='negative')
                self.timeframe_select.props(remove='disable')
                return
            
            # Handle empty folder
            if not fit_files:
                ui.notify('No .FIT files found in selected folder!', type='warning')
                self.timeframe_select.props(remove='disable')
                return
            
            # Show progress dialog
            with ui.dialog() as progress_dialog, ui.card().classes('bg-zinc-900 text-white p-6'):
                ui.label('ANALYZING FIT FILES...').classes('text-lg font-bold mb-4 text-white')
                progress_bar = ui.linear_progress(value=0).classes('w-96')
                progress_label = ui.label('Starting...').classes('mt-2 text-white')
            
            progress_dialog.open()
            
            # Initialize FitAnalyzer
            analyzer = FitAnalyzer()
            new_count = 0
            skipped_count = 0
            error_count = 0
            
            # Process each file
            for i, filepath in enumerate(fit_files):
                try:
                    # Calculate file hash for deduplication
                    f_hash = calculate_file_hash(filepath)
                    
                    # Check if hash exists in database
                    if self.db.activity_exists(f_hash):
                        skipped_count += 1
                    else:
                        # Analyze and import the file (run in thread to avoid UI freeze)
                        result = await run.io_bound(analyzer.analyze_file, filepath)
                        if result:
                            self.db.insert_activity(result, f_hash, self.current_session_id, filepath)
                            new_count += 1
                        else:
                            error_count += 1
                    
                    # Update progress bar after each file
                    progress = (i + 1) / len(fit_files)
                    progress_bar.value = progress
                    progress_label.text = f'Processing {i + 1}/{len(fit_files)} - {os.path.basename(filepath)}'
                    
                    # Allow UI to update
                    await asyncio.sleep(0)
                    
                except Exception as e:
                    error_count += 1
                    print(f"Error processing {filepath}: {e}")
            
            # Close dialog
            progress_dialog.close()
            
            # Update runs counter
            self.runs_count_label.text = f'{self.db.get_count()}'
            
            # Update status and switch timeframe if new activities imported
            if new_count > 0:
                self.status_label.text = f'Imported {new_count} new'
                self.status_label.style('color: #10B981;')
                
                # ADDED: Optional - Switch to "All Time" so you see EVERYTHING including new files
                if self.current_timeframe == 'Last Import':
                     self.current_timeframe = 'Last 30 Days'
                     self.timeframe_select.value = 'Last 30 Days'
                
                # Show success notification
                ui.notify(f'Import complete: {new_count} new activities', type='positive')
            else:
                # No new files imported
                self.status_label.text = 'No new runs'
                self.status_label.style('color: #ff9900;')
                
                # Show notification about duplicates
                ui.notify('All files were already in the database', type='info')
            
            # Re-enable timeframe dropdown
            self.timeframe_select.props(remove='disable')
            
            # Refresh all data views
            await self.refresh_data_view()
            
        except Exception as e:
            # Catch any unexpected errors
            ui.notify(f'Error during import: {e}', type='negative')
            print(f"Error in process_folder_async: {e}")
            import traceback
            traceback.print_exc()
            # Re-enable timeframe dropdown
            self.timeframe_select.props(remove='disable')
    
    def update_report_text(self):
        """Update report with Beautiful Cards. Fully restored styling."""
        self.report_container.clear()
        
        if not self.activities_data:
            with self.report_container:
                ui.label('No runs found for this timeframe.').classes('text-gray-600 italic')
            return
        
        # Calculate Average EF for context
        avg_ef = self.df['efficiency_factor'].mean() if self.df is not None else 0
        
        # Calculate long run threshold
        if self.df is not None and len(self.df) >= 5:
            long_run_threshold = self.df['distance_mi'].quantile(0.8)
        else:
            long_run_threshold = 10.0
        
        with self.report_container:
            for d in sorted(self.activities_data, key=lambda x: x.get('date', ''), reverse=True):
                
                # 1. Determine Border Color based on Cost
                cost = d.get('decoupling', 0)
                border_color = 'border-green-500' if cost <= 5 else 'border-red-500'
                
                # 2. Parse date string
                date_str = d.get('date', '')
                try:
                    from datetime import datetime
                    dt = datetime.strptime(date_str, '%Y-%m-%d %H:%M')
                    formatted_date = dt.strftime('%a, %b %-d')
                    formatted_time = dt.strftime('%-I:%M %p')
                except:
                    formatted_date = date_str
                    formatted_time = ''
                
                # 3. Classify run type
                run_type_tag = self.classify_run_type(d, long_run_threshold)
                
                # 3.5. Calculate Strain Score
                moving_time_min = d.get('moving_time_min', 0)
                avg_hr = d.get('avg_hr', 0)
                max_hr = d.get('max_hr', 185)
                
                intensity = avg_hr / max_hr if max_hr > 0 else 0
                
                if intensity < 0.65: factor = 1.0
                elif intensity < 0.75: factor = 1.5
                elif intensity < 0.85: factor = 3.0
                elif intensity < 0.92: factor = 6.0
                else: factor = 10.0
                
                strain = int(moving_time_min * factor)
                
                if strain < 75:
                    strain_label, strain_color, strain_text_color = "Recovery", "blue", "#60a5fa"
                elif strain < 150:
                    strain_label, strain_color, strain_text_color = "Maintenance", "#10B981", "#10B981"
                elif strain < 300:
                    strain_label, strain_color, strain_text_color = "Productive", "orange", "orange"
                else:
                    strain_label, strain_color, strain_text_color = "Overreaching", "red", "red"
                
                # 4. Create the "Hot Card" with user-preferred shadow settings
                activity_hash = d.get('db_hash')
                card = ui.card().classes(
                    'w-full p-4 bg-zinc-900 border border-zinc-800 '
                    'cursor-pointer relative overflow-hidden group '
                    'transform transition-all duration-300 ease-out '
                    'shadow-lg rounded-xl ' 
                    'hover:border-zinc-500 hover:shadow-[0_7px_17px_rgba(0,0,0,0.9)] hover:-translate-y-1 hover:bg-zinc-800'
                )
                
                if activity_hash:
                    card.on('click', lambda h=activity_hash: self.open_activity_detail_modal(h, from_feed=True))
                
                with card:
                    with ui.column().classes('w-full gap-1'):
                        # --- ROW 1: Header & Calories ---
                        with ui.row().classes('w-full justify-between items-start'):
                            with ui.column().classes('gap-0'):
                                ui.label(formatted_date).classes('font-bold text-zinc-200 text-sm group-hover:text-white transition-colors')
                                ui.label(formatted_time).classes('text-xs text-zinc-500 group-hover:text-zinc-400 transition-colors')
                            
                            if d.get('calories'):
                                with ui.row().classes('items-center gap-3 text-xs text-zinc-300 bg-zinc-800/50 px-2 py-1 rounded border border-zinc-700/50 group-hover:bg-zinc-700 transition-colors'):
                                    with ui.row().classes('items-center gap-1'):
                                        ui.icon('local_fire_department').classes('text-xs text-orange-400')
                                        ui.label(f"{d['calories']} cal")
                        
                        # --- ROW 2: Context Tags (FIXED: Duplicates Removed) ---
                        with ui.row().classes('w-full items-center gap-2 mt-2'):
                            for tag in run_type_tag.split(' | '):
                                ui.label(tag).classes('text-[10px] font-bold px-2 py-0.5 rounded bg-zinc-800 text-zinc-400 border border-zinc-700 tracking-wide')
                            
                            if d.get('te_label'):
                                te_color = d.get('te_label_color', 'text-zinc-400')
                                if 'text-purple-400' in te_color: bg_color, border_color = 'bg-purple-500/10', 'border-purple-500/30'
                                elif 'text-red-400' in te_color: bg_color, border_color = 'bg-red-500/10', 'border-red-500/30'
                                elif 'text-orange-400' in te_color: bg_color, border_color = 'bg-orange-500/10', 'border-orange-500/30'
                                elif 'text-emerald-400' in te_color: bg_color, border_color = 'bg-emerald-500/10', 'border-emerald-500/30'
                                elif 'text-blue-400' in te_color: bg_color, border_color = 'bg-blue-500/10', 'border-blue-500/30'
                                else: bg_color, border_color = 'bg-zinc-800', 'border-zinc-700'
                                
                                text_color = te_color.split()[0] if ' ' in te_color else te_color
                                ui.label(d['te_label']).classes(f"text-[10px] font-bold px-2 py-0.5 rounded {bg_color} border {border_color} {text_color} tracking-wide")

                        # --- ROW 3: Main Metrics Grid ---
                        with ui.row().classes('w-full gap-4 mb-1 items-center mt-3'):
                            with ui.column().classes('flex-1'):
                                with ui.grid(columns=5).classes('w-full gap-3'):
                                    # Distance
                                    with ui.column().classes('gap-0').style('line-height: 1.1;'):
                                        with ui.row().classes('items-center gap-1'):
                                            ui.icon('straighten').classes('text-blue-400 text-xs')
                                            ui.label('DISTANCE').classes('text-[10px] text-gray-500 font-bold tracking-wider')
                                        ui.label(f"{d.get('distance_mi', 0):.1f} mi").classes('text-lg font-bold').style('line-height: 1;')
                                    
                                    # Elevation
                                    with ui.column().classes('gap-0').style('line-height: 1.1;'):
                                        with ui.row().classes('items-center gap-1'):
                                            ui.icon('terrain').classes('text-green-400 text-xs')
                                            ui.label('ELEVATION').classes('text-[10px] text-gray-500 font-bold tracking-wider')
                                        ui.label(f"{d.get('elevation_ft', 0)} ft").classes('text-lg font-bold').style('line-height: 1;')
                                    
                                    # Pace
                                    with ui.column().classes('gap-0').style('line-height: 1.1;'):
                                        with ui.row().classes('items-center gap-1'):
                                            ui.icon('speed').classes('text-purple-400 text-xs')
                                            ui.label('PACE').classes('text-[10px] text-gray-500 font-bold tracking-wider')
                                        ui.label(d.get('pace', '--')).classes('text-lg font-bold').style('line-height: 1;')
                                        
                                    # Efficiency
                                    with ui.column().classes('gap-0').style('line-height: 1.1;'):
                                        with ui.row().classes('items-center gap-1'):
                                            ui.label('EFFICIENCY').classes('text-[10px] text-gray-500 font-bold tracking-wider')
                                            ui.icon('help_outline').classes('text-zinc-600 hover:text-white text-[10px] cursor-pointer').on('click.stop', lambda: self.show_ef_info())
                                        ui.label(f"{d.get('efficiency_factor', 0):.2f}").classes('text-lg font-bold').style('color: #10B981; line-height: 1;')
                                        
                                    # Decoupling
                                    with ui.column().classes('gap-0').style('line-height: 1.1;'):
                                        with ui.row().classes('items-center gap-1'):
                                            ui.label('DECOUPLING').classes('text-[10px] text-gray-500 font-bold tracking-wider')
                                            ui.icon('help_outline').classes('text-zinc-600 hover:text-white text-[10px] cursor-pointer').on('click.stop', lambda: self.show_cost_info())
                                        cost_color = '#10B981' if cost <= 5 else '#ff4d4d'
                                        ui.label(f"{cost:.1f}%").classes(f'text-lg font-bold').style(f'color: {cost_color}; line-height: 1;')
                            
                            ui.element('div').classes('h-full').style('width: 1px; background-color: #27272a; margin: 0 8px;')
                            
                            with ui.column().classes('items-center justify-center gap-1 mr-2'):
                                with ui.element('div').classes('relative'):
                                    ui.circular_progress(value=min(strain/500, 1.0), size='80px', color=strain_color, show_value=False)
                                    with ui.element('div').classes('absolute inset-0 flex items-center justify-center'):
                                        ui.label(str(strain)).classes('text-xl font-bold')
                                with ui.row().classes('items-center gap-1'):
                                    ui.icon('help_outline').classes('text-zinc-600 hover:text-white text-[10px] cursor-pointer').on('click.stop', lambda: self.show_load_info())
                                    ui.label('LOAD:').classes('text-xs text-zinc-300 font-bold uppercase tracking-widest')
                                    ui.label(strain_label).classes(f'text-sm font-bold').style(f'color: {strain_text_color};')

                        # --- ROW 4: Footer ---
                        ui.separator().classes('my-1').style('background-color: #52525b; height: 1px;')
                        with ui.row().classes('w-full justify-between items-center'):
                            with ui.row().classes('gap-6'):
                                # Avg HR
                                with ui.column().classes('items-center gap-0'):
                                    ui.label('AVG HR').classes('text-[9px] text-gray-500 font-bold tracking-wider mb-0.5')
                                    with ui.row().classes('items-center gap-1'):
                                        ui.icon('favorite').classes('text-pink-400 text-sm')
                                        ui.label(f"{d.get('avg_hr', 0)}").classes('text-sm font-bold text-white')
                                
                                # Cadence
                                with ui.column().classes('items-center gap-0'):
                                    ui.label('CADENCE').classes('text-[9px] text-gray-500 font-bold tracking-wider mb-0.5')
                                    with ui.row().classes('items-center gap-1'):
                                        ui.icon('directions_run').classes('text-blue-400 text-sm')
                                        ui.label(f"{d.get('avg_cadence', 0)}").classes('text-sm font-bold text-white')
                            
                            # Right: Form Status Pill
                            form = analyze_form(d.get('avg_cadence'), d.get('avg_stance_time'), d.get('avg_step_length'), d.get('avg_vertical_oscillation'))
                            if form['verdict'] != 'ANALYZING':
                                if form['verdict'] in ['ELITE FORM', 'GOOD FORM']:
                                    pill_bg, pill_border, pill_text = 'bg-emerald-500/10', 'border-emerald-700/30', 'text-emerald-400'
                                elif form['verdict'] in ['OVERSTRIDING', 'HEAVY FEET', 'INEFFICIENT']:
                                    pill_bg, pill_border, pill_text = 'bg-amber-500/10', 'border-amber-700/30', 'text-amber-400'
                                elif form['verdict'] in ['LOW CADENCE', 'PLODDING']:
                                    pill_bg, pill_border, pill_text = 'bg-red-500/10', 'border-red-700/30', 'text-red-400'
                                else:
                                    pill_bg, pill_border, pill_text = 'bg-slate-500/10', 'border-slate-700/30', 'text-slate-400'
                                
                                with ui.row().classes(f'items-center gap-2 px-3 py-1.5 rounded border {pill_bg} {pill_border}'):
                                    ui.label('🦶').classes('text-sm')
                                    ui.label('Form:').classes(f'text-xs font-bold {pill_text}')
                                    ui.label(form['verdict'].title()).classes(f'text-xs font-bold {pill_text}')
    
    def show_hrr_info(self):
        """
        Show informational modal about Heart Rate Recovery.
        Explains the science, interpretation, and common reasons for low readings.
        """
        with ui.dialog() as dialog, ui.card().classes('bg-zinc-900 text-white p-6 max-w-2xl'):
            # Title
            ui.label('Heart Rate Recovery (1-Min)').classes('text-xl font-bold mb-4')
            
            # Body explanation
            ui.label('Measures how fast your heart rate drops 60 seconds after pressing STOP. Faster recovery = better aerobic fitness.').classes('text-sm text-gray-300 mb-4')
            
            # Scale with color coding
            ui.label('Interpretation Scale:').classes('text-sm font-bold mb-2')
            with ui.column().classes('gap-2 mb-4'):
                with ui.row().classes('items-center gap-2'):
                    ui.label('🟢').classes('text-lg')
                    ui.label('> 30 bpm: Excellent').classes('text-sm text-green-400')
                
                with ui.row().classes('items-center gap-2'):
                    ui.label('🟡').classes('text-lg')
                    ui.label('20-30 bpm: Fair').classes('text-sm text-yellow-400')
                
                with ui.row().classes('items-center gap-2'):
                    ui.label('🔴').classes('text-lg')
                    ui.label('< 20 bpm: Poor / Fatigue').classes('text-sm text-red-400')
            # "Why is my number low?" Section
            ui.separator().classes('my-4 border-zinc-700')
            ui.label('Why is my number low?').classes('text-lg font-bold mb-3')
            
            ui.markdown('''
**1. The Cooldown Effect (Most Common)**  
If you jog or walk before pressing stop, your HR is already low, so the drop will be small. This is a "False Low." For a true test, stop your watch immediately after your hardest effort.

**2. Dehydration & Heat**  
Thicker blood keeps HR elevated after exercise.

**3. Accumulated Fatigue**  
A sign your body is struggling to recover from recent hard training.
            ''').classes('text-sm text-gray-300 mb-4')
            
            # Close button
            ui.button('Got it!', on_click=dialog.close).classes('w-full bg-green-600')
        
        dialog.open()
    
    def show_volume_info(self):
        """
        Show informational modal about Volume Classification logic.
        Explains the 'Ultra Trinity' (Terrain, HR, Mechanics).
        """
        with ui.dialog() as dialog, ui.card().classes('bg-zinc-900 text-white p-6 max-w-2xl border border-zinc-800'):
            # Title (Clean, no X)
            ui.label('Volume Quality Analysis').classes('text-xl font-bold text-white mb-2')
            
            # The 3 Buckets Grid
            with ui.column().classes('gap-6'):
                
                # 1. High Quality (Green)
                with ui.row().classes('gap-4 items-start'):
                    ui.icon('verified').classes('text-emerald-400 text-2xl mt-1')
                    with ui.column().classes('gap-1'):
                        ui.label('High Quality (The Engine)').classes('text-base font-bold text-emerald-400')
                        ui.label('Running on flat/rolling terrain with good mechanics (Cadence > 160) and honest effort. These miles build fitness without breaking the chassis.').classes('text-sm text-zinc-300')

                # 2. Structural (Blue)
                with ui.row().classes('gap-4 items-start'):
                    ui.icon('hiking').classes('text-blue-400 text-2xl mt-1')
                    with ui.column().classes('gap-1'):
                        ui.label('Structural (The Base)').classes('text-base font-bold text-blue-400')
                        ui.label('Valid volume that includes Hiking (Steep Grade), Recovery Shuffles (Low HR), or Walking. These miles build durability and aerobic base without the mechanical stress of fast running.').classes('text-sm text-zinc-300')

                # 3. Broken (Red)
                with ui.row().classes('gap-4 items-start'):
                    ui.icon('warning').classes('text-red-400 text-2xl mt-1')
                    with ui.column().classes('gap-1'):
                        ui.label('Broken (The Junk)').classes('text-base font-bold text-red-400')
                        ui.label('The "Danger Zone." You are working hard (High HR) but moving poorly (Low Cadence). This usually happens at the end of long runs when form falls apart. These miles cause injury.').classes('text-sm text-zinc-300')

            # The "Why" Footer
            ui.separator().classes('my-6 border-zinc-800')
            with ui.column().classes('gap-2 mb-4'):
                ui.label('How we decide:').classes('text-xs font-bold text-zinc-500 uppercase tracking-wider')
                ui.label('We analyze every single mile split against Terrain, Metabolic Cost, and Mechanics.').classes('text-sm text-zinc-400')

            # "Got it!" Button
            ui.button('Got it!', on_click=dialog.close).classes('w-full bg-green-600 hover:bg-green-500 text-white font-bold')

        dialog.open()

    def show_ef_info(self):
        """
        Show informational modal about Running Efficiency.
        Explains what Efficiency means and how to interpret it.
        """
        with ui.dialog() as dialog, ui.card().classes('bg-zinc-900 text-white p-6 max-w-2xl'):
            # Title
            ui.label('Running Efficiency').classes('text-xl font-bold mb-4')
            
            # Body explanation
            ui.label("Efficiency Factor (Speed / Heart Rate). Measures output per heartbeat. Higher is better.").classes('text-sm text-gray-300 mb-4')
            
            # Scale
            ui.label('Interpretation:').classes('text-sm font-bold mb-2')
            ui.markdown('''
📈 **Higher is Better**  
It means you are running faster at the same heart rate.

**How to Use It:**  
Compare this number to runs of similar intensity. If your Efficiency is improving over time, your aerobic engine is getting stronger.

**Typical Ranges:**
- 0.8-1.5: Recreational runners
- 1.5-2.5+: Elite runners
            ''').classes('text-sm text-gray-300 mb-4')
            
            # Close button
            ui.button('Got it!', on_click=dialog.close).classes('w-full bg-green-600')
        
        dialog.open()
    
    def show_cost_info(self):
        """
        Show informational modal about Aerobic Decoupling.
        Explains what Aerobic Decoupling means and how to interpret it.
        """
        with ui.dialog() as dialog, ui.card().classes('bg-zinc-900 text-white p-6 max-w-2xl'):
            # Title
            ui.label('Aerobic Decoupling').classes('text-xl font-bold mb-4')
            
            # Body explanation
            ui.label("Aerobic Decoupling (Pa:HR). Measures Cardiac Drift—how much HR rises while pace stays steady. Target < 5%.").classes('text-sm text-gray-300 mb-4')
            
            # Scale with color coding
            ui.label('Interpretation Scale:').classes('text-sm font-bold mb-2')
            with ui.column().classes('gap-2 mb-4'):
                with ui.row().classes('items-center gap-2'):
                    ui.label('✅').classes('text-lg')
                    ui.label('< 5%: Excellent aerobic endurance').classes('text-sm text-green-400')
                
                with ui.row().classes('items-center gap-2'):
                    ui.label('⚠️').classes('text-lg')
                    ui.label('5-10%: Moderate Drift').classes('text-sm text-yellow-400')
                
                with ui.row().classes('items-center gap-2'):
                    ui.label('🛑').classes('text-lg')
                    ui.label('> 10%: High Fatigue / Undeveloped Base').classes('text-sm text-red-400')
            
            ui.markdown('''
**What It Means:**  
Lower is better (<5% is solid). Your heart is working harder to maintain the same output (decoupling). This indicates a need for more aerobic base training or better fueling.

**High Aerobic Decoupling Indicates:**
- Insufficient aerobic base
- Running too fast for current fitness
- Accumulated fatigue from training
            ''').classes('text-sm text-gray-300 mb-4')
            
            # Close button
            ui.button('Got it!', on_click=dialog.close).classes('w-full bg-green-600')
        
        dialog.open()
    
    def show_load_info(self):
        """
        Show informational modal about Training Load.
        Explains how load is calculated and what each category means.
        """
        with ui.dialog() as dialog, ui.card().classes('bg-zinc-900 text-white p-6 max-w-2xl'):
            # Title
            ui.label('Training Load').classes('text-xl font-bold mb-4')
            
            # Body explanation
            ui.label("Training Load measures workout stress by analyzing duration and heart rate intensity. Higher intensity efforts are weighted more heavily.").classes('text-sm text-gray-300 mb-4')
            
            # Scale with color coding
            ui.label('Load Categories:').classes('text-sm font-bold mb-2')
            with ui.column().classes('gap-2 mb-4'):
                with ui.row().classes('items-center gap-2'):
                    ui.label('🔵').classes('text-lg')
                    ui.label('Recovery (<75): Easy effort that promotes adaptation and recovery.').classes('text-sm text-blue-400')
                
                with ui.row().classes('items-center gap-2'):
                    ui.label('🟢').classes('text-lg')
                    ui.label('Maintenance (75-150): Steady training that maintains your current fitness level.').classes('text-sm text-green-400')
                
                with ui.row().classes('items-center gap-2'):
                    ui.label('🟠').classes('text-lg')
                    ui.label('Productive (150-300): Hard work that builds fitness and improves performance.').classes('text-sm')
                
                with ui.row().classes('items-center gap-2'):
                    ui.label('🔴').classes('text-lg')
                    ui.label('Overreaching (300+): Very high stress that requires adequate recovery time.').classes('text-sm text-red-400')
            
            ui.markdown('''
**How to Use It:**  
Track your weekly load to balance hard training with recovery. Consistent productive loads build fitness, while too many overreaching sessions can lead to burnout.

**Training Tip:**  
Most of your runs should be Recovery or Maintenance, with Productive efforts 1-2x per week, and Overreaching reserved for key workouts or races.
            ''').classes('text-sm text-gray-300 mb-4')
            
            # Close button
            ui.button('Got it!', on_click=dialog.close).classes('w-full bg-green-600')
        
        dialog.open()
    
    def format_run_data(self, d, folder_avg_ef=0):
        """
        Format single activity data with comprehensive coaching context.
        
        This method formats an activity dictionary into a text block with:
        - Run type classification (stackable tags)
        - Distance, pace, EF, decoupling, HRR
        - Enriched metrics (Max HR, Max Speed, Elevation)
        - Mechanics (Cadence, Form Status)
        - Physiology (HR Zones, Load Score)
        - Weather context
        - Status indicators (✅ Excellent, ⚠️ Moderate, 🛑 High Fatigue)
        
        Requirements: 4.4, 4.5, 13.7
        """
        # Calculate long run threshold for classification
        if self.df is not None and len(self.df) >= 5:
            long_run_threshold = self.df['distance_mi'].quantile(0.8)
        else:
            long_run_threshold = 10.0
        
        # Get run type classification
        run_type = self.classify_run_type(d, long_run_threshold)
        
        # Get training effect label
        te_label = d.get('te_label', '')
        
        # Build type tags
        type_tags = [run_type]
        if te_label:
            type_tags.append(f"🔥 {te_label}")
        
        # Add weather tag if available (only if we have real temperature data)
        temp = d.get('avg_temp')
        if temp and temp > 0:  # Exclude 0 which indicates missing data
            temp_f = temp * 9/5 + 32  # Convert C to F
            if temp_f < 40:
                type_tags.append(f"🥶 Cold")
            elif temp_f > 80:
                type_tags.append(f"🥵 Hot")
        
        # Extract enriched metrics
        max_hr = d.get('max_hr', 0)
        avg_hr = d.get('avg_hr', 0)
        max_speed_mph = d.get('max_speed_mph', 0)
        elevation_ft = d.get('elevation_ft', 0)
        
        # Decoupling status
        decoupling = d.get('decoupling', 0)
        d_status = ""
        if decoupling < 5: 
            d_status = "✅ Excellent"
        elif decoupling <= 10: 
            d_status = "⚠️ Moderate"
        else: 
            d_status = "🛑 High Fatigue"
        
        # EF and HRR
        ef = d.get('efficiency_factor', 0)
        hrr_list = d.get('hrr_list', [])
        hrr_1min = hrr_list[0] if hrr_list and len(hrr_list) > 0 else None
        hrr_2min = hrr_list[1] if hrr_list and len(hrr_list) > 1 else None
        hrr_str = f"{hrr_1min}bpm (1min)" if hrr_1min else "--"
        if hrr_2min:
            hrr_str += f", {hrr_2min}bpm (2min)"
        
        # Duration - convert moving_time_min to HH:MM:SS format
        moving_time_min = d.get('moving_time_min', 0)
        if moving_time_min > 0:
            hours = int(moving_time_min // 60)
            minutes = int(moving_time_min % 60)
            seconds = int((moving_time_min % 1) * 60)
            if hours > 0:
                duration_str = f"{hours}:{minutes:02d}:{seconds:02d}"
            else:
                duration_str = f"{minutes}:{seconds:02d}"
        else:
            duration_str = "--"
        
        # Mechanics - Form Status
        cadence = d.get('avg_cadence', 0)
        gct = d.get('avg_stance_time', 0)  # Ground contact time in ms
        stride = d.get('avg_step_length', 0)  # Step length in meters
        bounce = d.get('avg_vertical_oscillation', 0)  # Vertical oscillation in cm
        
        form_analysis = analyze_form(cadence, gct, stride, bounce)
        form_verdict = form_analysis['verdict']
        
        # Add form emoji
        form_emoji = ""
        if 'ELITE' in form_verdict:
            form_emoji = "✅"
        elif 'GOOD' in form_verdict:
            form_emoji = "👍"
        elif 'OVERSTRIDING' in form_verdict:
            form_emoji = "🦶"
        elif 'PLODDING' in form_verdict or 'HEAVY' in form_verdict:
            form_emoji = "🐌"
        elif 'INEFFICIENT' in form_verdict:
            form_emoji = "⚠️"
        
        # Training Effect (Garmin's physiological load metric)
        training_effect = d.get('training_effect', 0)
        anaerobic_te = d.get('anaerobic_te', 0)
        
        # Power data
        avg_power = d.get('avg_power', 0)
        
        # Activity Breakdown (Run/Walk/Idle)
        # Note: This is approximate based on moving vs rest time
        # For accurate breakdown, would need speed stream analysis
        moving_time = d.get('moving_time_min', 0)
        rest_time = d.get('rest_time_min', 0)
        total_time = moving_time + rest_time
        
        if total_time > 0:
            moving_pct = (moving_time / total_time) * 100
            idle_pct = (rest_time / total_time) * 100
            # Assume most moving time is running (conservative for trail runs)
            activity_breakdown = f"Moving {moving_pct:.0f}% | Idle {idle_pct:.0f}%"
        else:
            activity_breakdown = "N/A"
        
        # Temperature
        temp_str = ""
        if temp and temp > 0:  # Only show if we have real temperature data
            temp_f = temp * 9/5 + 32
            temp_str = f"{temp_f:.0f}°F"
        
        # Build formatted output
        type_line = " | ".join([f"[{tag}]" for tag in type_tags])
        
        return f"""
RUN: {d.get('date')} {d.get('time', '')}
Type:       {type_line}
--------------------------------------------------
SUMMARY
Dist:       {d.get('distance_mi')} mi
Time:       {duration_str}
Pace:       {d.get('pace')} /mi
Elev Gain:  +{elevation_ft} ft
Temp:       {temp_str if temp_str else '--'}

PHYSIOLOGY (The Engine)
Avg HR:     {avg_hr} bpm
Max HR:     {max_hr} bpm{f'''
Avg Power:  {avg_power}W''' if avg_power > 0 else ''}
EF:         {ef:.2f} (Efficiency Factor)
Decoupling: {decoupling:.2f}% ({d_status})
HRR:        {hrr_str}
Training Effect: {training_effect:.1f} Aerobic, {anaerobic_te:.1f} Anaerobic

MECHANICS (The Chassis)
Avg Cadence: {cadence:.0f} spm
Form Tag:    [{form_emoji} {form_verdict}]
Max Speed:   {max_speed_mph:.1f} mph
Activity Breakdown: {activity_breakdown}
"""
    
    def update_activities_grid(self):
        """
        Update activities table with native NiceGUI table component.
        """
        # Clear grid container
        self.grid_container.clear()
        
        # Check if activities_data is empty
        if not self.activities_data:
            with self.grid_container:
                ui.label('No activities found. Import activities to view data.').classes(
                    'text-center text-gray-400 mt-20'
                )
            return

        # --- 1. PREPARE COLUMNS ---
        columns = [
            {'name': 'date', 'label': 'Date', 'field': 'date_sort', 'align': 'left', 'sortable': True},
            {'name': 'filename', 'label': 'Filename', 'field': 'filename', 'align': 'left', 'sortable': True},
            {'name': 'distance', 'label': 'Dist', 'field': 'distance', 'align': 'left', 'sortable': True},
            {'name': 'elevation', 'label': 'Elev', 'field': 'elevation', 'align': 'left', 'sortable': True},
            {'name': 'ef', 'label': 'Efficiency', 'field': 'ef', 'align': 'left', 'sortable': True},
            {'name': 'cost', 'label': 'Decoupling', 'field': 'cost', 'align': 'left', 'sortable': True},
            {'name': 'cadence', 'label': 'Cadence', 'field': 'cadence', 'align': 'left', 'sortable': True},
            {'name': 'actions', 'label': '', 'field': 'actions', 'align': 'center'},
        ]
        
        # --- 2. HELPER: SYSTEM TIMEZONE ---
        import time
        # Get local timezone abbreviation (e.g., EST, PDT)
        system_tz = time.tzname[time.daylight]

        # --- 3. TRANSFORM DATA ---
        rows = []
        for act in self.activities_data:
            cost = act.get('decoupling', 0)
            
            # --- DATE FORMATTING LOGIC ---
            raw_date = act.get('date', '') # e.g. "2026-02-22 17:06"
            try:
                dt = datetime.strptime(raw_date, '%Y-%m-%d %H:%M')
                
                # Manual formatting for cross-platform consistency
                hour = dt.strftime('%I').lstrip('0')
                minute = dt.strftime('%M')
                am_pm = dt.strftime('%p')
                month = dt.strftime('%m').lstrip('0')
                day = dt.strftime('%d').lstrip('0')
                year = dt.strftime('%Y')
                weekday = dt.strftime('%a')
                
                # FIXED: Added space before {system_tz}
                # Result: "Thu, 4/25/2026 5:06PM EDT"
                nice_date = f"{weekday}, {month}/{day}/{year} {hour}:{minute}{am_pm} {system_tz}"
            except:
                nice_date = raw_date

            rows.append({
                'date_sort': raw_date,      # ISO format for sorting
                'date_display': nice_date,  # Human format for display
                'filename': act.get('filename', '')[:30],
                'distance': f"{act.get('distance_mi', 0):.1f} mi",
                'elevation': f"{act.get('elevation_ft', 0)} ft",
                'ef': f"{act.get('efficiency_factor', 0):.2f}",
                'cost': f"{cost:.1f}%",
                'cost_value': cost,
                'cadence': f"{act.get('avg_cadence', 0)} spm",
                'hash': act.get('db_hash', ''),
                'full_filename': act.get('filename', ''),
            })
        
        # --- 4. RENDER TABLE ---
        with self.grid_container:
            with ui.card().classes('w-full bg-zinc-900 p-4 border-0').style('border-radius: 8px; overflow: hidden; box-shadow: 0px 4px 12px rgba(0, 0, 0, 0.4);'):
                table = ui.table(
                    columns=columns,
                    rows=rows,
                    row_key='hash'
                ).classes('w-full h-full')
                
                # --- SLOT: DATE (Two-line stack) ---
                table.add_slot('body-cell-date', '''
                    <q-td :props="props">
                        <div class="flex flex-col">
                            <span class="font-bold text-gray-200" style="font-size: 0.9rem;">
                                {{ props.row.date_display.split(', ')[0] }}, {{ props.row.date_display.split(', ')[1].split(' ')[0] }}
                            </span>
                            <span class="text-xs text-zinc-500 font-mono">
                                {{ props.row.date_display.split(' ')[2] }}{{ props.row.date_display.split(' ')[3] }} {{ props.row.date_display.split(' ')[4] }}
                            </span>
                        </div>
                    </q-td>
                ''')

                # --- SLOT: ACTIONS (With RESTORED CSS classes) ---
                table.add_slot('body-cell-actions', '''
                    <q-td :props="props">
                        <q-btn flat dense round icon="visibility" size="sm" class="view-btn" @click="$parent.$emit('view-row', props.row)"/>
                        <q-btn flat dense round icon="delete" size="sm" class="delete-btn" @click="$parent.$emit('delete-row', props.row)"/>
                    </q-td>
                ''')
                
                # --- SLOT: COST (Conditional Coloring) ---
                table.add_slot('body-cell-cost', '''
                    <q-td :props="props">
                        <span :style="props.row.cost_value > 5 ? 'color: #ff4d4d; font-weight: 600;' : 'color: #10B981; font-weight: 600;'">
                            {{ props.value }}
                        </span>
                    </q-td>
                ''')
                
                table.on('view-row', lambda e: self.open_activity_detail_modal(e.args['hash'], from_feed=True))
                table.on('delete-row', lambda e: self.delete_activity_inline(e.args['hash'], e.args['full_filename']))
                
                table.props('flat bordered dense dark')
                table.classes('bg-zinc-900 text-gray-200')

            # --- RESTORED CSS FOR HOVER EFFECTS ---
            ui.add_head_html('''
            <style>
            .q-table thead tr, .q-table tbody td {
                height: 48px;
            }
            .q-table th {
                background-color: #2a2a2a !important;
                color: #aaa !important;
                font-weight: 600 !important;
                border-bottom: 2px solid #444 !important;
            }
            .q-table td {
                color: #ddd !important;
                border-bottom: 1px solid #333 !important;
            }
            .q-table tbody tr:hover {
                background-color: #252525 !important;
            }
            /* RESTORED: Ghost button styling */
            .delete-btn.q-btn, .view-btn.q-btn {
                opacity: 0.3 !important;
                transition: all 0.2s ease !important;
            }
            .delete-btn.q-btn .q-icon, .view-btn.q-btn .q-icon {
                color: #9ca3af !important;
            }
            .delete-btn.q-btn:hover, .view-btn.q-btn:hover {
                opacity: 1 !important;
                background-color: rgba(255, 255, 255, 0.1) !important;
                transform: scale(1.1);
            }
            .delete-btn.q-btn:hover .q-icon, .view-btn.q-btn:hover .q-icon {
                color: #ffffff !important;
            }
            /* Disable Quasar's default hover overlay */
            .delete-btn.q-btn:before, .view-btn.q-btn:before {
                display: none !important;
            }
            </style>
            ''')
    
    async def delete_activity_inline(self, activity_hash, filename):
        """
        Delete activity from inline button click.
        
        This method:
        - Shows confirmation dialog
        - Deletes activity from database
        - Updates runs counter
        - Refreshes data view
        - Shows success notification
        
        Requirements: 5.4, 5.5, 5.6, 5.7, 5.8
        """
        # Show confirmation dialog
        result = await ui.run_javascript(
            f'confirm("Delete activity: {filename}?")', 
            timeout=10
        )
        
        # If confirmed, delete the activity
        if result:
            try:
                # Delete the activity
                self.db.delete_activity(activity_hash)
                
                # Update runs counter
                self.runs_count_label.text = f'{self.db.get_count()}'
                
                # Refresh all data views
                await self.refresh_data_view()
                
                # Show success notification
                ui.notify(f'Deleted: {filename[:30]}', type='positive')
                
            except Exception as e:
                ui.notify(f'Error deleting activity: {str(e)}', type='negative')
                print(f"Error in delete_activity_inline: {e}")
    
    async def delete_selected_activity(self):
        """
        Delete selected activity from grid.
        
        This method:
        - Gets selected row from AG Grid
        - Shows confirmation dialog
        - If confirmed, calls DatabaseManager.delete_activity()
        - Updates runs counter
        - Calls refresh_data_view()
        - Shows success notification
        
        Requirements: 5.4, 5.5, 5.6, 5.7, 5.8
        """
        # Get selected row from AG Grid
        selected = await self.activities_grid.get_selected_rows()
        
        # Check if a row is selected
        if not selected:
            ui.notify('No activity selected', type='warning')
            return
        
        # Get the activity hash from the selected row
        activity_hash = selected[0]['hash']
        activity_name = selected[0]['filename']
        
        # Show confirmation dialog
        result = await ui.run_javascript(
            f'confirm("Delete activity: {activity_name}?")', 
            timeout=10
        )
        
        # If confirmed, delete the activity
        if result:
            # Call DatabaseManager.delete_activity()
            self.db.delete_activity(activity_hash)
            
            # Update runs counter
            self.runs_count_label.text = f'{self.db.get_count()}'
            
            # Call refresh_data_view() to update all views
            await self.refresh_data_view()
            
            # Show success notification
            ui.notify('Activity deleted successfully', type='positive')
    
    def calculate_trend_stats(self, df_subset):
        """
        Calculate trend statistics for a DataFrame subset.
        
        Args:
            df_subset: DataFrame with 'date_obj' and 'efficiency_factor' columns
            
        Returns:
            tuple: (trend_msg, trend_color) - message string and hex color
        """
        if df_subset is None or df_subset.empty:
            return ("Trend: No Data", "silver")
        
        try:
            # Calculate linear regression trend for EF over time
            x_nums = (df_subset['date_obj'] - df_subset['date_obj'].min()).dt.total_seconds()
            y_ef = df_subset['efficiency_factor']
            from scipy.stats import linregress
            slope, intercept, r_value, p_value, std_err = linregress(x_nums, y_ef)
            
            # Determine trend message and color based on slope
            if slope > 0.0000001:
                trend_msg = f"📈 Trend: Engine Improving (+{slope*1e6:.2f} EF/day)"
                trend_color = "#2CC985"
            elif slope < -0.0000001:
                trend_msg = f"📉 Trend: Fitness Declining ({slope*1e6:.2f} EF/day)"
                trend_color = "#ff4d4d"
            else:
                trend_msg = "➡️ Trend: Fitness Stable"
                trend_color = "silver"
        except:
            trend_msg = "Trend: Insufficient Data"
            trend_color = "silver"
        
        return (trend_msg, trend_color)
    
    def classify_run_type(self, run_data, long_run_threshold):
        """
        Stackable classification system.
        Uses burst_count to distinguish true workouts from speed outliers.
        """
        tags = []
        
        # 1. Existing metric extraction
        distance_mi = run_data.get('distance_mi', 0)
        avg_hr = run_data.get('avg_hr', 0)
        max_hr = run_data.get('max_hr', 185)
        elevation_ft = run_data.get('elevation_ft', 0)
        avg_temp = run_data.get('avg_temp', 0)
        burst_count = run_data.get('burst_count', 0) # GET THE COUNT
        
        hr_ratio = avg_hr / max_hr if max_hr > 0 else 0
        grade = (elevation_ft / distance_mi) if distance_mi > 0 else 0
        
        # --- A. Primary Category ---
        if distance_mi > long_run_threshold: primary = "🦅 Long Run"
        elif distance_mi < 4.0 and hr_ratio < 0.75: primary = "🧘 Recovery"
        elif hr_ratio > 0.82: primary = "🔥 Tempo"
        elif hr_ratio > 0.75: primary = "🔷 Steady"
        else: primary = "🟡 Base"
        tags.append(primary)
        
        # --- B. Attributes ---
        if grade > 75: tags.append("⛰️ Hilly")
        
        # --- C. The New Interval Logic ---
        # Only tag as Intervals if we saw 3 or more sustained bursts
        if burst_count >= 4:
            tags.append("⚡ Intervals")
        elif burst_count >= 2:
            # Optional: Add a tag for short speed work like Strides
            tags.append("💨 Strides")
            
        if avg_temp and avg_temp > 25: tags.append("🥵 Hot")
        elif avg_temp and avg_temp < 5: tags.append("🥶 Cold")
        
        return " | ".join(tags)
    
    def generate_weekly_volume_chart(self):
        """
        Generate weekly volume chart. 
        Updates: 
        1. Passes 'Category Name' to click handler for better Modal Titles.
        2. Adds 'cursor-pointer' to the chart for better UX.
        """
        if self.df is None or self.df.empty:
            return None
        
        # ... (Data prep code remains the same until Chart Data construction) ...
        # [Use your existing data prep logic here]
        # Re-pasting the relevant data construction block for context:
        
        # Add week column
        self.df['week_start'] = self.df['date_obj'].dt.to_period('W-MON').dt.start_time
        
        # 1. PREPARE DATA
        split_data = []
        for activity in self.activities_data:
            date_obj = pd.to_datetime(activity.get('date'))
            week_start = date_obj.to_period('W-MON').start_time
            max_hr = activity.get('max_hr', 185)
            act_hash = activity.get('db_hash')
            act_date_str = date_obj.strftime('%-m/%-d')
            
            laps = activity.get('lap_data', [])
            if not laps:
                dist = activity.get('distance_mi', 0)
                if dist == 0: continue
                raw_cad = activity.get('avg_cadence', 0)
                cad = raw_cad * 2 if raw_cad < 100 else raw_cad
                laps = [{
                    'distance': dist * 1609.34, 'avg_cadence': raw_cad, 
                    'avg_hr': activity.get('avg_hr', 0), 'total_ascent': activity.get('elevation_ft', 0) / 3.28084
                }]

            for lap in laps:
                dist_m = lap.get('distance', 0)
                dist_mi = dist_m * 0.000621371
                if dist_mi < 0.1: continue
                raw_cad = lap.get('avg_cadence', 0)
                cadence = raw_cad * 2 if raw_cad < 120 else raw_cad
                hr = lap.get('avg_hr', 0)
                ascent = lap.get('total_ascent', 0) or 0
                grade = (ascent / dist_m) * 100 if dist_m > 0 else 0
                category = classify_split(cadence, hr, max_hr, grade)
                split_data.append({
                    'week_start': week_start, 'distance': dist_mi, 'category': category,
                    'hash': act_hash, 'date_str': act_date_str
                })

        if not split_data:
            return None
            
        df_splits = pd.DataFrame(split_data)
        
        # Verdict Data Storage
        weekly_vol = df_splits.groupby(['week_start', 'category'])['distance'].sum().unstack(fill_value=0)
        for col in ['HIGH QUALITY', 'STRUCTURAL', 'BROKEN']:
            if col not in weekly_vol.columns: weekly_vol[col] = 0
        self.weekly_volume_data = weekly_vol
        
        # 2. AGGREGATE FOR CHART
        grouped = df_splits.groupby(['week_start', 'category'])
        weeks = sorted(df_splits['week_start'].unique())
        categories = ['STRUCTURAL', 'HIGH QUALITY', 'BROKEN']
        
        chart_data = {cat: {'y': [], 'customdata': []} for cat in categories}
        week_labels = []
        
        for week in weeks:
            sunday = week - pd.Timedelta(days=1)
            saturday = week + pd.Timedelta(days=5)
            week_labels.append(f"{sunday.strftime('%b %-d')} - {saturday.strftime('%-d')}")
            
            for cat in categories:
                try:
                    cell = grouped.get_group((week, cat))
                    total_dist = cell['distance'].sum()
                    unique_acts = cell[['hash', 'date_str']].drop_duplicates()
                    
                    dates_list = unique_acts['date_str'].tolist()
                    if len(dates_list) > 3:
                        date_display = ", ".join(dates_list[:3]) + f" (+{len(dates_list)-3})"
                    else:
                        date_display = ", ".join(dates_list)
                    
                    hash_list = unique_acts['hash'].tolist()
                    chart_data[cat]['y'].append(total_dist)
                    
                    # UPDATED CUSTOMDATA: Added 'cat' (Category Name) at index 2
                    chart_data[cat]['customdata'].append([date_display, json.dumps(hash_list), cat])
                    
                except KeyError:
                    chart_data[cat]['y'].append(0)
                    chart_data[cat]['customdata'].append(["", "[]", cat])

        # 3. BUILD FIGURE
        fig = go.Figure()
        bar_style = dict(opacity=0.85, marker_line=dict(width=1, color='rgba(255,255,255,0.1)'))
        
        def add_trace(cat, name, color, desc, rank):
            fig.add_trace(go.Bar(
                x=week_labels, y=chart_data[cat]['y'], name=name, marker_color=color,
                customdata=chart_data[cat]['customdata'],
                hovertemplate=(
                    '<b>%{y:.1f} mi</b><br>'
                    '<span style="color: #cbd5e1; font-size: 12px;">' + desc + '</span><br>'
                    '<span style="color: rgba(255,255,255,0.8); font-size: 11px;">Runs: %{customdata[0]}</span>'
                    '<extra></extra>'
                ),
                
                hoverlabel=dict(font=dict(color='white'), bordercolor='white'),
                legendrank=rank, 
                **bar_style
            ))

        add_trace('STRUCTURAL', 'Structural', '#3b82f6', 'Valid Base/Hills', 1)
        add_trace('HIGH QUALITY', 'High Quality', '#10b981', 'Dialed Mechanics', 2)
        add_trace('BROKEN', 'Broken', '#f43f5e', 'Mechanical Failure', 3)
        
        fig.update_layout(
            barmode='stack', template='plotly_dark',
            paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
            height=300, margin=dict(l=40, r=20, t=20, b=80),
            showlegend=True, legend=dict(orientation="h", y=1.02, x=1, xanchor="right", traceorder="normal"),
            bargap=0.35, xaxis=dict(tickangle=0, showgrid=True, gridcolor='rgba(255,255,255,0.05)'),
            yaxis=dict(title='Miles', showgrid=True, gridcolor='rgba(255,255,255,0.05)'),
            clickmode='event'
        )
        fig.update_layout(modebar={'remove': ['zoom', 'pan', 'select', 'lasso2d', 'zoomIn', 'zoomOut', 'autoScale', 'resetScale', 'toImage']})
        
        return fig
    
    def generate_efficiency_decoupling_chart(self):
        """Generate Running Efficiency vs. Aerobic Decoupling chart."""
        if self.df is None or self.df.empty:
            return None
        
        # Calculate trend
        try:
            from scipy.stats import linregress
            x_nums = (self.df['date_obj'] - self.df['date_obj'].min()).dt.total_seconds()
            y_ef = self.df['efficiency_factor']
            slope, intercept, r_value, p_value, std_err = linregress(x_nums, y_ef)
            
            trend_msg, trend_color = self.calculate_trend_stats(self.df)
        except:
            slope = 0
            intercept = 0
            trend_msg = "Insufficient data"
            trend_color = "#888888"
        
        # Create figure with secondary y-axis
        fig = make_subplots(specs=[[{"secondary_y": True}]])
        
        # Separate positive and negative decoupling
        pos_d = self.df['decoupling'].copy()
        pos_d[pos_d < 0] = 0
        neg_d = self.df['decoupling'].copy()
        neg_d[neg_d > 0] = 0
        
        # Add decoupling filled areas
        fig.add_trace(
            go.Scatter(
                x=self.df['date_obj'], y=neg_d, name="Stable Zone",
                fill='tozeroy', mode='lines', line=dict(width=0),
                fillcolor='rgba(0, 128, 128, 0.2)', hoverinfo='skip', showlegend=False
            ), secondary_y=True
        )
        
        fig.add_trace(
            go.Scatter(
                x=self.df['date_obj'], y=pos_d, name="Cost Zone",
                fill='tozeroy', mode='lines',
                line=dict(color='rgba(255, 77, 77, 0.5)', width=1),
                fillcolor='rgba(255, 77, 77, 0.1)', hoverinfo='skip', showlegend=False
            ), secondary_y=True
        )
        
        # Add EF line
        fig.add_trace(
            go.Scatter(
                x=self.df['date_obj'], y=self.df['efficiency_factor'],
                name="EF Trend", mode='lines',
                line=dict(color='rgba(150, 150, 150, 0.3)', width=2, shape='spline'),
                hoverinfo='skip', showlegend=False
            ), secondary_y=False
        )
        
        # Add trend line
        if slope != 0:
            x_min = self.df['date_obj'].min().to_pydatetime()
            x_max = self.df['date_obj'].max().to_pydatetime()
            x_nums_for_line = [0, (self.df['date_obj'].max() - self.df['date_obj'].min()).total_seconds()]
            y_trend_line = [intercept + slope * x for x in x_nums_for_line]
            
            fig.add_trace(
                go.Scatter(
                    x=[x_min, x_max], y=y_trend_line, name="Fitness Trend",
                    mode='lines', line=dict(color=trend_color, width=2, dash='dash'),
                    opacity=0.6, hoverinfo='skip', showlegend=False
                ), secondary_y=False
            )
        
        # Add performance markers (simplified - just the data points)
        fig.add_trace(
            go.Scatter(
                x=self.df['date_obj'],
                y=self.df['efficiency_factor'],
                mode='markers',
                marker=dict(size=8, color='#10B981', line=dict(width=1, color='white')),
                name='Runs',
                hovertemplate='EF: %{y:.2f}<extra></extra>'
            ), secondary_y=False
        )
        
        # Update layout
        fig.update_layout(
            template='plotly_dark',
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            height=400,
            margin=dict(l=40, r=40, t=20, b=40),
            showlegend=False,
            hovermode='closest',
            hoverlabel=dict(bgcolor='#18181b', font_color='white')
        )
        
        fig.update_yaxes(
            title_text="Running Efficiency",
            color="#10B981",
            secondary_y=False,
            showgrid=True,
            gridcolor='rgba(255, 255, 255, 0.1)'
        )
        
        fig.update_yaxes(
            title_text="Decoupling (%)",
            color="#ff4d4d",
            secondary_y=True,
            range=[-5, max(20, self.df['decoupling'].max()+2)],
            showgrid=False
        )
        
        fig.update_xaxes(
            showgrid=True,
            gridcolor='rgba(255, 255, 255, 0.1)'
        )
        
        # Hide modebar
        fig.update_layout(
            modebar={'remove': ['zoom', 'pan', 'select', 'lasso2d', 'zoomIn', 'zoomOut', 'autoScale', 'resetScale', 'toImage']}
        )
        
        return fig
    
    def generate_cadence_trend_chart(self):
        """Generate cadence trend chart with form status in tooltip."""
        if self.df is None or self.df.empty:
            return None
        
        fig = go.Figure()
        
        # Calculate form status for each run
        form_verdicts = []
        form_colors = []
        
        for idx, row in self.df.iterrows():
            # FIX: Use correct field names from Garmin FIT standard
            form = analyze_form(
                row.get('avg_cadence', 0),
                row.get('avg_stance_time', 0),
                row.get('avg_step_length', 0),
                row.get('avg_vertical_oscillation', 0)
            )
            form_verdicts.append(form['verdict'])
            
            # Extract hex color
            color_class = form['color']
            if 'emerald' in color_class:
                form_colors.append('#10b981')
            elif 'blue' in color_class:
                form_colors.append('#3b82f6')
            elif 'yellow' in color_class:
                form_colors.append('#eab308')
            elif 'orange' in color_class:
                form_colors.append('#f97316')
            elif 'red' in color_class:
                form_colors.append('#ef4444')
            else:
                form_colors.append('#71717a')
        
        # Add cadence scatter (Restored: Blue Dotted Line + Colored Markers)
        fig.add_trace(
            go.Scatter(
                x=self.df['date_obj'],
                y=self.df['avg_cadence'],
                mode='markers+lines',
                marker=dict(
                    size=8, 
                    color=form_colors, # Data-driven colors
                    line=dict(width=1, color='white')
                ),
                # RESTORED: The crisp indigo dotted line
                line=dict(color='#6366f1', width=2, dash='dot'), 
                name='Cadence',
                customdata=list(zip(form_verdicts, form_colors, self.df['date_obj'].dt.strftime('%b %d, %Y'))),
                hovertemplate=(
                    '<span style="color:#a1a1aa; font-size:11px;">%{customdata[2]}</span><br>'
                    '<b>%{y:.0f} SPM</b><br>'
                    '<span style="color:%{customdata[1]};">●</span> %{customdata[0]}'
                    '<extra></extra>'
                )
            )
        )
        
        # Update layout
        fig.update_layout(
            template='plotly_dark',
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            height=300,
            margin=dict(l=40, r=20, t=20, b=40),
            showlegend=False,
            hovermode='closest',
            hoverlabel=dict(bgcolor='#18181b', font_color='white'),
            yaxis=dict(
                title='Cadence (SPM)',
                showgrid=True,
                gridcolor='rgba(255, 255, 255, 0.1)'
            ),
            xaxis=dict(
                showgrid=True,
                gridcolor='rgba(255, 255, 255, 0.1)'
            )
        )
        
        # Hide modebar
        fig.update_layout(
            modebar={'remove': ['zoom', 'pan', 'select', 'lasso2d', 'zoomIn', 'zoomOut', 'autoScale', 'resetScale', 'toImage']}
        )
        
        return fig
    
    def generate_plotly_figure(self):
        """
        Generate Plotly figure with stacked graphs (preserved logic from gui.py).
        
        This method creates an interactive Plotly dashboard with:
        - Linear regression trend for EF over time
        - Trend message and color based on slope
        - Marker colors based on EF and decoupling logic
        - Cadence colors based on cadence thresholds
        - Two stacked subplots with shared x-axis
        - Decoupling filled areas (red for positive, teal for negative)
        - EF line trace with colored markers
        - Cadence scatter trace
        - Dark theme layout
        - Static HTML annotation legend at bottom (y=-0.15)
        
        Returns:
            plotly.graph_objects.Figure: The generated figure object
        
        Requirements: 3.3, 3.4, 3.5, 3.6, 3.7, 3.8, 10.1, 10.12, 10.13, 13.5, 13.7
        """
        if self.df is None or self.df.empty:
            return None
        
        # --- 1. Smart Trend Logic (Linear Regression) ---
        try:
            # Calculate linear regression trend for EF over time
            x_nums = (self.df['date_obj'] - self.df['date_obj'].min()).dt.total_seconds()
            y_ef = self.df['efficiency_factor']
            from scipy.stats import linregress
            slope, intercept, r_value, p_value, std_err = linregress(x_nums, y_ef)
            
            # Determine trend message and color based on slope
            if slope > 0.0000001:
                trend_msg = f"📈 Trend: Engine Improving (+{slope*1e6:.2f} EF/day)"
                trend_color = "#2CC985"
            elif slope < -0.0000001:
                trend_msg = f"📉 Trend: Fitness Declining ({slope*1e6:.2f} EF/day)"
                trend_color = "#ff4d4d"
            else:
                trend_msg = "➡️ Trend: Fitness Stable"
                trend_color = "silver"
        except:
            trend_msg = "Trend: Insufficient Data"
            trend_color = "silver"
        
        # --- 2. Calculate long run threshold and classify runs ---
        # Long run threshold: 80th percentile of distance, fallback to 10 miles if < 5 runs
        if len(self.df) >= 5:
            long_run_threshold = self.df['distance_mi'].quantile(0.8)
        else:
            long_run_threshold = 10.0
        
        # Calculate mean EF and split data into 4 performance groups
        ef_mean = self.df['efficiency_factor'].mean()
        
        # Split data into 4 groups based on EF and decoupling
        df_green = self.df[(self.df['efficiency_factor'] >= ef_mean) & (self.df['decoupling'] <= 5)]  # Peak Efficiency
        df_yellow = self.df[(self.df['efficiency_factor'] < ef_mean) & (self.df['decoupling'] <= 5)]  # Base Maintenance
        df_orange = self.df[(self.df['efficiency_factor'] >= ef_mean) & (self.df['decoupling'] > 5)]  # Expensive Speed
        df_red = self.df[(self.df['efficiency_factor'] < ef_mean) & (self.df['decoupling'] > 5)]  # Struggling
        
        # --- 3. Calculate cadence colors based on cadence thresholds ---
        cad_colors = []
        for c in self.df['avg_cadence']:
            if c >= 170: 
                cad_colors.append('#10B981')  # Modern emerald - Efficient
            elif c >= 160: 
                cad_colors.append('#e6e600')  # Yellow - OK
            else: 
                cad_colors.append('#ff4d4d')  # Red - Sloppy
        
        # --- 4. Create subplots with make_subplots (3 rows, shared x-axis) ---
        fig = make_subplots(
            rows=3, cols=1, 
            shared_xaxes=True,
            vertical_spacing=0.08,
            subplot_titles=("Weekly Volume Composition", "Running Efficiency vs. Aerobic Decoupling", "Mechanics (Cadence Trend)"),
            specs=[[{"secondary_y": False}], [{"secondary_y": True}], [{"secondary_y": False}]],
            row_heights=[0.22, 0.48, 0.30]
        )
        
        # === ROW 1: WEEKLY VOLUME QUALITY ===
        
        # Add week column (Monday start)
        self.df['week_start'] = self.df['date_obj'].dt.to_period('W-MON').dt.start_time
        
        # Classify each run's form quality
        def classify_quality(row):
            cadence = row.get('avg_cadence', 0)
            gct = row.get('avg_gct', 0)
            stride = row.get('avg_stride', 0)
            bounce = row.get('avg_bounce', 0)
            
            form_result = analyze_form(cadence, gct, stride, bounce)
            verdict = form_result.get('verdict', '')
            
            if verdict in ['GOOD FORM', 'ELITE FORM']:
                return 'Dialed In'
            else:
                return 'Garbage Miles'
        
        self.df['quality'] = self.df.apply(classify_quality, axis=1)
        
        # Aggregate by week and quality
        weekly_data = self.df.groupby(['week_start', 'quality'])['distance_mi'].sum().unstack(fill_value=0)
        
        # Ensure both columns exist
        if 'Dialed In' not in weekly_data.columns:
            weekly_data['Dialed In'] = 0
        if 'Garbage Miles' not in weekly_data.columns:
            weekly_data['Garbage Miles'] = 0
        
        # Add "Dialed In" bars (bottom stack - green)
        fig.add_trace(go.Bar(
            x=weekly_data.index,
            y=weekly_data['Dialed In'],
            name='Dialed In',
            marker_color='#10b981',
            hovertemplate='%{y:.1f} mi<extra></extra>',
            showlegend=False,
            legendgroup='volume'
        ), row=1, col=1)
        
        # Add "Garbage Miles" bars (top stack - red)
        fig.add_trace(go.Bar(
            x=weekly_data.index,
            y=weekly_data['Garbage Miles'],
            name='Garbage Miles',
            marker_color='#f43f5e',
            hovertemplate='%{y:.1f} mi<extra></extra>',
            showlegend=False,
            legendgroup='volume'
        ), row=1, col=1)
        
        # === ROW 2: DURABILITY ===
        
        # --- 5. Add decoupling filled areas (red for positive, teal for negative) ---
        # Separate positive and negative decoupling values
        pos_d = self.df['decoupling'].copy()
        pos_d[pos_d < 0] = 0  # Zero out negative values
        neg_d = self.df['decoupling'].copy()
        neg_d[neg_d > 0] = 0  # Zero out positive values
        
        # Add teal filled area for negative decoupling (stable zone)
        fig.add_trace(
            go.Scatter(
                x=self.df['date_obj'], 
                y=neg_d, 
                name="Stable Zone", 
                fill='tozeroy', 
                mode='lines', 
                line=dict(width=0), 
                fillcolor='rgba(0, 128, 128, 0.2)',  # Teal
                hoverinfo='skip', 
                showlegend=False
            ), 
            row=2, col=1, secondary_y=True
        )
        
        # Add red filled area for positive decoupling (cost zone)
        fig.add_trace(
            go.Scatter(
                x=self.df['date_obj'], 
                y=pos_d, 
                name="Cost Zone", 
                fill='tozeroy', 
                mode='lines', 
                line=dict(color='rgba(255, 77, 77, 0.5)', width=1), 
                fillcolor='rgba(255, 77, 77, 0.1)',  # Red
                hoverinfo='skip', 
                showlegend=False
            ), 
            row=2, col=1, secondary_y=True
        )
        
        # --- 6. Add grey line trace for overall EF path (background) ---
        fig.add_trace(
            go.Scatter(
                x=self.df['date_obj'], 
                y=self.df['efficiency_factor'], 
                name="EF Trend Line", 
                mode='lines',
                line=dict(color='rgba(150, 150, 150, 0.3)', width=2, shape='spline'),
                hoverinfo='skip',
                showlegend=False
            ), 
            row=2, col=1, secondary_y=False
        )
        
        # --- 6.5. Add linear regression trend line (color-coded by trend direction) ---
        # Calculate trend line endpoints (convert Timestamp to datetime for JSON serialization)
        x_min = self.df['date_obj'].min().to_pydatetime()
        x_max = self.df['date_obj'].max().to_pydatetime()
        x_nums_for_line = [0, (self.df['date_obj'].max() - self.df['date_obj'].min()).total_seconds()]
        y_trend_line = [intercept + slope * x for x in x_nums_for_line]
        
        fig.add_trace(
            go.Scatter(
                x=[x_min, x_max],
                y=y_trend_line,
                name="Fitness Trend",
                mode='lines',
                line=dict(color=trend_color, width=2, dash='dash'),
                opacity=0.6,
                hoverinfo='skip',
                showlegend=False
            ),
            row=2, col=1, secondary_y=False
        )
        
        # --- 7. Add 4 separate marker traces for each performance category ---
        performance_groups = [
            (df_green, '#10B981', 'Peak Efficiency (Fast & Stable)', 'Peak Efficiency 🟢', 'High speed/output relative to HR'),
            (df_yellow, '#e6e600', 'Base Maintenance (Slow & Stable)', 'Base Maintenance 🟡', 'Building aerobic base'),
            (df_orange, '#ff9900', 'Expensive Speed (Fast but Drifted)', 'Expensive Speed 🟠', 'Fast but unsustainable'),
            (df_red, '#ff4d4d', 'Struggling (Slow & Drifted)', 'Struggling 🔴', 'Fatigue or high internal load')
        ]
        
        for df_group, color, legend_name, verdict, insight in performance_groups:
            if not df_group.empty:
                # Generate run type classifications for this group
                run_types = []
                max_hrs = []
                max_speeds = []
                elevations = []
                
                for idx, row in df_group.iterrows():
                    # Find the corresponding activity data
                    activity = next((a for a in self.activities_data if a.get('date') == row.get('date')), None)
                    if activity:
                        run_type = self.classify_run_type(activity, long_run_threshold)
                        run_types.append(run_type)
                        max_hrs.append(activity.get('max_hr', 0))
                        max_speeds.append(activity.get('max_speed_mph', 0))
                        elevations.append(activity.get('elevation_ft', 0))
                    else:
                        run_types.append("Unknown")
                        max_hrs.append(0)
                        max_speeds.append(0)
                        elevations.append(0)
                
                fig.add_trace(
                    go.Scatter(
                        x=df_group['date_obj'], 
                        y=df_group['efficiency_factor'], 
                        name=legend_name,
                        mode='markers',
                        marker=dict(
                            size=12, 
                            color=color,
                            line=dict(width=2, color='white')
                        ),
                        customdata=list(zip(
                            run_types,  # 0: Run type classification
                            [verdict] * len(df_group),  # 1: Performance verdict
                            df_group['decoupling'],  # 2: Decoupling %
                            df_group['pace'],  # 3: Pace
                            df_group['distance_mi'],  # 4: Distance
                            df_group['avg_hr'],  # 5: Avg HR
                            max_hrs,  # 6: Max HR
                            max_speeds,  # 7: Max Speed
                            elevations,  # 8: Elevation gain
                            [insight] * len(df_group),  # 9: Insight text
                            [color] * len(df_group)  # 10: Dot color
                        )),
                        hovertemplate=(
                            "<b style='font-size:14px'>%{customdata[0]}</b><br>"
                            "<span style='color:%{customdata[10]};'>●</span> <i style='color:#aaa'>%{customdata[9]}</i><br>"
                            "<span style='color:#444'>──────────────────</span><br>"
                            "<b>Efficiency:</b> %{y:.2f} | <b>Decoupling:</b> %{customdata[2]:.1f}%<br>"
                            "<b>Distance:</b> %{customdata[4]} mi @ %{customdata[3]}<br>"
                            "<b>Avg HR:</b> %{customdata[5]} bpm<br>"
                            "<span style='color:#444'>──────────────────</span><br>"
                            "📈 <b>Max HR:</b> %{customdata[6]} bpm<br>"
                            "🚀 <b>Max Speed:</b> %{customdata[7]:.1f} mph<br>"
                            "⛰️ <b>Elev Gain:</b> %{customdata[8]} ft<extra></extra>"
                        )
                    ), 
                    row=2, col=1, secondary_y=False
                )
        
        # === BOTTOM GRAPH: CADENCE ===
        
        # --- 8. Add cadence scatter trace with physics-based insights ---
        # Generate cadence insights based on SPM values (biomechanical feedback)
        cadence_insights = []
        for c in self.df['avg_cadence']:
            if c < 160:
                cadence_insights.append("High Impact / Overstriding Risk")
            elif c < 170:
                cadence_insights.append("Moderate Efficiency / Muscle Driven")
            elif c <= 185:
                cadence_insights.append("Optimal Efficiency / Low Impact")
            else:
                cadence_insights.append("High Turnover / Sprinter Style")
        
        fig.add_trace(
            go.Scatter(
                x=self.df['date_obj'], 
                y=self.df['avg_cadence'],
                name="Cadence", 
                mode='lines+markers',
                line=dict(color='#888', width=1, dash='dot'),
                marker=dict(
                    size=8, 
                    color=cad_colors,  # Color-coded by cadence thresholds
                    line=dict(width=1, color='white')
                ),
                customdata=list(zip(
                    cadence_insights,  # 0: Cadence insight text
                    cad_colors  # 1: Dot color
                )),
                hovertemplate=(
                    "Cadence: <b>%{y} spm</b><br>"
                    "<span style='color:%{customdata[1]};'>●</span> <i style='color:#aaa'>%{customdata[0]}</i>"
                    "<extra></extra>"
                ),
                showlegend=False  # Hide from legend (bottom subplot is self-explanatory)
            ), 
            row=3, col=1
        )
        
        # --- 9. Configure layout with dark theme and native legend ---
        # Get initial trend for title
        trend_msg, trend_color = self.calculate_trend_stats(self.df)
        
        fig.update_layout(
            title=dict(
                text=(
                    f"<b>Training Trends</b><br>"
                    f"<span style='font-size: 14px; color: {trend_color};' id='trend-subtitle'>{trend_msg}</span>"
                ),
            ),
            template="plotly_dark",
            height=900, 
            showlegend=True,
            legend=dict(
                orientation="v",  # Vertical layout
                yanchor="top",
                y=0.70,  # Position on row 2 (middle section)
                xanchor="right",
                x=0.99,  # Position at right edge
                bgcolor="rgba(26, 26, 26, 0.85)",  # Semi-transparent dark background
                bordercolor="rgba(255, 255, 255, 0.1)",  # Subtle border
                borderwidth=1,
                font=dict(size=11)
            ),
            margin=dict(l=60, r=20, t=100, b=40),  # Reduced right margin to eliminate gap
            hoverlabel=dict(
                bgcolor="#1F1F1F", 
                bordercolor="#444", 
                font_color="white"
            ),
            modebar={
                'remove': ['toImage', 'select2d', 'lasso2d', 'autoScale2d', 'resetScale2d', 'zoom', 'pan', 'zoomIn', 'zoomOut', 'autoScale', 'resetScale']
            },
            dragmode='zoom',
            hovermode='closest'
        )
        
        # Axis Styling
        fig.update_yaxes(
            title_text="[Gains] Running Efficiency", 
            color="#10B981",  # Modern emerald green
            row=2, col=1, secondary_y=False,
            showgrid=True,
            gridcolor='rgba(255, 255, 255, 0.1)'
        )
        fig.update_yaxes(
            title_text="Decoupling (%)", 
            color="#ff4d4d", 
            row=2, col=1, secondary_y=True, 
            range=[-5, max(20, self.df['decoupling'].max()+2)],
            showgrid=False
        )
        
        # Row 1 Y-axis (Weekly Volume)
        fig.update_yaxes(
            title_text="Miles",
            color="white",
            row=1, col=1,
            showgrid=True,
            gridcolor='rgba(255, 255, 255, 0.1)'
        )
        
        # Row 3 Y-axis (Cadence)
        fig.update_yaxes(
            title_text="Cadence (spm)", 
            color="white", 
            row=3, col=1,
            showgrid=True,
            gridcolor='rgba(255, 255, 255, 0.1)'
        )
        
        # Update layout to enable stacked bars for row 1
        fig.update_layout(barmode='stack')
        
        # X-axis styling - show dates on all subplots
        fig.update_xaxes(
            row=1, col=1,
            showticklabels=True,  # Show dates on top graph
            showgrid=True,
            gridcolor='rgba(255, 255, 255, 0.1)'
        )
        fig.update_xaxes(
            row=2, col=1,
            showticklabels=True,  # Show dates on middle graph
            showgrid=True,
            gridcolor='rgba(255, 255, 255, 0.1)'
        )
        fig.update_xaxes(
            row=3, col=1,
            showticklabels=True,  # Show dates on bottom graph
            showgrid=True,
            gridcolor='rgba(255, 255, 255, 0.1)'
        )
        
        # --- 10. Return figure object ---
        return fig
    
    def calculate_volume_verdict(self, df=None, start_index=None, end_index=None):
        """
        Verdict based on 'Broken Ratio' (Red Miles / Total Miles).
        Returns: HIGH QUALITY, STRUCTURAL, or BROKEN badge.
        """
        if self.weekly_volume_data is None:
            return 'N/A', '#71717a', 'bg-zinc-700'
            
        data = self.weekly_volume_data
        
        # Zoom Logic
        if start_index is not None and end_index is not None:
            start = max(0, int(round(start_index)))
            end = min(len(data), int(round(end_index)) + 1)
            if start < len(data) and end > start:
                data = data.iloc[start:end]
        
        # Calculate Sums using new key 'BROKEN'
        total_broken = data['BROKEN'].sum() if 'BROKEN' in data.columns else 0
        total_vol = data.sum().sum()
        
        if total_vol == 0: return 'N/A', '#71717a', 'bg-zinc-700'
        
        broken_ratio = (total_broken / total_vol) * 100
        
        # Verdict Thresholds (Matching the Legend Terms)
        if broken_ratio < 10:
            return 'HIGH QUALITY', '#10b981', 'bg-emerald-500/20'
        elif broken_ratio < 25:
            return 'STRUCTURAL', '#3b82f6', 'bg-blue-500/20'
        else:
            return 'BROKEN', '#ef4444', 'bg-red-500/20'
    
    def calculate_cadence_verdict(self, df):
        """
        Calculate verdict for Mechanics chart based on 'Form Score' Analysis.
        Returns: ELITE (90+), GOOD (60+), or BROKEN (<60).
        """
        if df is None or df.empty or len(df) < 2:
            return 'N/A', '#71717a', 'bg-zinc-700'
        
        # 1. Convert Form Verdicts to Numeric Scores
        scores = []
        dates = []
        
        for idx, row in df.iterrows():
            form = analyze_form(
                row.get('avg_cadence', 0),
                row.get('avg_stance_time', 0),
                row.get('avg_step_length', 0),
                row.get('avg_vertical_oscillation', 0)
            )
            verdict = form['verdict']
            
            # Scoring Logic (Weighted Average)
            if verdict == 'ELITE FORM':
                scores.append(100)
                dates.append(row['date_obj'])
            elif verdict == 'GOOD FORM':
                scores.append(80)
                dates.append(row['date_obj'])
            elif verdict in ['HIKING / REST', 'AEROBIC / MIXED']:
                # Skip structural miles - don't penalize strategy
                continue 
            elif verdict in ['PLODDING', 'LOW CADENCE']:
                scores.append(40)
                dates.append(row['date_obj'])
            elif verdict in ['OVERSTRIDING', 'HEAVY FEET', 'INEFFICIENT']:
                scores.append(0)
                dates.append(row['date_obj'])
                
        # If no valid running data (only hiking), return Neutral
        if not scores:
            return 'STRUCTURAL', '#3b82f6', 'bg-blue-500/20'
            
        # 2. Calculate Average Quality (The "State")
        avg_score = sum(scores) / len(scores)
        
        # 3. Calculate Trend (The "Trajectory")
        try:
            from scipy.stats import linregress
            # Convert dates to seconds for regression
            x_nums = [(d - min(dates)).total_seconds() for d in dates]
            slope, _, _, _, _ = linregress(x_nums, scores)
            # Normalize slope to "Points per week"
            slope_week = slope * 604800
        except:
            slope_week = 0

        # --- FINAL VERDICT LOGIC (Professor Grading Scale) ---
        
        # Scenario A: ELITE (Must be >= 90)
        # Your 83.75 will fail this check.
        if avg_score >= 90:
            if slope_week < -10: 
                return 'SLIPPING', '#fbbf24', 'bg-yellow-500/20'
            return 'ELITE', '#10b981', 'bg-emerald-500/20'
            
        # Scenario B: GOOD (60 - 89)
        # Your 83.75 lands here.
        elif avg_score >= 60:
            if slope_week > 10:
                return 'IMPROVING', '#10b981', 'bg-emerald-500/20'
            elif slope_week < -10:
                return 'SLIPPING', '#fbbf24', 'bg-yellow-500/20'
            return 'GOOD', '#3b82f6', 'bg-blue-500/20'
            
        # Scenario C: BROKEN (< 60)
        else:
            if slope_week > 10:
                return 'IMPROVING', '#3b82f6', 'bg-blue-500/20'
            return 'BROKEN', '#ef4444', 'bg-red-500/20'
    
    def calculate_efficiency_verdict(self, df):
        """
        Calculate verdict for Efficiency chart using hierarchical logic with tolerance buffers.
        
        Rule #1: Check Decoupling Slope (Safety First) - with 0.2 tolerance
        Rule #2: Check EF Slope (Stale Check) - with 0.1 tolerance
        Rule #3: Both good = SOLID
        
        Returns:
            tuple: (verdict_text, verdict_color, verdict_bg)
        """
        if df is None or df.empty or len(df) < 2:
            return 'N/A', '#71717a', 'bg-zinc-700'
        
        try:
            from scipy.stats import linregress
            x_nums = (df['date_obj'] - df['date_obj'].min()).dt.total_seconds()
            
            # Calculate Decoupling slope
            y_dec = df['decoupling']
            slope_dec, _, _, _, _ = linregress(x_nums, y_dec)
            slope_dec_per_week = slope_dec * 604800
            
            # Calculate EF slope
            y_ef = df['efficiency_factor']
            slope_ef, _, _, _, _ = linregress(x_nums, y_ef)
            slope_ef_per_week = (slope_ef * 604800) * 100
            
            # Rule #1: Safety First - Check Decoupling (with 0.2 tolerance)
            if slope_dec_per_week > 0.2:
                return 'GARBAGE', '#ef4444', 'bg-red-500/20'
            
            # Rule #2: Stale Check - Check EF (with 0.1 tolerance)
            if slope_ef_per_week < -0.1:
                return 'MEH', '#3b82f6', 'bg-blue-500/20'
            
            # Rule #3: The Win
            return 'SOLID', '#10b981', 'bg-emerald-500/20'
        except:
            return 'N/A', '#71717a', 'bg-zinc-700'
    
    def update_trends_chart(self):
        """
        Update Plotly chart in trends tab with zoom-responsive trend analysis.
        
        This method:
        - Clears plotly_container
        - Checks if df is not None and not empty
        - If data exists, calls generate_plotly_figure()
        - Creates ui.plotly() with figure and binds zoom event
        - Updates trend subtitle dynamically via JavaScript
        - If no data, shows placeholder message
        
        Requirements: 3.1, 3.2
        """
        # Clear plotly_container
        self.plotly_container.clear()
        
        # Check if df is not None and not empty
        with self.plotly_container:
            if self.df is not None and not self.df.empty:
                # Calculate trend stats for full dataset
                try:
                    from scipy.stats import linregress
                    x_nums = (self.df['date_obj'] - self.df['date_obj'].min()).dt.total_seconds()
                    y_ef = self.df['efficiency_factor']
                    slope, intercept, r_value, p_value, std_err = linregress(x_nums, y_ef)
                    trend_msg, trend_color = self.calculate_trend_stats(self.df)
                    
                    # Calculate stats for the stats bar
                    r_squared = r_value ** 2
                    # EF slope: convert to percentage change per week
                    # slope is in EF units per second, multiply by seconds in week, then by 100 for %
                    slope_pct_per_week = (slope * 604800) * 100
                except:
                    trend_msg = None
                    trend_color = "#888888"
                    r_squared = 0
                    slope_pct_per_week = 0
                
                # Determine badge color
                badge_color = 'grey'
                if trend_msg:
                    if 'Improving' in trend_msg or 'Building' in trend_msg:
                        badge_color = 'green'
                    elif 'Declining' in trend_msg or 'Regressing' in trend_msg:
                        badge_color = 'orange'
                    elif 'Stable' in trend_msg:
                        badge_color = 'grey'
                
                # 1. Weekly Volume Composition Card
                volume_fig = self.generate_weekly_volume_chart()
                if volume_fig:
                    # Calculate volume verdict
                    vol_verdict, vol_color, vol_bg = self.calculate_volume_verdict(self.df)
                    
                    # ADDED 'relative' class to the card so we can pin the icon
                    with ui.card().classes('w-full bg-zinc-900 border border-zinc-800 p-6 mb-8 relative').style('border-radius: 12px; box-shadow: 0 10px 15px -3px rgba(0, 0, 0, 0.3);'):
                        
                        # The Silicon Valley Info Icon (ABSOLUTE POSITIONED)
                        # top-5 right-5 puts it slightly higher/wider than the p-6 content padding
                        ui.icon('help_outline').classes('absolute top-5 right-5 text-zinc-500 hover:text-white transition-colors duration-200 cursor-pointer text-2xl').on(
                            'click', lambda: self.show_volume_info()
                        )

                        # Header Row (Now simpler, just Title + Badge)
                        with ui.row().classes('w-full items-center gap-3 mb-1'):
                            ui.label('Training Volume').classes('text-xl font-bold text-white')
                            self.volume_verdict_label = ui.label(f'[ {vol_verdict} ]').classes(f'text-sm font-bold px-3 py-1 rounded {vol_bg}').style(f'color: {vol_color};')
                            
                        ui.label('Breakdown in quality of miles (click any section to inspect runs)').classes('text-sm text-zinc-400 mb-4')
                        
                        # Chart with zoom binding
                        self.volume_chart = ui.plotly(volume_fig).classes('w-full').style('cursor: pointer')
                        self.volume_chart.on('plotly_relayout', self.handle_volume_zoom)
                        self.volume_chart.on('plotly_click', self.handle_bar_click)
                
                # 2. Efficiency & Decoupling Card (with unified dual-metric stats bar)
                efficiency_fig = self.generate_efficiency_decoupling_chart()
                if efficiency_fig:
                    # Calculate efficiency verdict
                    eff_verdict, eff_color, eff_bg = self.calculate_efficiency_verdict(self.df)
                    
                    with ui.card().classes('w-full bg-zinc-900 border border-zinc-800 p-6 mb-8').style('border-radius: 12px; box-shadow: 0 10px 15px -3px rgba(0, 0, 0, 0.3);'):
                        # Header with verdict badge
                        with ui.row().classes('w-full items-center gap-3 mb-1'):
                            ui.label('Aerobic Efficiency').classes('text-xl font-bold text-white')
                            self.efficiency_verdict_label = ui.label(f'[ {eff_verdict} ]').classes(f'text-sm font-bold px-3 py-1 rounded {eff_bg}').style(f'color: {eff_color};')
                        ui.label('Running efficiency vs. cardiovascular drift over time').classes('text-sm text-zinc-400 mb-4')
                        
                        # Calculate consistency text for EF
                        consistency_text = 'High' if r_squared > 0.7 else 'Moderate' if r_squared > 0.4 else 'Volatile'
                        
                        # Unified Dual-Metric Stats Bar
                        with ui.card().classes('w-full bg-zinc-800 p-4 mb-4').style('border-radius: 8px;'):
                            with ui.row().classes('w-full items-stretch gap-0'):
                                # Section A: Efficiency Factor (EF) - Left
                                with ui.column().classes('flex-1 gap-1'):
                                    ui.label('EFFICIENCY TREND').classes('text-xs text-zinc-400 font-bold tracking-wider')
                                    with ui.row().classes('items-center gap-2'):
                                        # Arrow icon based on slope direction
                                        ef_arrow = '↗' if slope_pct_per_week > 0 else '↘' if slope_pct_per_week < 0 else '→'
                                        # Color: Green for positive (improving), Red for negative
                                        ef_color = '#10b981' if slope_pct_per_week > 0 else '#ef4444' if slope_pct_per_week < 0 else '#71717a'
                                        self.ef_arrow_label = ui.label(ef_arrow).classes('text-3xl').style(f'color: {ef_color};')
                                        self.ef_trend_value_label = ui.label(f'{slope_pct_per_week:+.2f}% / week').classes('text-xl font-bold text-white')
                                    self.ef_consistency_label = ui.label(f'Consistency: {consistency_text} (R² = {r_squared:.2f})').classes('text-xs text-zinc-500')
                                
                                # Vertical Divider
                                ui.element('div').classes('h-full mx-4').style('width: 1px; background-color: #52525b;')
                                
                                # Section B: Aerobic Decoupling - Right
                                with ui.column().classes('flex-1 gap-1'):
                                    ui.label('DECOUPLING TREND').classes('text-xs text-zinc-400 font-bold tracking-wider')
                                    
                                    # Calculate decoupling trend (negative slope is good - less drift)
                                    try:
                                        x_nums_dec = (self.df['date_obj'] - self.df['date_obj'].min()).dt.total_seconds()
                                        y_dec = self.df['decoupling']
                                        slope_dec, intercept_dec, r_value_dec, p_value_dec, std_err_dec = linregress(x_nums_dec, y_dec)
                                        # Decoupling is already in % units, so just multiply by seconds in week
                                        # slope is in percentage points per second
                                        slope_dec_per_week = slope_dec * 604800
                                    except:
                                        slope_dec_per_week = 0
                                        r_value_dec = 0
                                    
                                    with ui.row().classes('items-center gap-2'):
                                        # Arrow icon based on slope direction
                                        dec_arrow = '↗' if slope_dec_per_week > 0 else '↘' if slope_dec_per_week < 0 else '→'
                                        # Color: Green for negative (improving - less drift), Red for positive
                                        dec_color = '#10b981' if slope_dec_per_week < 0 else '#ef4444' if slope_dec_per_week > 0 else '#71717a'
                                        self.dec_arrow_label = ui.label(dec_arrow).classes('text-3xl').style(f'color: {dec_color};')
                                        self.dec_trend_value_label = ui.label(f'{slope_dec_per_week:+.2f}% / week').classes('text-xl font-bold text-white')
                                    
                                    corr_text = 'Strong' if abs(r_value_dec) > 0.7 else 'Moderate' if abs(r_value_dec) > 0.4 else 'Weak'
                                    self.dec_correlation_label = ui.label(f'Correlation: {corr_text} (r = {r_value_dec:.2f})').classes('text-xs text-zinc-500')
                        
                        # Chart with zoom binding
                        self.efficiency_chart = ui.plotly(efficiency_fig).classes('w-full')
                        self.efficiency_chart.on('plotly_relayout', self.handle_efficiency_zoom)
                
                # 3. Cadence Trend Card
                cadence_fig = self.generate_cadence_trend_chart()
                if cadence_fig:
                    # Calculate cadence verdict
                    cad_verdict, cad_color, cad_bg = self.calculate_cadence_verdict(self.df)
                    
                    with ui.card().classes('w-full bg-zinc-900 border border-zinc-800 p-6 mb-8').style('border-radius: 12px; box-shadow: 0 10px 15px -3px rgba(0, 0, 0, 0.3);'):
                        # Header with verdict badge
                        with ui.row().classes('w-full items-center gap-3 mb-1'):
                            ui.label('Running Mechanics').classes('text-xl font-bold text-white')
                            self.cadence_verdict_label = ui.label(f'[ {cad_verdict} ]').classes(f'text-sm font-bold px-3 py-1 rounded {cad_bg}').style(f'color: {cad_color};')
                        ui.label('Cadence trend showing turnover consistency').classes('text-sm text-zinc-400 mb-4')
                        
                        # Chart with zoom binding
                        self.cadence_chart = ui.plotly(cadence_fig).classes('w-full')
                        self.cadence_chart.on('plotly_relayout', self.handle_cadence_zoom)
            else:
                # If no data, show placeholder message
                ui.label('No data available. Import activities to view trends.').classes(
                    'text-center text-gray-400 mt-20'
                )
    
    async def handle_efficiency_zoom(self, e):
        """
        Handle zoom events on the Efficiency chart and update stats dynamically.
        
        When user zooms into a date range, recalculate stats for that subset.
        """
        try:
            from scipy.stats import linregress
            
            # Check if this is a zoom event (has xaxis.range)
            if 'xaxis.range[0]' in e.args and 'xaxis.range[1]' in e.args:
                # Get the zoomed date range
                start_date = pd.to_datetime(e.args['xaxis.range[0]'])
                end_date = pd.to_datetime(e.args['xaxis.range[1]'])
                
                # Filter dataframe to zoomed range
                df_zoomed = self.df[
                    (self.df['date_obj'] >= start_date) & 
                    (self.df['date_obj'] <= end_date)
                ]
                
                if len(df_zoomed) < 2:
                    return
                
                # Recalculate EF stats for zoomed data
                x_nums = (df_zoomed['date_obj'] - df_zoomed['date_obj'].min()).dt.total_seconds()
                y_ef = df_zoomed['efficiency_factor']
                slope, intercept, r_value, p_value, std_err = linregress(x_nums, y_ef)
                
                r_squared = r_value ** 2
                slope_pct_per_week = (slope * 604800) * 100
                
                # Recalculate Decoupling stats for zoomed data
                y_dec = df_zoomed['decoupling']
                slope_dec, intercept_dec, r_value_dec, p_value_dec, std_err_dec = linregress(x_nums, y_dec)
                slope_dec_per_week = slope_dec * 604800  # Already in % units
                
            elif 'xaxis.autorange' in e.args:
                # User double-clicked to reset zoom - use full dataset
                x_nums = (self.df['date_obj'] - self.df['date_obj'].min()).dt.total_seconds()
                y_ef = self.df['efficiency_factor']
                slope, intercept, r_value, p_value, std_err = linregress(x_nums, y_ef)
                
                r_squared = r_value ** 2
                slope_pct_per_week = (slope * 604800) * 100
                
                # Recalculate Decoupling stats for full dataset
                y_dec = self.df['decoupling']
                slope_dec, intercept_dec, r_value_dec, p_value_dec, std_err_dec = linregress(x_nums, y_dec)
                slope_dec_per_week = slope_dec * 604800  # Already in % units
            else:
                return
            
            # Update EF stats
            self.ef_trend_value_label.set_text(f'{slope_pct_per_week:+.2f}% / week')
            consistency_text = 'High' if r_squared > 0.7 else 'Moderate' if r_squared > 0.4 else 'Volatile'
            self.ef_consistency_label.set_text(f'Consistency: {consistency_text} (R² = {r_squared:.2f})')
            
            # Update EF arrow and color
            ef_arrow = '↗' if slope_pct_per_week > 0 else '↘' if slope_pct_per_week < 0 else '→'
            ef_color = '#10b981' if slope_pct_per_week > 0 else '#ef4444' if slope_pct_per_week < 0 else '#71717a'
            self.ef_arrow_label.set_text(ef_arrow)
            self.ef_arrow_label.style(f'color: {ef_color};')
            
            # Update Decoupling stats
            self.dec_trend_value_label.set_text(f'{slope_dec_per_week:+.2f}% / week')
            corr_text = 'Strong' if abs(r_value_dec) > 0.7 else 'Moderate' if abs(r_value_dec) > 0.4 else 'Weak'
            self.dec_correlation_label.set_text(f'Correlation: {corr_text} (r = {r_value_dec:.2f})')
            
            # Update Decoupling arrow and color
            dec_arrow = '↗' if slope_dec_per_week > 0 else '↘' if slope_dec_per_week < 0 else '→'
            dec_color = '#10b981' if slope_dec_per_week < 0 else '#ef4444' if slope_dec_per_week > 0 else '#71717a'
            self.dec_arrow_label.set_text(dec_arrow)
            self.dec_arrow_label.style(f'color: {dec_color};')
            
            # Update Efficiency Verdict
            df_for_verdict = df_zoomed if 'xaxis.range[0]' in e.args else self.df
            eff_verdict, eff_color, eff_bg = self.calculate_efficiency_verdict(df_for_verdict)
            self.efficiency_verdict_label.set_text(f'[ {eff_verdict} ]')
            self.efficiency_verdict_label.classes(f'text-sm font-bold px-3 py-1 rounded {eff_bg}', remove='bg-emerald-500/20 bg-blue-500/20 bg-red-500/20 bg-zinc-700')
            self.efficiency_verdict_label.style(f'color: {eff_color};')
                
        except Exception as ex:
            # Silently catch errors
            pass
    
    def handle_volume_zoom(self, e):
        """
        Handle zoom events on Volume Chart (Categorical Axis).
        Translates index ranges to data slices.
        """
        try:
            if 'xaxis.range[0]' in e.args and 'xaxis.range[1]' in e.args:
                # Get indices (float)
                idx_start = e.args['xaxis.range[0]']
                idx_end = e.args['xaxis.range[1]']
                
                # Recalculate with slice
                vol_verdict, vol_color, vol_bg = self.calculate_volume_verdict(
                    start_index=idx_start, 
                    end_index=idx_end
                )
            else:
                # Reset / Autorange
                vol_verdict, vol_color, vol_bg = self.calculate_volume_verdict()
            
            # Update UI
            self.volume_verdict_label.set_text(f'[ {vol_verdict} ]')
            self.volume_verdict_label.classes(f'text-sm font-bold px-3 py-1 rounded {vol_bg}', remove='bg-emerald-500/20 bg-blue-500/20 bg-red-500/20 bg-zinc-700 bg-zinc-800 text-zinc-500')
            self.volume_verdict_label.style(f'color: {vol_color};')
            
        except Exception as ex:
            print(f"Volume Zoom Error: {ex}")
    
    async def handle_cadence_zoom(self, e):
        """
        Handle zoom events on the Cadence chart and update verdict dynamically.
        """
        try:
            # Check if this is a zoom event (has xaxis.range)
            if 'xaxis.range[0]' in e.args and 'xaxis.range[1]' in e.args:
                # Get the zoomed date range
                start_date = pd.to_datetime(e.args['xaxis.range[0]'])
                end_date = pd.to_datetime(e.args['xaxis.range[1]'])
                
                # Filter dataframe to zoomed range
                df_zoomed = self.df[
                    (self.df['date_obj'] >= start_date) & 
                    (self.df['date_obj'] <= end_date)
                ]
                
                if len(df_zoomed) < 2:
                    return
                
                # Recalculate verdict for zoomed data
                cad_verdict, cad_color, cad_bg = self.calculate_cadence_verdict(df_zoomed)
                
            elif 'xaxis.autorange' in e.args:
                # User double-clicked to reset zoom - use full dataset
                cad_verdict, cad_color, cad_bg = self.calculate_cadence_verdict(self.df)
            else:
                return
            
            # Update verdict label
            self.cadence_verdict_label.set_text(f'[ {cad_verdict} ]')
            self.cadence_verdict_label.classes(f'text-sm font-bold px-3 py-1 rounded {cad_bg}', remove='bg-emerald-500/20 bg-blue-500/20 bg-red-500/20 bg-zinc-700')
            self.cadence_verdict_label.style(f'color: {cad_color};')
                
        except Exception as ex:
            # Silently catch errors
            pass
    
    async def handle_chart_zoom(self, e):
        """
        Handle zoom events on the Plotly chart and update trend analysis.
        
        When user zooms into a date range, recalculate the trend for that subset.
        """
        try:
            # Check if this is a zoom event (has xaxis.range)
            if 'xaxis.range[0]' in e.args and 'xaxis.range[1]' in e.args:
                # Get the zoomed date range
                start_date = pd.to_datetime(e.args['xaxis.range[0]'])
                end_date = pd.to_datetime(e.args['xaxis.range[1]'])
                
                # Filter dataframe to zoomed range
                df_zoomed = self.df[
                    (self.df['date_obj'] >= start_date) & 
                    (self.df['date_obj'] <= end_date)
                ]
                
                # Recalculate trend for zoomed data
                trend_msg, trend_color = self.calculate_trend_stats(df_zoomed)
                
                # Calculate new trend line for zoomed range
                x_nums = (df_zoomed['date_obj'] - df_zoomed['date_obj'].min()).dt.total_seconds()
                y_ef = df_zoomed['efficiency_factor']
                from scipy.stats import linregress
                slope, intercept, r_value, p_value, std_err = linregress(x_nums, y_ef)
                
                # Calculate trend line endpoints
                x_min = df_zoomed['date_obj'].min().to_pydatetime()
                x_max = df_zoomed['date_obj'].max().to_pydatetime()
                x_nums_for_line = [0, (df_zoomed['date_obj'].max() - df_zoomed['date_obj'].min()).total_seconds()]
                y_trend_line = [intercept + slope * x for x in x_nums_for_line]
                
            elif 'xaxis.autorange' in e.args:
                # User double-clicked to reset zoom - use full dataset
                trend_msg, trend_color = self.calculate_trend_stats(self.df)
                
                # Calculate trend line for full dataset
                x_nums = (self.df['date_obj'] - self.df['date_obj'].min()).dt.total_seconds()
                y_ef = self.df['efficiency_factor']
                from scipy.stats import linregress
                slope, intercept, r_value, p_value, std_err = linregress(x_nums, y_ef)
                
                # Calculate trend line endpoints
                x_min = self.df['date_obj'].min().to_pydatetime()
                x_max = self.df['date_obj'].max().to_pydatetime()
                x_nums_for_line = [0, (self.df['date_obj'].max() - self.df['date_obj'].min()).total_seconds()]
                y_trend_line = [intercept + slope * x for x in x_nums_for_line]
            else:
                return
            
            # Update the chart title subtitle and trend line using Plotly's relayout
            new_title = (
                f"<b>Training Trends</b><br>"
                f"<span style='font-size: 14px; color: {trend_color};'>{trend_msg}</span>"
            )
            
            # Format dates and y-values for JavaScript (convert to ISO strings)
            x_min_iso = x_min.isoformat()
            x_max_iso = x_max.isoformat()
            
            await ui.run_javascript(f'''
                const plotDiv = document.querySelector('.js-plotly-plot');
                if (plotDiv) {{
                    // Update title
                    Plotly.relayout(plotDiv, {{'title.text': `{new_title}`}});
                    
                    // Update trend line (trace index 3 - after teal fill, red fill, and grey EF line)
                    Plotly.restyle(plotDiv, {{
                        'x': [['{x_min_iso}', '{x_max_iso}']],
                        'y': [[{y_trend_line[0]}, {y_trend_line[1]}]],
                        'line.color': ['{trend_color}']
                    }}, [3]);
                }}
            ''')
                
        except Exception as ex:
            # Silently catch timeout errors - the update still works
            pass
    
    async def export_csv(self):
        """
        Export activities to CSV with AI context - saves to Downloads folder.
        
        This method:
        - Checks if df is None
        - Exports DataFrame to CSV
        - Appends data dictionary section
        - Saves to Downloads folder
        - Shows success notification with file path
        
        Requirements: 8.1, 8.2, 8.3, 8.4, 8.5, 8.6, 13.6
        """
        # Check if df is None
        if self.df is None or self.df.empty:
            ui.notify('No data to export', type='warning')
            return
        
        try:
            # Export DataFrame to CSV with specified columns
            export_columns = [
                'date', 'filename', 'distance_mi', 'pace', 'gap_pace', 
                'efficiency_factor', 'decoupling', 'avg_hr', 'avg_resp', 
                'avg_temp', 'avg_power', 'avg_cadence', 'hrr_list', 
                'v_ratio', 'gct_balance', 'gct_change', 'elevation_ft', 
                'moving_time_min', 'rest_time_min'
            ]
            
            # Filter to only include columns that exist in the DataFrame
            available_columns = [col for col in export_columns if col in self.df.columns]
            
            # Export to CSV string
            csv_content = self.df[available_columns].to_csv(index=False)
            
            # Append data dictionary section
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
"""
            
            # Combine CSV and data dictionary
            full_content = csv_content + data_dictionary
            
            # Generate filename with timestamp
            from datetime import datetime
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"garmin_analysis_{timestamp}.csv"
            
            # Save to Downloads folder
            downloads_path = os.path.expanduser("~/Downloads")
            file_path = os.path.join(downloads_path, filename)
            
            # Write file
            with open(file_path, 'w') as f:
                f.write(full_content)
            
            # Show success notification with file path
            ui.notify(f'CSV saved! (check your Downloads folder)', type='positive', timeout=5000)
            print(f"CSV exported to: {file_path}")
            
        except Exception as e:
            # Handle export failure
            ui.notify(f'Error exporting CSV: {str(e)}', type='negative')
            print(f"Error in export_csv: {e}")
            import traceback
            traceback.print_exc()
    
    async def save_chart_to_downloads(self):
        """
        Save all Trends charts as a single combined PNG to Downloads folder.
        
        This method:
        - Shows loading dialog
        - Captures all 3 Plotly charts with titles added
        - Combines them vertically into a single PNG
        - Saves to ~/Downloads with timestamp
        - Shows success notification
        """
        # Show loading dialog
        loading_dialog = ui.dialog()
        with loading_dialog, ui.card().classes('bg-zinc-900 p-6').style('min-width: 300px; box-shadow: none;'):
            with ui.column().classes('items-center gap-4'):
                ui.spinner(size='lg', color='emerald')
                ui.label('Generating charts...').classes('text-lg text-white')
        
        loading_dialog.open()
        await asyncio.sleep(0.1)  # Let dialog render
        
        try:
            # Check if we have chart data
            if self.df is None or self.df.empty:
                loading_dialog.close()
                ui.notify('No chart data to save', type='warning')
                return
            
            # Generate the three charts with titles
            volume_fig = self.generate_weekly_volume_chart()
            efficiency_fig = self.generate_efficiency_decoupling_chart()
            cadence_fig = self.generate_cadence_trend_chart()
            
            if not volume_fig or not efficiency_fig or not cadence_fig:
                loading_dialog.close()
                ui.notify('Error generating charts', type='warning')
                return
            
            # Adjust efficiency chart colors for better static export appearance
            # Make the red decoupling fill less bright/more subtle
            efficiency_fig.data[1].fillcolor = 'rgba(180, 50, 50, 0.15)'  # Darker, more muted red
            efficiency_fig.data[1].line.color = 'rgba(180, 50, 50, 0.4)'
            
            # Add titles to each chart for export (left-aligned, larger text, better positioning)
            volume_fig.update_layout(
                title=dict(
                    text='<b>Training Volume</b><br><span style="font-size:14px; color:#a1a1aa;">Breakdown of quality miles vs. garbage miles</span>',
                    font=dict(size=24, color='white'),
                    x=0.02,
                    xanchor='left',
                    y=0.96,
                    yanchor='top'
                ),
                margin=dict(t=160, l=60, r=20, b=60)
            )
            
            efficiency_fig.update_layout(
                title=dict(
                    text='<b>Aerobic Efficiency</b><br><span style="font-size:14px; color:#a1a1aa;">Running efficiency vs. cardiovascular drift over time</span>',
                    font=dict(size=24, color='white'),
                    x=0.02,
                    xanchor='left',
                    y=0.96,
                    yanchor='top'
                ),
                margin=dict(t=160, l=60, r=60, b=60)
            )
            
            cadence_fig.update_layout(
                title=dict(
                    text='<b>Running Mechanics</b><br><span style="font-size:14px; color:#a1a1aa;">Cadence trend showing turnover consistency</span>',
                    font=dict(size=24, color='white'),
                    x=0.02,
                    xanchor='left',
                    y=0.96,
                    yanchor='top'
                ),
                margin=dict(t=160, l=60, r=20, b=60)
            )
            
            # Convert to images using Plotly's built-in export (run in thread to avoid blocking)
            import plotly.io as pio
            from PIL import Image
            import io
            
            def export_charts():
                # Export each chart as PNG bytes (increased height to accommodate titles)
                volume_img_bytes = pio.to_image(volume_fig, format='png', width=1200, height=520, scale=2)
                efficiency_img_bytes = pio.to_image(efficiency_fig, format='png', width=1200, height=620, scale=2)
                cadence_img_bytes = pio.to_image(cadence_fig, format='png', width=1200, height=520, scale=2)
                
                # Open as PIL Images
                volume_img = Image.open(io.BytesIO(volume_img_bytes))
                efficiency_img = Image.open(io.BytesIO(efficiency_img_bytes))
                cadence_img = Image.open(io.BytesIO(cadence_img_bytes))
                
                return [volume_img, efficiency_img, cadence_img]
            
            # Run export in background thread
            images = await run.io_bound(export_charts)
            
            # Calculate total height and max width, add spacing between charts
            spacing = 30  # pixels between charts
            total_height = sum(img.height for img in images) + (spacing * (len(images) - 1))
            max_width = max(img.width for img in images)
            
            # Create combined image with dark background
            combined = Image.new('RGB', (max_width, total_height), '#0f0f0f')
            
            # Paste images vertically with spacing
            y_offset = 0
            for img in images:
                combined.paste(img, (0, y_offset))
                y_offset += img.height + spacing
            
            # Generate filename with timestamp
            from datetime import datetime
            timestamp = datetime.now().strftime("%Y-%m-%d_%H%M%S")
            filename = f"Garmin_Trends_{timestamp}.png"
            
            # Save to Downloads folder
            downloads_path = os.path.expanduser("~/Downloads")
            file_path = os.path.join(downloads_path, filename)
            
            combined.save(file_path, 'PNG')
            
            # Close loading dialog and show notification
            loading_dialog.close()
            
            # Show success notification immediately after closing dialog
            ui.notify(
                f'Charts saved to Downloads! ({filename})',
                type='positive',
                position='top',
                timeout=5000
            )
            print(f"Combined chart saved to: {file_path}")
            
        except Exception as e:
            # Close loading dialog on error
            loading_dialog.close()
            
            # Handle save failure
            ui.notify(f'Error saving charts: {str(e)}', type='negative', position='top')
            print(f"Error in save_chart_to_downloads: {e}")
            import traceback
            traceback.print_exc()
    
    async def handle_bar_click(self, e):
        """
        Handle click on Volume Bar.
        Fixes:
        1. Pace Bug: Calculates pace from duration/distance (Fail-safe).
        2. Routing: Forces the 'Nice Modal' (from_feed=True).
        """
        try:
            point_data = e.args['points'][0]
            custom_data = point_data.get('customdata', [])
            
            if not custom_data or len(custom_data) < 3:
                return
                
            hash_list = json.loads(custom_data[1])
            category_raw = custom_data[2]
            
            if not hash_list: return
            
            # Scenario A: Single Run -> Open Nice Modal
            if len(hash_list) == 1:
                await self.open_activity_detail_modal(hash_list[0], from_feed=True)
                return

            # Scenario B: Selector Menu
            style_map = {
                'HIGH QUALITY': ('text-emerald-400', 'bg-emerald-500/20', 'border-emerald-500/30'),
                'STRUCTURAL': ('text-blue-400', 'bg-blue-500/20', 'border-blue-500/30'),
                'BROKEN': ('text-red-400', 'bg-red-500/20', 'border-red-500/30')
            }
            txt_col, bg_col, border_col = style_map.get(category_raw, ('text-zinc-400', 'bg-zinc-800', 'border-zinc-700'))
            category_label = category_raw.replace('_', ' ').title()

            async def pick_run(h, dlg):
                dlg.close()
                await asyncio.sleep(0.1)
                await self.open_activity_detail_modal(h, from_feed=True) # <--- Force Nice Modal

            with ui.dialog() as selector_dialog, ui.card().classes('bg-zinc-900 border border-zinc-800 p-0 min-w-[340px] shadow-2xl shadow-black'):
                
                # Header
                with ui.row().classes('w-full items-center justify-between p-4 border-b border-zinc-800 bg-zinc-900/50'):
                    with ui.column().classes('gap-2'):
                        ui.label('Inspect Runs').classes('text-xs font-bold text-zinc-500 uppercase tracking-wider')
                        ui.label(f'[ {category_label} ]').classes(f'text-sm font-bold px-3 py-1 rounded-md border {txt_col} {bg_col} {border_col}')
                
                # Scrollable List
                with ui.element('div').classes('w-full max-h-[350px] overflow-y-auto'):
                    with ui.column().classes('w-full gap-0'): 
                        for h in hash_list:
                            act = self.db.get_activity_by_hash(h)
                            if act:
                                date_obj = pd.to_datetime(act.get('date'))
                                nice_date = date_obj.strftime('%a, %-m/%-d')
                                dist = act.get('distance_mi', 0)
                                
                                # --- THE PACE FIX ---
                                # Calculate manually to avoid 0:00 errors
                                duration_min = act.get('moving_time_min', 0)
                                if dist > 0 and duration_min > 0:
                                    pace = duration_min / dist
                                    pace_fmt = f"{int(pace)}:{int((pace % 1) * 60):02d}"
                                else:
                                    # Fallback to DB string if calc fails
                                    pace_fmt = act.get('pace', '--:--')
                                # --------------------
                                
                                with ui.item().classes('w-full p-4 hover:bg-zinc-800 transition-colors border-b border-zinc-800/50 group') \
                                        .props('clickable v-ripple') \
                                        .on('click', lambda h=h: pick_run(h, selector_dialog)):
                                    
                                    with ui.row().classes('w-full justify-between items-center'):
                                        with ui.row().classes('items-center gap-3'):
                                            ui.label(nice_date).classes('text-zinc-300 font-medium text-sm w-20')
                                            with ui.row().classes('items-center gap-1'):
                                                ui.label('⏱️').classes('text-xs opacity-60')
                                                ui.label(f'{pace_fmt}/mi').classes('text-zinc-500 text-xs')

                                        with ui.row().classes('items-center gap-3'):
                                            with ui.row().classes('items-center gap-1'):
                                                ui.label('🏃‍♂️').classes('text-sm')
                                                ui.label(f'{dist:.1f} mi').classes('text-white font-bold text-sm')
                                            ui.icon('chevron_right').classes('text-zinc-600 group-hover:text-white transition-colors text-sm')

                # Ghost Button Footer
                with ui.row().classes('w-full p-3 bg-zinc-900/80 backdrop-blur-sm border-t border-zinc-800'):
                    ui.button('CANCEL', on_click=selector_dialog.close).classes(
                        'w-full bg-transparent border border-zinc-700 text-zinc-400 font-bold text-sm tracking-wide '
                        'hover:text-white hover:border-zinc-500 hover:bg-zinc-800 transition-all duration-200 rounded-lg'
                    )
            
            selector_dialog.open()
                
        except Exception as ex:
            print(f"Click Error: {ex}")

    async def copy_to_llm(self):
        """
        Copy activity data to clipboard with LLM context.
        
        Uses run.io_bound to keep WebSocket alive during heavy FIT parsing.
        
        Requirements: 9.1, 9.2, 9.3, 9.4, 9.5, 9.6, 13.6
        """
        # Check if we have data
        if not self.activities_data:
            ui.notify('No data to copy', type='warning')
            return
        
        # Show loading dialog immediately (client is alive)
        self.copy_loading_dialog.open()
        await asyncio.sleep(0.05)  # Let dialog render
        
        try:
            # Check if more than 20 activities (summary mode)
            if len(self.activities_data) > 20:
                # Summary mode: group by month and aggregate
                self.df['month'] = pd.to_datetime(self.df['date']).dt.to_period('M')
                monthly = self.df.groupby('month').agg({
                    'distance_mi': 'sum',
                    'efficiency_factor': 'mean',
                    'decoupling': 'mean',
                    'avg_cadence': 'mean',
                    'filename': 'count'
                }).round(2)
                
                # Format monthly summary
                report_text = "=== MONTHLY TRAINING SUMMARY ===\n\n"
                for month, row in monthly.iterrows():
                    report_text += f"{month}:\n"
                    report_text += f"  Runs: {row['filename']}\n"
                    report_text += f"  Total Distance: {row['distance_mi']:.1f} mi\n"
                    report_text += f"  Avg EF: {row['efficiency_factor']:.2f}\n"
                    report_text += f"  Avg Decoupling: {row['decoupling']:.1f}%\n"
                    report_text += f"  Avg Cadence: {row['avg_cadence']:.0f} spm\n\n"
            else:
                # Detailed mode: Parse FIT files using run.cpu_bound (keeps WebSocket alive!)
                avg_ef = self.df['efficiency_factor'].mean() if self.df is not None else 0
                
                # Sort activities by date
                sorted_activities = sorted(self.activities_data, 
                                          key=lambda x: x.get('date', ''), 
                                          reverse=True)
                
                # Prepare file paths for parsing
                activity_info = []
                for activity in sorted_activities:
                    activity_hash = activity.get('db_hash')
                    if activity_hash:
                        db_activity = self.db.get_activity_by_hash(activity_hash)
                        if db_activity:
                            fit_file_path = self._locate_fit_file(db_activity)
                            if fit_file_path:
                                activity_info.append({
                                    'path': fit_file_path,
                                    'activity': activity
                                })
                
                # Run heavy parsing in separate thread (WebSocket stays alive!)
                lap_splits_list = await run.io_bound(
                    _parse_fit_files_for_clipboard,
                    activity_info
                )
                
                # Build report with lap splits (client context is STILL alive!)
                report_lines = []
                for activity, lap_splits in zip(sorted_activities, lap_splits_list):
                    # Add activity summary
                    report_lines.append(self.format_run_data(activity, avg_ef))
                    
                    # Add lap splits if available
                    if lap_splits:
                        report_lines.append("\n[LAP SPLITS]")
                        for lap in lap_splits:
                            lap_num = lap.get('lap_number', 0)
                            distance_mi = lap.get('distance', 0) * 0.000621371
                            pace = lap.get('actual_pace', '--:--')
                            hr = int(lap['avg_hr']) if lap.get('avg_hr') else '--'
                            # Cadence: double it (Garmin stores one-foot cadence), handle None
                            raw_cadence = lap.get('avg_cadence')
                            if raw_cadence and raw_cadence > 0:
                                cadence = int(raw_cadence * 2)
                            else:
                                cadence = '--'
                            elev_ft = lap.get('total_ascent', 0) * 3.28084 if lap.get('total_ascent') else 0
                            
                            report_lines.append(
                                f"{lap_num} | {distance_mi:.2f}mi | {pace}/mi | {hr}bpm | {cadence}spm | +{int(elev_ft)}ft"
                            )
                        report_lines.append("")
                    
                    report_lines.append('=' * 50)
                
                report_text = '\n'.join(report_lines)
            
            # Append LLM context dictionary
            llm_context = """

=== CONTEXT FOR AI COACHING ===

I'm sharing my running data for analysis. Please help me understand:
1. Training trends and patterns
2. Areas for improvement
3. Signs of overtraining or fatigue
4. Recommendations for future training

KEY METRICS EXPLAINED:

EFFICIENCY FACTOR (EF):
- Measures aerobic efficiency (normalized speed / heart rate)
- Higher = better aerobic fitness
- Typical: 0.8-1.5 recreational, 1.5-2.5+ elite
- Improving EF = getting faster at same effort

AEROBIC DECOUPLING:
- Measures aerobic durability (% efficiency loss over run)
- Lower = better endurance
- <5%: Excellent, 5-10%: Moderate, >10%: High fatigue
- High decoupling = cardiovascular drift, need more base training

HEART RATE RECOVERY (HRR):
- BPM drop 60 seconds after peak efforts
- Higher = better cardiovascular fitness
- >30: Excellent, 20-30: Good, <20: Poor/fatigued

TRAINING ZONES:
- 🟢 Green (Peak Efficiency): High EF + Low Decoupling = Your output (Speed) was high relative to your input (Heart Rate)
- 🟡 Yellow (Base Maintenance): Low EF + Low Decoupling = Building base
- 🟠 Orange (Expensive Speed): High EF + High Decoupling = Fast but unsustainable
- 🔴 Red (Struggling): Low EF + High Decoupling = Fatigue/overtraining

FORM METRICS:
- Cadence: Target 170-180 spm for efficiency
- Vertical Ratio: Lower = less wasted vertical motion
- GCT Balance: Target 50/50 left/right symmetry
"""
            
            # Combine report and context
            clipboard_content = report_text + llm_context
            
            # Copy to clipboard (client context is STILL alive!)
            try:
                import pyperclip
                pyperclip.copy(clipboard_content)
                
                # Close loading dialog (safe now!)
                self.copy_loading_dialog.close()
                
                # Show success notification with modern styling
                ui.notify(
                    '✅ Data copied to clipboard!', 
                    type='positive',
                    position='top',
                    close_button=True
                )
                
                print("✅ Data successfully copied to clipboard!")
                
            except ImportError:
                self.copy_loading_dialog.close()
                ui.notify('Please install pyperclip: pip3 install pyperclip', type='negative')
                print("❌ Error: pyperclip not installed. Run: pip3 install pyperclip")
            
        except Exception as e:
            # Close loading dialog on error
            self.copy_loading_dialog.close()
            ui.notify(f'Error: {str(e)}', type='negative')
            print(f"❌ Error copying to clipboard: {str(e)}")
            import traceback
            traceback.print_exc()


# --- STANDALONE FUNCTION FOR IO-BOUND PARSING ---
def _parse_fit_files_for_clipboard(activity_info_list):
    """
    Parse FIT files to extract lap data (runs in separate thread via run.io_bound).
    
    This runs in a ThreadPool to keep the WebSocket heartbeat alive during
    heavy file I/O operations.
    
    Args:
        activity_info_list: List of dicts with 'path' and 'activity' keys
        
    Returns:
        List of enhanced lap data (or None for each activity)
    """
    import fitparse
    import pandas as pd
    from analyzer import minetti_cost_of_running
    
    results = []
    
    for info in activity_info_list:
        try:
            fit_file_path = info['path']
            
            # Parse FIT file
            fitfile = fitparse.FitFile(fit_file_path)
            
            # Extract lap data
            lap_data = []
            for lap_msg in fitfile.get_messages("lap"):
                vals = lap_msg.get_values()
                
                total_distance = vals.get('total_distance')
                total_timer_time = vals.get('total_timer_time')
                
                # Calculate speed directly for consistency
                if total_distance and total_timer_time and total_timer_time > 0:
                    avg_speed = total_distance / total_timer_time
                else:
                    avg_speed = vals.get('enhanced_avg_speed') or vals.get('avg_speed')
                
                lap_data.append({
                    'lap_number': len(lap_data) + 1,
                    'distance': total_distance,
                    'avg_speed': avg_speed,
                    'avg_hr': vals.get('avg_heart_rate'),
                    'avg_cadence': vals.get('avg_cadence'),
                    'total_ascent': vals.get('total_ascent'),
                    'start_time': vals.get('start_time'),
                    'total_elapsed_time': vals.get('total_elapsed_time')
                })
            
            # Extract elevation, cadence, and timestamp streams for GAP and cadence calculation
            elevation_stream = []
            cadence_stream = []
            timestamps = []
            for record in fitfile.get_messages("record"):
                vals = record.get_values()
                
                # --- UTC TO LOCAL FIX ---
                ts = vals.get('timestamp')
                if ts:
                    ts = ts.replace(tzinfo=timezone.utc).astimezone()
                timestamps.append(ts)
                # ------------------------

                elevation_stream.append(
                    vals.get('enhanced_altitude') or vals.get('altitude')
                )
                cadence_stream.append(vals.get('cadence'))
            
            # Calculate GAP and average cadence for each lap
            enhanced_laps = []
            for lap in lap_data:
                if not lap.get('start_time') or not lap.get('total_elapsed_time'):
                    enhanced_laps.append({
                        **lap,
                        'gap_pace': '--:--',
                        'actual_pace': '--:--',
                        'is_steep': False,
                        'avg_gradient': 0,
                        'avg_cadence': None
                    })
                    continue
                
                avg_speed = lap.get('avg_speed')
                if avg_speed is None or avg_speed == 0:
                    distance = lap.get('distance', 0)
                    elapsed_time = lap.get('total_elapsed_time', 0)
                    if distance > 0 and elapsed_time > 0:
                        avg_speed = distance / elapsed_time
                    else:
                        avg_speed = 0
                
                # Calculate average gradient and cadence for this lap
                lap_start = lap['start_time']
                lap_end = lap_start + pd.Timedelta(seconds=lap['total_elapsed_time'])
                
                lap_elevations = []
                lap_cadences = []
                for i, ts in enumerate(timestamps):
                    if ts is None:
                        continue
                    if lap_start <= ts <= lap_end:
                        # Collect elevation data
                        if elevation_stream[i] is not None:
                            if i > 0 and elevation_stream[i-1] is not None:
                                elev_diff = elevation_stream[i] - elevation_stream[i-1]
                                dist_diff = avg_speed
                                
                                if dist_diff > 0:
                                    gradient = elev_diff / dist_diff
                                    lap_elevations.append(gradient)
                        
                        # Collect cadence data
                        if cadence_stream[i] is not None and cadence_stream[i] > 0:
                            lap_cadences.append(cadence_stream[i])
                
                avg_gradient = sum(lap_elevations) / len(lap_elevations) if lap_elevations else 0
                avg_lap_cadence = sum(lap_cadences) / len(lap_cadences) if lap_cadences else None
                
                # Apply Minetti formula
                flat_cost = 3.6
                terrain_cost = minetti_cost_of_running(avg_gradient)
                cost_multiplier = terrain_cost / flat_cost
                
                # Calculate GAP
                if avg_speed and avg_speed > 0:
                    gap_speed = avg_speed / cost_multiplier
                    gap_pace_min = 26.8224 / gap_speed
                    gap_pace_str = f"{int(gap_pace_min)}:{int((gap_pace_min % 1) * 60):02d}"
                    
                    actual_pace_min = 26.8224 / avg_speed
                    actual_pace_str = f"{int(actual_pace_min)}:{int((actual_pace_min % 1) * 60):02d}"
                    
                    pace_diff_seconds = abs(gap_pace_min - actual_pace_min) * 60
                    is_steep = pace_diff_seconds > 15
                else:
                    gap_pace_str = "--:--"
                    actual_pace_str = "--:--"
                    is_steep = False
                
                enhanced_laps.append({
                    **lap,
                    'gap_pace': gap_pace_str,
                    'actual_pace': actual_pace_str,
                    'is_steep': is_steep,
                    'avg_gradient': avg_gradient,
                    'avg_cadence': avg_lap_cadence
                })
            
            results.append(enhanced_laps)
            
        except Exception as e:
            print(f"Error parsing FIT file: {e}")
            results.append(None)
    
    return results


def main():
    """Application entry point."""
    # Instantiate the application
    app = GarminAnalyzerApp()
    
    # Run in native mode with specified window configuration
    ui.run(
        native=True,
        window_size=(1200, 900),
        title="Garmin Analyzer Pro",
        reload=False,
        dark=True  # Force dark mode for native window
    )


if __name__ == "__main__":
    main()
