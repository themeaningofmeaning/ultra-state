"""
Garmin FIT Analyzer
"""

# Standard library imports
import os
import time
import json
import copy
import hashlib
import sqlite3
import asyncio
import logging
from datetime import datetime, timezone

# Third-party imports
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from nicegui import ui, run

# Local imports
from analyzer import FitAnalyzer, analyze_form, classify_split
from db import DatabaseManager, calculate_file_hash


class MuteFrameworkNoise(logging.Filter):
    def filter(self, record):
        # Filter out the specific NiceGUI warning about event listeners
        return "Event listeners changed after initial definition" not in record.getMessage()


MAP_PAYLOAD_VERSION = 4


# --- MAIN APPLICATION CLASS ---
class GarminAnalyzerApp:
    """Main application class for the Garmin FIT Analyzer."""
    
    def __init__(self):
        """Initialize the application with database and state."""
        self.db = DatabaseManager()
        # Load the last session ID from DB on startup so "Last Import" works after restart
        self.current_session_id = self.db.get_last_session_id()
        self.current_timeframe = "Last 30 Days"
        self.current_sort_by = 'date'
        self.current_sort_desc = True # True = DESC, False = ASC
        self.activities_data = []
        self.df = None
        self.import_in_progress = False
        self.focus_mode_active = False
        self._entering_focus_mode = False
        self.activities_table = None
        self.active_filters = set()
        self.volume_lens = 'quality'  # 'quality', 'mix', 'load'
        self.volume_card_container = None  # for surgical card refresh

        # Define taxonomy of filter tags
        self.TAG_CONFIG = {
            # CONTEXT TAGS
            'Long Run':   {'icon': '🏃', 'color': 'purple'},
            'Tempo':      {'icon': '🔥', 'color': 'orange'},
            'Intervals':  {'icon': '⚡', 'color': 'orange'},
            'Hill Sprints': {'icon': '⛰️', 'color': 'emerald'},
            'Hills':      {'icon': '⛰️', 'color': 'emerald'},
            'Recovery':   {'icon': '🧘', 'color': 'blue'},
            'Base':       {'icon': '🔷', 'color': 'blue'},
            'Fartlek':    {'icon': '💨', 'color': 'orange'},
            'Steady':     {'icon': '⚓', 'color': 'cyan'},
            
            # PHYSIO TAGS
            'VO2 MAX':    {'icon': '🫀', 'color': 'fuchsia'},
            'ANAEROBIC':  {'icon': '🔋', 'color': 'orange'},
            'THRESHOLD':  {'icon': '📈', 'color': 'emerald'},
            'MAX POWER':  {'icon': '🚀', 'color': 'purple'},
            }

        # Initialize the volume data container
        self.weekly_volume_data = None

        # LRU-ish cache for parsed activity detail payloads (avoids reparsing FIT on reopen)
        self.activity_detail_cache = {}
        self.activity_detail_cache_size = 24
        self.activity_detail_cache_version = 2

        # Background backfill task for legacy map payloads
        self._map_backfill_task = None
        
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

    def _normalize_bounds(self, bounds_value):
        """Validate and normalize bounds into [[min_lat, min_lon], [max_lat, max_lon]]."""
        try:
            if not isinstance(bounds_value, (list, tuple)) or len(bounds_value) != 2:
                return None
            if not isinstance(bounds_value[0], (list, tuple)) or not isinstance(bounds_value[1], (list, tuple)):
                return None
            if len(bounds_value[0]) != 2 or len(bounds_value[1]) != 2:
                return None

            lat1 = float(bounds_value[0][0])
            lon1 = float(bounds_value[0][1])
            lat2 = float(bounds_value[1][0])
            lon2 = float(bounds_value[1][1])

            min_lat, max_lat = sorted([lat1, lat2])
            min_lon, max_lon = sorted([lon1, lon2])

            if not (-90 <= min_lat <= 90 and -90 <= max_lat <= 90):
                return None
            if not (-180 <= min_lon <= 180 and -180 <= max_lon <= 180):
                return None
            if abs(max_lat - min_lat) < 1e-6:
                min_lat -= 0.0005
                max_lat += 0.0005
            if abs(max_lon - min_lon) < 1e-6:
                min_lon -= 0.0005
                max_lon += 0.0005

            return [[min_lat, min_lon], [max_lat, max_lon]]
        except Exception:
            return None

    def _hex_to_rgb(self, color):
        """Convert #RRGGBB to (r, g, b), or None for invalid input."""
        if not isinstance(color, str):
            return None
        color = color.strip()
        if not color.startswith('#') or len(color) != 7:
            return None
        try:
            return (
                int(color[1:3], 16),
                int(color[3:5], 16),
                int(color[5:7], 16),
            )
        except Exception:
            return None

    def _rgb_to_hex(self, r, g, b):
        """Convert RGB ints to #RRGGBB."""
        r = max(0, min(255, int(r)))
        g = max(0, min(255, int(g)))
        b = max(0, min(255, int(b)))
        return f'#{r:02x}{g:02x}{b:02x}'

    def _multi_color_from_t(self, t):
        """Map normalized t (0..1) to Garmin 5-color gradient."""
        t = max(0.0, min(1.0, float(t)))
        colors = [
            (0, 0, 255),    # Blue
            (0, 255, 0),    # Green
            (255, 255, 0),  # Yellow
            (255, 165, 0),  # Orange
            (255, 0, 0),    # Red
        ]
        idx = int(t * 4)
        if idx >= 4:
            idx = 3
        ratio = (t * 4) - idx
        c1, c2 = colors[idx], colors[idx + 1]
        r = int(c1[0] + (c2[0] - c1[0]) * ratio)
        g = int(c1[1] + (c2[1] - c1[1]) * ratio)
        b = int(c1[2] + (c2[2] - c1[2]) * ratio)
        return self._rgb_to_hex(r, g, b)

    def _projection_on_dual_tone(self, rgb):
        """Project RGB onto dual-tone Zinc->Emerald ramp and return (t, distance)."""
        start = (113.0, 113.0, 122.0)  # Zinc-500
        end = (52.0, 211.0, 153.0)     # Emerald-400
        vx, vy, vz = (end[0] - start[0], end[1] - start[1], end[2] - start[2])
        wx, wy, wz = (rgb[0] - start[0], rgb[1] - start[1], rgb[2] - start[2])
        vv = vx * vx + vy * vy + vz * vz
        if vv <= 0:
            return 0.5, float('inf')
        t = (wx * vx + wy * vy + wz * vz) / vv
        t = max(0.0, min(1.0, t))
        px = start[0] + t * vx
        py = start[1] + t * vy
        pz = start[2] + t * vz
        dist = ((rgb[0] - px) ** 2 + (rgb[1] - py) ** 2 + (rgb[2] - pz) ** 2) ** 0.5
        return t, dist

    def _looks_like_dual_tone_segments(self, segments):
        """Heuristic: detect old Zinc->Emerald palette so we can migrate to 5-color scale."""
        if not isinstance(segments, list) or not segments:
            return False

        sample = segments[: min(120, len(segments))]
        valid = 0
        near_line = 0
        for seg in sample:
            if not isinstance(seg, (list, tuple)) or len(seg) < 5:
                continue
            rgb = self._hex_to_rgb(seg[4])
            if rgb is None:
                continue
            valid += 1
            _, dist = self._projection_on_dual_tone(rgb)
            if dist <= 10.0:
                near_line += 1

        if valid < 10:
            return False
        return (near_line / valid) >= 0.85

    def _convert_dual_tone_segments_to_multicolor(self, segments):
        """Convert old dual-tone segment colors to the Garmin 5-color palette."""
        converted = []
        for seg in segments:
            if not isinstance(seg, (list, tuple)) or len(seg) < 5:
                converted.append(seg)
                continue
            rgb = self._hex_to_rgb(seg[4])
            if rgb is None:
                converted.append(list(seg))
                continue
            t, _ = self._projection_on_dual_tone(rgb)
            converted.append([seg[0], seg[1], seg[2], seg[3], self._multi_color_from_t(t)])
        return converted

    def _build_map_payload_from_segments(self, route_segments, max_segments=500):
        """Convert legacy route_segments into versioned map payload with robust bounds."""
        if not isinstance(route_segments, list) or not route_segments:
            return None

        cleaned = []
        for seg in route_segments:
            try:
                if not isinstance(seg, (list, tuple)) or len(seg) < 4:
                    continue
                lat1, lon1, lat2, lon2 = float(seg[0]), float(seg[1]), float(seg[2]), float(seg[3])
                if not (-90 <= lat1 <= 90 and -180 <= lon1 <= 180 and -90 <= lat2 <= 90 and -180 <= lon2 <= 180):
                    continue
                if (abs(lat1) < 1e-6 and abs(lon1) < 1e-6) or (abs(lat2) < 1e-6 and abs(lon2) < 1e-6):
                    continue
                pace_color = seg[4] if len(seg) > 4 and isinstance(seg[4], str) and seg[4].startswith('#') else '#10b981'
                hr_color = seg[5] if len(seg) > 5 and isinstance(seg[5], str) and seg[5].startswith('#') else None
                normalized = [lat1, lon1, lat2, lon2, pace_color]
                if hr_color:
                    normalized.append(hr_color)
                cleaned.append(normalized)
            except Exception:
                continue

        if not cleaned:
            return None

        # Migrate old dual-tone color payloads to restored multi-color gradient.
        if self._looks_like_dual_tone_segments(cleaned):
            cleaned = self._convert_dual_tone_segments_to_multicolor(cleaned)

        if len(cleaned) > max_segments:
            step = max(1, len(cleaned) // max_segments)
            sampled = cleaned[::step]
            if sampled[-1] != cleaned[-1]:
                sampled.append(cleaned[-1])
            cleaned = sampled

        lats = [s[0] for s in cleaned] + [s[2] for s in cleaned]
        lons = [s[1] for s in cleaned] + [s[3] for s in cleaned]
        if not lats or not lons:
            return None

        sorted_lats = sorted(lats)
        sorted_lons = sorted(lons)

        def percentile(values, p):
            if not values:
                return 0.0
            idx = (len(values) - 1) * p
            lo = int(idx)
            hi = min(lo + 1, len(values) - 1)
            if lo == hi:
                return float(values[lo])
            frac = idx - lo
            return float(values[lo] + (values[hi] - values[lo]) * frac)

        if len(sorted_lats) > 20:
            min_lat = percentile(sorted_lats, 0.005)
            max_lat = percentile(sorted_lats, 0.995)
            min_lon = percentile(sorted_lons, 0.005)
            max_lon = percentile(sorted_lons, 0.995)
        else:
            min_lat, max_lat = float(sorted_lats[0]), float(sorted_lats[-1])
            min_lon, max_lon = float(sorted_lons[0]), float(sorted_lons[-1])

        bounds = self._normalize_bounds([[min_lat, min_lon], [max_lat, max_lon]])
        if not bounds:
            return None

        center = [
            (bounds[0][0] + bounds[1][0]) / 2,
            (bounds[0][1] + bounds[1][1]) / 2,
        ]

        return {
            'v': MAP_PAYLOAD_VERSION,
            'segments': cleaned,
            'bounds': bounds,
            'center': center,
            'segment_count': len(cleaned),
        }

    def _activity_needs_map_payload_backfill(self, activity):
        """Decide whether an activity needs map payload migration."""
        if not isinstance(activity, dict):
            return False
        map_payload = activity.get('map_payload')
        if not isinstance(map_payload, dict):
            return True
        version = int(activity.get('map_payload_version') or map_payload.get('v') or 1)
        if version < MAP_PAYLOAD_VERSION:
            return True
        if not isinstance(map_payload.get('segments'), list) or not map_payload.get('segments'):
            return True
        if not self._normalize_bounds(map_payload.get('bounds')):
            return True
        return False

    def _get_or_backfill_map_payload(self, activity):
        """Return normalized map payload, creating/persisting current-version payload when needed."""
        if not isinstance(activity, dict):
            return None

        map_payload = activity.get('map_payload')
        if isinstance(map_payload, dict):
            bounds = self._normalize_bounds(map_payload.get('bounds'))
            segments = map_payload.get('segments')
            version = int(activity.get('map_payload_version') or map_payload.get('v') or 1)
            if version >= MAP_PAYLOAD_VERSION and bounds and isinstance(segments, list) and segments:
                map_payload['bounds'] = bounds
                activity['map_payload'] = map_payload
                activity['map_payload_version'] = version
                return map_payload

        legacy_segments = activity.get('route_segments')
        if (not legacy_segments) and isinstance(map_payload, dict):
            legacy_segments = map_payload.get('segments')

        new_payload = self._build_map_payload_from_segments(legacy_segments, max_segments=500)
        if not new_payload:
            return None

        activity['map_payload'] = new_payload
        activity['map_payload_version'] = new_payload.get('v', MAP_PAYLOAD_VERSION)
        activity['route_segments'] = new_payload.get('segments', [])
        activity['bounds'] = new_payload.get('bounds', [[0, 0], [0, 0]])

        activity_hash = activity.get('hash') or activity.get('db_hash')
        if activity_hash:
            try:
                self.db.update_activity_map_payload(
                    activity_hash,
                    new_payload,
                    route_segments=new_payload.get('segments'),
                    bounds=new_payload.get('bounds'),
                    map_payload_version=new_payload.get('v', MAP_PAYLOAD_VERSION),
                )
            except Exception as ex:
                print(f"Map payload backfill warning for {activity_hash}: {ex}")

        return new_payload

    async def _backfill_map_payloads_for_loaded_activities(self):
        """Background migration for loaded activities to versioned map payloads."""
        if not self.activities_data:
            return

        for idx, activity in enumerate(self.activities_data):
            try:
                if self._activity_needs_map_payload_backfill(activity):
                    self._get_or_backfill_map_payload(activity)
            except Exception as ex:
                print(f"Background map payload backfill warning: {ex}")

            if idx % 20 == 0:
                await asyncio.sleep(0)
    
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

        # Ensure versioned map payload exists/normalized for this activity metadata.
        # This is lightweight and avoids stale legacy bounds in modal rendering.
        try:
            self._get_or_backfill_map_payload(activity)
        except Exception as ex:
            print(f"Map payload prep warning for {activity_hash}: {ex}")

        # 2. Locate FIT file on disk
        fit_file_path = self._locate_fit_file(activity)
        if not fit_file_path:
            return {'error': 'file_not_found', 'activity': activity}

        file_mtime = os.path.getmtime(fit_file_path) if os.path.exists(fit_file_path) else None

        # 2.5 Fast path: return cached detail payload when source file has not changed
        cached = self.activity_detail_cache.get(activity_hash)
        if (
            cached
            and cached.get('file_path') == fit_file_path
            and cached.get('file_mtime') == file_mtime
            and cached.get('cache_version') == self.activity_detail_cache_version
        ):
            payload = copy.deepcopy(cached.get('payload', {}))
            if payload:
                payload['activity_metadata'] = activity
                return payload

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
                
                # --- NEW: Advanced Form Streams ---
                vertical_oscillation = []
                stance_time = []
                vertical_ratio = []
                step_length = [] # aka stride_length
                
                # (Route extraction removed - using pre-calculated DB data)
                # ---------------------------------------------------------

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
                    
                    # --- NEW: Extract Advanced Metrics ---
                    # Keep as None if missing (do not coerce to 0)
                    vertical_oscillation.append(vals.get('vertical_oscillation'))
                    stance_time.append(vals.get('stance_time') or vals.get('ground_contact_time'))
                    vertical_ratio.append(vals.get('vertical_ratio'))
                    step_length.append(vals.get('step_length') or vals.get('stride_length'))
                    
                    # (Route point extraction removed)
                    # -----------------------------------------------------

                # 5. Extract lap data
                lap_data = []
                for lap_msg in fitfile.get_messages("lap"):
                    vals = lap_msg.get_values()
                    if not vals:
                        continue
                    
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
                    'session_data': session_data,
                    # --- NEW: Advanced Form Streams ---
                    'vertical_oscillation': vertical_oscillation,
                    'stance_time': stance_time,
                    'vertical_ratio': vertical_ratio,
                    'step_length': step_length,
                    'step_length': step_length,
                    # --- NEW: Route Data with Speed ---
                    # ----------------------------------
                    # ----------------------------------
                    # ----------------------------------
                }

            # Run parsing in background thread
            result = await run.io_bound(parse_fit)
            
            # Check if parsing failed
            if result is None or not isinstance(result, dict):
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

            # Save detail payload cache keyed by activity hash + file version
            self.activity_detail_cache[activity_hash] = {
                'file_path': fit_file_path,
                'file_mtime': file_mtime,
                'cache_version': self.activity_detail_cache_version,
                'payload': copy.deepcopy(result),
            }
            if len(self.activity_detail_cache) > self.activity_detail_cache_size:
                oldest_key = next(iter(self.activity_detail_cache))
                if oldest_key != activity_hash:
                    self.activity_detail_cache.pop(oldest_key, None)

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

        # Define color palette (unified with trends HR Zones lens)
        colors = [
            '#60a5fa',  # Zone 1 - Blue (easy/warmup)
            '#34d399',  # Zone 2 - Emerald (aerobic base)
            '#fbbf24',  # Zone 3 - Amber (threshold/gray zone)
            '#f97316',  # Zone 4 - Orange (hard)
            '#ef4444'   # Zone 5 - Red (max effort)
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
            xaxis_title='Time (minutes)',
            yaxis_title='',
            template='plotly_dark',
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            height=300,
            margin=dict(l=20, r=20, t=20, b=20), # Reduced top margin since title is gone
            showlegend=False,
            font=dict(color='#a1a1aa') # Zinc-400
        )
        
        # Completely hide the modebar
        fig.update_layout(modebar={'remove': ['zoom', 'pan', 'select', 'lasso2d', 'zoomIn', 'zoomOut', 'autoScale', 'resetScale', 'toImage']})

        return fig
    
    def create_form_analysis_chart(self, distance_stream, cadence_stream, elevation_stream, 
                                 timestamps=None, vertical_oscillation=None, stance_time=None, 
                                 vertical_ratio=None, step_length=None, use_miles=True):
        """
        Create Pro Form Analysis Chart (Apple Health Style).
        Visualizes Cadence (Grey Line) + Efficiency Status (Colored Markers).
        """
        from plotly.subplots import make_subplots
        import plotly.graph_objects as go
        import pandas as pd
        import numpy as np

        # 1. VALIDATION & ALIGNMENT
        # We perform an intersection of valid data. 
        # Crucial: Arrays might be different lengths, so crop to minimum.
        min_len = min(len(distance_stream), len(cadence_stream), len(elevation_stream))
        
        # Prepare arrays (crop to min_len)
        dist = np.array(distance_stream[:min_len])
        cad = np.array(cadence_stream[:min_len])
        elev = np.array(elevation_stream[:min_len])
        
        # Optional Streams (Handle None or Mismatched lengths)
        def prepare_optional(stream, default_val=None):
            if stream is None or len(stream) == 0: return np.full(min_len, default_val)
            # Crop or Pad
            arr = np.array(stream[:min_len])
            if len(arr) < min_len:
                return np.pad(arr, (0, min_len - len(arr)), constant_values=default_val)
            return arr

        vo = prepare_optional(vertical_oscillation) # mm
        gct = prepare_optional(stance_time)         # ms
        vr = prepare_optional(vertical_ratio)       # %
        sl = prepare_optional(step_length)          # mm (needs /1000 for m)

        # Timestamps alignment
        time_labels = []
        if timestamps:
            valid_timestamps = timestamps[:min_len]
            if valid_timestamps:
                start_time = valid_timestamps[0]
                for ts in valid_timestamps:
                    if ts:
                        delta = ts - start_time
                        total_seconds = int(delta.total_seconds())
                        hours = total_seconds // 3600
                        minutes = (total_seconds % 3600) // 60
                        seconds = total_seconds % 60
                        if hours > 0:
                            time_labels.append(f"⏱️ Time: {hours}:{minutes:02d}:{seconds:02d}")
                        else:
                            time_labels.append(f"⏱️ Time: {minutes}:{seconds:02d}")
                    else:
                        time_labels.append("⏱️ Time: --:--")
        
        if not time_labels:
            time_labels = ["⏱️ Time: --:--" for _ in range(min_len)]

        # 2. DATA CLEANING & SMOOTHING (Grey Line)
        # Filter out stops (Cadence <= 0)
        # Using a mask allows us to break the line (connectgaps=False)
        cad_float = np.array(cad, dtype=float)
        cad_float[cad_float <= 0] = np.nan
        
        # Apply 15s Rolling Average (Smooth the line)
        # We smooth even lightly to make it look "Apple-like"
        cad_series = pd.Series(cad_float)
        cad_smoothed = cad_series.rolling(window=15, min_periods=1, center=True).mean()
        
        # Re-apply gaps (Stops should be visual breaks)
        mask_gaps = (cad_float <= 0) | (np.isnan(cad_float))
        cad_smoothed[mask_gaps] = np.nan

        # 3. WATERFALL LOGIC (Marker Colors & Verdicts)
        marker_colors = []
        verdicts = []
        why_metrics = [] # Diagnostic string
        
        # Constants
        C_GREEN = '#32D74B' 
        C_YELLOW = '#FFD60A'
        C_RED = '#FF453A'
        C_GREY = '#8E8E93' # Fallback

        # Iterate through the aligned data
        for i in range(min_len):
            # Safe access to current values
            c_val = cad_smoothed[i] if not np.isnan(cad_smoothed[i]) else 0
            vr_val = vr[i]
            gct_val = gct[i]
            
            # --- Logic ---
            status = None
            
            # Priority 1: Vertical Ratio
            if vr_val is not None and vr_val > 0:
                if vr_val < 8.0:
                    status = (C_GREEN, "✅ Elite Efficiency")
                elif vr_val <= 10.0:
                    status = (C_YELLOW, "⚖️ Good Form")
                else:
                    status = (C_RED, "⚠️ High Bounce")
            
            # Priority 2: GCT (if Ratio missing)
            elif gct_val is not None and gct_val > 0:
                if gct_val < 250:
                    status = (C_GREEN, "🚀 Fast Reactivity")
                elif gct_val <= 300:
                    status = (C_YELLOW, "🏃 Balanced")
                else:
                    status = (C_RED, "🛑 Long Ground Contact")
                    
            # Priority 3: Cadence (Fallback)
            elif c_val > 0:
                if c_val > 170:
                    status = (C_GREEN, "✅ Optimal Cadence")
                elif c_val >= 160:
                    status = (C_YELLOW, "🛠 Improvement Zone")
                else:
                    status = (C_RED, "⚠️ Overstriding Risk")
            
            else:
                status = (C_GREY, "⏸️ Stopped")

            marker_colors.append(status[0])
            verdicts.append(status[1])
            
            # Diagnostic String
            # "📏 Stride: 1.1m | ⏱️ GCT: 240ms | ↕️ Vert Osc: 8.2cm"
            diag = []
            if sl[i] and sl[i] > 0: diag.append(f"📏 Stride: {sl[i]/1000:.2f}m")
            if gct_val and gct_val > 0: diag.append(f"⏱️ GCT: {int(gct_val)}ms")
            if vo[i] and vo[i] > 0: diag.append(f"↕️ Vert Osc: {vo[i]/10:.1f}cm")
            
            why_metrics.append(" | ".join(diag) if diag else "No advanced metrics")

        # 4. MARKER OPTIMIZATION
        # If activity > 2 hours (~7200 points), take every 5th point for markers
        # The Line (Trace 1) stays full resolution. Markers (Trace 2) get subsampled.
        step = 5 if min_len > 7200 else 1
        
        # Subsampled Indices for Scatter Trace
        idx_sub = np.arange(0, min_len, step)
        
        # 5. CONVERT UNITS
        if use_miles:
            dist_units = "mi"
            dist_conv = dist * 0.000621371
            elev_units = "ft"
            elev_conv = elev * 3.28084
        else:
            dist_units = "km"
            dist_conv = dist / 1000
            elev_conv = elev
            elev_units = "m"

        # 6. BUILD FIGURE
        fig = make_subplots(specs=[[{"secondary_y": True}]])
        
        # Trace 1: Elevation (Area - Background)
        fig.add_trace(
            go.Scatter(
                x=dist_conv,
                y=elev_conv,
                name="Elevation",
                fill='tozeroy',
                mode='lines',
                line=dict(color='#3f3f46', width=0), # Zinc-700ish, very subtle
                fillcolor='rgba(63, 63, 70, 0.3)',
                hoverinfo='skip' # Tooltip handled by the main trace
            ),
            secondary_y=True
        )

        # Trace 2: The Grey Line (Smoothed Cadence)
        fig.add_trace(
            go.Scatter(
                x=dist_conv,
                y=cad_smoothed,
                name="Trend",
                mode='lines',
                line=dict(color='#525252', width=2), # Neutral Grey
                connectgaps=False,
                hoverinfo='skip' # Tooltip mainly on markers
            ),
            secondary_y=False
        )
        
        # Trace 3: The Status Markers (Overlay)
        # Using subsampled data
        fig.add_trace(
            go.Scatter(
                x=dist_conv[idx_sub],
                y=cad_smoothed[idx_sub],
                name="Form Status",
                mode='markers',
                marker=dict(
                    color=[marker_colors[i] for i in idx_sub],
                    size=4, # Subtle but visible
                    opacity=0.8
                ),
                # Custom Data for Tooltip
                customdata=np.stack((
                    [time_labels[i] for i in idx_sub],      # 0: Time
                    [verdicts[i] for i in idx_sub],         # 1: Verdict
                    [why_metrics[i] for i in idx_sub],      # 2: Diagnostic
                    [elev_conv[i] for i in idx_sub]         # 3: Elev
                ), axis=-1),
                hovertemplate=(
                    "<b>%{customdata[0]}</b><br>" +
                    f"📏 Distance: %{{x:.2f}} {dist_units}<br>" +
                    f"🏔️ Elevation: %{{customdata[3]:.0f}} {elev_units}<br>" +
                    "👟 Cadence: <b>%{y:.0f}</b> spm<br>" +
                    "<br>" +
                    "<i>%{customdata[2]}</i><br>" + # Diagnostic
                    "<b>%{customdata[1]}</b>" +     # Verdict
                    "<extra></extra>"
                )
            ),
            secondary_y=False
        )

        # Layout
        fig.update_layout(
            template='plotly_dark',
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            height=350,
            margin=dict(l=20, r=20, t=20, b=40), # Reduced top margin
            hovermode="x unified",
            showlegend=False,
            font=dict(family="Inter, sans-serif", size=12, color="#a1a1aa") # Zinc-400
        )
        
        # Axes
        fig.update_yaxes(
            title_text="Cadence (SPM)", 
            title_font=dict(color='#A1A1AA'),
            tickfont=dict(color='#D4D4D8'),
            secondary_y=False, 
            gridcolor='#27272a', 
            zeroline=False
        )
        fig.update_yaxes(
            title_text=f"Elevation ({elev_units})", 
            title_font=dict(color='#A1A1AA'),
            tickfont=dict(color='#D4D4D8'),
            secondary_y=True, 
            showgrid=False, 
            zeroline=False
        )
        fig.update_xaxes(
            title_text=f"Distance ({dist_units})", 
            title_font=dict(color='#A1A1AA'),
            tickfont=dict(color='#D4D4D8'),
            gridcolor='#27272a', 
            zeroline=False
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
                        {{ props.value === 'HIGH QUALITY' ? 'High Quality' : props.value === 'STRUCTURAL' ? 'Base' : 'Broken' }}
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

    def create_decoupling_card(self, decoupling_data, efficiency_factor=None):
        """
        Create card displaying aerobic decoupling metrics.

        Args:
            decoupling_data: Dictionary with decoupling metrics
            efficiency_factor: Optional EF value from the activity

        Returns:
            NiceGUI card component
        """
        with ui.card().classes('bg-zinc-900 p-4 border border-zinc-800 h-full') as card:
            card.style('border-radius: 8px; box-shadow: 0px 4px 12px rgba(0, 0, 0, 0.4);')

            # Title with info icon
            with ui.row().classes('items-center gap-2 mb-3'):
                ui.label('AEROBIC EFFICIENCY').classes('text-lg font-bold text-white')
                ae_info_icon = ui.icon('help_outline').classes('text-zinc-500 hover:text-white cursor-pointer text-base transition-colors')
                ae_info_icon.on('click', lambda: self.show_aerobic_efficiency_info())

            # Data Section: Efficiency Factor (Top Priority)
            if efficiency_factor and efficiency_factor > 0:
                with ui.column().classes('gap-0'):
                    ui.label('EFFICIENCY FACTOR').classes('text-xs text-zinc-500 uppercase tracking-wider font-semibold')
                    with ui.row().classes('items-baseline gap-2'):
                        ui.label(f'{efficiency_factor:.2f}').classes('text-3xl font-bold text-white')
                        ui.label('speed/HR').classes('text-xs text-zinc-500 font-bold')
            
            ui.separator().classes('bg-zinc-800 my-3')

            # Data Section: Decoupling
            with ui.column().classes('gap-0'):
                ui.label('AEROBIC DECOUPLING').classes('text-xs text-zinc-500 uppercase tracking-wider font-semibold')
                with ui.row().classes('items-baseline gap-2'):
                    ui.label(f"{decoupling_data['decoupling_pct']:.1f}%").classes('text-2xl font-bold').style(
                        f"color: {decoupling_data['color']};"
                    )
                    ui.label(decoupling_data['status']).classes('text-base font-bold').style(
                        f"color: {decoupling_data['color']};"
                    )

            # Detail Metrics (Small)
            with ui.column().classes('mt-auto gap-0.5 pt-2'):
                ui.label(f"1st Half: {decoupling_data['ef_first_half']:.4f}").classes('text-[10px] text-zinc-600')
                ui.label(f"2nd Half: {decoupling_data['ef_second_half']:.4f}").classes('text-[10px] text-zinc-600')

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
            
            with ui.column().classes('gap-3 h-full justify-between w-full'):
                
                # --- TOP SECTION (Anchored Top) ---
                with ui.column().classes('gap-3 w-full'):
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
                
                # --- BOTTOM SECTION (Anchored Bottom) ---
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
                        
                        with ui.column().classes('gap-0 w-full'):
                            ui.separator().classes('bg-zinc-800 mb-3')
                            
                            with ui.row().classes('items-center gap-2 mb-1'):
                                ui.label('HR RECOVERY (1-MIN)').classes('text-xs text-zinc-500 uppercase tracking-wider font-semibold')
                                ui.icon('help_outline').classes('text-zinc-600 hover:text-white text-xs cursor-pointer').on('click', lambda: self.show_hrr_info())
                            
                            with ui.row().classes('items-baseline gap-2'):
                                ui.label(f"{score}").classes('text-3xl font-bold font-mono leading-none').style(f'color: {hrr_color};')
                                ui.label('bpm').classes('text-xs text-zinc-500 font-bold')
        
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
        
        # Calculate Vertical Ratio (if possible)
        ver_ratio = (v_osc_cm / stride_m) * 100 if v_osc_cm and stride_m else None

        # --- 3. METRIC DEFINITIONS AND COLORING ---
        # This list defines each metric, its display, and traffic light logic
        metrics = [
            {
                'key': 'CADENCE',
                'value': f'{int(cadence)}' if cadence else '-',
                'unit': 'spm',
                'range': '> 170spm',
                'verdict': 'High' if cadence and cadence > 170 else 'Low',
                'color': 'text-emerald-400' if cadence and cadence > 170 else ('text-blue-400' if cadence and cadence > 160 else 'text-orange-400'),
                'icon': 'directions_run',
                'meaning': 'Steps per minute (SPM). Higher cadence generally means shorter ground contact, less braking force, and more efficient energy transfer. Most elite runners are 170–185 SPM.'
            },
            {
                'key': 'BOUNCE',
                'value': f'{bounce:.1f}' if bounce else '-',
                'unit': 'cm',
                'range': '< 8cm',
                'verdict': 'Good' if bounce and bounce < 8 else 'High',
                'color': 'text-emerald-400' if bounce and bounce < 8 else ('text-yellow-400' if bounce and bounce < 10 else 'text-red-400'),
                'icon': 'height',
                'meaning': 'Vertical Oscillation. Lower is better. Measures how much "bounce" for each step forward. Excess bounce wastes energy.'
            },
            {
                'key': 'CONTACT',
                'value': f'{gct:.0f}' if gct else '-',
                'unit': 'ms',
                'range': '< 250ms',
                'verdict': 'Short' if gct and gct < 250 else 'Long',
                'color': 'text-emerald-400' if gct and gct < 250 else ('text-blue-400' if gct and gct < 270 else 'text-orange-400'),
                'icon': 'timer',
                'meaning': 'Ground Contact Time. How long your foot stays on the ground. Shorter = better toggle and less braking.'
            },
            {
                'key': 'STRIDE',
                'value': f'{stride_m:.2f}' if stride_m else '-',
                'unit': 'm',
                'range': '> 1.0m',
                'verdict': 'Long' if stride_m and stride_m > 1.0 else 'Short',
                'color': 'text-blue-400',
                'icon': 'straighten',
                'meaning': 'Stride Length. Distance covered in one step. Should increase naturally with speed, not by overreaching.'
            }
        ]

        # --- 4. RENDER THE CARD ---
        with ui.card().classes('bg-zinc-900 p-4 border border-zinc-800 h-full') as card:
            card.style('border-radius: 8px; box-shadow: 0px 4px 12px rgba(0, 0, 0, 0.4);')
            
            # Header
            with ui.row().classes('items-center gap-2 mb-3'):
                ui.label('RUNNING MECHANICS').classes('text-lg font-bold text-white')
                rm_info_icon = ui.icon('help_outline').classes('text-zinc-500 hover:text-white cursor-pointer text-base transition-colors')
                rm_info_icon.on('click', lambda: self.show_form_info())
            
            # Form Verdict Badge
            cadence_val = cadence if cadence else 0
            with ui.column().classes('w-full items-center py-2 gap-1'):
                with ui.row().classes('items-center gap-2'):
                    ui.icon(diagnosis['icon']).classes(f"text-2xl {diagnosis['color']}")
                    ui.label(diagnosis['verdict']).classes(f"text-xl font-black {diagnosis['color']} tracking-tight")
                ui.label(diagnosis['prescription']).classes('text-xs text-zinc-400 italic text-center px-4')
            
            ui.separator().classes('bg-zinc-800 my-3')
            
            # Data Grid (Dynamic based on 'metrics' list)
            with ui.row().classes('w-full justify-between text-center'):
                for metric in metrics:
                    with ui.column().classes('gap-0'):
                        ui.label(metric['key']).classes('text-[9px] font-bold text-zinc-600')
                        ui.label(metric['value']).classes(f"text-sm font-bold font-mono {metric['color']}")
                        ui.label(metric['unit']).classes('text-[9px] text-zinc-600')
        
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
            # 1. Gather Averages
            avg_cad = activity.get('avg_cadence', 0)
            avg_hr = activity.get('avg_hr', 0)
            
            # Use the max_hr from the detailed parse if available, else DB
            max_hr = detail_data.get('max_hr') or activity.get('max_hr', 185)
            
            # 2. Calculate Average Grade
            dist_mi = activity.get('distance_mi', 0)
            elev_ft = activity.get('elevation_ft', 0)
            if dist_mi > 0:
                # Rise (ft) / Run (ft) * 100
                avg_grade = (elev_ft / (dist_mi * 5280)) * 100
            else:
                avg_grade = 0
            
            # 3. Use the SAME logic as the Chart (classify_split)
            # This ensures Z2 runs are correctly labeled "Structural"
            verdict_text = classify_split(avg_cad, avg_hr, max_hr, avg_grade)
            
            # 4. Map Verdict to Badge
            if verdict_text == 'HIGH QUALITY':
                v_label = 'High Quality Miles' 
                v_color = 'text-emerald-400'
                v_bg = 'bg-emerald-500/20 border-emerald-500/30'
            elif verdict_text == 'STRUCTURAL':
                 v_label = 'Structural Miles'
                 v_color = 'text-blue-400'
                 v_bg = 'bg-blue-500/20 border-blue-500/30'
            elif verdict_text == 'BROKEN':
                v_label = 'Broken Miles'
                v_color = 'text-red-400'
                v_bg = 'bg-red-500/20 border-red-500/30'
            else:
                v_label = None
                
        except Exception as e:
            print(f"Badge Calc Error: {e}")
            v_label = None
        # -------------------------------

        # Create main modal
        modal_map = None
        modal_fit_bounds = None
        modal_map_card = None

        with ui.dialog() as detail_dialog:
            detail_dialog.props('transition-show=none transition-hide=none')
            with ui.card().classes('w-full max-w-[900px] p-0 bg-zinc-950 h-full border border-zinc-800'):
                # Close button
                with ui.row().classes('w-full justify-end p-2'):
                    close_btn = ui.button(icon='close', on_click=detail_dialog.close, color=None).props('flat round dense')
                    close_btn.style('color: #9ca3af !important;')

                # Content container
                with ui.column().classes('w-full gap-4 px-4 pb-4'):

                    # --- NEW: Cinematic Route Map (Leaflet) ---
                    # Uses pre-calculated data from "First Principles" Analyzer (TinyDB)
                    activity_meta = detail_data.get('activity_metadata', {})
                    map_payload = self._get_or_backfill_map_payload(activity_meta) or {}

                    route_segments = map_payload.get('segments', []) if isinstance(map_payload, dict) else []
                    final_bounds = self._normalize_bounds(map_payload.get('bounds')) if isinstance(map_payload, dict) else None

                    map_center = None
                    if isinstance(map_payload, dict):
                        center_val = map_payload.get('center')
                        if isinstance(center_val, (list, tuple)) and len(center_val) == 2:
                            try:
                                c_lat = float(center_val[0])
                                c_lon = float(center_val[1])
                                if -90 <= c_lat <= 90 and -180 <= c_lon <= 180:
                                    map_center = (c_lat, c_lon)
                            except Exception:
                                map_center = None

                    if map_center is None and final_bounds:
                        map_center = (
                            (final_bounds[0][0] + final_bounds[1][0]) / 2,
                            (final_bounds[0][1] + final_bounds[1][1]) / 2,
                        )

                    if route_segments and map_center:
                        # Map Container
                        with ui.card().classes('w-full h-80 bg-zinc-950 p-0 border-none no-shadow overflow-hidden relative group').style('opacity: 0; transition: opacity 0.18s ease;') as map_card:
                            modal_map_card = map_card
                            # Initialize Leaflet with NO controls for cinematic look
                            m = ui.leaflet(center=map_center, zoom=13, options={
                                'zoomControl': False, 
                                'attributionControl': False
                            }).classes('w-full h-full leaflet-seam-fix')
                            
                            # Hybrid Tiles (helps hide seam artifacts on some satellite tile rows)
                            m.tile_layer(
                                url_template=r'https://mt1.google.com/vt/lyrs=y&x={x}&y={y}&z={z}',
                                options={'maxZoom': 20, 'detectRetina': True}
                            )

                            # Route color mode state + shared render pipeline
                            map_color_mode = {'value': 'pace'}
                            map_route_palette = {
                                'pace': 'linear-gradient(to right, #0000ff, #00ff00, #ffff00, #ffa500, #ff0000)',
                                'hr': 'linear-gradient(to right, #3b82f6, #10b981, #f97316, #ef4444)',
                            }
                            map_legend_left = {'pace': 'Slower', 'hr': 'Lower HR'}
                            map_legend_right = {'pace': 'Faster', 'hr': 'Higher HR'}
                            route_weight = 4

                            def _segment_color(seg, mode):
                                try:
                                    if mode == 'hr' and len(seg) > 5 and isinstance(seg[5], str) and seg[5].startswith('#'):
                                        return seg[5]
                                    if len(seg) > 4 and isinstance(seg[4], str) and seg[4].startswith('#'):
                                        return seg[4]
                                except Exception:
                                    pass
                                return '#10b981'

                            def _render_route(mode='pace'):
                                if not route_segments:
                                    return
                                try:
                                    m.clear_layers()
                                except Exception as ex:
                                    print(f"Map clear layers warning: {ex}")

                                # Re-add base tile layer after clear
                                m.tile_layer(
                                    url_template=r'https://mt1.google.com/vt/lyrs=y&x={x}&y={y}&z={z}',
                                    options={'maxZoom': 20, 'detectRetina': True}
                                )

                                # Group segments by selected color mode to reduce layer count
                                segments_by_color = {}
                                for seg in route_segments:
                                    if not isinstance(seg, (list, tuple)) or len(seg) < 4:
                                        continue
                                    color = _segment_color(seg, mode)
                                    if color not in segments_by_color:
                                        segments_by_color[color] = []
                                    segments_by_color[color].append([[seg[0], seg[1]], [seg[2], seg[3]]])

                                for color, latlngs in segments_by_color.items():
                                    m.generic_layer(
                                        name='polyline',
                                        args=[
                                            latlngs,
                                            {
                                                'color': color,
                                                'weight': route_weight,
                                                'opacity': 1.0,
                                                'lineCap': 'round',
                                                'lineJoin': 'round'
                                            }
                                        ]
                                    )

                            # Initial draw (default pace)
                            _render_route('pace')

                            # Start / End markers (derived from first and last route segments)
                            start_point = None
                            end_point = None
                            try:
                                if route_segments and len(route_segments[0]) >= 4:
                                    start_point = [float(route_segments[0][0]), float(route_segments[0][1])]
                                if route_segments and len(route_segments[-1]) >= 4:
                                    end_point = [float(route_segments[-1][2]), float(route_segments[-1][3])]
                            except Exception:
                                start_point = None
                                end_point = None

                            def _valid_latlng(pt):
                                return (
                                    isinstance(pt, (list, tuple))
                                    and len(pt) == 2
                                    and -90 <= pt[0] <= 90
                                    and -180 <= pt[1] <= 180
                                )

                            # Pro start/end markers using Leaflet divIcon with crisp inline SVG
                            if _valid_latlng(start_point) or _valid_latlng(end_point):
                                icon_start_svg = """
<svg width="16" height="16" viewBox="0 0 24 24" fill="white" xmlns="http://www.w3.org/2000/svg">
    <path d="M8 5V19L19 12L8 5Z" />
</svg>
""".strip()
                                icon_end_svg = """
<svg width="14" height="14" viewBox="0 0 24 24" fill="white" xmlns="http://www.w3.org/2000/svg">
    <rect x="4" y="4" width="16" height="16" rx="2" />
</svg>
""".strip()

                                async def _add_pro_markers_once_ready(_map=m, _start=start_point, _end=end_point, _svg_start=icon_start_svg, _svg_end=icon_end_svg):
                                    try:
                                        await _map.initialized()
                                        await asyncio.sleep(0.02)
                                    except Exception as ex:
                                        print(f"Pro marker init wait failed: {ex}")
                                        return

                                    if _valid_latlng(_start):
                                        try:
                                            start_js = (
                                                "L.marker(["
                                                f"{_start[0]}, {_start[1]}"
                                                "], {"
                                                "interactive:false, keyboard:false, zIndexOffset:1000, "
                                                "icon:L.divIcon({"
                                                "className:'pro-marker pm-start',"
                                                f"html:{json.dumps(_svg_start)},"
                                                "iconSize:[30,30],"
                                                "iconAnchor:[15,15]"
                                                "})"
                                                "})"
                                            )
                                            _map.run_map_method(':addLayer', start_js)
                                        except Exception as ex:
                                            print(f"Pro start marker add failed: {ex}")

                                    if _valid_latlng(_end):
                                        try:
                                            end_js = (
                                                "L.marker(["
                                                f"{_end[0]}, {_end[1]}"
                                                "], {"
                                                "interactive:false, keyboard:false, zIndexOffset:1000, "
                                                "icon:L.divIcon({"
                                                "className:'pro-marker pm-end',"
                                                f"html:{json.dumps(_svg_end)},"
                                                "iconSize:[30,30],"
                                                "iconAnchor:[15,15]"
                                                "})"
                                                "})"
                                            )
                                            _map.run_map_method(':addLayer', end_js)
                                        except Exception as ex:
                                            print(f"Pro end marker add failed: {ex}")

                                asyncio.create_task(_add_pro_markers_once_ready())

                            # Explicit Attribution (Minimalist Text)
                            with ui.element('div').classes('absolute bottom-1 right-2 z-[9999] pointer-events-none text-[10px] text-zinc-400 font-mono mix-blend-plus-lighter'):
                                ui.html('&copy; Google 2026')

                            # Custom monochrome map mode slider (PACE <-> HR)
                            toggle_id = f'map_mode_toggle_{activity_hash}'

                            async def _on_map_mode_checkbox_change(e):
                                checked = False
                                try:
                                    checked = bool(await ui.run_javascript(
                                        f'Boolean(document.getElementById({json.dumps(toggle_id)})?.checked)'
                                    ))
                                except Exception:
                                    raw = e.args
                                    if isinstance(raw, dict):
                                        if isinstance(raw.get('checked'), bool):
                                            checked = raw.get('checked', False)
                                        elif isinstance(raw.get('target'), dict):
                                            checked = bool(raw.get('target', {}).get('checked', False))
                                        elif 'value' in raw:
                                            value_str = str(raw.get('value', '')).strip().lower()
                                            checked = value_str in {'1', 'true', 'on', 'checked', 'hr'}
                                    elif isinstance(raw, bool):
                                        checked = raw
                                _set_map_mode('hr' if checked else 'pace')

                            with ui.element('div').classes('mono-toggle-wrapper'):
                                ui.element('input').props(
                                    f'type="checkbox" id="{toggle_id}" class="mono-toggle-checkbox"'
                                ).on('change', _on_map_mode_checkbox_change)

                                with ui.element('label').props(f'for="{toggle_id}" class="mono-toggle-label"'):
                                    ui.label('PACE').classes('mono-toggle-text-pace')
                                    ui.label('HR').classes('mono-toggle-text-hr')

                            # Custom Zoom Controls (Top Left - Dark Glass)
                            with ui.column().classes('absolute top-2 left-2 z-[9999] gap-2'):
                                # Zoom In
                                with ui.button(icon='add').props('round dense flat no-caps ripple=false').classes('map-zoom-btn no-ripple').style('background-color: rgba(9, 9, 11, 0.72) !important; color: #ffffff !important; border: 1px solid rgba(255, 255, 255, 0.14) !important;').on('click', lambda: m.run_map_method("zoomIn")):
                                    pass
                                # Zoom Out
                                with ui.button(icon='remove').props('round dense flat no-caps ripple=false').classes('map-zoom-btn no-ripple').style('background-color: rgba(9, 9, 11, 0.72) !important; color: #ffffff !important; border: 1px solid rgba(255, 255, 255, 0.14) !important;').on('click', lambda: m.run_map_method("zoomOut")):
                                    pass

                            # Speed Legend (Glassmorphism - Top Right)
                            with ui.column().classes('absolute top-2 right-2 z-[9999] p-3 rounded-xl bg-black/60 backdrop-blur-md border border-white/10 shadow-lg gap-1'):
                                # Gradient Bar
                                legend_bar = ui.element('div').classes('w-32 h-2 rounded-full').style(f"background: {map_route_palette['pace']}")
                                # Labels
                                with ui.row().classes('w-full justify-between text-[10px] text-zinc-300 font-medium tracking-wide'):
                                    legend_left = ui.label(map_legend_left['pace'])
                                    legend_right = ui.label(map_legend_right['pace'])

                            def _set_map_mode(mode):
                                if mode not in ('pace', 'hr'):
                                    mode = 'pace'
                                if map_color_mode['value'] == mode:
                                    return
                                map_color_mode['value'] = mode

                                # Route redraw
                                _render_route(mode)

                                # Re-add start/end markers (route redraw clears all map layers)
                                if _valid_latlng(start_point) or _valid_latlng(end_point):
                                    try:
                                        asyncio.create_task(_add_pro_markers_once_ready())
                                    except Exception:
                                        pass

                                # Legend update
                                legend_bar.style(f"background: {map_route_palette.get(mode, map_route_palette['pace'])}")
                                legend_left.set_text(map_legend_left.get(mode, map_legend_left['pace']))
                                legend_right.set_text(map_legend_right.get(mode, map_legend_right['pace']))

                            if final_bounds:
                                # defer fit until modal is visible; execute after dialog open lifecycle
                                modal_map = m
                                modal_fit_bounds = final_bounds

                    
                    # -------------------------------------------
                            
                    # -------------------------------------------

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
                            
                            ui.label(metrics_str).classes('text-zinc-400 font-sans tabular-nums text-sm tracking-wide')

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
                        
                        # Title
                        ui.label('TIME IN HEART RATE ZONES').classes('text-lg font-bold text-white mb-2')
                        
                        hr_chart = self.create_hr_zone_chart(hr_zones)
                        ui.plotly(hr_chart).classes('w-full')

                    # 3. Body Response Row
                    session_data = detail_data.get('session_data', {})
                    if not session_data.get('avg_cadence'):
                        session_data['avg_cadence'] = activity.get('avg_cadence', 0)
                    
                    has_dynamics = session_data and any([session_data.get('avg_vertical_oscillation'), session_data.get('avg_stance_time')])
                    
                    with ui.row().classes('w-full gap-3 items-stretch'):
                        if has_dynamics:
                            with ui.column().classes('flex-1 min-w-0'):
                                self.create_running_dynamics_card(session_data)
                        with ui.column().classes('flex-1 min-w-0'):
                            self.create_decoupling_card(decoupling, efficiency_factor=activity.get('efficiency_factor', 0))
                        if session_data:
                            with ui.column().classes('flex-1 min-w-0'):
                                self.create_physiology_card(session_data, activity)

                    # 4. Cadence-Elevation Chart
                    if detail_data.get('cadence_stream') and detail_data.get('distance_stream'):
                        with ui.card().classes('w-full bg-zinc-900 p-4 border border-zinc-800').style('border-radius: 8px; box-shadow: 0px 4px 12px rgba(0, 0, 0, 0.4);'):
                            # Title
                            ui.label('CADENCE & FORM ANALYSIS').classes('text-lg font-bold text-white mb-2')
                            
                            cadence_chart = self.create_form_analysis_chart(
                                detail_data['distance_stream'], 
                                detail_data['cadence_stream'], 
                                detail_data['elevation_stream'], 
                                timestamps=detail_data.get('timestamps'), 
                                vertical_oscillation=detail_data.get('vertical_oscillation'),
                                stance_time=detail_data.get('stance_time'),
                                vertical_ratio=detail_data.get('vertical_ratio'),
                                step_length=detail_data.get('step_length'),
                                use_miles=True
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

        if modal_map and modal_fit_bounds:
            fit_options = {'padding': [26, 26], 'maxZoom': 18, 'animate': False}

            def _stabilize_and_fit():
                try:
                    modal_map.run_map_method('invalidateSize')
                    modal_map.run_map_method('fitBounds', modal_fit_bounds, fit_options)
                except Exception as ex:
                    print(f"Map fitBounds retry failed: {ex}")

            def _finalize_map_open():
                _stabilize_and_fit()
                if modal_map_card:
                    modal_map_card.style('opacity: 1; transition: opacity 0.18s ease;')

            # Two-pass strategy for smooth first paint:
            # 1) fit while map is still transparent
            # 2) refit after dialog settles, then fade in
            ui.timer(0.02, _stabilize_and_fit, once=True)
            ui.timer(0.08, _finalize_map_open, once=True)

 
    def build_ui(self):
        """Construct the complete UI layout with fixed sidebar."""
        
        # 1. INTEGRATED CSS CONFIGURATION
        ui.add_head_html('''
        <style>
        /**************************************************/
        /* 1. THE APPLE-STYLE HOVER FIX (DARK MODE)       */
        /**************************************************/

        /* Hide Plotly Modebar */
        .modebar {
            display: none !important;
        }
        
        /* Disable Quasar's default 'ripple' overlay */
        .q-btn.no-ripple .q-focus-helper {
            display: none !important;
        }
        
        /* Manual hover state: Glassy highlight for Dark Mode */
        /* Exclude filter-active buttons — they get their own hover below */
        .q-btn.no-ripple:not(.filter-active):hover {
            /* This creates a subtle lightening effect on dark backgrounds */
            background-color: rgba(255, 255, 255, 0.1) !important; 
            transition: background-color 0.2s ease;
        }
        
        /* Active filter buttons: darken slightly on hover instead of whitening */
        .q-btn.filter-active:hover {
            filter: brightness(1.15) !important;
            transition: filter 0.2s ease;
        }
        
        /* Subtle tactile scale effect on click */
        .q-btn.no-ripple:active {
            transform: scale(0.97);
        }

        /**************************************************/
        /* 2. PLOTLY & LAYOUT RESETS                      */
        /**************************************************/
        .modebar-btn--logo, .plotly .notifier, .js-plotly-plot .notifier { 
            display: none !important; 
        }
        
        body, .q-page, .nicegui-content { margin: 0 !important; padding: 0 !important; }
        
        html, body, #app, .q-page-container, .q-page {
            background-color: #09090b !important;
            color: #e4e4e7 !important;
        }
        
        body {
            background-image: radial-gradient(circle at 50% 0%, #27272a 0%, transparent 60%) !important;
            background-attachment: fixed !important;
        }
        
        html, body { overscroll-behavior: none !important; overflow: hidden !important; }
        .q-page { overflow-y: auto !important; }
        
        /* Ensure table actions are clickable */
        .q-table tbody td { position: relative; }

        /**************************************************/
        /* 3. TOAST NOTIFICATION ANIMATIONS (PRESERVED)   */
        /**************************************************/
        @keyframes slideIn {
            from { transform: translateX(400px); opacity: 0; }
            to { transform: translateX(0); opacity: 1; }
        }
        
        @keyframes slideOut {
            from { transform: translateX(0); opacity: 1; }
            to { transform: translateX(400px); opacity: 0; }
        }

        /**************************************************/
        /* 3.5 "DARK GLASS" ACTIVITY CARD                 */
        /**************************************************/
        .glass-card {
            /* Base Dynamic Properties driven by python-injected variables */
            background: radial-gradient(circle at top right, var(--strain-bg), transparent 70%), #1b1b1b !important;
            border: 1px solid var(--strain-border) !important;
            box-shadow: inset 0 1px 0 0 rgba(255, 255, 255, 0.05), 0 4px 20px var(--strain-shadow) !important;
            border-radius: 12px !important;
            transition: transform 0.2s ease, box-shadow 0.2s ease, border-color 0.2s ease !important;
        }
        .glass-card:hover {
            transform: translateY(-2px) !important;
            /* Intensify glow to 20% and brighter top edge */
            box-shadow: inset 0 1px 0 0 rgba(255, 255, 255, 0.1), 0 8px 30px var(--strain-shadow-hover) !important;
            border-color: var(--strain-border-hover) !important;
        }

        /* 3.6 INTERACTIVE "LIFT & GLOW" EFFECT */
        .interactive-card {
            transition: all 0.3s ease-out !important;
            /* Base state matches glass-card but explicit transition ensures smoothness */
        }
        
        .interactive-card:hover {
            /* Lift Effect */
            transform: translateY(-4px) !important;
            /* Expanded Glow Shadow using injected RGB variable */
            /* We use !important to override the base glass-card hover shadow if needed, or blend them */
            box-shadow: 0 12px 40px rgba(var(--theme-color-rgb), 0.3) !important;
            /* Brightened Border */
            border-color: rgba(255, 255, 255, 0.3) !important;
        }

        /**************************************************/
        /* 4. TAB TRANSITIONS (PRESERVED)                 */
        /**************************************************/
        .q-tab-panels { overflow: hidden; position: relative; }
        .q-tab-panel {
            transition: transform 0.8s cubic-bezier(0.16, 1, 0.3, 1) !important, 
                        opacity 0.8s cubic-bezier(0.16, 1, 0.3, 1) !important;
            will-change: transform, opacity;
        }
        
        .q-tab-panel[aria-hidden="true"] {
            position: absolute; top: 0; left: 0; width: 100%;
            opacity: 0; pointer-events: none; transform: translateX(30px);
        }
        
        .q-tab-panel[aria-hidden="false"] {
            position: relative; opacity: 1; pointer-events: auto; transform: translateX(0);
        }

        /**************************************************/
        /* 5. COPY SPLITS BUTTON STYLING (PRESERVED)      */
        /**************************************************/
        .copy-splits-btn.q-btn {
            opacity: 0.3 !important;
            transition: all 0.15s ease !important;
        }
        .copy-splits-btn.q-btn .q-icon {
            color: #9ca3af !important;
        }
        .copy-splits-btn.q-btn:hover {
            opacity: 1 !important;
            background-color: rgba(255, 255, 255, 0.1) !important;
        }
        .copy-splits-btn.q-btn:hover .q-icon {
            color: #ffffff !important;
        }

        /**************************************************/
        /* 5.5 MAP ZOOM BUTTONS (ACTIVITY MODAL)         */
        /**************************************************/
        .map-zoom-btn.q-btn {
            min-width: 2.25rem !important;
            min-height: 2.25rem !important;
            width: 2.25rem !important;
            height: 2.25rem !important;
            padding: 0 !important;
            border-radius: 9999px !important;
            background: rgba(0, 0, 0, 0.60) !important;
            border: 1px solid rgba(255, 255, 255, 0.10) !important;
            color: #ffffff !important;
            box-shadow: 0 8px 20px rgba(0, 0, 0, 0.35) !important;
            backdrop-filter: blur(12px) saturate(120%);
            -webkit-backdrop-filter: blur(12px) saturate(120%);
            transition: transform 0.18s ease, background-color 0.18s ease, border-color 0.18s ease, box-shadow 0.18s ease;
        }
        .map-zoom-btn.q-btn.bg-primary,
        .map-zoom-btn.q-btn.text-primary {
            background: rgba(9, 9, 11, 0.72) !important;
            color: #ffffff !important;
        }
        .map-zoom-btn.q-btn .q-focus-helper {
            display: none !important;
        }
        .map-zoom-btn.q-btn .q-btn__content {
            color: #ffffff !important;
        }
        .map-zoom-btn.q-btn .q-icon {
            color: #ffffff !important;
            font-size: 18px !important;
        }
        .map-zoom-btn.q-btn:hover {
            transform: translateY(-1px) !important;
            background: rgba(24, 24, 27, 0.68) !important;
            border-color: rgba(255, 255, 255, 0.18) !important;
            box-shadow: 0 10px 24px rgba(0, 0, 0, 0.42) !important;
        }
        .map-zoom-btn.q-btn:active {
            transform: scale(0.97) !important;
        }

        /**************************************************/
        /* 5.55 MAP LAYER TOGGLE (CUSTOM MONO SLIDER)      */
        /**************************************************/
        .mono-toggle-wrapper {
            position: absolute;
            top: 10px;
            left: 50%;
            transform: translateX(-50%);
            z-index: 9999;
            pointer-events: auto;
        }
        .mono-toggle-checkbox {
            display: none !important;
        }
        .mono-toggle-label {
            width: 140px;
            height: 34px;
            display: block;
            position: relative;
            cursor: pointer;
            border-radius: 9999px;
            overflow: hidden;
            background: rgba(0, 0, 0, 0.58);
            backdrop-filter: blur(10px);
            -webkit-backdrop-filter: blur(10px);
            border: 1px solid rgba(255, 255, 255, 0.14);
            box-shadow: 0 6px 16px rgba(0, 0, 0, 0.34);
        }
        .mono-toggle-label::before {
            content: 'PACE';
            position: absolute;
            top: 2px;
            left: 2px;
            width: 68px;
            height: 28px;
            border-radius: 9999px;
            background: #ffffff;
            color: #059669;
            font-size: 11px;
            font-weight: 800;
            letter-spacing: 0.08em;
            display: flex;
            align-items: center;
            justify-content: center;
            box-shadow: 0 2px 8px rgba(16, 185, 129, 0.30);
            z-index: 2;
            transition: transform 420ms cubic-bezier(0.47, -0.3, 0.21, 1.33);
        }
        .mono-toggle-checkbox:checked + .mono-toggle-label::before {
            transform: translateX(68px);
            content: 'HR';
            background: #ffffff;
            color: #dc2626;
            box-shadow: 0 2px 8px rgba(239, 68, 68, 0.30);
        }
        .mono-toggle-text-pace,
        .mono-toggle-text-hr {
            position: absolute;
            top: 50%;
            transform: translateY(-50%);
            font-size: 10px;
            font-weight: 700;
            letter-spacing: 0.08em;
            color: rgba(255, 255, 255, 0.60);
            pointer-events: none;
            z-index: 1;
            transition: color 0.25s ease;
            text-transform: uppercase;
        }
        .mono-toggle-text-pace { left: 18px; }
        .mono-toggle-text-hr { right: 24px; }

        .mono-toggle-checkbox:not(:checked) + .mono-toggle-label .mono-toggle-text-pace {
            color: rgba(255, 255, 255, 0.00);
        }
        .mono-toggle-checkbox:not(:checked) + .mono-toggle-label .mono-toggle-text-hr {
            color: rgba(255, 255, 255, 0.66);
        }
        .mono-toggle-checkbox:checked + .mono-toggle-label .mono-toggle-text-pace {
            color: rgba(255, 255, 255, 0.66);
        }
        .mono-toggle-checkbox:checked + .mono-toggle-label .mono-toggle-text-hr {
            color: rgba(255, 255, 255, 0.00);
        }

        /**************************************************/
        /* 5.7 PRO START/END MARKERS (LEAFLET DIVICON)    */
        /**************************************************/
        .pro-marker {
            display: flex !important;
            align-items: center;
            justify-content: center;
            border-radius: 9999px;
            border: 2px solid #ffffff;
            box-shadow: 0 4px 12px rgba(0, 0, 0, 0.4);
            transition: transform 0.2s ease;
            background-clip: padding-box;
        }
        .pro-marker:hover {
            transform: scale(1.08);
            z-index: 1000 !important;
        }
        .pm-start { background-color: #10b981; }
        .pm-end { background-color: #ef4444; }
        .pro-marker svg {
            display: block;
        }

        /**************************************************/
        /* 5.6 LEAFLET TILE SEAM FIX (PHANTOM LINE)      */
        /**************************************************/
        .leaflet-seam-fix,
        .leaflet-seam-fix .leaflet-container {
            background: #09090b !important;
        }
        .leaflet-seam-fix .leaflet-pane,
        .leaflet-seam-fix .leaflet-tile-container,
        .leaflet-seam-fix .leaflet-tile {
            -webkit-backface-visibility: hidden;
            backface-visibility: hidden;
        }
        .leaflet-seam-fix img,
        .leaflet-seam-fix .leaflet-tile {
            border: 0 !important;
            max-width: none !important;
            will-change: transform;
        }

        /**************************************************/
        /* 6. DROPDOWN & SELECT STYLING (PRESERVED)       */
        /**************************************************/
        .q-menu .q-item { color: white !important; background-color: #1F1F1F !important; }
        .q-menu .q-item:hover { background-color: #2CC985 !important; }
        .q-field__native, .q-field__input, .q-field__append .q-icon { color: white !important; }
        .q-field--outlined .q-field__control:before { border-color: rgba(255, 255, 255, 0.3) !important; }
        .q-field--outlined .q-field__control:hover:before { border-color: rgba(255, 255, 255, 0.6) !important; }
        </style>
        ''')
        
        # 2. COLOR SCHEME
        ui.colors(primary='#10B981', secondary='#1F1F1F', accent='#ff9900', 
                  dark='#09090b', positive='#10B981', negative='#ff4d4d', 
                  info='#3b82f6', warning='#ff9900')
        
        ui.query('body').classes('bg-zinc-950')
        
        # 3. MAIN LAYOUT
        with ui.row().classes('w-full h-screen m-0 p-0 gap-0 no-wrap overflow-hidden'):
            self.build_sidebar()
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
            ).classes('w-full mb-4 bg-zinc-900').style('color: white;').props('outlined dense dark behavior="menu"')
            
            # Action buttons - Modern solid style with Apple aesthetics
            # Action buttons - Reverted to Dark/Subtle Style for Utilities
            with ui.row().classes('w-full gap-2 mt-4 mb-2'):
                # Import Button: Dark Grey, White Text, Subtle Hover
                ui.button('IMPORT FOLDER', on_click=self.select_folder, icon='folder_open').classes('flex-1 bg-zinc-800 text-white hover:bg-zinc-700').props('flat')
            
            # Export Button: Dark Grey, White Text, Subtle Hover
            self.export_btn = ui.button('EXPORT CSV', on_click=self.export_csv, icon='download').classes('w-full bg-zinc-800 text-white hover:bg-zinc-700 mb-2').props('flat disable')
            
            # Visual Separator
            ui.separator().classes('my-2 bg-zinc-800')

            # "Copy for AI" Button (HERO STYLE: Gradient + Glow + Margin)
            # Added mt-4 for adjusted isolation (Total ~2rem with separator)
            self.copy_btn = ui.button('COPY FOR AI', on_click=self.copy_to_llm, icon='content_copy').classes(
                'w-full text-white font-bold tracking-wide transform transition-transform duration-300 hover:scale-105 mt-4'
            ).props('disable').style(
                'background: linear-gradient(to right, #10b981, #14b8a6); box-shadow: 0 0 15px rgba(16, 185, 129, 0.4); border: none;'
            )
            
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
        # Outer wrapper: No padding, flush to edges with dark background
        with ui.column().classes('flex-1 h-screen overflow-hidden p-0 gap-0'):
            # Inner container: Adds padding for content breathing room (no bottom padding)
            # Increased pt-6 -> pt-10 for global vertical rhythm
            with ui.column().classes('w-full min-h-full pt-6 px-6 pb-0 gap-4'):
                # Create tabs row with absolute positioned Save Chart button
                with ui.row().classes('w-full items-center mb-0 relative pb-3'):
                    # Tabs centered, taking full width
                    with ui.tabs(on_change=lambda e: self.toggle_save_chart_button(e.value)).classes('w-full justify-center').props('active-color="white" indicator-color="orange" align="center" content-class="text-zinc-500"') as tabs:
                        trends_tab = ui.tab('Trends')
                        report_tab = ui.tab('FEED')
                        activities_tab = ui.tab('ACTIVITIES')
                    
                    # Save Chart button absolutely positioned on the right (minimal, icon-focused)
                    # Add transition for smooth fade effect
                    self.save_chart_btn = ui.button(icon='download', on_click=self.save_chart_to_downloads, color=None).classes('text-white absolute right-0 top-0 z-10').style('background-color: #27272a; border-radius: 6px; border: none; padding: 8px; min-width: 40px; transition: opacity 0.3s ease-in-out, background-color 0.2s ease;').props('flat dense').tooltip('Save Chart')
                
                # Create tab panels with transparent background
                with ui.tab_panels(tabs, value=trends_tab).classes('w-full flex-1').props('transparent'):
                    # Trends tab panel
                    with ui.tab_panel(trends_tab).classes('p-0'):
                        self.build_trends_tab()
                    
                    # Report tab panel
                    with ui.tab_panel(report_tab).classes('p-0'):
                        self.build_report_tab()
                    
                    # Activities tab panel
                    with ui.tab_panel(activities_tab).classes('p-0 h-full flex flex-col'):
                        self.build_activities_tab()
                
        # Floating Action Bar (Lives at root level to float above everything)
        self.fab_container = ui.row().classes(
            'fixed bottom-8 left-1/2 transform -translate-x-1/2 ' # Centered at bottom
            'bg-zinc-800/90 backdrop-blur-md text-white px-6 py-2 rounded-full '
            'shadow-2xl border border-zinc-700 z-50 items-center gap-4 '
            'transition-all duration-300 translate-y-[150%] opacity-0 pointer-events-none'
        )
    
    def build_trends_tab(self):
        """Create trends tab with embedded Plotly chart."""
        # Store the plotly_container as an instance variable so it can be updated later
        # Added 'p-4' to match Feed container's padding for perfect vertical alignment
        self.plotly_container = ui.column().classes('w-full p-8').style('min-height: 900px;')
        
        with self.plotly_container:
            # Show placeholder message when no data is available
            ui.label('No data available. Import activities to view trends.').classes(
                'text-center text-zinc-400 mt-20'
            )
    
    def build_report_tab(self):
        """Create report tab with Card View."""
        # RETRY FIX: Feed Card Hover Cut-off
        # Reverted pt-10 to standard p-4. Global container padding handles the rhythm now.
        with ui.scroll_area().classes('w-full h-full'):
            # Removed 'pt-10', keeping 'p-4' for internal spacing
            self.feed_container = ui.column().classes('w-full max-w-4xl mx-auto gap-4 p-4 pb-8')
    
    def build_activities_tab(self):
        """Create activities tab with Filter, Table, and Floating Action Bar."""
        # 1. Filter Bar
        # Added 'p-4' to wrapper so it starts at same vertical height as Feed/Trends
        # Note: We need a wrapping column to hold the padding, or apply it to the row
        with ui.column().classes('w-full flex-1 overflow-hidden p-8 gap-2'):
            self.filter_container = ui.row().classes('w-full mb-0 gap-2')
            
            # 2. Table (Moved inside this wrapper)
            self.grid_container = ui.column().classes('w-full flex-1 overflow-hidden')
        
        # 3. Initial Render
        self.update_filter_bar()
        self.update_activities_grid()

    def show_floating_action_bar(self, selected_rows):
        """Shows the floating bar when rows are selected."""

        # FIX: Explicit check for empty list
        if not selected_rows or len(selected_rows) == 0:
            self.hide_floating_action_bar()
            return
            
        count = len(selected_rows)
        self.fab_container.clear()
        self.fab_container.classes(
            remove='translate-y-[150%] opacity-0 pointer-events-none', 
            add='translate-y-0 opacity-100 pointer-events-auto'
        )
        
        with self.fab_container:
            # COUNT BADGE — Emerald tint
            with ui.element('div').classes(
                'flex items-center justify-center rounded-full px-3 py-0.5 mr-1'
            ).style('background: rgba(16, 185, 129, 0.12); border: 1px solid rgba(16, 185, 129, 0.3);'):
                ui.label(f"{count}").classes('font-bold text-base text-emerald-400')
            ui.label('Selected').classes('text-sm text-zinc-400 mr-3 font-medium')
            
            # FOCUS BUTTON — Emerald accent, unified with brand
            ui.button('Focus', icon='center_focus_strong', color=None,
                      on_click=lambda: self.enter_focus_mode(selected_rows)
            ).props('flat dense no-caps').classes(
                'text-emerald-400 hover:text-emerald-300 font-semibold text-sm'
            )
            
            # SEPARATOR
            ui.element('div').classes('w-px h-5 mx-1').style('background: rgba(255,255,255,0.1);')
            
            # DOWNLOAD — White icon
            ui.button(icon='download', color=None,
                      on_click=lambda: self.bulk_download(selected_rows)
            ).props('flat round dense').classes('text-zinc-400 hover:text-white')
            
            # DELETE — Red on hover
            ui.button(icon='delete_outline', color=None,
                      on_click=lambda: self.bulk_delete(selected_rows)
            ).props('flat round dense').classes('text-zinc-400 hover:text-red-400')

    def hide_floating_action_bar(self):
        """Hides the floating bar."""
        self.fab_container.classes(
            remove='translate-y-0 opacity-100 pointer-events-auto', 
            add='translate-y-[150%] opacity-0 pointer-events-none'
        )

    def enter_focus_mode(self, selected_rows):
        """Filters the ENTIRE APP to just these rows."""
        # 1. Get the hashes (robustly)
        hashes = []
        for row in selected_rows:
            if isinstance(row, dict):
                # Grab the real DB hash we stored in the row
                h = row.get('hash')
                if h:
                    hashes.append(h)
            elif isinstance(row, str):
                # If it's just a string ID, use it (unless it's a temp ID)
                if not row.startswith('temp_'):
                    hashes.append(row)
        
        # Safety check
        if not hashes:
            ui.notify("No valid activities selected", type='warning')
            return

        # 2. Filter the main data list in memory
        self.activities_data = [act for act in self.activities_data if act.get('db_hash') in hashes]
        
        # 3. Update DataFrame (Critical for charts to work)
        if self.activities_data:
            self.df = pd.DataFrame(self.activities_data)
            # Ensure date objects exist for the charts
            if 'date' in self.df.columns:
                 self.df['date_obj'] = pd.to_datetime(self.df['date'])
        else:
             self.df = None
        
        # 4. Update State
        self.focus_mode_active = True
        focus_label = f"🎯 Focus ({len(hashes)})"
        self.current_timeframe = focus_label
        
        # 5. Inject Focus Mode into the dropdown and style it
        # Guard flag prevents on_filter_change from immediately undoing our work
        self._entering_focus_mode = True
        current_options = list(self.timeframe_select.options)
        if focus_label not in current_options:
            current_options.append(focus_label)
            self.timeframe_select.options = current_options
        self.timeframe_select.value = focus_label
        self._entering_focus_mode = False
        # Add a neon glow to the dropdown so user knows they're in a special state
        self.timeframe_select.style(add='border: 1px solid #34d399; box-shadow: 0 0 12px rgba(16, 185, 129, 0.4); border-radius: 8px;')
        
        # 6. Refresh All Views (Feed, Charts, and Table)
        self.update_trends_chart()
        self.update_report_text()
        self.update_activities_grid()
        
        # 7. Hide the bar
        self.hide_floating_action_bar()
        ui.notify("Focus Mode — select a timeframe to exit", type='info', icon='center_focus_strong')

    async def exit_focus_mode(self):
        """Exit Focus Mode — restore original timeframe and reload data."""
        self.focus_mode_active = False
        
        # Remove the Focus option from dropdown and restore styling
        self._entering_focus_mode = True
        current_options = [o for o in self.timeframe_select.options if not str(o).startswith('🎯')]
        self.timeframe_select.options = current_options
        self.timeframe_select.value = 'Last 30 Days'
        self._entering_focus_mode = False
        self.timeframe_select.style(remove='border: 1px solid #34d399; box-shadow: 0 0 12px rgba(16, 185, 129, 0.4); border-radius: 8px;')
        
        # Update timeframe and reload data
        self.current_timeframe = 'Last 30 Days'
        await self.refresh_data_view()
        ui.notify("Exited Focus Mode", type='info', icon='zoom_out')

    async def bulk_download(self, rows):
        """Download FIT files for all selected rows to ~/Downloads."""
        import shutil
        count = 0
        errors = 0
        for row in rows:
            try:
                if isinstance(row, dict):
                    source = row.get('file_path', '')
                    if not source:
                        print(f"DEBUG bulk_download: no file_path in row {row.get('id', '?')}")
                        errors += 1
                        continue
                    if not os.path.exists(source):
                        print(f"DEBUG bulk_download: file not found: {source}")
                        errors += 1
                        continue
                    dest = os.path.join(os.path.expanduser('~/Downloads'), os.path.basename(source))
                    shutil.copy2(source, dest)
                    count += 1
            except Exception as ex:
                print(f"DEBUG bulk_download error: {ex}")
                errors += 1
        # Clear selection and hide FAB
        if hasattr(self, 'activities_table') and self.activities_table:
            self.activities_table.selected.clear()
            self.activities_table.update()
        self.hide_floating_action_bar()
        if count > 0:
            ui.notify(f"Saved {count} file{'s' if count != 1 else ''} to Downloads", type='positive', icon='download')
        if errors > 0:
            ui.notify(f"{errors} file{'s' if errors != 1 else ''} could not be downloaded", type='warning')

    async def bulk_delete(self, rows):
        """Delete all selected activities after confirmation."""
        count = len(rows)
        word = "activities" if count > 1 else "activity"
        result = await ui.run_javascript(
            f'confirm("Delete {count} {word}?")',
            timeout=10
        )
        if result:
            deleted = 0
            for row in rows:
                if isinstance(row, dict):
                    h = row.get('hash')
                    if h:
                        self.db.delete_activity(h)
                        deleted += 0
            self.runs_count_label.text = f'{self.db.get_count()}'
            # Clear selection and hide FAB
            if hasattr(self, 'activities_table') and self.activities_table:
                self.activities_table.selected.clear()
            self.activities_table.update()
            self.hide_floating_action_bar()
            await self.refresh_data_view()
            ui.notify(f"Deleted {deleted} activit{'ies' if deleted != 1 else 'y'}", type='positive', icon='delete')
    
    async def refresh_data_view(self):
        """
        Refresh data using DB-side sorting.
        """
        # [Keep existing sorting logic...]
        sort_order = 'desc' if self.current_sort_desc else 'asc'

        self.activities_data = self.db.get_activities(
            self.current_timeframe, 
            self.current_session_id,
            sort_by=self.current_sort_by,
            sort_order=sort_order
        )
        
        if self.activities_data:
            self.df = pd.DataFrame(self.activities_data)
            self.df['date_obj'] = pd.to_datetime(self.df['date'])
        else:
            self.df = None
        
        # --- CRITICAL FIX: Update Filter Bar AFTER data load ---
        self.update_report_text() 
        self.update_activities_grid() 
        self.update_filter_bar() # <--- ADD THIS LINE
        self.update_trends_chart() 
        
        # Toggle buttons
        has_data = bool(self.activities_data)
        if has_data:
            self.export_btn.props(remove='disable')
            self.copy_btn.props(remove='disable')
        else:
            self.export_btn.props(add='disable')
            self.copy_btn.props(add='disable')

        # Background map payload migration for older activities (non-blocking)
        if self.activities_data:
            if self._map_backfill_task is None or self._map_backfill_task.done():
                self._map_backfill_task = asyncio.create_task(self._backfill_map_payloads_for_loaded_activities())
    
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
        # Guard: skip if we're programmatically setting focus mode value
        if self._entering_focus_mode:
            return
        
        # If we're in focus mode and user picks a DIFFERENT timeframe, exit focus mode
        if self.focus_mode_active and not str(e.value).startswith('🎯'):
            self.focus_mode_active = False
            # Remove the Focus option from dropdown and restore styling
            self._entering_focus_mode = True
            new_options = [o for o in self.timeframe_select.options if not str(o).startswith('🎯')]
            self.timeframe_select.options = new_options
            self._entering_focus_mode = False
            self.timeframe_select.style(remove='border: 1px solid #34d399; box-shadow: 0 0 12px rgba(16, 185, 129, 0.4); border-radius: 8px;')
        elif self.focus_mode_active and str(e.value).startswith('🎯'):
            # User re-selected the focus option itself, no-op
            return
        
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
            
            # Show Premium Progress Dialog
            progress_dialog = ui.dialog().props('persistent transition-show="scale" transition-hide="scale" backdrop-filter="blur(4px)"')
            with progress_dialog, ui.card().classes('bg-zinc-900/95 backdrop-blur-xl border border-white/10 p-8 shadow-2xl rounded-2xl items-center text-center w-[400px] gap-6'):
                
                # Header Section with Icon
                with ui.column().classes('items-center gap-2'):
                     # Pulsing folder icon
                     ui.icon('folder_open', size='3rem', color='emerald-500').classes('mb-2 opacity-90 animate-pulse')
                     ui.label('Importing Your Data').classes('text-xl font-bold tracking-tight text-white')
                     ui.label('Calculating metrics & analyzing GPS tracks...').classes('text-sm text-zinc-400 font-medium')

                # Progress Section
                with ui.column().classes('w-full gap-2'):
                    # Linear Progress with custom track color
                    # Explicitly disable show-value to prevent raw float display (the "weird text")
                    progress_bar = ui.linear_progress(value=0, show_value=False).classes('rounded-full h-2').props('color="emerald-500" track-color="grey-9" size="8px"')
                    
                    # Status Row
                    with ui.row().classes('w-full justify-between text-xs font-mono text-zinc-500'):
                         progress_label = ui.label('Initializing...')
                         percent_label = ui.label('0%')

                # Footer / Tech Detail
                # (Removed per user request)
            
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
                    percent_label.text = f'{int(progress * 100)}%'
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
    
    @staticmethod
    def hex_to_rgb(hex_color):
        """Convert #RRGGBB to 'R, G, B' string for CSS variables."""
        hex_color = hex_color.lstrip('#')
        return f"{int(hex_color[0:2], 16)}, {int(hex_color[2:4], 16)}, {int(hex_color[4:6], 16)}"

    def update_report_text(self):
        """Update report with Beautiful Cards. Fully restored styling."""
        self.feed_container.clear()
        
        if not self.activities_data:
            with self.feed_container:
                ui.label('No runs found for this timeframe.').classes('text-gray-600 italic')
            return
        
        # Calculate Average EF for context
        avg_ef = self.df['efficiency_factor'].mean() if self.df is not None else 0
        
        # Calculate long run threshold
        if self.df is not None and len(self.df) >= 5:
            long_run_threshold = self.df['distance_mi'].quantile(0.8)
        else:
            long_run_threshold = 10.0
        
        with self.feed_container:
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
                    strain_label, strain_color, strain_text_color = "Recovery", "#60a5fa", "#60a5fa" # Blue
                elif strain < 150:
                    strain_label, strain_color, strain_text_color = "Maintenance", "#10B981", "#10B981" # Green
                elif strain < 300:
                    strain_label, strain_color, strain_text_color = "Productive", "#f97316", "#f97316" # Orange
                else:
                    strain_label, strain_color, strain_text_color = "Overreaching", "#ef4444", "#ef4444" # Red
                
                # 4. Create the "Hot Card" with Dark Glass styling
                # Generate CSS variables for this specific card's theme
                
                # Helper to convert hex to rgba-like string for our CSS variables
                # Note: We are constructing the hex+alpha strings directly as requested
                # Productive (#f97316) -> BG: #f9731626, Border: #f973164D, etc.
                
                strain_bg = f"{strain_color}26"          # 15%
                strain_border = f"{strain_color}4D"      # 30%
                strain_border_hover = f"{strain_color}80"# 50%
                strain_shadow = f"{strain_color}1A"      # 10%
                strain_shadow_hover = f"{strain_color}33"# 20%
                
                # Convert strain_color to RGB triplet for dynamic shadow
                strain_rgb = self.hex_to_rgb(strain_color)
                
                activity_hash = d.get('db_hash')
                
                # Apply the .glass-card AND .interactive-card classes
                card = ui.card().classes(
                    'w-full p-4 glass-card interactive-card cursor-pointer relative overflow-hidden group'
                ).style(
                    f'max-width: 720px; margin: 0 auto; '
                    f'--theme-color-rgb: {strain_rgb}; ' # Inject RGB triplet for hover shadow
                    f'--strain-bg: {strain_bg}; '
                    f'--strain-border: {strain_border}; '
                    f'--strain-border-hover: {strain_border_hover}; '
                    f'--strain-shadow: {strain_shadow}; '
                    f'--strain-shadow-hover: {strain_shadow_hover};'
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
                        
                        # --- ROW 2: Context Tags ---
                        with ui.row().classes('w-full items-center gap-2 mt-2'):
                            for tag in run_type_tag.split(' | '):
                                ui.label(tag).classes(
                                    'text-[10px] font-bold px-2 py-0.5 rounded bg-zinc-800 text-zinc-400 border border-zinc-700 tracking-wide '
                                    'cursor-pointer hover:brightness-125 hover:-translate-y-px hover:shadow-lg transition-all duration-200'
                                )
                            
                            # --- UPDATED CHECK ---
                            # Check if the label exists AND is not the string "None"
                            te_label = d.get('te_label')
                            if te_label and str(te_label) != "None":
                                te_color = d.get('te_label_color', 'text-zinc-400')
                                
                                # Determine styling based on color class
                                if 'text-purple-400' in te_color: bg_color, border_color = 'bg-purple-500/10', 'border-purple-500/30'
                                elif 'text-red-400' in te_color: bg_color, border_color = 'bg-red-500/10', 'border-red-500/30'
                                elif 'text-orange-400' in te_color: bg_color, border_color = 'bg-orange-500/10', 'border-orange-500/30'
                                elif 'text-emerald-400' in te_color: bg_color, border_color = 'bg-emerald-500/10', 'border-emerald-500/30'
                                elif 'text-blue-400' in te_color: bg_color, border_color = 'bg-blue-500/10', 'border-blue-500/30'
                                else: bg_color, border_color = 'bg-zinc-800', 'border-zinc-700'
                                
                                text_color = te_color.split()[0] if ' ' in te_color else te_color
                                
                                physio_tag = ui.label(te_label).classes(
                                    f"text-[10px] font-bold px-2 py-0.5 rounded {bg_color} border {border_color} {text_color} "
                                    "tracking-wide cursor-pointer hover:brightness-125 transition-all"
                                )
                                physio_tag.on('click.stop', lambda t=te_label: self.show_training_effect_info(t))

                        # --- ROW 3: Main Metrics Grid ---
                        with ui.row().classes('w-full gap-4 mb-1 items-center mt-3'):
                            with ui.column().classes('flex-1'):
                                with ui.grid(columns=3).classes('w-full gap-3'):
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
                            
                            # Right side: verdict pills
                            with ui.row().classes('items-center gap-2 flex-wrap'):
                                # Aerobic Efficiency Verdict Pill
                                run_ef = d.get('efficiency_factor', 0)
                                run_cost = d.get('decoupling', 0)
                                ef_above_avg = run_ef >= avg_ef if avg_ef > 0 else False
                                low_decouple = run_cost <= 5
                                
                                if ef_above_avg and low_decouple:
                                    aero_verdict, aero_bg, aero_border, aero_text = 'Efficient', 'bg-emerald-500/10', 'border-emerald-700/30', 'text-emerald-400'
                                elif not ef_above_avg and low_decouple:
                                    aero_verdict, aero_bg, aero_border, aero_text = 'Base', 'bg-blue-500/10', 'border-blue-700/30', 'text-blue-400'
                                elif ef_above_avg and not low_decouple:
                                    aero_verdict, aero_bg, aero_border, aero_text = 'Pushing', 'bg-orange-500/10', 'border-orange-700/30', 'text-orange-400'
                                else:
                                    aero_verdict, aero_bg, aero_border, aero_text = 'Fatigued', 'bg-red-500/10', 'border-red-700/30', 'text-red-400'
                                
                                verdict_emojis = {
                                    'Efficient': '⚡', 'Base': '🧱', 'Pushing': '🔥', 'Fatigued': '⚠️'
                                }
                                aero_icon = verdict_emojis.get(aero_verdict, '⚡')
                                
                                # Fix Hover: Use explicit bg-opacity instead of brightness to avoid fading issue
                                hover_bg = aero_bg.replace('/10', '/30') 

                                aero_pill = ui.row().classes(f'items-center gap-2 px-3 py-1.5 rounded border {aero_bg} {aero_border} cursor-pointer hover:{hover_bg} transition-all')
                                with aero_pill:
                                    ui.label(aero_icon).classes('text-sm')
                                    ui.label('Efficiency:').classes(f'text-xs font-bold {aero_text}')
                                    ui.label(aero_verdict).classes(f'text-xs font-bold {aero_text}')
                                aero_pill.on('click.stop', lambda av=aero_verdict: self.show_aerobic_efficiency_info(highlight_verdict=av))

                                # Form Status Pill
                                form = analyze_form(d.get('avg_cadence'), d.get('avg_stance_time'), d.get('avg_step_length'), d.get('avg_vertical_oscillation'))
                                if form['verdict'] != 'ANALYZING':
                                    # Match colors to Analyzer.py / Legend
                                    if form['verdict'] == 'ELITE FORM':
                                        pill_bg, pill_border, pill_text = 'bg-emerald-500/10', 'border-emerald-700/30', 'text-emerald-400'
                                    elif form['verdict'] == 'GOOD FORM':
                                        # Fix: Good Form is Blue in legend/analyzer, not Emerald
                                        pill_bg, pill_border, pill_text = 'bg-blue-500/10', 'border-blue-700/30', 'text-blue-400'
                                    elif form['verdict'] == 'HEAVY FEET':
                                        pill_bg, pill_border, pill_text = 'bg-orange-500/10', 'border-orange-700/30', 'text-orange-400'
                                    elif form['verdict'] == 'PLODDING':
                                        # Fix: Plodding is Yellow in legend/analyzer, not Red
                                        pill_bg, pill_border, pill_text = 'bg-yellow-500/10', 'border-yellow-700/30', 'text-yellow-400'
                                    elif form['verdict'] == 'HIKING / REST':
                                        pill_bg, pill_border, pill_text = 'bg-blue-500/10', 'border-blue-700/30', 'text-blue-400'
                                    else:
                                        # Fallback for unexpected verdicts
                                        pill_bg, pill_border, pill_text = 'bg-slate-500/10', 'border-slate-700/30', 'text-slate-400'
                                    
                                    form_hover_bg = pill_bg.replace('/10', '/30')
                                    form_pill = ui.row().classes(f'items-center gap-2 px-3 py-1.5 rounded border {pill_bg} {pill_border} cursor-pointer hover:{form_hover_bg} transition-all')
                                    with form_pill:
                                        ui.label('🦶').classes('text-sm')
                                        ui.label('Form:').classes(f'text-xs font-bold {pill_text}')
                                        ui.label(form['verdict'].title()).classes(f'text-xs font-bold {pill_text}')
                                    form_pill.on('click.stop', lambda fv=form['verdict']: self.show_form_info(highlight_verdict=fv))
    
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
        Show informational modal about the current Volume lens.
        Content adapts based on self.volume_lens.
        """
        with ui.dialog() as dialog, ui.card().classes('bg-zinc-900 text-white p-6 max-w-2xl border border-zinc-800'):
            
            if self.volume_lens == 'quality':
                # === QUALITY LENS ===
                ui.label('Volume Quality Analysis').classes('text-xl font-bold text-white mb-2')
                with ui.column().classes('gap-6'):
                    with ui.row().classes('gap-4 items-start'):
                        ui.icon('verified').classes('text-emerald-400 text-2xl mt-1')
                        with ui.column().classes('gap-1'):
                            ui.label('High Quality (The Engine)').classes('text-base font-bold text-emerald-400')
                            ui.label('Running on flat/rolling terrain with good mechanics (Cadence > 160) and honest effort. These miles build fitness without breaking the chassis.').classes('text-sm text-zinc-300')
                    with ui.row().classes('gap-4 items-start'):
                        ui.icon('hiking').classes('text-blue-400 text-2xl mt-1')
                        with ui.column().classes('gap-1'):
                            ui.label('Structural (The Base)').classes('text-base font-bold text-blue-400')
                            ui.label('Valid volume that includes Hiking (Steep Grade), Recovery Shuffles (Low HR), or Walking. These miles build durability and aerobic base without the mechanical stress of fast running.').classes('text-sm text-zinc-300')
                    with ui.row().classes('gap-4 items-start'):
                        ui.icon('warning').classes('text-red-400 text-2xl mt-1')
                        with ui.column().classes('gap-1'):
                            ui.label('Broken (The Junk)').classes('text-base font-bold text-red-400')
                            ui.label('The "Danger Zone." You are working hard (High HR) but moving poorly (Low Cadence). This usually happens at the end of long runs when form falls apart. These miles cause injury.').classes('text-sm text-zinc-300')
                ui.separator().classes('my-6 border-zinc-800')
                with ui.column().classes('gap-2 mb-4'):
                    ui.label('How we decide:').classes('text-xs font-bold text-zinc-500 uppercase tracking-wider')
                    ui.label('We analyze every single mile split against Terrain, Metabolic Cost, and Mechanics.').classes('text-sm text-zinc-400')

            elif self.volume_lens == 'mix':
                # === TRAINING MIX LENS (verdict-focused) ===
                ui.label('Training Mix Analysis').classes('text-xl font-bold text-white mb-2')
                ui.label('Shows how your weekly mileage breaks down by run type (Easy, Tempo, Hard). Your verdict reflects the balance:').classes('text-sm text-zinc-400 mb-4')
                with ui.column().classes('gap-4'):
                    # Verdict: POLARIZED (ideal)
                    with ui.element('div').classes('rounded-lg p-3').style('background: rgba(16, 185, 129, 0.1); border: 1px solid rgba(16, 185, 129, 0.25);'):
                        with ui.row().classes('items-center gap-2 mb-1'):
                            ui.icon('check_circle').classes('text-emerald-400 text-lg')
                            ui.label('POLARIZED').classes('text-sm font-bold text-emerald-400')
                        ui.label('The gold standard — 80%+ easy miles with purposeful hard sessions. You\'re building a big aerobic engine while sharpening speed. Keep it up.').classes('text-sm text-zinc-300')
                    # Verdict: BALANCED
                    with ui.element('div').classes('rounded-lg p-3').style('background: rgba(59, 130, 246, 0.1); border: 1px solid rgba(59, 130, 246, 0.25);'):
                        with ui.row().classes('items-center gap-2 mb-1'):
                            ui.icon('balance').classes('text-blue-400 text-lg')
                            ui.label('BALANCED').classes('text-sm font-bold text-blue-400')
                        ui.label('A decent variety of run types. Not bad, but pushing more volume into easy runs would unlock better aerobic gains with less fatigue.').classes('text-sm text-zinc-300')
                    # Verdict: TEMPO HEAVY (warning)
                    with ui.element('div').classes('rounded-lg p-3').style('background: rgba(245, 158, 11, 0.1); border: 1px solid rgba(245, 158, 11, 0.25);'):
                        with ui.row().classes('items-center gap-2 mb-1'):
                            ui.icon('warning_amber').classes('text-amber-400 text-lg')
                            ui.label('TEMPO HEAVY').classes('text-sm font-bold text-amber-400')
                        ui.label('Too much time in the moderate zone without enough easy. This leads to chronic fatigue without the recovery to absorb it. Swap some tempo runs for true easy days.').classes('text-sm text-zinc-300')
                    # Verdict: MONOTONE
                    with ui.element('div').classes('rounded-lg p-3').style('background: rgba(249, 115, 22, 0.1); border: 1px solid rgba(249, 115, 22, 0.25);'):
                        with ui.row().classes('items-center gap-2 mb-1'):
                            ui.icon('repeat').classes('text-orange-400 text-lg')
                            ui.label('MONOTONE').classes('text-sm font-bold text-orange-400')
                        ui.label('Nearly all your miles are the same type. Add variety — even one tempo or interval session per week creates a stronger training stimulus.').classes('text-sm text-zinc-300')

            elif self.volume_lens == 'load':
                # === LOAD LENS (matches chart legend: Recovery/Maintenance/Productive/Overreaching) ===
                ui.label('Training Load Analysis').classes('text-xl font-bold text-white mb-2')
                ui.label('Each bar shows your weekly mileage broken down by training stress. Strain is calculated from duration, intensity, and heart rate for each run:').classes('text-sm text-zinc-400 mb-4')
                with ui.column().classes('gap-4'):
                    # Legend/Verdict: MAINTAINING (healthy balance)
                    with ui.element('div').classes('rounded-lg p-3').style('background: rgba(16, 185, 129, 0.1); border: 1px solid rgba(16, 185, 129, 0.25);'):
                        with ui.row().classes('items-center gap-2 mb-1'):
                            ui.icon('check_circle').classes('text-emerald-400 text-lg')
                            ui.label('MAINTAINING').classes('text-sm font-bold text-emerald-400')
                        ui.label('A healthy mix of easy and hard runs — your training load is sustainable and balanced. The sweet spot for steady progress without burnout.').classes('text-sm text-zinc-300')
                    # Legend: Recovery
                    with ui.element('div').classes('rounded-lg p-3').style('background: rgba(96, 165, 250, 0.1); border: 1px solid rgba(96, 165, 250, 0.25);'):
                        with ui.row().classes('items-center gap-2 mb-1'):
                            ui.icon('spa').classes('text-blue-400 text-lg')
                            ui.label('UNDERTRAINED').classes('text-sm font-bold text-blue-400')
                        ui.label('Nearly all recovery-level runs with no real stimulus. Great after a race, but sustained easy-only training won\'t build fitness. Add one harder session per week.').classes('text-sm text-zinc-300')
                    # Legend: Productive
                    with ui.element('div').classes('rounded-lg p-3').style('background: rgba(249, 115, 22, 0.1); border: 1px solid rgba(249, 115, 22, 0.25);'):
                        with ui.row().classes('items-center gap-2 mb-1'):
                            ui.icon('trending_up').classes('text-orange-400 text-lg')
                            ui.label('PRODUCTIVE').classes('text-sm font-bold text-orange-400')
                        ui.label('Heavy productive volume — you\'re pushing hard. This builds fitness fast but can\'t be sustained indefinitely. Plan a step-back week every 3-4 weeks.').classes('text-sm text-zinc-300')
                    # Legend: Overreaching
                    with ui.element('div').classes('rounded-lg p-3').style('background: rgba(239, 68, 68, 0.1); border: 1px solid rgba(239, 68, 68, 0.25);'):
                        with ui.row().classes('items-center gap-2 mb-1'):
                            ui.icon('warning').classes('text-red-400 text-lg')
                            ui.label('OVERREACHING').classes('text-sm font-bold text-red-400')
                        ui.label('Too many high-stress sessions. An occasional spike is fine, but repeated overreaching leads to injury and staleness. Follow with an easy week.').classes('text-sm text-zinc-300')

            elif self.volume_lens == 'zones':
                # === HR ZONES LENS (5 zone-anchored verdicts) ===
                ui.label('Heart Rate Zones Analysis').classes('text-xl font-bold text-white mb-2')
                ui.label('Shows weekly time distribution across heart rate zones. The 80/20 rule says ~80% of training should be easy (Zone 1-2) and ~20% hard (Zone 4-5):').classes('text-sm text-zinc-400 mb-4')
                with ui.column().classes('gap-4'):
                    # Verdict: 80/20 BALANCED (ideal)
                    with ui.element('div').classes('rounded-lg p-3').style('background: rgba(16, 185, 129, 0.1); border: 1px solid rgba(16, 185, 129, 0.25);'):
                        with ui.row().classes('items-center gap-2 mb-1'):
                            ui.icon('check_circle').classes('text-emerald-400 text-lg')
                            ui.label('80/20 BALANCED').classes('text-sm font-bold text-emerald-400')
                        ui.label('The gold standard — ~80% easy, ~20% hard. You\'re building a massive aerobic engine while sharpening speed with controlled intensity. This is how elites train.').classes('text-sm text-zinc-300')
                    # Verdict: ZONE 2 BASE
                    with ui.element('div').classes('rounded-lg p-3').style('background: rgba(59, 130, 246, 0.1); border: 1px solid rgba(59, 130, 246, 0.25);'):
                        with ui.row().classes('items-center gap-2 mb-1'):
                            ui.icon('favorite').classes('text-blue-400 text-lg')
                            ui.label('ZONE 2 BASE').classes('text-sm font-bold text-blue-400')
                        ui.label('Nearly all Zone 1-2. This is where mitochondrial magic happens — fat oxidation, capillary density, cardiac efficiency. Great for base building, but one hard session per week rounds it out.').classes('text-sm text-zinc-300')
                    # Verdict: ZONE 3 JUNK (cautionary)
                    with ui.element('div').classes('rounded-lg p-3').style('background: rgba(245, 158, 11, 0.1); border: 1px solid rgba(245, 158, 11, 0.25);'):
                        with ui.row().classes('items-center gap-2 mb-1'):
                            ui.icon('pause_circle').classes('text-amber-400 text-lg')
                            ui.label('ZONE 3 JUNK').classes('text-sm font-bold text-amber-400')
                        ui.label('Too much time in the grey zone — not easy enough to recover, not hard enough to force adaptation. This is wasted effort. Slow your easy runs and make hard days truly hard.').classes('text-sm text-zinc-300')
                    # Verdict: ZONE 4 THRESHOLD ADDICT
                    with ui.element('div').classes('rounded-lg p-3').style('background: rgba(249, 115, 22, 0.1); border: 1px solid rgba(249, 115, 22, 0.25);'):
                        with ui.row().classes('items-center gap-2 mb-1'):
                            ui.icon('fitness_center').classes('text-orange-400 text-lg')
                            ui.label('ZONE 4 THRESHOLD ADDICT').classes('text-sm font-bold text-orange-400')
                        ui.label('Excessive threshold grinding. Zone 4 builds lactate clearance, but more than ~2 sessions/week accumulates fatigue without enough recovery. Add more easy volume between hard days.').classes('text-sm text-zinc-300')
                    # Verdict: ZONE 5 REDLINING
                    with ui.element('div').classes('rounded-lg p-3').style('background: rgba(239, 68, 68, 0.1); border: 1px solid rgba(239, 68, 68, 0.25);'):
                        with ui.row().classes('items-center gap-2 mb-1'):
                            ui.icon('local_fire_department').classes('text-red-400 text-lg')
                            ui.label('ZONE 5 REDLINING').classes('text-sm font-bold text-red-400')
                        ui.label('Too much VO2max-level effort. Zone 5 is powerful but demands ~48h recovery between sessions. Back off and rebuild your aerobic base — the speed will come back faster.').classes('text-sm text-zinc-300')


            # "Got it!" Button (all lenses)
            ui.button('Got it!', on_click=dialog.close).classes('w-full bg-green-600 hover:bg-green-500 text-white font-bold')

        dialog.open()

    def show_training_effect_info(self, label):
        """Explains the physiological takeaway of the Garmin Label."""
        # Normalize labels for dictionary lookup
        lookup = label.upper().strip()
        
        info = {
            'VO2 MAX': {
                'title': '🚀 VO2 MAX (Aerobic Capacity)',
                'takeaway': 'You pushed your aerobic ceiling. This run improved your body\'s ability to oxygenate muscles during high-sustained efforts.',
                'tip': 'Expect higher cardiac strain. This builds "race pace" durability.'
            },
            'MAX POWER': {
                'title': '⚡ MAX POWER (Anaerobic)',
                'takeaway': 'Short, explosive bursts. You worked the anaerobic system to improve raw speed and sprinting torque.',
                'tip': 'Metabolically expensive. Prioritize sleep tonight.'
            },
            'THRESHOLD': {
                'title': '🔥 THRESHOLD',
                'takeaway': 'Working at the edge of lactate clearance. This improves how long you can hold a fast pace before "burning out."',
                'tip': 'Crucial for half-marathon and marathon performance.'
            },
            'MAINTAINING': {
                'title': '🔷 MAINTAINING',
                'takeaway': 'The sweet spot for aerobic health. You hit the right volume to hold your fitness without overtaxing the system.',
                'tip': 'The backbone of a successful training block.'
            },
            'RECOVERY': {
                'title': '🧘 RECOVERY',
                'takeaway': 'Easy effort to promote blood flow. This facilitates repair without adding new structural damage.',
                'tip': 'If your heart rate felt high during this "Recovery" run, you might be over-fatigued.'
            },
            'BASE': {
                'title': '🟡 BASE',
                'takeaway': 'Low-intensity endurance work. This builds mitochondrial density and teaches your body to burn fat efficiently.',
                'tip': 'Stay slow to grow. Don\'t rush these runs.'
            }
        }
        
        # Fallback if the label isn't in our dictionary
        data = info.get(lookup, {
            'title': label, 
            'takeaway': 'This represents your primary physiological adaptation for this session.', 
            'tip': 'Consistency is the key to long-term gains.'
        })
        
        with ui.dialog() as dialog, ui.card().classes('p-6 bg-zinc-900 border border-zinc-800 rounded-xl shadow-2xl'):
            with ui.column().classes('gap-2'):
                ui.label(data['title']).classes('text-xl font-bold text-white')
                ui.separator().classes('bg-zinc-700')
                ui.label('TRAINING FOCUS:').classes('text-[10px] font-bold text-zinc-500 tracking-widest mt-2')
                ui.label(data['takeaway']).classes('text-zinc-300 leading-relaxed')
                
                if data['tip']:
                    with ui.row().classes('items-start gap-2 mt-4 p-3 bg-blue-500/10 rounded-lg border border-blue-500/20'):
                        ui.icon('lightbulb').classes('text-blue-400 text-sm mt-0.5')
                        ui.label(data['tip']).classes('text-sm italic text-blue-200 flex-1')
                
                ui.button('GOT IT', on_click=dialog.close).classes('mt-6 w-full bg-zinc-800 hover:bg-zinc-700 text-white font-bold')
        
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
    
    def show_aerobic_efficiency_info(self, highlight_verdict=None):
        """
        Show informational modal explaining Aerobic Efficiency, EF, Decoupling, and all 4 verdicts.
        If highlight_verdict is provided, that verdict section gets visually emphasized.
        """
        verdicts = [
            {
                'key': 'Efficient',
                'condition': 'High EF + Low Drift',
                'meaning': 'Your output is high relative to your effort. This is the sweet spot of training.',
                'color': 'text-emerald-400',
                'bg': 'bg-emerald-500/10',
                'border': 'border-emerald-500/30',
                'icon': '⚡'
            },
            {
                'key': 'Pushing',
                'condition': 'High EF + High Drift',
                'meaning': 'You were putting in effort and intensity, this is good training.',
                'color': 'text-orange-400',
                'bg': 'bg-orange-500/10',
                'border': 'border-orange-500/30',
                'icon': '🔥'
            },
            {
                'key': 'Base',
                'condition': 'Low EF + Low Drift',
                'meaning': 'This run is building your foundation.',
                'color': 'text-blue-400',
                'bg': 'bg-blue-500/10',
                'border': 'border-blue-500/30',
                'icon': '🧱'
            },
            {
                'key': 'Fatigued',
                'condition': 'Low EF + High Drift',
                'meaning': 'Clear warning label: output is low AND heart rate is rising. Your body is tired.',
                'color': 'text-red-400',
                'bg': 'bg-red-500/10',
                'border': 'border-red-500/30',
                'icon': '⚠️'
            }
        ]
        
        with ui.dialog() as dialog, ui.card().classes('p-6 bg-zinc-900 border border-zinc-800 rounded-xl shadow-2xl max-w-xl'):
            with ui.column().classes('gap-3'):
                ui.label('Understanding Aerobic Efficiency').classes('text-xl font-bold text-white')
                ui.separator().classes('bg-zinc-700')
                
                # EF Explanation
                with ui.column().classes('gap-1'):
                    ui.label('EFFICIENCY FACTOR (EF)').classes('text-[10px] font-bold text-zinc-500 tracking-widest')
                    ui.label('Speed ÷ Heart Rate — how much pace you get per heartbeat. Higher is better. Improving EF over weeks means your aerobic engine is growing.').classes('text-sm text-zinc-300 leading-relaxed')
                
                # Decoupling Explanation
                with ui.column().classes('gap-1 mt-1'):
                    ui.label('AEROBIC DECOUPLING').classes('text-[10px] font-bold text-zinc-500 tracking-widest')
                    ui.label('Cardiac drift — how much your heart rate rises while pace stays steady. Compares 1st half EF to 2nd half EF. Target < 5% for endurance fitness.').classes('text-sm text-zinc-300 leading-relaxed')
                
                ui.separator().classes('bg-zinc-700 mt-1')
                
                # Verdict Cards
                ui.label('VERDICTS').classes('text-[10px] font-bold text-zinc-500 tracking-widest')
                
                for v in verdicts:
                    is_highlighted = highlight_verdict and v['key'] == highlight_verdict
                    ring = 'ring-2 ring-white/30' if is_highlighted else ''
                    scale = 'scale-[1.02]' if is_highlighted else 'opacity-80'
                    
                    with ui.row().classes(f'items-start gap-3 p-3 rounded-lg border {v["bg"]} {v["border"]} {ring} {scale} transition-all'):
                        ui.label(v['icon']).classes('text-lg mt-0.5')
                        with ui.column().classes('gap-0.5 flex-1'):
                            with ui.row().classes('items-center gap-2'):
                                ui.label(v['key']).classes(f'text-sm font-bold {v["color"]}')
                                ui.label(f'— {v["condition"]}').classes('text-xs text-zinc-500')
                            ui.label(v['meaning']).classes('text-xs text-zinc-400 leading-relaxed')
                
                # Tip
                with ui.row().classes('items-start gap-2 mt-2 p-3 bg-blue-500/10 rounded-lg border border-blue-500/20'):
                    ui.icon('lightbulb').classes('text-blue-400 text-sm mt-0.5')
                    ui.label('The trend graph plots each run as a dot using these two axes. Click any dot to see that activity\'s details.').classes('text-sm italic text-blue-200 flex-1')
                
                ui.button('GOT IT', on_click=dialog.close).classes('mt-3 w-full bg-zinc-800 hover:bg-zinc-700 text-white font-bold')
        
        dialog.open()
    
    def show_form_info(self, highlight_verdict=None):
        """
        Show informational modal explaining Running Form verdicts.
        If highlight_verdict is provided, that verdict section gets visually emphasized.
        """
        verdicts = [
            {
                'key': 'Elite Form',
                'raw': 'ELITE FORM',
                'icon': '✅',
                'color': 'text-emerald-400',
                'bg': 'bg-emerald-500/10',
                'border': 'border-emerald-500/30',
                'range': '170+ SPM',
                'meaning': 'Pro-level mechanics. Fast turnover minimizes ground contact, reduces braking forces, and lowers injury risk. This is the goal.'
            },
            {
                'key': 'Good Form',
                'raw': 'GOOD FORM',
                'icon': '👍',
                'color': 'text-blue-400',
                'bg': 'bg-blue-500/10',
                'border': 'border-blue-500/30',
                'range': '160–169 SPM',
                'meaning': 'Solid, balanced mechanics. You\'re in a healthy cadence zone. Small gains still possible with drills and strides.'
            },
            {
                'key': 'Heavy Feet',
                'raw': 'HEAVY FEET',
                'icon': '⚠️',
                'color': 'text-orange-400',
                'bg': 'bg-orange-500/10',
                'border': 'border-orange-500/30',
                'range': '155–159 SPM',
                'meaning': 'Cadence is low. You\'re spending too long on the ground each step, wasting energy on vertical bounce. Focus on quick, light turnover.'
            },
            {
                'key': 'Plodding',
                'raw': 'PLODDING',
                'icon': '🐢',
                'color': 'text-yellow-400',
                'bg': 'bg-yellow-500/10',
                'border': 'border-yellow-500/30',
                'range': '135–154 SPM',
                'meaning': 'Sluggish turnover with high ground contact. This pattern increases impact forces and wastes energy. Try cadence drills or a metronome.'
            },
            {
                'key': 'Hiking / Rest',
                'raw': 'HIKING / REST',
                'icon': '🥾',
                'color': 'text-blue-400',
                'bg': 'bg-blue-500/10',
                'border': 'border-blue-500/30',
                'range': '< 135 SPM',
                'meaning': 'Power hiking or recovery interval. Low cadence here is expected — this isn\'t a running gait.'
            }
        ]
        
        with ui.dialog() as dialog, ui.card().classes('p-6 bg-zinc-900 border border-zinc-800 rounded-xl shadow-2xl max-w-xl'):
            with ui.column().classes('gap-3'):
                ui.label('Understanding Running Form').classes('text-xl font-bold text-white')
                ui.separator().classes('bg-zinc-700')
                
                with ui.column().classes('gap-1'):
                    ui.label('WHAT IS CADENCE?').classes('text-[10px] font-bold text-zinc-500 tracking-widest')
                    ui.label('Steps per minute (SPM). Higher cadence generally means shorter ground contact, less braking force, and more efficient energy transfer. Most elite runners are 170–185 SPM.').classes('text-sm text-zinc-300 leading-relaxed')
                
                ui.separator().classes('bg-zinc-700 mt-1')
                
                ui.label('VERDICTS').classes('text-[10px] font-bold text-zinc-500 tracking-widest')
                
                for v in verdicts:
                    is_highlighted = highlight_verdict and (v['key'] == highlight_verdict or v['raw'] == highlight_verdict)
                    ring = 'ring-2 ring-white/30' if is_highlighted else ''
                    scale = 'scale-[1.02]' if is_highlighted else 'opacity-80'
                    
                    with ui.row().classes(f'items-start gap-3 p-3 rounded-lg border {v["bg"]} {v["border"]} {ring} {scale} transition-all'):
                        ui.label(v['icon']).classes('text-lg mt-0.5')
                        with ui.column().classes('gap-0.5 flex-1'):
                            with ui.row().classes('items-center gap-2'):
                                ui.label(v['key']).classes(f'text-sm font-bold {v["color"]}')
                                ui.label(f'({v["range"]})').classes('text-xs text-zinc-500')
                            ui.label(v['meaning']).classes('text-xs text-zinc-400 leading-relaxed')
                
                with ui.row().classes('items-start gap-2 mt-2 p-3 bg-blue-500/10 rounded-lg border border-blue-500/20'):
                    ui.icon('lightbulb').classes('text-blue-400 text-sm mt-0.5')
                    ui.label('Cadence is pace-dependent — slower paces naturally have lower cadence. Compare cadence at similar paces for the most useful signal.').classes('text-sm italic text-blue-200 flex-1')
                
                ui.button('GOT IT', on_click=dialog.close).classes('mt-3 w-full bg-zinc-800 hover:bg-zinc-700 text-white font-bold')
        
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
    
    def _get_unique_tags_from_current_data(self):
        """
        Scans all currently loaded activities to find which tags actually exist.
        Returns two sets: context_tags (Tempo, etc.) and physio_tags (VO2, etc.)
        """
        context_tags = set()
        physio_tags = set()
        
        # Threshold for 'Long Run' calculation
        if self.df is not None and len(self.df) >= 5:
            lrt = self.df['distance_mi'].quantile(0.8)
        else:
            lrt = 10.0

        for act in self.activities_data:
            # 1. Get Context Tags (Tempo, Long Run, Hills, etc.)
            # We use the EXACT same function the Feed Cards use
            c_tags = self.classify_run_type(act, lrt) 
            if c_tags:
                # classify_run_type returns a string like "Long Run | Hills", split it
                for t in c_tags.split(' | '):
                    # Strip existing emojis so we get just "Hills" or "Tempo"
                    clean_t = t.replace('⛰️', '').replace('🔥', '').replace('🦅', '').replace('⚡', '').replace('🏃', '').replace('🧘', '').replace('🔷', '').strip()
                    if clean_t: context_tags.add(clean_t)
            
            # 2. Get Physio Tags (VO2 Max, Threshold, etc.)
            p_tag = act.get('te_label')
            if p_tag and str(p_tag) != "None":
                physio_tags.add(p_tag)
                
        return sorted(list(context_tags)), sorted(list(physio_tags))

    def update_filter_bar(self):
        """
        Renders the 'Apple Native' Filter Bar.
        Layout: STACKED
        Row 1: Distance (Scope)
        Row 2: Tags (Context & Characteristics)
        """
        self.filter_container.clear()
        
        # 1. Get available tags
        avail_context, avail_physio = self._get_unique_tags_from_current_data()
        
        # 2. Distance Definitions
        dist_filters = [
            {'id': 'all', 'label': 'All'},
            {'id': 'short', 'label': 'Short'},
            {'id': 'med', 'label': 'Medium'},
            {'id': 'long_dist', 'label': 'Long'},
        ]

        with self.filter_container:
            # CHANGE: Main container is now a COLUMN (Vertical Stack)
            # gap-3 gives nice breathing room between the Distance row and Tags row
            with ui.column().classes('w-full gap-3 mb-2'):
                
                # --- ROW 1: DISTANCE (The Scope) ---
                with ui.row().classes('w-full items-center'):
                    with ui.row().classes('bg-zinc-900 border border-zinc-800 p-1 rounded-lg gap-1'):
                        for f in dist_filters:
                            if f['id'] == 'all':
                                is_active = not any(k in self.active_filters for k in ['short', 'med', 'long_dist'])
                            else:
                                is_active = f['id'] in self.active_filters
                            
                            if is_active:
                                classes = "bg-zinc-800 text-white shadow-lg shadow-zinc-900/50 border border-zinc-700 font-bold"
                            else:
                                classes = "bg-transparent text-zinc-500 hover:text-zinc-300 font-medium"
                            
                            ui.button(f['label'], on_click=lambda id=f['id']: self.toggle_filter(id), color=None).props('flat dense no-caps ripple=False').classes(
                                f"rounded-md px-4 py-1 text-xs transition-all duration-200 no-ripple {classes}"
                            )

                # --- ROW 2: TAGS (The Details) ---
                # Full width container, tags wrap naturally
                with ui.row().classes('w-full gap-2 wrap items-center'):
                    
                    # GROUP: CONTEXT TAGS
                    for tag_name in avail_context:
                        config = self.TAG_CONFIG.get(tag_name, {'icon': '🏷️', 'color': 'zinc'})
                        color = config['color']
                        label = tag_name if any(c in tag_name for c in ['🏃','🔥','⚡','⛰️','🧘','🔷']) else f"{config['icon']} {tag_name}"
                        is_active = tag_name in self.active_filters
                        
                        if is_active:
                            classes = f"filter-active bg-{color}-500 text-white shadow-md border border-{color}-600/20 transform scale-105"
                        else:
                            # Context Tags: "Glass Capsule" Style
                            # Unselected: Visible container with grey border
                            # Hover: Glows with tag's specific color
                            classes = f"bg-zinc-800/20 text-zinc-300 border border-zinc-700 hover:bg-white/10 hover:border-{color}-500 hover:text-{color}-400 hover:shadow-[0_0_15px_-3px_currentColor] hover:-translate-y-px transition-all duration-300"

                        # Use 'label' instead of 'tag_name' to show the emoji icon
                        ui.button(label.replace('🏷️', '🏔️'), on_click=lambda t=tag_name: self.toggle_filter(t), color=None).props('flat dense no-caps ripple=False').classes(
                            f"rounded-full px-4 py-1 text-xs font-bold transition-all duration-200 no-ripple {classes}"
                        )

                    # GROUP: PHYSIO TAGS
                    for tag_name in avail_physio:
                        color = 'emerald'
                        if 'MAX' in tag_name or 'VO2' in tag_name: color = 'fuchsia'
                        elif 'ANAEROBIC' in tag_name: color = 'orange'
                        elif 'THRESHOLD' in tag_name: color = 'amber'
                        
                        is_active = tag_name in self.active_filters

                        if is_active:
                            classes = f"filter-active bg-{color}-500 text-white shadow-md border border-{color}-600/20"
                        else:
                            # Refined Dark Theme: Outlined with Colored Glow
                            classes = f"bg-zinc-900 text-{color}-400 border border-{color}-500/50 hover:border-{color}-400 hover:shadow-[0_0_10px_currentColor] hover:-translate-y-px transition-all duration-300"

                        ui.button(tag_name, on_click=lambda t=tag_name: self.toggle_filter(t), color=None).props('flat dense no-caps ripple=False').classes(
                            f"rounded-md px-3 py-1 text-[10px] font-bold tracking-wider uppercase transition-all duration-200 no-ripple {classes}"
                        )

    def toggle_filter(self, filter_id):
        """Toggles a filter on/off with STRICT logic."""
        
        dist_keys = {'short', 'med', 'long_dist'}
        
        # 1. Handle Distance Mutex
        if filter_id == 'all':
            # CLEAR all distance filters
            self.active_filters -= dist_keys
            
        elif filter_id in dist_keys:
            # If clicking the ACTIVE distance filter -> Do nothing (enforce one selection)
            # OR allow toggling off to go back to "All"
            if filter_id in self.active_filters:
                self.active_filters.remove(filter_id) # Clicking active -> goes to 'All'
            else:
                self.active_filters -= dist_keys # Clear others
                self.active_filters.add(filter_id) # Set new
        
        # 2. Handle Tags (Standard Toggle) - No changes here
        else:
            if filter_id in self.active_filters:
                self.active_filters.remove(filter_id)
            else:
                self.active_filters.add(filter_id)
        
        self.update_filter_bar() 
        self.update_activities_grid()
    
    def update_activities_grid(self):
        """
        Update activities table.
        FIXED:
        1. IDs are now DETERMINISTIC (Fixes Ghost Selection/FAB issues).
        2. Preserves 'virtual-scroll' and Dark UI.
        3. Uses standard $emit for buttons.
        """
        self.grid_container.clear()
        
        # --- 1. FILTER LOGIC ---
        filtered_data = []
        if self.df is not None and len(self.df) >= 5:
            lrt = self.df['distance_mi'].quantile(0.8)
        else:
            lrt = 10.0

        if not self.activities_data:
            filtered_data = []
        else:
            for act in self.activities_data:
                dist = act.get('distance_mi', 0)
                
                # Tags
                run_tags_set = set()
                c_tags_str = self.classify_run_type(act, lrt)
                if c_tags_str:
                    for t in c_tags_str.split(' | '):
                        run_tags_set.add(t.replace('⛰️', '').replace('🔥', '').replace('🦅', '').replace('⚡', '').replace('🏃', '').replace('🧘', '').replace('🔷', '').strip())
                
                p_tag = act.get('te_label')
                if p_tag and str(p_tag) != "None": run_tags_set.add(str(p_tag).strip())
                
                # Filter Check
                include = True
                if self.active_filters:
                    dist_keys = {'short', 'med', 'long_dist', 'all'}
                    active_tag_filters = {f for f in self.active_filters if f not in dist_keys}
                    
                    if 'short' in self.active_filters and not (dist < 5): include = False
                    if 'med' in self.active_filters and not (5 <= dist <= 10): include = False
                    if 'long_dist' in self.active_filters and not (dist > 10): include = False
                    
                    if active_tag_filters and not active_tag_filters.issubset(run_tags_set): 
                        include = False
                
                if include: 
                    filtered_data.append(act)

        # --- EMPTY STATE ---
        if not filtered_data:
            with self.grid_container:
                 with ui.card().classes('w-full bg-zinc-900/80 p-0 border border-zinc-800 shadow-2xl overflow-hidden').style('border-radius: 12px; min-height: 200px;'):
                     ui.label('No runs match these filters.').classes('w-full text-center text-zinc-500 mt-20 font-mono')
            return

        # --- 2. PREPARE ROWS (With STABLE IDs) ---
        rows = []
        for i, act in enumerate(filtered_data):
            db_hash = act.get('db_hash')
            if db_hash:
                row_id = db_hash
            else:
                # [CRITICAL FIX] STABLE ID GENERATION - Use filename + date if hash missing
                # Ensure no Python objects (like lists) are in the ID to prevent serialization issues
                safe_filename = act.get('filename', 'unk').replace('.fit', '')
                row_id = f"temp_{safe_filename}"
            
            # Format Data
            raw_date = act.get('date', '') 
            try:
                dt = datetime.strptime(raw_date, '%Y-%m-%d %H:%M')
                n_date, n_time = dt.strftime('%a, %m/%d'), dt.strftime('%I:%M%p').lstrip('0')
            except: n_date, n_time = raw_date, ""

            ts = self.classify_run_type(act, lrt)
            te = act.get('te_label')
            tags_str = f"{ts} | {te}" if te and str(te) != "None" else ts
            
            p_raw = act.get('pace', 0)
            def fmt_pace(v):
                try: return v if ':' in str(v) else f"{int(float(v))}:{int((float(v)%1)*60):02d}"
                except: return "--:--"
            
            rows.append({
                'id': row_id,             # The Quasar Row Key
                'hash': db_hash,          # The Real Data Key
                'filename': act.get('filename', 'Unknown'),
                'date_d': n_date, 
                'date_t': n_time,
                'date_sort': raw_date,
                'distance': f"{act.get('distance_mi', 0):.1f}",
                'dist_sort': act.get('distance_mi', 0),
                'pace': fmt_pace(p_raw),
                'pace_sort': float(p_raw) if isinstance(p_raw, (int, float)) else 0,
                'elev_d': f"{int(act.get('elevation_ft', 0)):,}",
                'elev_sort': act.get('elevation_ft', 0),
                'type_display': tags_str, # Keep as STRING to prevent NiceGUI intervention

                'full_filename': act.get('filename', ''),
                'file_path': act.get('file_path', '')
            })

        # --- 3. COLUMNS ---
        columns = [
            {'name': 'date', 'label': 'Date', 'field': 'date_sort', 'align': 'center', 'sortable': True, 'style': 'width: 100px'},
            {'name': 'distance', 'label': 'Dist', 'field': 'dist_sort', 'align': 'left', 'sortable': True, 'style': 'width: 70px'},
            {'name': 'pace', 'label': 'Pace', 'field': 'pace_sort', 'align': 'left', 'sortable': True, 'style': 'width: 80px'},
            {'name': 'elevation', 'label': 'Elev', 'field': 'elev_sort', 'align': 'left', 'sortable': True, 'style': 'width: 85px'},
            {'name': 'type', 'label': 'Run Context', 'field': 'type_sort', 'align': 'left', 'sortable': True}, 
            {'name': 'actions', 'label': '', 'field': 'actions', 'align': 'right', 'style': 'width: 130px'},
        ]

        # --- 4. RENDER TABLE ---
        with self.grid_container:
            with ui.card().classes('w-full bg-zinc-900/80 p-0 border border-zinc-800 shadow-2xl overflow-hidden').style('border-radius: 12px;'):

                # --- EVENT HANDLERS ---

                def on_selection(e):
                    # Use table.selected (NiceGUI's binding) - NOT e.args (raw Quasar event)
                    # table.selected always reflects the current cumulative selection state.
                    self.show_floating_action_bar(table.selected)

                async def handle_row_action(e):
                    payload = e.args if isinstance(e.args, dict) else {}
                    action = payload.get('action')
                    row = payload.get('row') if isinstance(payload.get('row'), dict) else None

                    if not row:
                        return

                    if action == 'analyze':
                        activity_hash = row.get('hash')
                        if activity_hash:
                            await self.open_activity_detail_modal(activity_hash, from_feed=True)
                    elif action == 'download':
                        await self.download_fit_file(row)
                    elif action == 'delete':
                        activity_hash = row.get('hash')
                        filename = row.get('full_filename')
                        await self.delete_activity_inline(activity_hash, filename)
                
                table = ui.table(
                    columns=columns, 
                    rows=rows, 
                    row_key='id',       # Matches our STABLE ID
                    selection='multiple',
                    pagination={'rowsPerPage': 0, 'sortBy': self.current_sort_by, 'descending': self.current_sort_desc},
                    on_select=on_selection,
                ).classes('w-full h-full text-sm sticky-header-table')
                self.activities_table = table  # Store ref for selection clearing

                # Preserved Dark Mode - REMOVED virtual-scroll to fix event trapping
                table.props('flat bordered dense dark')

                # Bind row action events emitted from custom action slot
                table.on('row-action', handle_row_action)

                # --- SLOTS ---
                
                table.add_slot('body-cell-date', '<q-td :props="props"><div class="flex flex-col items-center justify-center py-1 leading-tight"><span class="font-bold text-gray-200 text-[13px]">{{ props.row.date_d }}</span><span class="text-[11px] text-zinc-500 font-mono mt-0.5">{{ props.row.date_t }}</span></div></q-td>')
                table.add_slot('body-cell-distance', '<q-td :props="props"><div class="flex items-baseline gap-0.5"><span class="font-bold text-white text-[13px]">{{ props.row.distance }}</span><span class="text-xs text-zinc-500">mi</span></div></q-td>')
                table.add_slot('body-cell-pace', '<q-td :props="props"><div class="flex items-baseline gap-0.5"><span class="font-mono text-zinc-300 text-[12px]">{{ props.row.pace }}</span><span class="text-[10px] text-zinc-600">/mi</span></div></q-td>')
                table.add_slot('body-cell-elevation', '<q-td :props="props"><div class="flex items-baseline gap-1"><span class="text-[10px] brightness-125">⛰️</span><span class="font-bold text-zinc-300 text-[12px]">{{ props.row.elev_d }}</span><span class="text-[10px] text-zinc-500">ft</span></div></q-td>')
                
                table.add_slot('body-cell-type', '''
                    <q-td :props="props">
                        <div class="flex flex-wrap gap-2 items-center py-1">
                            <span v-for="tag in props.row.type_display.split(' | ')" :key="tag"  
                                  class="px-2 py-1 rounded-md text-[11px] font-bold tracking-wide border whitespace-nowrap bg-zinc-800 text-zinc-400 border-zinc-700">
                                {{ tag }}
                            </span>
                        </div>
                    </q-td>
                ''')

                # ACTIONS SLOT
                # CRITICAL: Must use $parent.$emit (not $emit) to reach NiceGUI's event listener.
                # $emit in a Quasar slot emits on the slot's own context (goes nowhere).
                # $parent.$emit emits on the parent QTable component (which NiceGUI listens to).
                table.add_slot('body-cell-actions', '''
                    <q-td :props="props">
                        <div class="flex flex-nowrap items-center justify-end gap-1 relative z-10">
                            <q-btn flat dense round icon="visibility" size="sm" 
                                class="action-btn no-ripple text-zinc-400 hover:text-white" :ripple="false"
                                @click.stop="$parent.$emit('row-action', { action: 'analyze', row: props.row })">
                                <q-tooltip>Analyze</q-tooltip>
                            </q-btn>
                            <q-btn flat dense round icon="download" size="sm" 
                                class="action-btn no-ripple text-zinc-400 hover:text-white" :ripple="false"
                                @click.stop="$parent.$emit('row-action', { action: 'download', row: props.row })">
                                <q-tooltip>Save .FIT</q-tooltip>
                            </q-btn>
                            <q-btn flat dense round icon="delete" size="sm" 
                                class="action-btn no-ripple text-zinc-400 hover:text-white" :ripple="false"
                                @click.stop="$parent.$emit('row-action', { action: 'delete', row: props.row })">
                                <q-tooltip>Delete</q-tooltip>
                            </q-btn>
                        </div>
                    </q-td>
                ''')
                
                table.props('flat bordered dense dark')
                table.classes('bg-zinc-900 text-gray-200')
    
    async def handle_table_request(self, e):
        """
        Handle server-side sort requests from the UI table.
        """
        # 1. Extract new sort settings from the event
        pagination = e.args.get('pagination', {})
        new_sort_by = pagination.get('sortBy')
        new_descending = pagination.get('descending')
        
        # 2. Update App State
        if new_sort_by:
            self.current_sort_by = new_sort_by
            self.current_sort_desc = new_descending
            
            # 3. Reload Data (Server-Side Sort)
            await self.refresh_data_view()
            
            # 4. Notify user (Optional polish)
            direction = "↓" if new_descending else "↑"
            ui.notify(f"Sorted by {new_sort_by} {direction}", type='info', timeout=1000)

    async def download_fit_file(self, row_data):
        """
        Copy the FIT file to the user's Downloads folder.
        """
        try:
            # 1. Get Source Path from the row data
            source_path = row_data.get('file_path')
            
            # Validation
            if not source_path:
                ui.notify("File path missing in database.", type='negative')
                return
            
            if not os.path.exists(source_path):
                ui.notify(f"File not found on disk: {os.path.basename(source_path)}", type='negative')
                return

            # 2. Determine Destination
            filename = os.path.basename(source_path)
            downloads_path = os.path.expanduser("~/Downloads")
            dest_path = os.path.join(downloads_path, filename)
            
            # 3. Copy File (shutil.copy2 preserves metadata)
            import shutil
            shutil.copy2(source_path, dest_path)
            
            ui.notify(f"Saved to Downloads: {filename}", type='positive', icon='download')
            
        except Exception as e:
            ui.notify(f"Download failed: {e}", type='negative')
    
    async def delete_activity_inline(self, activity_hash, filename):
        """
        Delete activity from inline button click.
        FIXED: Notify BEFORE refreshing to prevent context loss crash.
        """
        # Show confirmation dialog
        result = await ui.run_javascript(
            f'confirm("Delete activity: {filename}?")', 
            timeout=10
        )
        
        # If confirmed, delete the activity
        if result:
            try:
                # 1. Delete the activity from DB
                self.db.delete_activity(activity_hash)
                
                # 2. Show notification NOW (while the button/table still exists)
                ui.notify(f'Deleted: {filename[:30]}', type='positive')
                
                # 3. Update runs counter
                self.runs_count_label.text = f'{self.db.get_count()}'
                
                # 4. Refresh views (This wipes the table, which is fine now)
                await self.refresh_data_view()
                
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
        FIXED: Checks for at least 2 data points to avoid SmallSampleWarning.
        """
        # Need at least 2 points to draw a line
        if df_subset is None or len(df_subset) < 2:
            return ("Trend: Insufficient Data", "silver")
        
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
        categories = ['HIGH QUALITY', 'STRUCTURAL', 'BROKEN']
        
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
                    '<span style="font-size: 12px; font-weight: 600;">' + name + '</span><br>'
                    '<span style="color: #e2e8f0; font-size: 12px;">' + desc + '</span><br>'
                    '<span style="color: rgba(255,255,255,0.9); font-size: 11px;">Runs: %{customdata[0]}</span>'
                    '<extra></extra>'
                ),
                
                hoverlabel=dict(font=dict(color='white'), bordercolor='white'),
                legendrank=rank, 
                **bar_style
            ))

        add_trace('HIGH QUALITY', 'High Quality', '#10b981', 'Dialed Mechanics', 1)
        add_trace('STRUCTURAL', 'Structural', '#3b82f6', 'Valid Base/Hills', 2)
        add_trace('BROKEN', 'Broken', '#f43f5e', 'Mechanical Failure', 3)
        
        fig.update_layout(
            barmode='stack', template='plotly_dark',
            paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
            height=300, margin=dict(l=40, r=20, t=20, b=80),
            showlegend=True, legend=dict(orientation="h", y=1.02, x=1, xanchor="right", traceorder="normal", font=dict(color='#a1a1aa')),
            bargap=0.35, 
            xaxis=dict(tickangle=0, showgrid=True, gridcolor='#27272a', tickfont=dict(color='#a1a1aa')),
            yaxis=dict(title=dict(text='Miles', font=dict(color='#a1a1aa')), showgrid=True, gridcolor='#27272a', tickfont=dict(color='#a1a1aa')),
            clickmode='event',
            font=dict(color='#a1a1aa')
        )
        fig.update_layout(modebar={'remove': ['zoom', 'pan', 'select', 'lasso2d', 'zoomIn', 'zoomOut', 'autoScale', 'resetScale', 'toImage']})
        
        return fig
    
    def generate_training_mix_chart(self):
        """Generate weekly volume chart grouped by RUN TYPE (Training Mix lens).
        Uses purple/pink/indigo palette to visually distinguish from Quality lens."""
        if self.df is None or self.df.empty or not self.activities_data:
            return None
        
        # Calculate long run threshold
        distances = [a.get('distance_mi', 0) for a in self.activities_data if a.get('distance_mi', 0) > 0]
        long_run_threshold = max(10, sorted(distances)[int(len(distances) * 0.8)] if len(distances) >= 5 else 10)
        
        # Build per-activity data with primary run type only
        mix_data = []
        for activity in self.activities_data:
            date_obj = pd.to_datetime(activity.get('date'))
            week_start = date_obj.to_period('W-MON').start_time
            dist = activity.get('distance_mi', 0)
            if dist < 0.1:
                continue
            
            act_hash = activity.get('db_hash')
            act_date_str = date_obj.strftime('%-m/%-d')

            # Get primary tag only (first tag before any ' | ')
            full_tag = self.classify_run_type(activity, long_run_threshold)
            primary = full_tag.split(' | ')[0] if full_tag else '🟡 Base'
            # Strip emoji prefix for clean category name
            clean_primary = primary.split(' ', 1)[-1] if ' ' in primary else primary
            
            mix_data.append({
                'week_start': week_start, 'distance': dist, 'category': clean_primary,
                'hash': act_hash, 'date_str': act_date_str
            })
        
        if not mix_data:
            return None
        
        df_mix = pd.DataFrame(mix_data)
        
        # Define categories and colors (purple/pink/indigo palette — distinct from Quality)
        categories = ['Recovery', 'Base', 'Steady', 'Long Run', 'Tempo']
        colors = {
            'Recovery': '#60a5fa',    # Blue-400 (easy)
            'Base':     '#a78bfa',    # Violet-400 (easy)
            'Steady':   '#f59e0b',    # Amber-500 (moderate)
            'Long Run': '#34d399',    # Emerald-400 (trails/endurance)
            'Tempo':    '#f43f5e',    # Rose-500 (hard)
        }
        descriptions = {
            'Recovery': 'Easy effort, promotes adaptation',
            'Base':     'Aerobic foundation miles',
            'Steady':   'Moderate effort, gray zone',
            'Long Run': 'Duration-focused endurance',
            'Tempo':    'High intensity speed work',
        }
        
        # Group and build chart data
        grouped = df_mix.groupby(['week_start', 'category'])
        weeks = sorted(df_mix['week_start'].unique())
        
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
                    date_display = ", ".join(dates_list[:3]) + (f" (+{len(dates_list)-3})" if len(dates_list) > 3 else "")
                    hash_list = unique_acts['hash'].tolist()
                    chart_data[cat]['y'].append(total_dist)
                    chart_data[cat]['customdata'].append([date_display, json.dumps(hash_list), cat])
                except KeyError:
                    chart_data[cat]['y'].append(0)
                    chart_data[cat]['customdata'].append(["", "[]", cat])
        
        # Build figure
        fig = go.Figure()
        bar_style = dict(opacity=0.85, marker_line=dict(width=1, color='rgba(255,255,255,0.1)'))
        
        for cat in categories:
            if sum(chart_data[cat]['y']) > 0:  # Only add trace if there's data
                fig.add_trace(go.Bar(
                    x=week_labels, y=chart_data[cat]['y'], name=cat,
                    marker_color=colors.get(cat, '#888'),
                    customdata=chart_data[cat]['customdata'],
                    hovertemplate=(
                        '<b>%{y:.1f} mi</b><br>'
                        f'<span style="color: {colors.get(cat, "#888")}; font-size: 12px; font-weight: 600;">{cat}</span><br>'
                        f'<span style="color: #e2e8f0; font-size: 12px;">{descriptions.get(cat, "")}</span><br>'
                        '<span style="color: rgba(255,255,255,0.9); font-size: 11px;">Runs: %{customdata[0]}</span>'
                        '<extra></extra>'
                    ),
                    hoverlabel=dict(font=dict(color='white'), bordercolor='white'),
                    **bar_style
                ))
        
        fig.update_layout(
            barmode='stack', template='plotly_dark',
            paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
            height=300, margin=dict(l=40, r=20, t=20, b=80),
            showlegend=True, legend=dict(orientation="h", y=1.02, x=1, xanchor="right", traceorder="normal", font=dict(color='#a1a1aa')),
            bargap=0.35, 
            xaxis=dict(tickangle=0, showgrid=True, gridcolor='#27272a', tickfont=dict(color='#a1a1aa')),
            yaxis=dict(title=dict(text='Miles', font=dict(color='#a1a1aa')), showgrid=True, gridcolor='#27272a', tickfont=dict(color='#a1a1aa')),
            clickmode='event',
            font=dict(color='#a1a1aa')
        )
        fig.update_layout(modebar={'remove': ['zoom', 'pan', 'select', 'lasso2d', 'zoomIn', 'zoomOut', 'autoScale', 'resetScale', 'toImage']})
        
        return fig

    def _calculate_strain(self, activity):
        """Calculate strain score for an activity (shared logic)."""
        moving_time_min = activity.get('moving_time_min', 0)
        avg_hr = activity.get('avg_hr', 0)
        max_hr = activity.get('max_hr', 185)
        intensity = avg_hr / max_hr if max_hr > 0 else 0
        
        if intensity < 0.65: factor = 1.0
        elif intensity < 0.75: factor = 1.5
        elif intensity < 0.85: factor = 3.0
        elif intensity < 0.92: factor = 6.0
        else: factor = 10.0
        
        return int(moving_time_min * factor)

    def generate_load_chart(self):
        """Generate weekly volume chart grouped by LOAD CATEGORY (Load lens).
        Uses teal/cyan palette to visually distinguish from Quality and Mix lenses."""
        if self.df is None or self.df.empty or not self.activities_data:
            return None
        
        load_data = []
        for activity in self.activities_data:
            date_obj = pd.to_datetime(activity.get('date'))
            week_start = date_obj.to_period('W-MON').start_time
            dist = activity.get('distance_mi', 0)
            if dist < 0.1:
                continue
            
            act_hash = activity.get('db_hash')
            act_date_str = date_obj.strftime('%-m/%-d')
            
            strain = self._calculate_strain(activity)
            if strain < 75: load_cat = 'Recovery'
            elif strain < 150: load_cat = 'Maintenance'
            elif strain < 300: load_cat = 'Productive'
            else: load_cat = 'Overreaching'
            
            load_data.append({
                'week_start': week_start, 'distance': dist, 'category': load_cat,
                'hash': act_hash, 'date_str': act_date_str, 'strain': strain
            })
        
        if not load_data:
            return None
        
        df_load = pd.DataFrame(load_data)
        
        # Colors match feed card Load colors for consistency
        categories = ['Recovery', 'Maintenance', 'Productive', 'Overreaching']
        colors = {
            'Recovery':     '#60a5fa',  # Blue — matches feed card
            'Maintenance':  '#10B981',  # Green — matches feed card
            'Productive':   '#f97316',  # Orange — matches feed card
            'Overreaching': '#ef4444',  # Red — matches feed card
        }
        descriptions = {
            'Recovery':     'Low stress, promotes adaptation',
            'Maintenance':  'Steady load, maintains fitness',
            'Productive':   'Hard effort, builds fitness',
            'Overreaching': 'Very high stress, needs recovery',
        }
        
        grouped = df_load.groupby(['week_start', 'category'])
        weeks = sorted(df_load['week_start'].unique())
        
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
                    date_display = ", ".join(dates_list[:3]) + (f" (+{len(dates_list)-3})" if len(dates_list) > 3 else "")
                    hash_list = unique_acts['hash'].tolist()
                    chart_data[cat]['y'].append(total_dist)
                    chart_data[cat]['customdata'].append([date_display, json.dumps(hash_list), cat])
                except KeyError:
                    chart_data[cat]['y'].append(0)
                    chart_data[cat]['customdata'].append(["", "[]", cat])
        
        fig = go.Figure()
        bar_style = dict(opacity=0.85, marker_line=dict(width=1, color='rgba(255,255,255,0.1)'))
        
        for cat in categories:
            if sum(chart_data[cat]['y']) > 0:
                fig.add_trace(go.Bar(
                    x=week_labels, y=chart_data[cat]['y'], name=cat,
                    marker_color=colors.get(cat, '#888'),
                    customdata=chart_data[cat]['customdata'],
                    hovertemplate=(
                        '<b>%{y:.1f} mi</b><br>'
                        f'<span style="color: {colors.get(cat, "#888")}; font-size: 12px; font-weight: 600;">{cat}</span><br>'
                        f'<span style="color: #e2e8f0; font-size: 12px;">{descriptions.get(cat, "")}</span><br>'
                        '<span style="color: rgba(255,255,255,0.9); font-size: 11px;">Runs: %{customdata[0]}</span>'
                        '<extra></extra>'
                    ),
                    hoverlabel=dict(font=dict(color='white'), bordercolor='white'),
                    **bar_style
                ))
        
        fig.update_layout(
            barmode='stack', template='plotly_dark',
            paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
            height=300, margin=dict(l=40, r=20, t=20, b=80),
            showlegend=True, legend=dict(orientation="h", y=1.02, x=1, xanchor="right", traceorder="normal", font=dict(color='#a1a1aa')),
            bargap=0.35, 
            xaxis=dict(tickangle=0, showgrid=True, gridcolor='#27272a', tickfont=dict(color='#a1a1aa')),
            yaxis=dict(title=dict(text='Miles', font=dict(color='#a1a1aa')), showgrid=True, gridcolor='#27272a', tickfont=dict(color='#a1a1aa')),
            clickmode='event',
            font=dict(color='#a1a1aa')
        )
        fig.update_layout(modebar={'remove': ['zoom', 'pan', 'select', 'lasso2d', 'zoomIn', 'zoomOut', 'autoScale', 'resetScale', 'toImage']})
        
        return fig

    def calculate_mix_verdict(self, df):
        """Calculate verdict for Training Mix lens — checks 80/20 distribution."""
        if df is None or df.empty or not self.activities_data:
            return 'N/A', '#71717a', 'bg-zinc-700'
        
        try:
            distances = [a.get('distance_mi', 0) for a in self.activities_data if a.get('distance_mi', 0) > 0]
            long_run_threshold = max(10, sorted(distances)[int(len(distances) * 0.8)] if len(distances) >= 5 else 10)
            
            total_miles = 0
            easy_miles = 0   # Recovery + Base
            hard_miles = 0   # Tempo + Intervals
            steady_miles = 0
            long_miles = 0
            
            for activity in self.activities_data:
                dist = activity.get('distance_mi', 0)
                if dist < 0.1:
                    continue
                total_miles += dist
                
                full_tag = self.classify_run_type(activity, long_run_threshold)
                primary = full_tag.split(' | ')[0] if full_tag else '🟡 Base'
                clean_primary = primary.split(' ', 1)[-1] if ' ' in primary else primary
                
                if clean_primary in ('Recovery', 'Base'):
                    easy_miles += dist
                elif clean_primary == 'Tempo':
                    hard_miles += dist
                elif clean_primary == 'Steady':
                    steady_miles += dist
                elif clean_primary == 'Long Run':
                    long_miles += dist
            
            if total_miles < 1:
                return 'N/A', '#71717a', 'bg-zinc-700'
            
            easy_pct = (easy_miles / total_miles) * 100
            hard_pct = (hard_miles / total_miles) * 100
            tempo_threshold_pct = ((hard_miles + steady_miles) / total_miles) * 100
            
            # Tempo Heavy: >40% combined Tempo+Steady without enough easy
            if tempo_threshold_pct > 40 and easy_pct < 50:
                return 'TEMPO HEAVY', '#ef4444', 'bg-red-500/20'
            
            # Polarized: >=60% easy + some hard work (great for ultra runners)
            if easy_pct >= 60 and hard_miles > 0:
                return 'POLARIZED', '#10b981', 'bg-emerald-500/20'
            
            # Monotone: almost all one type
            if easy_pct > 85:
                return 'MONOTONE', '#f97316', 'bg-orange-500/20'
            
            # Default: balanced
            return 'BALANCED', '#3b82f6', 'bg-blue-500/20'
            
        except:
            return 'N/A', '#71717a', 'bg-zinc-700'

    def calculate_load_verdict(self, df):
        """Calculate verdict for Load lens — checks stress distribution."""
        if df is None or df.empty or not self.activities_data:
            return 'N/A', '#71717a', 'bg-zinc-700'
        
        try:
            total_runs = 0
            recovery_runs = 0
            maintenance_runs = 0
            productive_runs = 0
            overreaching_runs = 0
            
            for activity in self.activities_data:
                dist = activity.get('distance_mi', 0)
                if dist < 0.1:
                    continue
                total_runs += 1
                strain = self._calculate_strain(activity)
                
                if strain < 75: recovery_runs += 1
                elif strain < 150: maintenance_runs += 1
                elif strain < 300: productive_runs += 1
                else: overreaching_runs += 1
            
            if total_runs < 2:
                return 'N/A', '#71717a', 'bg-zinc-700'
            
            overreach_pct = (overreaching_runs / total_runs) * 100
            productive_pct = (productive_runs / total_runs) * 100
            easy_pct = ((recovery_runs + maintenance_runs) / total_runs) * 100
            
            # Overreaching: too much overreaching
            if overreach_pct > 25:
                return 'OVERREACHING', '#ef4444', 'bg-red-500/20'
            
            # Productive: heavy productive volume
            if productive_pct > 40:
                return 'PRODUCTIVE', '#f97316', 'bg-orange-500/20'
            
            # Undertrained: all easy, no stimulus
            if easy_pct > 90 and productive_runs == 0:
                return 'UNDERTRAINED', '#3b82f6', 'bg-blue-500/20'
            
            # Maintaining: healthy balance
            return 'MAINTAINING', '#10b981', 'bg-emerald-500/20'
            
        except:
            return 'N/A', '#71717a', 'bg-zinc-700'

    def generate_hr_zones_chart(self):
        """Generate weekly volume chart grouped by HEART RATE ZONE (HR Zones lens).
        Y-axis = minutes. Each activity assigned to dominant zone via avg_hr/max_hr ratio."""
        if self.df is None:
            return None
        if self.df.empty:
            return None
        if not self.activities_data:
            return None
        
        zone_data = []
        for activity in self.activities_data:
            date_obj = pd.to_datetime(activity.get('date'))
            week_start = date_obj.to_period('W-MON').start_time
            moving_time = activity.get('moving_time_min', 0)
            if moving_time < 1:
                continue
            
            avg_hr = activity.get('avg_hr', 0)
            max_hr = activity.get('max_hr', 185)
            if not avg_hr or not max_hr or max_hr == 0:
                continue
            
            act_hash = activity.get('db_hash')
            act_date_str = date_obj.strftime('%-m/%-d')
            
            # Classify into HR zone based on avg_hr / max_hr ratio
            ratio = avg_hr / max_hr
            if ratio < 0.60:
                zone = 'Zone 1'
            elif ratio < 0.70:
                zone = 'Zone 2'
            elif ratio < 0.80:
                zone = 'Zone 3'
            elif ratio < 0.90:
                zone = 'Zone 4'
            else:
                zone = 'Zone 5'
            
            zone_data.append({
                'week_start': week_start, 'minutes': moving_time, 'zone': zone,
                'hash': act_hash, 'date_str': act_date_str
            })
        
        if not zone_data:
            return None
        
        df_zones = pd.DataFrame(zone_data)
        
        # Unified HR zone color palette (consistent across modal + trends)
        categories = ['Zone 1', 'Zone 2', 'Zone 3', 'Zone 4', 'Zone 5']
        colors = {
            'Zone 1': '#60a5fa',  # Blue — easy/warmup
            'Zone 2': '#34d399',  # Emerald — aerobic base
            'Zone 3': '#fbbf24',  # Amber — threshold/gray zone ⚠️
            'Zone 4': '#f97316',  # Orange — hard
            'Zone 5': '#ef4444',  # Red — max effort
        }
        descriptions = {
            'Zone 1': 'Easy (<60% max HR)',
            'Zone 2': 'Aerobic (60-70% max HR)',
            'Zone 3': 'Threshold (70-80% max HR)',
            'Zone 4': 'Hard (80-90% max HR)',
            'Zone 5': 'Max effort (>90% max HR)',
        }
        
        grouped = df_zones.groupby(['week_start', 'zone'])
        weeks = sorted(df_zones['week_start'].unique())
        
        chart_data = {cat: {'y': [], 'customdata': []} for cat in categories}
        week_labels = []
        
        for week in weeks:
            sunday = week - pd.Timedelta(days=1)
            saturday = week + pd.Timedelta(days=5)
            week_labels.append(f"{sunday.strftime('%b %-d')} - {saturday.strftime('%-d')}")
            
            for cat in categories:
                try:
                    cell = grouped.get_group((week, cat))
                    total_mins = cell['minutes'].sum()
                    unique_acts = cell[['hash', 'date_str']].drop_duplicates()
                    dates_list = unique_acts['date_str'].tolist()
                    date_display = ", ".join(dates_list[:3]) + (f" (+{len(dates_list)-3})" if len(dates_list) > 3 else "")
                    hash_list = unique_acts['hash'].tolist()
                    chart_data[cat]['y'].append(total_mins)
                    chart_data[cat]['customdata'].append([date_display, json.dumps(hash_list), cat])
                except KeyError:
                    chart_data[cat]['y'].append(0)
                    chart_data[cat]['customdata'].append(["", "[]", cat])
        
        fig = go.Figure()
        bar_style = dict(opacity=0.85, marker_line=dict(width=1, color='rgba(255,255,255,0.1)'))
        
        for cat in categories:
            if sum(chart_data[cat]['y']) > 0:
                fig.add_trace(go.Bar(
                    x=week_labels, y=chart_data[cat]['y'], name=cat,
                    marker_color=colors.get(cat, '#888'),
                    customdata=chart_data[cat]['customdata'],
                    hovertemplate=(
                        '<b>%{y:.0f} min</b><br>'
                        f'<span style="color: {colors.get(cat, "#888")}; font-size: 12px; font-weight: 600;">{cat}</span><br>'
                        f'<span style="color: #e2e8f0; font-size: 12px;">{descriptions.get(cat, "")}</span><br>'
                        '<span style="color: rgba(255,255,255,0.9); font-size: 11px;">Runs: %{customdata[0]}</span>'
                        '<extra></extra>'
                    ),
                    hoverlabel=dict(font=dict(color='white'), bordercolor='white'),
                    **bar_style
                ))
        
        fig.update_layout(
            barmode='stack', template='plotly_dark',
            paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
            height=300, margin=dict(l=40, r=20, t=20, b=80),
            showlegend=True, legend=dict(orientation="h", y=1.02, x=1, xanchor="right", traceorder="normal", font=dict(color='#a1a1aa')),
            bargap=0.35, 
            xaxis=dict(tickangle=0, showgrid=True, gridcolor='#27272a', tickfont=dict(color='#a1a1aa')),
            yaxis=dict(title=dict(text='Minutes', font=dict(color='#a1a1aa')), showgrid=True, gridcolor='#27272a', tickfont=dict(color='#a1a1aa')),
            clickmode='event',
            font=dict(color='#a1a1aa')
        )
        fig.update_layout(modebar={'remove': ['zoom', 'pan', 'select', 'lasso2d', 'zoomIn', 'zoomOut', 'autoScale', 'resetScale', 'toImage']})
        
        return fig

    def calculate_hr_zones_verdict(self, df):
        """Calculate verdict for HR Zones lens — checks zone distribution."""
        if df is None or df.empty or not self.activities_data:
            return 'N/A', '#71717a', 'bg-zinc-700'
        
        try:
            z1_time = 0
            z2_time = 0
            z3_time = 0
            z4_time = 0
            z5_time = 0
            total_time = 0
            
            for activity in self.activities_data:
                moving_time = activity.get('moving_time_min', 0)
                avg_hr = activity.get('avg_hr', 0)
                max_hr = activity.get('max_hr', 185)
                if not avg_hr or not max_hr or max_hr == 0 or moving_time < 1:
                    continue
                
                total_time += moving_time
                ratio = avg_hr / max_hr
                if ratio < 0.60:
                    z1_time += moving_time
                elif ratio < 0.70:
                    z2_time += moving_time
                elif ratio < 0.80:
                    z3_time += moving_time
                elif ratio < 0.90:
                    z4_time += moving_time
                else:
                    z5_time += moving_time
            
            if total_time < 1:
                return 'N/A', '#71717a', 'bg-zinc-700'
            
            easy_pct = ((z1_time + z2_time) / total_time) * 100
            z3_pct = (z3_time / total_time) * 100
            z4_pct = (z4_time / total_time) * 100
            z5_pct = (z5_time / total_time) * 100
            
            # Zone 5 Redlining: >15% in Zone 5 (truly maximal effort)
            if z5_pct > 15:
                return 'ZONE 5 REDLINING', '#ef4444', 'bg-red-500/20'
            
            # Zone 4 Threshold Addict: >25% in Zone 4
            if z4_pct > 25:
                return 'ZONE 4 THRESHOLD ADDICT', '#f97316', 'bg-orange-500/20'
            
            # Zone 3 Junk: >30% in the gray zone
            if z3_pct > 30:
                return 'ZONE 3 JUNK', '#fbbf24', 'bg-amber-500/20'
            
            # Zone 2 Base: >=85% in Z1+Z2 with very little intensity
            if easy_pct >= 85:
                return 'ZONE 2 BASE', '#3b82f6', 'bg-blue-500/20'
            
            # 80/20 Balanced: healthy distribution
            return '80/20 BALANCED', '#10b981', 'bg-emerald-500/20'
            
        except:
            return 'N/A', '#71717a', 'bg-zinc-700'

    def refresh_volume_card(self):
        """Surgically refresh only the volume card content based on current lens."""
        if self.volume_card_container is None:
            return
        
        self.volume_card_container.clear()
        with self.volume_card_container:
            # Generate chart based on current lens
            if self.volume_lens == 'mix':
                fig = self.generate_training_mix_chart()
                verdict, v_color, v_bg = self.calculate_mix_verdict(self.df)
                subtitle = 'Weekly distribution by run type'
            elif self.volume_lens == 'load':
                fig = self.generate_load_chart()
                verdict, v_color, v_bg = self.calculate_load_verdict(self.df)
                subtitle = 'Weekly distribution by training stress'
            elif self.volume_lens == 'zones':
                fig = self.generate_hr_zones_chart()
                verdict, v_color, v_bg = self.calculate_hr_zones_verdict(self.df)
                subtitle = 'Weekly time in each heart rate zone'
            else:
                fig = self.generate_weekly_volume_chart()
                verdict, v_color, v_bg = self.calculate_volume_verdict(self.df)
                subtitle = 'Breakdown in quality of miles (click any section to inspect runs)'
            
            # Update verdict badge
            if hasattr(self, 'volume_verdict_label'):
                self.volume_verdict_label.text = f'{verdict}'
                self.volume_verdict_label.style(replace=f'color: {v_color};')
            
            # Update subtitle
            if hasattr(self, 'volume_subtitle_label'):
                self.volume_subtitle_label.text = subtitle
            
            # Render chart
            if fig:
                self.volume_chart = ui.plotly(fig).classes('w-full').style('cursor: pointer')
                self.volume_chart.on('plotly_relayout', self.handle_volume_zoom)
                self.volume_chart.on('plotly_click', self.handle_bar_click)
            else:
                ui.label('No data available for this view').classes('text-zinc-500 text-center py-8')
    
    def generate_efficiency_decoupling_chart(self):
        """Generate Running Efficiency vs. Aerobic Decoupling chart with QUADRANT LOGIC."""
        if self.df is None or self.df.empty:
            return None
        
        # --- 1. Calculate Trends ---
        try:
            if len(self.df) >= 2:
                from scipy.stats import linregress
                x_nums = (self.df['date_obj'] - self.df['date_obj'].min()).dt.total_seconds()
                y_ef = self.df['efficiency_factor']
                slope, intercept, r_value, p_value, std_err = linregress(x_nums, y_ef)
                trend_msg, trend_color = self.calculate_trend_stats(self.df)
            else:
                slope, intercept = 0, 0
                trend_msg, trend_color = "Trend: Insufficient Data", "#888888"
        except:
            slope, intercept = 0, 0
            trend_msg, trend_color = "Insufficient data", "#888888"
        
        # --- 2. Create Figure ---
        fig = make_subplots(specs=[[{"secondary_y": True}]])
        
        # --- 3. Decoupling Background Areas ---
        pos_d = self.df['decoupling'].copy()
        pos_d[pos_d < 0] = 0
        neg_d = self.df['decoupling'].copy()
        neg_d[neg_d > 0] = 0
        
        # Teal zone (stable/negative decoupling)
        fig.add_trace(go.Scatter(
            x=self.df['date_obj'], y=neg_d, hoverinfo='skip', showlegend=False,
            fill='tozeroy', mode='lines', line=dict(width=0),
            fillcolor='rgba(0, 128, 128, 0.15)'
        ), secondary_y=True)
        
        # Red zone (positive decoupling = cardiac drift)
        fig.add_trace(go.Scatter(
            x=self.df['date_obj'], y=pos_d, hoverinfo='skip', showlegend=False,
            fill='tozeroy', mode='lines',
            line=dict(color='rgba(255, 77, 77, 0.3)', width=1),
            fillcolor='rgba(255, 77, 77, 0.08)'
        ), secondary_y=True)
        
        # --- 4. EF Trend Line (subtle connector) ---
        fig.add_trace(go.Scatter(
            x=self.df['date_obj'], y=self.df['efficiency_factor'],
            name="EF Line", mode='lines', hoverinfo='skip', showlegend=False,
            line=dict(color='rgba(150, 150, 150, 0.25)', width=2, shape='spline')
        ), secondary_y=False)
        
        # --- 5. QUADRANT LOGIC (Colored Dots) ---
        # Threshold: user's mean EF splits "fast" vs "slow", 5% decoupling splits "stable" vs "drifting"
        mean_ef = self.df['efficiency_factor'].mean()
        decouple_threshold = 5.0
        
        groups = {
            'Efficient': self.df[(self.df['efficiency_factor'] >= mean_ef) & (self.df['decoupling'] <= decouple_threshold)],
            'Base':      self.df[(self.df['efficiency_factor'] < mean_ef) & (self.df['decoupling'] <= decouple_threshold)],
            'Pushing':   self.df[(self.df['efficiency_factor'] >= mean_ef) & (self.df['decoupling'] > decouple_threshold)],
            'Fatigued':  self.df[(self.df['efficiency_factor'] < mean_ef) & (self.df['decoupling'] > decouple_threshold)],
        }
        
        configs = {
            'Efficient': {'color': '#10B981', 'label': 'Efficient', 'symbol': 'circle',      'desc': 'High EF & Stable'},
            'Base':      {'color': '#3b82f6', 'label': 'Base',      'symbol': 'circle',      'desc': 'Building Foundation'},
            'Pushing':   {'color': '#f97316', 'label': 'Pushing',   'symbol': 'triangle-up', 'desc': 'Higher Output, Drift'},
            'Fatigued':  {'color': '#ef4444', 'label': 'Fatigued',  'symbol': 'x',           'desc': 'Fatigue / Warning'},
        }
        
        for key, group in groups.items():
            if not group.empty:
                cfg = configs[key]
                # Construct customdata: [decoupling, db_hash] - FIX: Use db_hash for click handler
                custom_data = list(zip(group['decoupling'], group.get('db_hash', [''] * len(group))))
                
                fig.add_trace(go.Scatter(
                    x=group['date_obj'], y=group['efficiency_factor'],
                    mode='markers', name=cfg['desc'],
                    marker=dict(
                        size=9, color=cfg['color'], symbol=cfg['symbol'],
                        line=dict(width=1, color='white')
                    ),
                    hovertemplate=(
                        f"<b>{cfg['label']}</b><br>"
                        f"<i style='color:#aaa'>{cfg['desc']}</i><br>"
                        "EF: %{y:.2f}<br>"
                        "Decoupling: %{customdata[0]:.1f}%"
                        "<extra></extra>"
                    ),
                    customdata=custom_data
                ), secondary_y=False)
        
        # --- 6. Trend Dashed Line ---
        if slope != 0:
            x_min = self.df['date_obj'].min().to_pydatetime()
            x_max = self.df['date_obj'].max().to_pydatetime()
            x_nums_for_line = [0, (self.df['date_obj'].max() - self.df['date_obj'].min()).total_seconds()]
            y_trend_line = [intercept + slope * x for x in x_nums_for_line]
            
            fig.add_trace(go.Scatter(
                x=[x_min, x_max], y=y_trend_line, name="Fitness Trend",
                mode='lines', line=dict(color=trend_color, width=2, dash='dash'),
                opacity=0.6, hoverinfo='skip', showlegend=False
            ), secondary_y=False)
        
        # --- 7. Layout ---
        fig.update_layout(
            template='plotly_dark',
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            height=400,
            margin=dict(l=40, r=40, t=20, b=40),
            showlegend=False,
            hovermode='closest',
            hoverlabel=dict(bgcolor='#18181b', font_color='white'),
            font=dict(color='#a1a1aa')
        )
        
        fig.update_yaxes(
            title_text="Running Efficiency", title_font=dict(color="#10B981"), tickfont=dict(color="#a1a1aa"),
            secondary_y=False, showgrid=True, gridcolor='#27272a'
        )
        fig.update_yaxes(
            title_text="Decoupling (%)", title_font=dict(color="#ff4d4d"), tickfont=dict(color="#a1a1aa"),
            secondary_y=True, range=[-5, max(20, self.df['decoupling'].max() + 2)],
            showgrid=False
        )
        fig.update_xaxes(showgrid=True, gridcolor='#27272a', tickfont=dict(color="#a1a1aa"))
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
                customdata=list(zip(
                    form_verdicts,  # 0
                    form_colors,  # 1
                    self.df['date_obj'].dt.strftime('%b %d, %Y'),  # 2
                    self.df['db_hash'] if 'db_hash' in self.df.columns else [''] * len(self.df)  # 3: Activity hash
                )),
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
            font=dict(color='#a1a1aa'),
            yaxis=dict(
                title=dict(text='Cadence (SPM)', font=dict(color='#a1a1aa')),
                tickfont=dict(color='#a1a1aa'),
                showgrid=True,
                gridcolor='#27272a'
            ),
            xaxis=dict(
                tickfont=dict(color='#a1a1aa'),
                showgrid=True,
                gridcolor='#27272a'
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
            # SAFETY CHECK: Need at least 2 points
            if len(self.df) < 2:
                raise ValueError("Not enough data")

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
                
                # Collect db_hash for click-to-modal
                hashes = df_group['db_hash'].tolist() if 'db_hash' in df_group.columns else [''] * len(df_group)
                
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
                            [color] * len(df_group),  # 10: Dot color
                            hashes  # 11: Activity hash for click-to-modal
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
                    cad_colors,  # 1: Dot color
                    self.df['db_hash'] if 'db_hash' in self.df.columns else [''] * len(self.df)  # 2: Activity hash
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
                font=dict(color='#e4e4e7')
            ),
            template="plotly_dark",
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            height=900, 
            showlegend=True,
            legend=dict(
                orientation="v",  # Vertical layout
                yanchor="top",
                y=0.70,  # Position on row 2 (middle section)
                xanchor="right",
                x=0.99,  # Position at right edge
                bgcolor="rgba(24, 24, 27, 0.8)",  # Zinc-900 transparent
                bordercolor="#27272a",  # Zinc-800
                borderwidth=1,
                font=dict(size=11, color='#a1a1aa')
            ),
            margin=dict(l=60, r=20, t=100, b=40),  # Reduced right margin to eliminate gap
            hoverlabel=dict(
                bgcolor="#18181b", 
                bordercolor="#27272a", 
                font_color="#e4e4e7"
            ),
            modebar={
                'remove': ['toImage', 'select2d', 'lasso2d', 'autoScale2d', 'resetScale2d', 'zoom', 'pan', 'zoomIn', 'zoomOut', 'autoScale', 'resetScale']
            },
            dragmode='zoom',
            hovermode='closest',
            font=dict(family="Inter, sans-serif", color='#a1a1aa')
        )
        
        # Axis Styling
        fig.update_yaxes(
            title_text="[Gains] Running Efficiency", 
            title_font=dict(color="#10B981"), tickfont=dict(color="#a1a1aa"),
            row=2, col=1, secondary_y=False,
            showgrid=True,
            gridcolor='#27272a'
        )
        fig.update_yaxes(
            title_text="Decoupling (%)", 
            title_font=dict(color="#ff4d4d"), tickfont=dict(color="#a1a1aa"),
            row=2, col=1, secondary_y=True, 
            range=[-5, max(20, self.df['decoupling'].max()+2)],
            showgrid=False
        )
        
        # Row 1 Y-axis (Weekly Volume)
        fig.update_yaxes(
            title_text="Miles",
            title_font=dict(color="#a1a1aa"), tickfont=dict(color="#a1a1aa"),
            row=1, col=1,
            showgrid=True,
            gridcolor='#27272a'
        )
        
        # Row 3 Y-axis (Cadence)
        fig.update_yaxes(
            title_text="Cadence (spm)", 
            title_font=dict(color="#a1a1aa"), tickfont=dict(color="#a1a1aa"),
            row=3, col=1,
            showgrid=True,
            gridcolor='#27272a'
        )
        
        # Update layout to enable stacked bars for row 1
        fig.update_layout(barmode='stack')
        
        # X-axis styling - show dates on all subplots
        fig.update_xaxes(
            row=1, col=1,
            showticklabels=True,  # Show dates on top graph
            showgrid=True,
            gridcolor='#27272a',
            tickfont=dict(color="#a1a1aa")
        )
        fig.update_xaxes(
            row=2, col=1,
            showticklabels=True,  # Show dates on middle graph
            showgrid=True,
            gridcolor='#27272a',
            tickfont=dict(color="#a1a1aa")
        )
        fig.update_xaxes(
            row=3, col=1,
            gridcolor='#27272a',
            tickfont=dict(color="#a1a1aa")
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
            return 'High Quality Miles', '#10b981', 'bg-emerald-500/20'
        elif broken_ratio < 25:
            return 'Structural Miles', '#3b82f6', 'bg-blue-500/20'
        else:
            return 'Broken Miles', '#ef4444', 'bg-red-500/20'
    
    def calculate_cadence_verdict(self, df):
        """
        Calculate verdict for Mechanics chart based on 'Form Score' Analysis.
        FIXED: Returns 'N/A' if fewer than 2 runs (prevents SmallSampleWarning).
        """
        # SAFETY CHECK: Need at least 2 runs to calculate a trend
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
            
            # Scoring Logic
            if verdict == 'ELITE FORM':
                scores.append(100)
                dates.append(row['date_obj'])
            elif verdict == 'GOOD FORM':
                scores.append(80)
                dates.append(row['date_obj'])
            elif verdict in ['HIKING / REST', 'AEROBIC / MIXED']:
                continue 
            elif verdict in ['PLODDING', 'LOW CADENCE']:
                scores.append(40)
                dates.append(row['date_obj'])
            elif verdict in ['OVERSTRIDING', 'HEAVY FEET', 'INEFFICIENT']:
                scores.append(0)
                dates.append(row['date_obj'])
                
        if not scores or len(scores) < 2:
            return 'STRUCTURAL', '#3b82f6', 'bg-blue-500/20'
            
        # 2. Calculate Average Quality
        avg_score = sum(scores) / len(scores)
        
        # 3. Calculate Trend
        try:
            from scipy.stats import linregress
            x_nums = [(d - min(dates)).total_seconds() for d in dates]
            # SAFETY CHECK again for the regression input
            if len(x_nums) < 2:
                slope_week = 0
            else:
                slope, _, _, _, _ = linregress(x_nums, scores)
                slope_week = slope * 604800
        except:
            slope_week = 0

        # --- FINAL VERDICT LOGIC ---
        if avg_score >= 90:
            if slope_week < -10: return 'SLIPPING', '#fbbf24', 'bg-yellow-500/20'
            return 'ELITE', '#10b981', 'bg-emerald-500/20'
        elif avg_score >= 60:
            if slope_week > 10: return 'IMPROVING', '#10b981', 'bg-emerald-500/20'
            elif slope_week < -10: return 'SLIPPING', '#fbbf24', 'bg-yellow-500/20'
            return 'GOOD', '#3b82f6', 'bg-blue-500/20'
        else:
            if slope_week > 10: return 'IMPROVING', '#3b82f6', 'bg-blue-500/20'
            return 'BROKEN', '#ef4444', 'bg-red-500/20'
    
    def calculate_efficiency_verdict(self, df):
        """
        Calculate verdict for Efficiency chart using nuanced logic.
        Understands that rising EF with moderate drift = progressive overload, not garbage.
        """
        if df is None or df.empty or len(df) < 2:
            return 'N/A', '#71717a', 'bg-zinc-700'
        
        try:
            from scipy.stats import linregress
            x_nums = (df['date_obj'] - df['date_obj'].min()).dt.total_seconds()
            
            # Calculate slopes
            y_dec = df['decoupling']
            slope_dec, _, _, _, _ = linregress(x_nums, y_dec)
            slope_dec_per_week = slope_dec * 604800
            
            y_ef = df['efficiency_factor']
            slope_ef, _, _, _, _ = linregress(x_nums, y_ef)
            slope_ef_per_week = (slope_ef * 604800) * 100
            
            # --- VERDICT LOGIC ---
            
            # 1. The Holy Grail: EF rising, decoupling stable or dropping
            if slope_ef_per_week > 0.5 and slope_dec_per_week < 0.3:
                return 'PEAKING', '#10b981', 'bg-emerald-500/20'
            
            # 2. Building Speed: EF rising fast, some drift is acceptable (progressive overload)
            if slope_ef_per_week > 1.0 and slope_dec_per_week > 0.3:
                return 'BUILDING', '#f97316', 'bg-orange-500/20'
            
            # 3. Fitness Loss: EF clearly declining
            if slope_ef_per_week < -0.5:
                return 'DETRAINING', '#ef4444', 'bg-red-500/20'
            
            # 4. Pure Drift: EF flat but decoupling increasing significantly
            if abs(slope_ef_per_week) <= 0.5 and slope_dec_per_week > 1.0:
                return 'DRIFTING', '#ef4444', 'bg-red-500/20'
            
            # 5. Default: modest gains or flat, manageable drift
            return 'STABLE', '#3b82f6', 'bg-blue-500/20'
            
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
                    # SAFETY CHECK: Need at least 2 points for regression
                    if len(self.df) < 2:
                        raise ValueError("Not enough data")

                    from scipy.stats import linregress
                    x_nums = (self.df['date_obj'] - self.df['date_obj'].min()).dt.total_seconds()
                    y_ef = self.df['efficiency_factor']
                    slope, intercept, r_value, p_value, std_err = linregress(x_nums, y_ef)
                    trend_msg, trend_color = self.calculate_trend_stats(self.df)
                    
                    # Calculate stats for the stats bar
                    r_squared = r_value ** 2
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
                
                # 1. Weekly Volume Composition Card (with Lens Switcher)
                # Generate initial chart based on current lens
                if self.volume_lens == 'mix':
                    volume_fig = self.generate_training_mix_chart()
                    vol_verdict, vol_color, vol_bg = self.calculate_mix_verdict(self.df)
                    vol_subtitle = 'Weekly distribution by run type'
                elif self.volume_lens == 'load':
                    volume_fig = self.generate_load_chart()
                    vol_verdict, vol_color, vol_bg = self.calculate_load_verdict(self.df)
                    vol_subtitle = 'Weekly distribution by training stress'
                elif self.volume_lens == 'zones':
                    volume_fig = self.generate_hr_zones_chart()
                    vol_verdict, vol_color, vol_bg = self.calculate_hr_zones_verdict(self.df)
                    vol_subtitle = 'Weekly time in each heart rate zone'
                else:
                    volume_fig = self.generate_weekly_volume_chart()
                    vol_verdict, vol_color, vol_bg = self.calculate_volume_verdict(self.df)
                    vol_subtitle = 'Breakdown in quality of miles (click any section to inspect runs)'
                
                if volume_fig:
                    with ui.card().classes('w-full bg-zinc-900 border border-zinc-800 p-6 mb-8').style('border-radius: 12px; box-shadow: 0 10px 15px -3px rgba(0, 0, 0, 0.3);'):
                        
                        # Header Row: Title + ❓ Icon + Verdict Badge (INLINE)
                        with ui.row().classes('w-full items-center gap-3 mb-1'):
                            ui.label('Training Volume').classes('text-xl font-bold text-white')
                            
                            # Verdict Badge
                            self.volume_verdict_label = ui.label(f'{vol_verdict}').classes(f'text-sm font-bold px-3 py-1 rounded {vol_bg}').style(f'color: {vol_color};')
                            
                            # Info Icon (inline)
                            ui.icon('help_outline').classes('text-zinc-500 hover:text-white transition-colors duration-200 cursor-pointer text-xl ml-auto').on(
                                'click', lambda: self.show_volume_info()
                            )
                        
                        # Subtitle (lens-adaptive)
                        self.volume_subtitle_label = ui.label(vol_subtitle).classes('text-sm text-zinc-400 mb-3')
                        
                        # === SEGMENTED TOGGLE (Pill Group) ===
                        def switch_lens(lens):
                            self.volume_lens = lens
                            # Update button styles
                            for l, btn in self._lens_buttons.items():
                                if l == lens:
                                    btn.classes(replace='text-sm px-4 py-1.5 rounded-full font-medium transition-all duration-200 bg-zinc-700 text-white')
                                    btn.style(replace='min-height: 0; line-height: 1; color: white !important;')
                                else:
                                    btn.classes(replace='text-sm px-4 py-1.5 rounded-full font-medium transition-all duration-200')
                                    btn.style(replace='min-height: 0; line-height: 1; color: #a1a1aa !important;')
                            self.refresh_volume_card()
                        
                        self._lens_buttons = {}
                        with ui.row().classes('w-full gap-1 mb-4 p-1 bg-zinc-800/50 rounded-full').style('width: fit-content;'):
                            for lens_key, lens_label in [('quality', 'Quality'), ('mix', 'Training Mix'), ('load', 'Load'), ('zones', 'HR Zones')]:
                                is_active = self.volume_lens == lens_key
                                btn = ui.button(lens_label, on_click=lambda lk=lens_key: switch_lens(lk)).props('flat no-caps')
                                if is_active:
                                    btn.classes('text-sm px-4 py-1.5 rounded-full font-medium transition-all duration-200 bg-zinc-700 text-white')
                                    btn.style('min-height: 0; line-height: 1; color: white !important;')
                                else:
                                    btn.classes('text-sm px-4 py-1.5 rounded-full font-medium transition-all duration-200')
                                    btn.style('min-height: 0; line-height: 1; color: #a1a1aa !important;')
                                self._lens_buttons[lens_key] = btn
                        
                        # === CHART CONTAINER (fixed height to prevent page jump) ===
                        self.volume_card_container = ui.column().classes('w-full').style('min-height: 320px;')
                        with self.volume_card_container:
                            self.volume_chart = ui.plotly(volume_fig).classes('w-full').style('cursor: pointer')
                            self.volume_chart.on('plotly_relayout', self.handle_volume_zoom)
                            self.volume_chart.on('plotly_click', self.handle_bar_click)
                
                # 2. Efficiency & Decoupling Card (with unified dual-metric stats bar)
                efficiency_fig = self.generate_efficiency_decoupling_chart()
                if efficiency_fig:
                    # Calculate efficiency verdict
                    eff_verdict, eff_color, eff_bg = self.calculate_efficiency_verdict(self.df)
                    
                    with ui.card().classes('w-full bg-zinc-900 border border-zinc-800 p-6 mb-8 h-full flex flex-col justify-between').style('border-radius: 12px; box-shadow: 0 10px 15px -3px rgba(0, 0, 0, 0.3);'):
                        # Header with verdict badge
                        with ui.row().classes('w-full items-center gap-3 mb-1'):
                            ui.label('Aerobic Efficiency').classes('text-xl font-bold text-white')
                            self.efficiency_verdict_label = ui.label(f'{eff_verdict}').classes(f'text-sm font-bold px-3 py-1 rounded {eff_bg}').style(f'color: {eff_color};')
                            ae_info_icon = ui.icon('help_outline').classes('text-zinc-500 hover:text-white cursor-pointer text-lg transition-colors')
                            ae_info_icon.on('click', lambda: self.show_aerobic_efficiency_info())
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
                                        # Decoupling is in percentage points per second
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
                        self.efficiency_chart.on('plotly_click', self.handle_efficiency_click)
                
                # 3. Cadence Trend Card
                cadence_fig = self.generate_cadence_trend_chart()
                if cadence_fig:
                    # Calculate cadence verdict
                    cad_verdict, cad_color, cad_bg = self.calculate_cadence_verdict(self.df)
                    
                    with ui.card().classes('w-full bg-zinc-900 border border-zinc-800 p-6 mb-8 h-full flex flex-col justify-between').style('border-radius: 12px; box-shadow: 0 10px 15px -3px rgba(0, 0, 0, 0.3);'):
                        # Header with verdict badge
                        with ui.row().classes('w-full items-center gap-3 mb-1'):
                            ui.label('Running Mechanics').classes('text-xl font-bold text-white')
                            self.cadence_verdict_label = ui.label(f'{cad_verdict}').classes(f'text-sm font-bold px-3 py-1 rounded {cad_bg}').style(f'color: {cad_color};')
                            form_info_icon = ui.icon('help_outline').classes('text-zinc-500 hover:text-white cursor-pointer text-lg transition-colors')
                            form_info_icon.on('click', lambda: self.show_form_info())
                        ui.label('Cadence trend showing turnover consistency').classes('text-sm text-zinc-400 mb-4')
                        
                        # Chart with zoom binding
                        self.cadence_chart = ui.plotly(cadence_fig).classes('w-full')
                        self.cadence_chart.on('plotly_relayout', self.handle_cadence_zoom)
                        self.cadence_chart.on('plotly_click', self.handle_cadence_click)
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
            self.efficiency_verdict_label.set_text(f'{eff_verdict}')
            self.efficiency_verdict_label.classes(f'text-sm font-bold px-3 py-1 rounded {eff_bg}', remove='bg-emerald-500/20 bg-blue-500/20 bg-red-500/20 bg-zinc-700')
            self.efficiency_verdict_label.style(f'color: {eff_color};')
                
        except Exception as ex:
            # Silently catch errors
            pass
    
    def handle_volume_zoom(self, e):
        """
        Handle zoom events on Volume Chart (Categorical Axis).
        Translates index ranges to data slices.
        Lens-aware: uses the correct verdict calculator for the current lens.
        """
        try:
            # Determine the correct verdict calculator based on current lens
            if self.volume_lens == 'mix':
                vol_verdict, vol_color, vol_bg = self.calculate_mix_verdict(self.df)
            elif self.volume_lens == 'load':
                vol_verdict, vol_color, vol_bg = self.calculate_load_verdict(self.df)
            elif self.volume_lens == 'zones':
                vol_verdict, vol_color, vol_bg = self.calculate_hr_zones_verdict(self.df)
            elif 'xaxis.range[0]' in e.args and 'xaxis.range[1]' in e.args:
                # Quality lens with zoom — recalculate with slice
                idx_start = e.args['xaxis.range[0]']
                idx_end = e.args['xaxis.range[1]']
                vol_verdict, vol_color, vol_bg = self.calculate_volume_verdict(
                    start_index=idx_start, 
                    end_index=idx_end
                )
            else:
                # Quality lens — reset / autorange
                vol_verdict, vol_color, vol_bg = self.calculate_volume_verdict()
            
            # Update UI
            self.volume_verdict_label.set_text(f'{vol_verdict}')
            self.volume_verdict_label.classes(f'text-sm font-bold px-3 py-1 rounded {vol_bg}', remove='bg-emerald-500/20 bg-blue-500/20 bg-red-500/20 bg-orange-500/20 bg-zinc-700 bg-zinc-800 text-zinc-500')
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
            self.cadence_verdict_label.set_text(f'{cad_verdict}')
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
                
                # SAFETY CHECK
                if len(df_zoomed) < 2:
                    return

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
                'High Quality Miles': ('text-emerald-400', 'bg-emerald-500/20', 'border-emerald-500/30'),
                'Structural Miles': ('text-blue-400', 'bg-blue-500/20', 'border-blue-500/30'),
                'Broken Miles': ('text-red-400', 'bg-red-500/20', 'border-red-500/30')
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

    async def handle_efficiency_click(self, e):
        """Handle click on Efficiency/Cadence scatter data point → open activity detail modal.
        
        The multi-subplot chart has two trace types:
        - Efficiency traces: customdata has 12 items, hash at index 11
        - Cadence trace: customdata has 3 items, hash at index 2
        Note: Trendline/shading traces may have scalar customdata (float) — skip those.
        """
        try:
            point_data = e.args['points'][0]
            custom_data = point_data.get('customdata')
            if custom_data is None or not isinstance(custom_data, (list, tuple)):
                return
            
            activity_hash = None
            if len(custom_data) >= 2 and isinstance(custom_data[1], str):
             # Efficiency scatter trace: [decoupling, filename] -> filename at index 1
                activity_hash = custom_data[1]
            elif len(custom_data) >= 12:
                # Legacy/Other scatter trace (hash at index 11)
                activity_hash = custom_data[11]
            elif len(custom_data) >= 3:
                # Cadence subplot trace (hash at index 2)
                activity_hash = custom_data[2]
            
            if activity_hash and isinstance(activity_hash, str):
                await self.open_activity_detail_modal(activity_hash, from_feed=True)
        except Exception as ex:
            print(f"Efficiency Click Error: {ex}")

    async def handle_cadence_click(self, e):
        """Handle click on Cadence/Mechanics scatter data point → open activity detail modal."""
        try:
            point_data = e.args['points'][0]
            custom_data = point_data.get('customdata')
            if custom_data is None or not isinstance(custom_data, (list, tuple)) or len(custom_data) < 4:
                return
            activity_hash = custom_data[3]
            if activity_hash and isinstance(activity_hash, str):
                await self.open_activity_detail_modal(activity_hash, from_feed=True)
        except Exception as ex:
            print(f"Cadence Click Error: {ex}")

    async def copy_to_llm(self):
        """
        Copy detailed activity data to clipboard with LLM context.
        
        Upgrade: Uses "Cinematic Glass" Progress Modal + Sequential Processing.
        """
        # Check if we have data
        if not self.activities_data:
            ui.notify('No data to copy', type='warning')
            return
            
        # --- UI: Cinematic Glass Modal ---
        with ui.dialog() as progress_dialog, ui.card().classes(
            'bg-zinc-900/95 backdrop-blur-xl border border-white/10 rounded-2xl '
            'shadow-2xl shadow-emerald-500/20 min-w-[360px] p-6 items-start'
        ):
            with ui.column().classes('w-full gap-4'):
                # Header
                ui.label('Constructing Analysis...').classes('text-white font-medium tracking-wide text-lg')
                
                # Progress Bar (The Pill)
                with ui.element('div').classes('w-full h-1.5 bg-zinc-800 rounded-full overflow-hidden'):
                    progress_bar = ui.element('div').classes(
                        'h-full bg-gradient-to-r from-emerald-500 to-teal-400 transition-all duration-300'
                    ).style('width: 0%')
                
                # Status Label
                status_label = ui.label('Initializing...').classes('font-mono text-xs text-zinc-400')
                
        progress_dialog.open()
        await asyncio.sleep(0.1) # Let UI render
        
        try:
            # --- 1. PREPARE ACTIVITIES ---
            # Sort by date (Newest First)
            sorted_activities = sorted(self.activities_data, 
                                      key=lambda x: x.get('date', ''), 
                                      reverse=True)
            
            # Prepare list of metadata for loop
            files_to_process = []
            for activity in sorted_activities:
                activity_hash = activity.get('db_hash')
                if activity_hash:
                    db_activity = self.db.get_activity_by_hash(activity_hash)
                    if db_activity:
                        fit_file_path = self._locate_fit_file(db_activity)
                        if fit_file_path:
                            files_to_process.append({
                                'path': fit_file_path,
                                'activity': activity
                            })
            
            total_files = len(files_to_process)
            lap_splits_list = []
            
            # --- 2. SEQUENTIAL PROCESSING LOOP ---
            for i, info in enumerate(files_to_process):
                # Update UI
                count = i + 1
                percent = int((count / total_files) * 100)
                progress_bar.style(f'width: {percent}%')
                status_label.set_text(f"Parsing activity {count} of {total_files}...")
                
                # Process Single File (Background Thread)
                # Ensure we pass a LIST to match the function signature
                batch_result = await run.io_bound(_parse_fit_files_for_clipboard, [info])
                
                # Result is a list of lists (because we sent a batch of 1)
                # So we take the first item
                if batch_result:
                    lap_splits_list.append(batch_result[0])
                else:
                    lap_splits_list.append(None)
            
            # --- 3. BUILD REPORT TEXT ---
            status_label.set_text("Formatting report...")
            await asyncio.sleep(0.05)
            
            report_lines = []
            
            # We need to match activities to results. 
            # Note: files_to_process might be smaller than sorted_activities if some files were missing.
            # But here we iterate strictly over what we processed.
            for info, lap_splits in zip(files_to_process, lap_splits_list):
                activity = info['activity']
                
                # Extract Standard Metrics
                date = activity.get('date', '')
                time_str = activity.get('time', '')
                icon = activity.get('context_emoji', '🏃')
                context = activity.get('context_name', 'Run')
                dist = activity.get('distance_mi', 0)
                pace = activity.get('pace', '--:--')
                elev = activity.get('elevation_ft', 0)
                
                # Physiology Metrics
                avg_hr = activity.get('avg_hr', '--')
                max_hr = activity.get('max_hr', '--')
                power = activity.get('avg_power', '--')
                ef = activity.get('efficiency_factor', 0)
                decoupling = activity.get('decoupling', 0)
                
                # Decoupling Label Logic
                if decoupling < 5: dec_label = "Excellent"
                elif decoupling < 10: dec_label = "⚠️ Moderate"
                else: dec_label = "🛑 High Fatigue"
                
                # HRR Logic
                hrr_list = activity.get('hrr_list', [])
                hrr_str = "--"
                if hrr_list and len(hrr_list) > 0:
                    hrr_str = f"{hrr_list[0]}bpm (1min)"
                
                te_aerobic = activity.get('aerobic_te', '--')
                te_anaerobic = activity.get('anaerobic_te', '--')
                
                # Mechanics Logic
                cadence = activity.get('avg_cadence', '--')
                form_tag = "[✅ ELITE FORM]" if (cadence != '--' and int(cadence) > 170) else "[👍 GOOD FORM]"
                
                # --- FORMAT THE BLOCK ---
                block = f"""
RUN: {date} {time_str}
Type:        [{icon} {context}]
--------------------------------------------------
SUMMARY
Dist:        {dist:.2f} mi
Pace:        {pace} /mi
Elev Gain:   +{elev} ft

PHYSIOLOGY (The Engine)
Avg HR:      {avg_hr} bpm
Max HR:      {max_hr} bpm
Avg Power:   {power}W
EF:          {ef:.2f} (Efficiency Factor)
Decoupling:  {decoupling:.2f}% ({dec_label})
HRR:         {hrr_str}
Training Effect: {te_aerobic} Aerobic, {te_anaerobic} Anaerobic

MECHANICS (The Chassis)
Avg Cadence: {cadence} spm
Form Tag:    {form_tag}

[LAP SPLITS]"""
                report_lines.append(block)
                
                # Add Splits
                if lap_splits:
                    for lap in lap_splits:
                        l_num = lap.get('lap_number', 0)
                        l_dist = lap.get('distance', 0) * 0.000621371
                        l_pace = lap.get('actual_pace', '--:--')
                        l_hr = int(lap['avg_hr']) if lap.get('avg_hr') else '--'
                        raw_cad = lap.get('avg_cadence')
                        l_cad = int(raw_cad * 2) if raw_cad else '--'
                        l_elev = lap.get('total_ascent', 0) * 3.28084 if lap.get('total_ascent') else 0
                        
                        report_lines.append(f"{l_num} | {l_dist:.2f}mi | {l_pace}/mi | {l_hr}bpm | {l_cad}spm | +{int(l_elev)}ft")
                
                report_lines.append("\\n" + "="*50 + "\\n")
            
            # --- 4. ADD CONTEXT & COPY ---
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
            full_content = "\\n".join(report_lines) + llm_context
            
            import pyperclip
            pyperclip.copy(full_content)
            
            ui.notify('✅ Analysis Data Copied!', type='positive', close_button=True)
            
        except Exception as e:
            ui.notify(f'Error: {str(e)}', type='negative')
            print(f"❌ Error copying to clipboard: {str(e)}")
            import traceback
            traceback.print_exc()
        finally:
            progress_dialog.close()


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

                # --- Make lap_start timezone-aware if needed ---
                if lap_start and lap_start.tzinfo is None:
                    lap_start = lap_start.replace(tzinfo=timezone.utc).astimezone()

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
    # Suppress known NiceGUI framework listener-churn warning noise
    nicegui_logger = logging.getLogger('nicegui')
    nicegui_logger.addFilter(MuteFrameworkNoise())

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
