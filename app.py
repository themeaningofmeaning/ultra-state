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
import subprocess
import sys
from datetime import datetime, timezone

# Third-party imports
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from nicegui import ui, run

# Local imports
from analyzer import (
    FitAnalyzer,
    analyze_form,
    build_map_payload_from_streams,
    classify_split,
    compute_training_load_and_zones,
    gradient_color_from_t,
)
from db import DatabaseManager
from library_manager import LibraryManager
from hr_zones import (
    HR_ZONE_COLORS,
    HR_ZONE_DESCRIPTIONS,
    HR_ZONE_MAP_LEGEND_GRADIENT,
    HR_ZONE_ORDER,
    HR_ZONE_RANGE_LABELS,
    classify_hr_zone,
    classify_hr_zone_by_ratio,
)
from constants import (
    LOAD_CATEGORY,
    LOAD_CATEGORY_COLORS,
    LOAD_CATEGORY_DESCRIPTIONS,
    LOAD_CATEGORY_EMOJI,
    LOAD_MIX_VERDICT,
    SPLIT_BUCKET,
    TE_ICON_MAP,
    DEFAULT_TIMEFRAME,
    MODAL_TITLES,
    UI_COPY,
)
from state import AppState
from core.data_manager import DataManager
from core.llm_export import LLMExporter
from components.activity_modal import ActivityModal
from components.library_modal import LibraryModal
from components.analysis_view import AnalysisView
from components.layout import AppShell
from components.cards import (
    create_physiology_card, create_decoupling_card,
    create_running_dynamics_card, create_strategy_row, create_lap_splits_table,
)
from components.charts import create_hr_zone_chart, build_terrain_graph
from components.run_card import create_run_card
from analyzer import (
    calculate_hr_zones, calculate_gap_for_laps, calculate_aerobic_decoupling,
    calculate_run_walk_stats, calculate_terrain_stats,
)


class MuteFrameworkNoise(logging.Filter):
    def filter(self, record):
        # Filter out the specific NiceGUI warning about event listeners
        return "Event listeners changed after initial definition" not in record.getMessage()


MAP_PAYLOAD_VERSION = 5


# --- MAIN APPLICATION CLASS ---
class UltraStateApp:
    """Main application class for the Garmin FIT Analyzer."""
    
    def __init__(self):
        """Initialize the application with database and state."""
        self.db = DatabaseManager()
        self.library_manager = LibraryManager(db=self.db)

        # ── Centralized session state (Tier 1) ──────────────────────────
        # See state.py for field definitions and the observer API.
        # session_id is loaded from DB so 'Last Import' survives restarts.
        self.state = AppState()
        self.state.session_id = self.db.get_last_session_id()

        # ── Data lifecycle controller (Phase 3 logic decoupling) ─────────
        self.data_manager = DataManager(db=self.db, state=self.state)

        # ── Focus-mode save/restore (not observed — simple R/W pair) ────
        self.pre_focus_timeframe = DEFAULT_TIMEFRAME

        # ── UI widget handles ────────────────────────────────────────────
        self.activities_table = None

        # Define taxonomy of filter tags
        self.TAG_CONFIG = {
            # CONTEXT TAGS
            'Long Run':   {'icon': '🏃', 'color': 'purple'},
            'Tempo':      {'icon': '🔥', 'color': 'orange'},
            'Intervals':  {'icon': '⚡', 'color': 'orange'},
            'Hill Sprints': {'icon': '⛰️', 'color': 'emerald'},
            'Hills':      {'icon': '⛰️', 'color': 'emerald'},
            'Recovery':   {'icon': LOAD_CATEGORY_EMOJI['Recovery'], 'color': 'blue'},
            'Base':       {'icon': LOAD_CATEGORY_EMOJI['Base'],     'color': 'blue'},
            'Fartlek':    {'icon': '💨', 'color': 'orange'},
            'Steady':     {'icon': '⚓', 'color': 'cyan'},

            # PHYSIO TAGS (keys and icons from constants.TE_ICON_MAP)
            'VO2 MAX':    {'icon': TE_ICON_MAP['VO2 MAX']['icon'],   'color': TE_ICON_MAP['VO2 MAX']['color']},
            'ANAEROBIC':  {'icon': TE_ICON_MAP['ANAEROBIC']['icon'], 'color': TE_ICON_MAP['ANAEROBIC']['color']},
            'THRESHOLD':  {'icon': TE_ICON_MAP['THRESHOLD']['icon'], 'color': TE_ICON_MAP['THRESHOLD']['color']},
            'MAX POWER':  {'icon': TE_ICON_MAP['MAX POWER']['icon'], 'color': TE_ICON_MAP['MAX POWER']['color']},
            }

        # LRU detail cache is now owned by ActivityModal (moved in Phase 2 Step 3)

        # Background backfill task for legacy map payloads
        self._map_backfill_task = None

        # ── LibraryModal component (Phase 2 Step 5) ──────────────────────
        self.library_modal = LibraryModal(
            db=self.db,
            library_manager=self.library_manager,
            state=self.state,
            on_library_changed_cb=self.refresh_data_view,
            disable_controls_cb=self._set_sidebar_controls_disabled,
        )

        # ── AnalysisView component (Phase 2 Step 7) ──────────────────────
        self.analysis_view = AnalysisView(
            state=self.state,
            db=self.db,
            df=self.df,
            activities_data=self.activities_data,
            callbacks={
                'classify_run_type': self.data_manager.classify_run_type,
                'show_volume_info': self.show_volume_info,
                'show_aerobic_efficiency_info': self.show_aerobic_efficiency_info,
                'show_form_info': self.show_form_info,
                'handle_bar_click': self.handle_bar_click,
                'handle_efficiency_click': self.handle_efficiency_click,
                'handle_cadence_click': self.handle_cadence_click,
            },
        )

        # ── AppShell component (Phase 2 final step) ─────────────────────
        self.layout = AppShell(
            state=self.state,
            library_modal=self.library_modal,
            tag_config=self.TAG_CONFIG,
            callbacks={
                'on_timeframe_change': self.on_filter_change,
                'on_exit_focus_mode': self.exit_focus_mode,
                'on_export_csv': self.export_csv,
                'on_copy_to_llm': self.copy_to_llm,
                'on_save_chart': self.save_chart_to_downloads,
                'on_build_trends_tab': self.analysis_view.build,
                'on_update_activities_grid': self.update_activities_grid,
                'on_focus_selected': self.enter_focus_mode,
                'on_bulk_download': self.bulk_download,
                'on_bulk_delete': self.bulk_delete,
                'classify_run_type': self.data_manager.classify_run_type,
            },
        )

        # ── LLM export service (Phase 3 Step 2) ──────────────────────────
        self.llm_exporter = LLMExporter(
            db=self.db,
            library_manager=self.library_manager,
            layout=self.layout,
        )
        
        # Build UI
        self.build_ui()

        # ── Reactive subscriptions ───────────────────────────────────────
        # These replace imperative refresh calls scattered through the code.
        # The subscriber is called automatically whenever the field is written.
        #
        # NOTE: refresh_data_view is async; we wrap it via ui.timer(0, ..., once=True)
        # so it is scheduled onto the NiceGUI event loop rather than called inline,
        # which would block synchronously inside the __setattr__ call chain.
        def _schedule_data_refresh(value):
            """Schedule an async data refresh on the next event-loop tick.

            Guard: 'Focus' is a synthetic sentinel set by enter_focus_mode(),
            which manages its own in-memory data slice and fires explicit renders.
            A DB reload here would overwrite the filtered subset immediately after
            it was set, restoring all activities.
            """
            if value == 'Focus':
                return   # Focus Mode owns its data pipeline — do not reload from DB
            ui.timer(0, self.refresh_data_view, once=True)

        self.state.subscribe('timeframe', _schedule_data_refresh)
        self.state.subscribe('volume_lens', lambda _: self.analysis_view.refresh_volume_card())

        # ── ActivityModal component (Phase 2 Step 3) ─────────────────────
        # Instantiated after build_ui so all render-helper methods exist as
        # bound methods when passed into the callbacks dict.
        self.activity_modal = ActivityModal(
            db=self.db,
            state=self.state,
            callbacks=self._build_modal_callbacks(),
        )

        # Start background library services after event loop starts (script-mode safe).
        ui.timer(0.12, self.start_library_services, once=True)
        # Initial render pass.
        ui.timer(0.1, self.refresh_data_view, once=True)
        ui.timer(0.25, self.library_modal.refresh_status, once=True)
        ui.timer(1.0, self.library_modal.refresh_status)

        # Show Save Chart button initially with fade (Trends tab is default)
        ui.timer(0.05, lambda: self.save_chart_btn.style('opacity: 1; pointer-events: auto;'), once=True)

        # Check for updates in the background (Option A: notification only).
        # Runs 3 seconds after startup so it never competes with initial render.
        from updater import check_and_notify
        ui.timer(3.0, check_and_notify, once=True)

    @property
    def df(self):
        return self.data_manager.df

    @df.setter
    def df(self, value):
        self.data_manager.df = value

    @property
    def activities_data(self):
        return self.data_manager.activities_data

    @activities_data.setter
    def activities_data(self, value):
        self.data_manager.activities_data = value or []

    def _build_modal_callbacks(self) -> dict:
        """
        Build the callbacks dict injected into ActivityModal.

        All references are captured as bound methods of this UltraStateApp
        instance. ActivityModal calls them without importing UltraStateApp,
        keeping the dependency boundary clean.

        Must be called after build_ui() so widget-referencing helpers exist.
        """
        return {
            # ── Data helpers ────────────────────────────────────────────────
            'locate_fit_cb':       self.llm_exporter._locate_file,
            'map_payload_cb':      self._get_or_backfill_map_payload,
            'normalize_bounds_cb': self._normalize_bounds,
            'calc_distance_cb':    self._calculate_distance_from_speed,
            # ── Compute helpers ─────────────────────────────────────────────
            'hr_zones_cb':         calculate_hr_zones,
            'gap_laps_cb':         calculate_gap_for_laps,
            'decoupling_cb':       calculate_aerobic_decoupling,
            'run_walk_cb':         calculate_run_walk_stats,
            'terrain_cb':          calculate_terrain_stats,
            # ── Render helpers ──────────────────────────────────────────────
            'physiology_card_cb':  create_physiology_card,
            'hr_zone_chart_cb':    create_hr_zone_chart,
            'lap_splits_cb':       create_lap_splits_table,
            'decoupling_card_cb':  lambda data, efficiency_factor=None: create_decoupling_card(
                data, efficiency_factor,
                aerobic_verdict_cb=self.classify_single_run_aerobic_verdict,
                aerobic_info_cb=self.show_aerobic_efficiency_info
            ),
            'dynamics_card_cb':    lambda data: create_running_dynamics_card(
                data,
                form_info_cb=self.show_form_info
            ),
            'strategy_row_cb':     create_strategy_row,
            'terrain_graph_cb':    build_terrain_graph,
            # ── Action helpers ──────────────────────────────────────────────
            'copy_splits_cb':      self.copy_splits_to_clipboard,
            # ── Navigation (recursive self-reference via shim) ──────────────
            'open_modal_cb':       self.open_activity_detail_modal,
        }

    def _sync_layout_refs(self):
        """Expose AppShell-owned widgets on app attributes for compatibility."""
        self.timeframe_select = self.layout.timeframe_select
        self.focus_token = self.layout.focus_token
        self.focus_token_label = self.layout.focus_token_label
        self.export_btn = self.layout.export_btn
        self.copy_btn = self.layout.copy_btn
        self.copy_btn_label = self.layout.copy_btn_label
        self.copy_loading_dialog = self.layout.copy_loading_dialog
        self.save_chart_btn = self.layout.save_chart_btn
        self.feed_container = self.layout.feed_container
        self.filter_container = self.layout.filter_container
        self.grid_container = self.layout.grid_container
        self.fab_container = self.layout.fab_container

    def _set_sidebar_controls_disabled(self, disabled: bool):
        """Enable/disable top-level sidebar controls during heavy library operations."""
        if not getattr(self, 'timeframe_select', None):
            return
        if disabled:
            self.timeframe_select.props(add='disable')
        else:
            self.timeframe_select.props(remove='disable')

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

    # NOTE: _multi_color_from_t() was removed — use gradient_color_from_t(t)
    # from analyzer.py, which is the single authoritative implementation of
    # the Garmin 5-color gradient. See ARCHITECTURE.md > Color Gradient.

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
            converted.append([seg[0], seg[1], seg[2], seg[3], gradient_color_from_t(t)])
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
        segments = map_payload.get('segments')
        bounds = self._normalize_bounds(map_payload.get('bounds'))
        if not isinstance(segments, list) or not segments:
            return True
        if not bounds:
            return True
        # v5 relies on FIT-stream-aware regeneration; do not "upgrade" v4 payloads from legacy segments.
        if version < MAP_PAYLOAD_VERSION:
            return version < 4
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
            if bounds and isinstance(segments, list) and segments:
                map_payload['bounds'] = bounds
                activity['map_payload'] = map_payload
                activity['map_payload_version'] = version
                if version >= MAP_PAYLOAD_VERSION:
                    return map_payload
                if version >= 4:
                    # Keep valid v4 payloads until we can rebuild from source FIT streams.
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
        Thin shim — extracted to ActivityModal._fetch_detail() in Phase 2 Step 3.
        Kept for backward compatibility with any direct callers.
        """
        return await self.activity_modal._fetch_detail(activity_hash)

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
    
    def create_form_analysis_chart(self, distance_stream, cadence_stream, elevation_stream,
                                 timestamps=None, vertical_oscillation=None, stance_time=None,
                                 vertical_ratio=None, step_length=None, use_miles=True):
        """Legacy wrapper – delegates to _build_terrain_graph with metric='Cadence'."""
        detail_data = {
            'distance_stream': distance_stream,
            'cadence_stream': cadence_stream,
            'elevation_stream': elevation_stream,
            'timestamps': timestamps,
            'vertical_oscillation': vertical_oscillation,
            'stance_time': stance_time,
            'vertical_ratio': vertical_ratio,
            'step_length': step_length,
        }
        return build_terrain_graph(detail_data, metric='Cadence', use_miles=use_miles)
    
    def classify_single_run_aerobic_verdict(self, run_ef, run_decoupling, avg_ef=None):
        """
        Classify a single run's aerobic efficiency verdict.

        Returns:
            tuple: (verdict, bg_class, border_class, text_class, icon)
        """
        if avg_ef is None:
            avg_ef = self.df['efficiency_factor'].mean() if self.df is not None else 0
        if pd.isna(avg_ef):
            avg_ef = 0

        ef_above_avg = run_ef >= avg_ef if avg_ef > 0 else False
        low_decouple = run_decoupling <= 5

        if ef_above_avg and low_decouple:
            return 'Efficient', 'bg-emerald-500/10', 'border-emerald-700/30', 'text-emerald-400', '⚡'
        if not ef_above_avg and low_decouple:
            return 'Base', 'bg-blue-500/10', 'border-blue-700/30', 'text-blue-400', '🧱'
        if ef_above_avg and not low_decouple:
            return 'Pushing', 'bg-orange-500/10', 'border-orange-700/30', 'text-orange-400', '🔥'
        return 'Fatigued', 'bg-red-500/10', 'border-red-700/30', 'text-red-400', '⚠️'

    async def open_activity_detail_modal(self, activity_hash, from_feed=False, navigation_list=None):
        """
        Thin shim — extracted to ActivityModal.open() in Phase 2 Step 3.
        Kept as a forwarding method so nav closures (prev/next) keep working
        without circular import issues inside the component.
        """
        await self.activity_modal.open(
            activity_hash,
            from_feed=from_feed,
            navigation_list=navigation_list,
        )

    def build_ui(self):
        """Construct the complete UI layout with fixed sidebar."""
        
        # 1. INTEGRATED CSS CONFIGURATION
        ui.add_head_html('''
        <style>
        /**************************************************/
        /* 1. THE APPLE-STYLE HOVER FIX (DARK MODE)       */
        /**************************************************/

        /* Aurora Gradient Button Force Override */
        .ai-gradient-btn {
            background: linear-gradient(135deg, #059669 0%, #34D399 100%) !important;
            box-shadow: 0 0 12px rgba(52, 211, 153, 0.4) !important;
            border: none !important;
            color: white !important;
        }

        /* Primary Tab Indicator */
        .ember-tabs .q-tab__indicator {
            background: linear-gradient(135deg, #059669 0%, #34D399 100%) !important;
            box-shadow: 0 0 10px rgba(52, 211, 153, 0.5) !important;
            height: 3px !important;
            border-radius: 2px 2px 0 0;
        }
        .ai-gradient-btn:hover {
            filter: brightness(1.1);
            transform: scale(1.02);
        }

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

        /* Library status row + modal controls */
        .library-status-row.q-btn {
            background-color: transparent !important;
            border: none !important;
            box-shadow: none !important;
            color: #e4e4e7 !important;
        }
        .library-status-row.q-btn:hover {
            background-color: rgba(39, 39, 42, 0.55) !important;
        }
        .library-status-row.q-btn .q-focus-helper {
            display: none !important;
        }
        .library-modal-resync-icon.q-btn {
            background-color: transparent !important;
            color: #a1a1aa !important;
            border: 1px solid #3f3f46 !important;
            min-height: 28px !important;
            min-width: 28px !important;
        }
        .library-modal-resync-icon.q-btn .q-icon {
            transition: transform 0.25s ease;
        }
        .library-modal-resync-icon.q-btn:hover {
            background-color: #27272a !important;
            color: #e4e4e7 !important;
        }
        .library-modal-resync-icon.q-btn:hover .q-icon {
            transform: rotate(180deg);
        }
        .library-modal-resync-icon.q-btn .q-focus-helper {
            display: none !important;
        }
        .library-modal-secondary.q-btn {
            background-color: transparent !important;
            color: #a1a1aa !important;
            border: 1px solid #3f3f46 !important;
        }
        .library-modal-secondary.q-btn:hover {
            background-color: #27272a !important;
            color: #d4d4d8 !important;
        }
        .library-modal-primary.q-btn {
            background-color: #10b981 !important;
            color: #ffffff !important;
            border: 1px solid #10b981 !important;
        }
        .library-modal-primary.q-btn:hover {
            background-color: #34d399 !important;
            color: #ffffff !important;
            border-color: #34d399 !important;
        }
        .library-modal-secondary.q-btn .q-focus-helper,
        .library-modal-primary.q-btn .q-focus-helper {
            display: none !important;
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
            box-shadow: inset 0 1px 0 0 rgba(255, 255, 255, 0.05), 0 2px 12px var(--strain-shadow) !important;
            border-radius: 12px !important;
            transition: transform 0.2s ease, box-shadow 0.2s ease, border-color 0.2s ease !important;
        }
        .glass-card:hover {
            transform: translateY(-2px) !important;
            /* Intensify glow to 20% and brighter top edge */
            box-shadow: inset 0 1px 0 0 rgba(255, 255, 255, 0.1), 0 4px 16px var(--strain-shadow-hover) !important;
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
            box-shadow: 0 6px 20px rgba(var(--theme-color-rgb), 0.3) !important;
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

        /**************************************************/
        /* 7. FOCUS TOKEN BLOOM ANIMATION                 */
        /**************************************************/
        @keyframes bloom-pulse {
            0% {
                box-shadow: 0 0 12px rgba(255, 255, 255, 0.05);
                border-color: rgba(63, 63, 70, 0.8); /* zinc-700 */
            }
            40% {
                box-shadow: 0 0 25px rgba(255, 255, 255, 0.3);
                border-color: rgba(161, 161, 170, 0.8); /* zinc-400 */
            }
            100% {
                box-shadow: 0 0 0px rgba(0, 0, 0, 0);
                border-color: rgba(63, 63, 70, 1.0); /* zinc-700 */
            }
        }
        .animate-bloom {
            animation: bloom-pulse 600ms ease-out 1 forwards;
        }
        </style>
        ''')
        
        # 2. COLOR SCHEME
        ui.colors(primary='#10B981', secondary='#1F1F1F', accent='#ff9900', 
                  dark='#09090b', positive='#10B981', negative='#ff4d4d', 
                  info='#3b82f6', warning='#ff9900')
        
        ui.query('body').classes('bg-zinc-950')
        
        # 3. MAIN LAYOUT
        self.layout.build()
        self._sync_layout_refs()
    

    async def start_library_services(self):
        """Start library manager loop after NiceGUI app startup."""
        try:
            await self.library_manager.start()
            self.state.session_id = self.db.get_last_session_id()
            self.library_modal.update_run_count()
            await self.library_modal.refresh_status()
        except Exception as e:
            ui.notify(f'Library startup error: {e}', type='warning')

    async def stop_library_services(self):
        """Stop library manager loop on app shutdown."""
        await self.library_manager.stop()

    @staticmethod
    def _move_file_to_trash(file_path):
        """Best-effort cross-platform move to Trash/Recycle Bin."""
        if not file_path or not os.path.exists(file_path):
            return False, 'File not found on disk.'

        try:
            if sys.platform == 'darwin':
                escaped = file_path.replace('\\', '\\\\').replace('"', '\\"')
                script = f'tell application "Finder" to delete POSIX file "{escaped}"'
                subprocess.run(
                    ['osascript', '-e', script],
                    check=True,
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.PIPE,
                )
                return True, None

            if sys.platform.startswith('win'):
                safe_path = file_path.replace("'", "''")
                powershell = (
                    "Add-Type -AssemblyName Microsoft.VisualBasic; "
                    "[Microsoft.VisualBasic.FileIO.FileSystem]::DeleteFile("
                    f"'{safe_path}', 'OnlyErrorDialogs', 'SendToRecycleBin')"
                )
                subprocess.run(
                    ['powershell', '-NoProfile', '-Command', powershell],
                    check=True,
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.PIPE,
                )
                return True, None

            for cmd in (['gio', 'trash', file_path], ['trash-put', file_path]):
                try:
                    subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
                    return True, None
                except Exception:
                    continue

            return False, 'Trash utility not available on this platform.'
        except Exception as exc:
            return False, str(exc)

    def _delete_activity_with_library_cleanup(self, activity_hash, file_path=None):
        """Delete activity row + library index row; attempt to move source file to Trash."""
        resolved_path = file_path or self.db.get_activity_file_path(activity_hash)
        trashed = False
        trash_error = None

        if resolved_path:
            trashed, trash_error = self._move_file_to_trash(resolved_path)

        self.db.delete_activity(activity_hash)
        self.db.delete_library_file(activity_hash)
        return trashed, trash_error, resolved_path
    
    
    
    



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

        # 2. Filter the main data list in memory and rebuild DataFrame cache
        focused_activities = [act for act in self.activities_data if act.get('db_hash') in hashes]
        self.data_manager.set_activities_data(focused_activities)

        # 3. Sync extracted views with focused data
        self.analysis_view.set_data(self.df, self.activities_data)
        self.layout.set_data(self.df, self.activities_data)
        self.layout.update_filter_bar()
        
        # 4. Update State
        self.state.focus_mode_active = True
        self.state.timeframe = 'Focus'
        
        # 5. Swap out the UI
        # Save previous timeframe so we can restore it on exit
        if not str(self.timeframe_select.value).startswith('🎯'):
            self.pre_focus_timeframe = self.timeframe_select.value
            
        self.timeframe_select.classes(add='hidden')
        self.focus_token_label.set_text(f"🎯 Focus ({len(hashes)})")
        
        # Unhide the token and trigger the bloom animation
        self.focus_token.classes(remove='hidden', add='animate-bloom')
        # 6. Refresh All Views (Feed, Charts, and Table)
        self.analysis_view.update_trends_chart()
        self.update_report_text()
        self.update_activities_grid()
        
        # 7. Hide the bar
        self.layout.hide_floating_action_bar()
        ui.notify("Focus Mode — select a timeframe to exit", type='info', icon='center_focus_strong')

    async def exit_focus_mode(self):
        """Exit Focus Mode — restore original timeframe UI and reload data."""
        self.state.focus_mode_active = False
        
        # Swap out the UI and clear the animation so it can trigger next time
        self.focus_token.classes(add='hidden', remove='animate-bloom')
        self.timeframe_select.classes(remove='hidden')
        
        # Guard flag prevents on_filter_change from immediately looping
        self.state.entering_focus_mode = True
        self.timeframe_select.value = self.pre_focus_timeframe
        self.state.entering_focus_mode = False
        
        # Update timeframe and reload data
        self.state.timeframe = self.pre_focus_timeframe
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
        self.layout.hide_floating_action_bar()
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
            trashed = 0
            trash_failures = 0
            for row in rows:
                if isinstance(row, dict):
                    h = row.get('hash')
                    if h:
                        file_path = row.get('file_path')
                        moved, _trash_error, _resolved_path = self._delete_activity_with_library_cleanup(
                            h,
                            file_path=file_path,
                        )
                        deleted += 1
                        if moved:
                            trashed += 1
                        elif _resolved_path:
                            trash_failures += 1
            self.library_modal.update_run_count()
            # Clear selection and hide FAB
            if hasattr(self, 'activities_table') and self.activities_table:
                self.activities_table.selected.clear()
            self.activities_table.update()
            self.layout.hide_floating_action_bar()
            await self.refresh_data_view()
            if trash_failures > 0:
                ui.notify(
                    f"Deleted {deleted} activit{'ies' if deleted != 1 else 'y'}; {trash_failures} file(s) could not be moved to Trash.",
                    type='warning',
                    icon='delete',
                )
            else:
                ui.notify(
                    f"Deleted {deleted} activit{'ies' if deleted != 1 else 'y'} (trashed {trashed} file(s)).",
                    type='positive',
                    icon='delete',
                )
    
    async def refresh_data_view(self):
        """
        Refresh data using DB-side sorting.
        """
        self.data_manager.load_data()
        self.analysis_view.set_data(self.df, self.activities_data)
        self.layout.set_data(self.df, self.activities_data)
        
        # --- CRITICAL FIX: Update Filter Bar AFTER data load ---
        self.update_report_text() 
        self.update_activities_grid() 
        self.layout.update_filter_bar()
        self.analysis_view.update_trends_chart() 
        
        # Toggle buttons and LLM safety lock
        has_data = bool(self.activities_data)
        if has_data:
            self.export_btn.props(remove='disable')
            # LLM Safety Lock: large timeframes contain too much data for an LLM context window
            if self.state.timeframe in ('All Time', 'This Year'):
                self.copy_btn.style('opacity: 0.5; pointer-events: none;')
                self.copy_btn_label.text = 'Too much data for LLM'
            else:
                self.copy_btn.style('opacity: 1; pointer-events: auto; cursor: pointer;')
                self.copy_btn_label.text = 'COPY FOR AI'
        else:
            self.export_btn.props(add='disable')
            self.copy_btn.style('opacity: 0.5; pointer-events: none;')

        # Background map payload migration for older activities (non-blocking)
        if self.activities_data:
            if self._map_backfill_task is None or self._map_backfill_task.done():
                self._map_backfill_task = asyncio.create_task(self._backfill_map_payloads_for_loaded_activities())
    
    
    # Placeholder methods for handlers (to be implemented in later tasks)
    async def on_filter_change(self, e):
        """
        Handle timeframe filter changes from the dropdown.

        Writing self.state.timeframe automatically triggers refresh_data_view
        via the subscriber registered in __init__.  This method only handles
        side-effects that are tightly bound to the *user gesture* itself
        (focus mode cleanup, UI class swaps).
        """
        # Guard: skip if we're programmatically setting the select during focus mode transitions
        if self.state.entering_focus_mode:
            return

        # If we're in focus mode and user picks a DIFFERENT timeframe, exit focus mode cleanly
        if self.state.focus_mode_active and not str(e.value).startswith('🎯'):
            self.state.focus_mode_active = False
            self.focus_token.classes(add='hidden')
            self.timeframe_select.classes(remove='hidden')

        # Writing timeframe fires the subscriber which schedules refresh_data_view.
        # The LLM safety lock is applied inside refresh_data_view itself.
        self.state.timeframe = e.value
    
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
            _feed_sorted = sorted(self.activities_data, key=lambda x: x.get('date', ''), reverse=True)
            _feed_nav_list = [d.get('db_hash') for d in _feed_sorted if d.get('db_hash')]
            card_callbacks = {
                'on_click': lambda activity_hash, nav_list: self.open_activity_detail_modal(
                    activity_hash,
                    from_feed=True,
                    navigation_list=nav_list,
                ),
                'on_copy': lambda activity_hash: self.llm_exporter.generate_export(
                    self.data_manager.activities_data,
                    target_activity_id=activity_hash,
                ),
                'on_te_info': self.show_training_effect_info,
                'on_eff_info': lambda verdict: self.show_aerobic_efficiency_info(highlight_verdict=verdict),
                'on_load_info': self.show_load_info,
                'on_form_info': lambda verdict: self.show_form_info(highlight_verdict=verdict),
            }

            for activity in _feed_sorted:
                create_run_card(
                    activity,
                    avg_ef=avg_ef,
                    long_run_threshold=long_run_threshold,
                    navigation_list=_feed_nav_list,
                    classify_run_type_cb=self.data_manager.classify_run_type,
                    classify_aerobic_verdict_cb=self.classify_single_run_aerobic_verdict,
                    callbacks=card_callbacks,
                )
    
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
    
    def show_volume_info(self, highlight_verdict=None):
        """
        Show informational modal about the current Volume lens.
        Content adapts based on self.state.volume_lens and highlights the active verdict.
        """
        current_verdict = highlight_verdict
        if not current_verdict and getattr(self.analysis_view, 'volume_verdict_label', None):
            current_verdict = getattr(self.analysis_view.volume_verdict_label, 'text', None)

        if not current_verdict:
            if self.state.volume_lens == 'mix':
                current_verdict, _, _ = self.analysis_view.calculate_mix_verdict()
            elif self.state.volume_lens == 'load':
                current_verdict, _, _ = self.analysis_view.calculate_load_verdict()
            elif self.state.volume_lens == 'zones':
                current_verdict, _, _ = self.analysis_view.calculate_hr_zones_verdict()
            else:
                current_verdict, _, _ = self.analysis_view.calculate_volume_verdict()

        with ui.dialog() as dialog, ui.card().classes('bg-zinc-900 text-white p-6 max-w-2xl border border-zinc-800'):

            def style_card(label_verdict, base_color, text_color):
                is_active = (current_verdict == label_verdict)
                if is_active:
                    if label_verdict == 'ZONE 3 JUNK':
                        return (
                            'background: rgba(71, 85, 105, 0.42); '
                            'border: 2px solid rgba(148, 163, 184, 0.98); '
                            'box-shadow: 0 0 0 1px rgba(226, 232, 240, 0.30) inset, 0 0 24px rgba(148, 163, 184, 0.24);',
                            'text-slate-100'
                        )
                    return f'background: {base_color}2A; border: 2px solid {base_color}90;', text_color
                return f'background: {base_color}14; border: 1px solid {base_color}40;', text_color

            if self.state.volume_lens == 'quality':
                ui.label('Volume Quality Analysis').classes('text-xl font-bold text-white mb-2')
                with ui.column().classes('gap-4'):
                    s, t = style_card('High Quality Miles', '#10b981', 'text-emerald-400')
                    with ui.element('div').classes('rounded-lg p-3').style(s):
                        with ui.row().classes('gap-3 items-start'):
                            ui.icon('verified').classes(f'{t} text-xl mt-0.5')
                            with ui.column().classes('gap-1'):
                                ui.label('High Quality (The Engine)').classes(f'text-base font-bold {t}')
                                ui.label('Running on flat/rolling terrain with good mechanics (Cadence > 160) and honest effort. These miles build fitness without breaking the chassis.').classes('text-sm text-zinc-300')

                    s, t = style_card('Structural Miles', '#3b82f6', 'text-blue-400')
                    with ui.element('div').classes('rounded-lg p-3').style(s):
                        with ui.row().classes('gap-3 items-start'):
                            ui.icon('hiking').classes(f'{t} text-xl mt-0.5')
                            with ui.column().classes('gap-1'):
                                ui.label('Structural (The Base)').classes(f'text-base font-bold {t}')
                                ui.label('Valid volume that includes Hiking (Steep Grade), Recovery Shuffles (Low HR), or Walking. These miles build durability and aerobic base without the mechanical stress of fast running.').classes('text-sm text-zinc-300')

                    s, t = style_card('Broken Miles', '#ef4444', 'text-red-400')
                    with ui.element('div').classes('rounded-lg p-3').style(s):
                        with ui.row().classes('gap-3 items-start'):
                            ui.icon('warning').classes(f'{t} text-xl mt-0.5')
                            with ui.column().classes('gap-1'):
                                ui.label('Broken (The Junk)').classes(f'text-base font-bold {t}')
                                ui.label('The "Danger Zone." You are working hard (High HR) but moving poorly (Low Cadence). This usually happens at the end of long runs when form falls apart. These miles cause injury.').classes('text-sm text-zinc-300')

                ui.separator().classes('my-6 border-zinc-800')
                with ui.column().classes('gap-2 mb-4'):
                    ui.label('How we decide:').classes('text-xs font-bold text-zinc-500 uppercase tracking-wider')
                    ui.label('We analyze every single mile split against Terrain, Metabolic Cost, and Mechanics.').classes('text-sm text-zinc-400')

            elif self.state.volume_lens == 'mix':
                ui.label('Training Mix Analysis').classes('text-xl font-bold text-white mb-2')
                ui.label('Shows how your weekly mileage breaks down by run type (Easy, Tempo, Hard). Your verdict reflects the balance:').classes('text-sm text-zinc-400 mb-4')
                with ui.column().classes('gap-4'):
                    s, t = style_card('POLARIZED', '#10b981', 'text-emerald-400')
                    with ui.element('div').classes('rounded-lg p-3').style(s):
                        with ui.row().classes('items-center gap-2 mb-1'):
                            ui.icon('check_circle').classes(f'{t} text-lg')
                            ui.label('POLARIZED').classes(f'text-sm font-bold {t}')
                        ui.label('The gold standard — 80%+ easy miles with purposeful hard sessions. You\'re building a big aerobic engine while sharpening speed. Keep it up.').classes('text-sm text-zinc-300')

                    s, t = style_card('BALANCED', '#3b82f6', 'text-blue-400')
                    with ui.element('div').classes('rounded-lg p-3').style(s):
                        with ui.row().classes('items-center gap-2 mb-1'):
                            ui.icon('balance').classes(f'{t} text-lg')
                            ui.label('BALANCED').classes(f'text-sm font-bold {t}')
                        ui.label('A decent variety of run types. Not bad, but pushing more volume into easy runs would unlock better aerobic gains with less fatigue.').classes('text-sm text-zinc-300')

                    s, t = style_card('TEMPO HEAVY', '#ef4444', 'text-red-400')
                    with ui.element('div').classes('rounded-lg p-3').style(s):
                        with ui.row().classes('items-center gap-2 mb-1'):
                            ui.icon('warning_amber').classes(f'{t} text-lg')
                            ui.label('TEMPO HEAVY').classes(f'text-sm font-bold {t}')
                        ui.label('Too much time in the moderate zone without enough easy. This leads to chronic fatigue without the recovery to absorb it. Swap some tempo runs for true easy days.').classes('text-sm text-zinc-300')

                    s, t = style_card('MONOTONE', '#f97316', 'text-orange-400')
                    with ui.element('div').classes('rounded-lg p-3').style(s):
                        with ui.row().classes('items-center gap-2 mb-1'):
                            ui.icon('repeat').classes(f'{t} text-lg')
                            ui.label('MONOTONE').classes(f'text-sm font-bold {t}')
                        ui.label('Nearly all your miles are the same type. Add variety — even one tempo or interval session per week creates a stronger training stimulus.').classes('text-sm text-zinc-300')

            elif self.state.volume_lens == 'load':
                ui.label('Training Load Analysis').classes('text-xl font-bold text-white mb-2')
                ui.label('Each bar shows your weekly mileage broken down by training stress. Strain is calculated from duration, intensity, and heart rate for each run:').classes('text-sm text-zinc-400 mb-4')
                with ui.column().classes('gap-4'):
                    s, t = style_card('MAINTAINING', '#10b981', 'text-emerald-400')
                    with ui.element('div').classes('rounded-lg p-3').style(s):
                        with ui.row().classes('items-center gap-2 mb-1'):
                            ui.icon('check_circle').classes(f'{t} text-lg')
                            ui.label('MAINTAINING').classes(f'text-sm font-bold {t}')
                        ui.label('A healthy mix of easy and hard runs — your training load is sustainable and balanced. The sweet spot for steady progress without burnout.').classes('text-sm text-zinc-300')

                    s, t = style_card('UNDERTRAINED', '#3b82f6', 'text-blue-400')
                    with ui.element('div').classes('rounded-lg p-3').style(s):
                        with ui.row().classes('items-center gap-2 mb-1'):
                            ui.icon('spa').classes(f'{t} text-lg')
                            ui.label('UNDERTRAINED').classes(f'text-sm font-bold {t}')
                        ui.label('Nearly all recovery-level runs with no real stimulus. Great after a race, but sustained easy-only training won\'t build fitness. Add one harder session per week.').classes('text-sm text-zinc-300')

                    s, t = style_card('PRODUCTIVE', '#f97316', 'text-orange-400')
                    with ui.element('div').classes('rounded-lg p-3').style(s):
                        with ui.row().classes('items-center gap-2 mb-1'):
                            ui.icon('trending_up').classes(f'{t} text-lg')
                            ui.label('PRODUCTIVE').classes(f'text-sm font-bold {t}')
                        ui.label('Heavy productive volume — you\'re pushing hard. This builds fitness fast but can\'t be sustained indefinitely. Plan a step-back week every 3-4 weeks.').classes('text-sm text-zinc-300')

                    s, t = style_card('OVERREACHING', '#ef4444', 'text-red-400')
                    with ui.element('div').classes('rounded-lg p-3').style(s):
                        with ui.row().classes('items-center gap-2 mb-1'):
                            ui.icon('warning').classes(f'{t} text-lg')
                            ui.label('OVERREACHING').classes(f'text-sm font-bold {t}')
                        ui.label('Too many high-stress sessions. An occasional spike is fine, but repeated overreaching leads to injury and staleness. Follow with an easy week.').classes('text-sm text-zinc-300')

            elif self.state.volume_lens == 'zones':
                ui.label('Heart Rate Zones Analysis').classes('text-xl font-bold text-white mb-2')
                ui.label('Shows weekly time distribution across heart rate zones. The 80/20 rule says ~80% of training should be easy (Zone 1-2) and ~20% hard (Zone 4-5):').classes('text-sm text-zinc-400 mb-4')

                with ui.column().classes('gap-4'):
                    s, t = style_card('80/20 BALANCED', '#10b981', 'text-emerald-400')
                    with ui.element('div').classes('rounded-lg p-3').style(s):
                        with ui.row().classes('items-center gap-2 mb-1'):
                            ui.icon('check_circle').classes(f'{t} text-lg')
                            ui.label('80/20 BALANCED').classes(f'text-sm font-bold {t}')
                        ui.label('The gold standard — ~80% easy, ~20% hard. You\'re building a massive aerobic engine while sharpening speed with controlled intensity. This is how elites train.').classes('text-sm text-zinc-300')

                    s, t = style_card('ZONE 2 BASE', '#3b82f6', 'text-blue-400')
                    with ui.element('div').classes('rounded-lg p-3').style(s):
                        with ui.row().classes('items-center gap-2 mb-1'):
                            ui.icon('favorite').classes(f'{t} text-lg')
                            ui.label('ZONE 2 BASE').classes(f'text-sm font-bold {t}')
                        ui.label('Nearly all Zone 1-2. This is where mitochondrial magic happens — fat oxidation, capillary density, cardiac efficiency. Great for base building, but one hard session per week rounds it out.').classes('text-sm text-zinc-300')

                    s, t = style_card('ZONE 3 JUNK', '#64748b', 'text-slate-400')
                    with ui.element('div').classes('rounded-lg p-3').style(s):
                        with ui.row().classes('items-center gap-2 mb-1'):
                            ui.icon('pause_circle').classes(f'{t} text-lg')
                            ui.label('ZONE 3 JUNK').classes(f'text-sm font-bold {t}')
                        ui.label('Too much time in the grey zone — not easy enough to recover, not hard enough to force adaptation. This is wasted effort. Slow your easy runs and make hard days truly hard.').classes('text-sm text-zinc-300')

                    s, t = style_card('ZONE 4 THRESHOLD ADDICT', '#f97316', 'text-orange-400')
                    with ui.element('div').classes('rounded-lg p-3').style(s):
                        with ui.row().classes('items-center gap-2 mb-1'):
                            ui.icon('fitness_center').classes(f'{t} text-lg')
                            ui.label('ZONE 4 THRESHOLD ADDICT').classes(f'text-sm font-bold {t}')
                        ui.label('Excessive threshold grinding. Zone 4 builds lactate clearance, but more than ~2 sessions/week accumulates fatigue without enough recovery. Add more easy volume between hard days.').classes('text-sm text-zinc-300')

                    s, t = style_card('ZONE 5 REDLINING', '#ef4444', 'text-red-400')
                    with ui.element('div').classes('rounded-lg p-3').style(s):
                        with ui.row().classes('items-center gap-2 mb-1'):
                            ui.icon('local_fire_department').classes(f'{t} text-lg')
                            ui.label('ZONE 5 REDLINING').classes(f'text-sm font-bold {t}')
                        ui.label('Too much VO2max-level effort. Zone 5 is powerful but demands ~48h recovery between sessions. Back off and rebuild your aerobic base — the speed will come back faster.').classes('text-sm text-zinc-300')

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
    
    def show_aerobic_efficiency_info(self, highlight_verdict=None, from_trends=False):
        """
        Show informational modal explaining Aerobic Efficiency, EF, Decoupling, and all 4 verdicts.
        If highlight_verdict is provided, that verdict section gets visually emphasized.
        """
        normalized_highlight = (highlight_verdict or '').strip().upper()
        verdict_aliases = {
            'PEAKING': 'EFFICIENT',
            'BUILDING': 'PUSHING',
            'STABLE': 'BASE',
            'DETRAINING': 'FATIGUED',
            'DRIFTING': 'FATIGUED',
        }
        normalized_highlight = verdict_aliases.get(normalized_highlight, normalized_highlight)

        verdicts = [
            {
                'key': 'Efficient',
                'concept': 'Peak Durability',
                'description': 'You are generating high speed with minimal cardiac stress. Your heart rate stayed stable relative to your pace.',
                'takeaway': 'This is your sustainable race mode. You are converting fuel to motion efficiently with zero wasted energy.',
                'color': 'text-emerald-400',
                'bg': 'bg-emerald-500/10',
                'border': 'border-emerald-500/30',
                'icon': '⚡'
            },
            {
                'key': 'Pushing',
                'concept': 'Building Engine',
                'description': 'High output, but with cardiac drift (>5%). You ran hard, but your heart rate rose faster than your pace later in the run.',
                'takeaway': 'This provides a strong fitness stimulus (Tempo/Threshold), but carries a higher recovery cost. Treat these as "hard days" and do not do them back-to-back.',
                'color': 'text-orange-400',
                'bg': 'bg-orange-500/10',
                'border': 'border-orange-500/30',
                'icon': '🔥'
            },
            {
                'key': 'Base',
                'concept': 'Volume Capacity',
                'description': 'Moderate output with rock-solid stability. Your heart rate did not drift, meaning you stayed purely aerobic.',
                'takeaway': 'This is the safe zone for adding mileage. These runs strengthen your chassis without taxing your recovery. This state should make up the majority (~80%) of your weekly training.',
                'color': 'text-blue-400',
                'bg': 'bg-blue-500/10',
                'border': 'border-blue-500/30',
                'icon': '🧱'
            },
            {
                'key': 'Fatigued',
                'concept': 'Compromised / Warning',
                'description': 'Low output with high cardiac drift. You were running slower than usual, yet your heart rate continued to climb aggressively.',
                'takeaway': 'This is a mechanical warning light. It often signals dehydration, lack of sleep, or accumulated fatigue. Prioritize rest or sleep over intensity tomorrow.',
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

                # Neutral context block (kept intentionally low-contrast)
                with ui.element('div').classes('w-full rounded-lg border border-zinc-800 bg-zinc-950/45 p-4'):
                    with ui.column().classes('gap-3'):
                        with ui.column().classes('gap-1'):
                            ui.label('EFFICIENCY FACTOR (EF)').classes('text-[10px] font-bold text-zinc-500 tracking-[0.18em]')
                            ui.label('Speed ÷ Heart Rate — how much pace you get per heartbeat. Higher is better. Improving EF over weeks means your aerobic engine is growing.').classes('text-sm text-zinc-300 leading-relaxed')

                        ui.separator().classes('bg-zinc-800')

                        with ui.column().classes('gap-1'):
                            ui.label('AEROBIC DECOUPLING').classes('text-[10px] font-bold text-zinc-500 tracking-[0.18em]')
                            ui.label('Cardiac drift — how much your heart rate rises while pace stays steady. Compares 1st half EF to 2nd half EF. Target < 5% for endurance fitness.').classes('text-sm text-zinc-300 leading-relaxed')

                ui.separator().classes('bg-zinc-700')
                
                # Verdict Cards
                ui.label('VERDICTS').classes('text-[10px] font-bold text-zinc-500 tracking-widest')
                
                for v in verdicts:
                    is_highlighted = bool(normalized_highlight) and v['key'].upper() == normalized_highlight
                    if is_highlighted:
                        row_classes = (
                            f'items-start gap-3 p-3 rounded-lg border {v["bg"]} {v["border"]} '
                            'ring-2 ring-white/25 scale-[1.01] shadow-lg shadow-black/20 transition-all'
                        )
                        desc_classes = 'text-xs text-zinc-100 leading-relaxed mt-0.5'
                        takeaway_wrap = 'w-full mt-2 rounded-md border border-white/12 bg-black/20 px-2.5 py-2'
                        takeaway_label = f'text-[10px] font-bold uppercase tracking-wider {v["color"]} mb-1'
                        takeaway_text = 'text-xs text-zinc-100 leading-relaxed'
                    else:
                        row_classes = 'items-start gap-3 p-3 rounded-lg border bg-zinc-900/35 border-zinc-700/80 transition-all'
                        desc_classes = 'text-xs text-zinc-400 leading-relaxed mt-0.5'
                        takeaway_wrap = 'w-full mt-2 rounded-md border border-zinc-700/70 bg-zinc-900/60 px-2.5 py-2'
                        takeaway_label = 'text-[10px] font-bold uppercase tracking-wider text-zinc-400 mb-1'
                        takeaway_text = 'text-xs text-zinc-300 leading-relaxed'

                    with ui.row().classes(row_classes):
                        ui.label(v['icon']).classes('text-lg mt-0.5')
                        with ui.column().classes('gap-0.5 flex-1'):
                            with ui.row().classes('items-center gap-2 flex-wrap'):
                                ui.label(v['key']).classes(f'text-sm font-bold {v["color"]}')
                            ui.label(v['description']).classes(desc_classes)
                            with ui.element('div').classes(takeaway_wrap):
                                ui.label('Takeaway').classes(takeaway_label)
                                ui.label(v['takeaway']).classes(takeaway_text)
                
                # Tip
                if from_trends:
                    with ui.row().classes('items-start gap-2 mt-4 p-3 bg-blue-500/10 rounded-lg border border-blue-500/20'):
                        ui.icon('lightbulb').classes('text-blue-400 text-sm mt-0.5')
                        ui.label('The trend graph plots each run as a dot using these two axes. Click any dot to see that activity\'s details.').classes('text-sm italic text-blue-200 flex-1')
                
                ui.button('GOT IT', on_click=dialog.close).classes('mt-6 w-full bg-zinc-800 hover:bg-zinc-700 text-white font-bold')
        
        dialog.open()
    
    def show_form_info(self, highlight_verdict=None):
        """
        Show informational modal explaining Running Form verdicts.
        If highlight_verdict is provided, that verdict section gets visually emphasized.
        """
        normalized_highlight = (highlight_verdict or '').strip().upper()
        verdict_aliases = {
            'ELITE': 'ELITE FORM',
            'GOOD': 'GOOD FORM',
            'IMPROVING': 'GOOD FORM',
            'SLIPPING': 'HEAVY FEET',
            'BROKEN': 'HEAVY FEET',
            'LOW CADENCE': 'HEAVY FEET',
            'OVERSTRIDING': 'HEAVY FEET',
            'INEFFICIENT': 'HEAVY FEET',
            'AEROBIC / MIXED': 'HIKING / REST',
        }
        normalized_highlight = verdict_aliases.get(normalized_highlight, normalized_highlight)

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
                    is_highlighted = bool(normalized_highlight) and (
                        v['raw'] == normalized_highlight or v['key'].upper() == normalized_highlight
                    )
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
            with ui.column().classes('gap-3 mb-4'):
                with ui.row().classes('items-start no-wrap gap-2'):
                    ui.label('🔵').classes('text-lg leading-none mt-0.5')
                    ui.html('<b>Recovery (<75):</b> Easy effort that promotes adaptation and recovery.').classes('text-sm text-blue-400 leading-snug')
                
                with ui.row().classes('items-start no-wrap gap-2'):
                    ui.label('🟢').classes('text-lg leading-none mt-0.5')
                    ui.html('<b>Base (75-150):</b> Builds and strengthens your aerobic foundation and endurance capacity. This is the goal state for 80% of your volume. It is highly constructive.').classes('text-sm text-green-400 leading-snug')
                
                with ui.row().classes('items-start no-wrap gap-2'):
                    ui.label('🟠').classes('text-lg leading-none mt-0.5')
                    ui.html('<b>Overload (150-300):</b> Hard work that builds fitness and improves performance.').classes('text-sm text-orange-400 leading-snug')
                
                with ui.row().classes('items-start no-wrap gap-2'):
                    ui.label('🔴').classes('text-lg leading-none mt-0.5')
                    ui.html('<b>Overreaching (300+):</b> Very high stress that requires adequate recovery time.').classes('text-sm text-red-400 leading-snug')
            
            ui.markdown('''
**How to Use It:**  
Track your weekly load to balance hard training with recovery. Consistent overload builds fitness, while too many overreaching sessions can lead to burnout.

**Training Tip:**  
Most of your runs should be Recovery or Base, with Overload efforts 1-2x per week, and Overreaching reserved for key workouts or races.
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
        run_type = self.data_manager.classify_run_type(d, long_run_threshold)
        
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
        Update activities table.
        FIXED:
        1. IDs are now DETERMINISTIC (Fixes Ghost Selection/FAB issues).
        2. Preserves 'virtual-scroll' and Dark UI.
        3. Uses standard $emit for buttons.
        """
        self.grid_container.clear()
        
        # --- 1. FILTER LOGIC ---
        if self.df is not None and len(self.df) >= 5:
            lrt = self.df['distance_mi'].quantile(0.8)
        else:
            lrt = 10.0

        filtered_data = self.data_manager.get_filtered_data()

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

            ts = self.data_manager.classify_run_type(act, lrt)
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
                    self.layout.show_floating_action_bar(table.selected)

                async def handle_row_action(e):
                    payload = e.args if isinstance(e.args, dict) else {}
                    action = payload.get('action')
                    row = payload.get('row') if isinstance(payload.get('row'), dict) else None

                    if not row:
                        return

                    if action == 'analyze':
                        activity_hash = row.get('hash')
                        if activity_hash:
                            _table_nav = [r.get('hash') for r in rows if r.get('hash')]
                            await self.open_activity_detail_modal(activity_hash, from_feed=True, navigation_list=_table_nav)
                    elif action == 'download':
                        await self.download_fit_file(row)
                    elif action == 'delete':
                        activity_hash = row.get('hash')
                        filename = row.get('full_filename')
                        file_path = row.get('file_path')
                        await self.delete_activity_inline(activity_hash, filename, file_path)
                
                table = ui.table(
                    columns=columns, 
                    rows=rows, 
                    row_key='id',       # Matches our STABLE ID
                    selection='multiple',
                    pagination={'rowsPerPage': 0, 'sortBy': self.state.sort_by, 'descending': self.state.sort_desc},
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
            self.state.sort_by = new_sort_by
            self.state.sort_desc = new_descending
            
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
    
    async def delete_activity_inline(self, activity_hash, filename, file_path=None):
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
                # 1. Delete from activity + library index and attempt trash.
                moved, trash_error, resolved_path = self._delete_activity_with_library_cleanup(
                    activity_hash,
                    file_path=file_path,
                )

                # 2. Show notification NOW (while the button/table still exists)
                if moved:
                    ui.notify(f'Deleted: {filename[:30]} (moved to Trash)', type='positive')
                elif resolved_path and trash_error:
                    ui.notify(
                        f'Deleted activity, but failed to move file to Trash: {filename[:30]}',
                        type='warning',
                    )
                else:
                    ui.notify(f'Deleted: {filename[:30]}', type='positive')
                
                # 3. Update runs counter
                self.library_modal.update_run_count()
                
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
        file_path = selected[0].get('file_path')
        
        # Show confirmation dialog
        result = await ui.run_javascript(
            f'confirm("Delete activity: {activity_name}?")', 
            timeout=10
        )
        
        # If confirmed, delete the activity
        if result:
            # Delete from activity + library index and attempt trash.
            moved, trash_error, resolved_path = self._delete_activity_with_library_cleanup(
                activity_hash,
                file_path=file_path,
            )
            
            # Update runs counter
            self.library_modal.update_run_count()
            
            # Call refresh_data_view() to update all views
            await self.refresh_data_view()
            
            # Show success notification
            if moved:
                ui.notify('Activity deleted and file moved to Trash', type='positive')
            elif resolved_path and trash_error:
                ui.notify('Activity deleted, but file could not be moved to Trash', type='warning')
            else:
                ui.notify('Activity deleted successfully', type='positive')
    
    
    def apply_export_chart_header(self, fig, title, subtitle, verdict=None, badge_color=None, margin=None):
        """Apply static export header with optional inline-styled verdict badge."""
        if margin is None:
            margin = dict(t=185, l=60, r=20, b=60)

        verdict_text = str(verdict).strip() if verdict is not None else ''
        
        if verdict_text and verdict_text.upper() != 'N/A':
            formatted_title = f"{title}  [ {verdict_text.upper()} ]"
        else:
            formatted_title = title

        fig.update_layout(
            title=dict(
                text=f'<b>{formatted_title}</b><br><span style="font-size:14px; color:#a1a1aa;">{subtitle}</span>',
                font=dict(size=24, color='white'),
                x=0.02,
                xanchor='left',
                y=0.96,
                yanchor='top'
            ),
            margin=margin
        )

        existing_annotations = list(fig.layout.annotations) if fig.layout.annotations else []
        annotations = [a for a in existing_annotations if getattr(a, 'name', None) != 'export_verdict_badge']
        if len(annotations) != len(existing_annotations):
            fig.update_layout(annotations=annotations)
    
    
    
    
    
    
    
    
    
    
    
    
    async def export_csv(self):
        """Export current data snapshot via DataManager and show UI notifications."""
        if self.df is None or self.df.empty:
            ui.notify('No data to export', type='warning')
            return

        try:
            file_path = self.data_manager.export_csv()
            ui.notify('CSV saved! (check your Downloads folder)', type='positive', timeout=5000)
            print(f"CSV exported to: {file_path}")
        except ValueError:
            ui.notify('No data to export', type='warning')
        except Exception as e:
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
            
            # Generate the three charts with titles (use active Training Volume lens + verdict context)
            volume_fig, volume_verdict, volume_verdict_color, _, volume_subtitle = self.analysis_view.get_active_volume_lens_state()
            efficiency_fig = self.analysis_view.generate_efficiency_decoupling_chart()
            cadence_fig = self.analysis_view.generate_cadence_trend_chart()
            
            if not volume_fig or not efficiency_fig or not cadence_fig:
                loading_dialog.close()
                ui.notify('Error generating charts', type='warning')
                return
            
            # Adjust efficiency chart colors for better static export appearance
            # Make the red decoupling fill less bright/more subtle
            efficiency_fig.data[1].fillcolor = 'rgba(180, 50, 50, 0.15)'  # Darker, more muted red
            efficiency_fig.data[1].line.color = 'rgba(180, 50, 50, 0.4)'
            
            # Resolve verdict badges for all three charts in export header
            efficiency_verdict, efficiency_verdict_color, _ = self.analysis_view.calculate_efficiency_verdict(self.df)
            cadence_verdict, cadence_verdict_color, _ = self.analysis_view.calculate_cadence_verdict(self.df)

            # Add titles + inline-styled badges to each chart for export
            self.apply_export_chart_header(
                fig=volume_fig,
                title=f'Training Volume | {self.analysis_view.get_volume_lens_label()}',
                subtitle=volume_subtitle,
                verdict=volume_verdict,
                badge_color=volume_verdict_color,
                margin=dict(t=185, l=60, r=20, b=60),
            )
            
            self.apply_export_chart_header(
                fig=efficiency_fig,
                title='Aerobic Efficiency',
                subtitle='Running efficiency vs. cardiovascular drift over time',
                verdict=efficiency_verdict,
                badge_color=efficiency_verdict_color,
                margin=dict(t=185, l=60, r=60, b=60),
            )
            
            self.apply_export_chart_header(
                fig=cadence_fig,
                title='Running Mechanics',
                subtitle='Cadence trend showing turnover consistency',
                verdict=cadence_verdict,
                badge_color=cadence_verdict_color,
                margin=dict(t=185, l=60, r=20, b=60),
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
            style_config = {
                # Quality Lens
                'HIGH QUALITY': ('High Quality Miles', '#10b981', 'bg-emerald-500/20'),
                'STRUCTURAL':   ('Structural Miles',   '#3b82f6', 'bg-blue-500/20'),
                'BROKEN':       ('Broken Miles',       '#ef4444', 'bg-red-500/20'),
                
                # Training Mix Lens
                'Recovery': ('Recovery Miles', '#60a5fa', 'bg-blue-500/20'),
                'Base':     ('Base Miles',     '#a78bfa', 'bg-violet-500/20'),
                'Steady':   ('Steady Miles',   '#f59e0b', 'bg-amber-500/20'),
                'Long Run': ('Long Run Miles', '#34d399', 'bg-emerald-500/20'),
                'Tempo':    ('Tempo Miles',    '#f43f5e', 'bg-rose-500/20'),
                
                # Load Lens
                'Base':         ('Base',         LOAD_CATEGORY_COLORS['Base'],         'bg-emerald-500/20'),
                'Overload':     ('Overload',     LOAD_CATEGORY_COLORS['Overload'],     'bg-orange-500/20'),
                'Overreaching': ('Overreaching', LOAD_CATEGORY_COLORS['Overreaching'], 'bg-red-500/20'),
                # 'Recovery' is shared with Mix but consistent color from LOAD_CATEGORY_COLORS
                
                # HR Zones Lens
                'Zone 1': ('Zone 1 (Easy)',      '#60a5fa', 'bg-blue-500/20'),
                'Zone 2': ('Zone 2 (Aerobic)',   '#34d399', 'bg-emerald-500/20'),
                'Zone 3': ('The Grey Zone',      '#64748b', 'bg-slate-500/20'),
                'Zone 4': ('Zone 4 (Hard)',      '#f97316', 'bg-orange-500/20'),
                'Zone 5': ('Zone 5 (Max)',       '#ef4444', 'bg-red-500/20'),
            }
            
            # Map category to style (Fallback to Zinc)
            label_text, text_hex, bg_class = style_config.get(
                category_raw, 
                (category_raw.replace('_', ' ').title(), '#a1a1aa', 'bg-zinc-800')
            )

            async def pick_run(h, dlg):
                dlg.close()
                await asyncio.sleep(0.1)
                await self.open_activity_detail_modal(h, from_feed=True) # <--- Force Nice Modal

            with ui.dialog() as selector_dialog, ui.card().classes('bg-zinc-900 border border-zinc-800 p-0 min-w-[340px] shadow-2xl shadow-black'):
                
                # Header
                with ui.row().classes('w-full items-center justify-between p-4 border-b border-zinc-800 bg-zinc-900/50'):
                    with ui.column().classes('gap-2'):
                        ui.label('Inspect Runs').classes('text-xs font-bold text-zinc-500 uppercase tracking-wider')
                        ui.label(label_text).classes(f'text-sm font-bold px-3 py-1 rounded {bg_class}').style(f'color: {text_hex};')
                
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
        """Backward-compatible sidebar handler."""
        await self.llm_exporter.generate_export(self.data_manager.activities_data)


def main():
    """Application entry point."""
    # Suppress known NiceGUI framework listener-churn warning noise
    nicegui_logger = logging.getLogger('nicegui')
    nicegui_logger.addFilter(MuteFrameworkNoise())

    # Instantiate the application
    UltraStateApp()
    
    # Run in native mode with specified window configuration
    try:
        ui.run(
            native=True,
            window_size=(1200, 900),
            title="Ultra State",
            reload=False,
            dark=True  # Force dark mode for native window
        )
    except KeyboardInterrupt:
        # Graceful terminal interrupt during local development.
        pass
    except RuntimeError as e:
        msg = str(e)
        if 'Cannot close a running event loop' in msg or 'this event loop is already running' in msg:
            # uvloop teardown can surface this after Ctrl+C; treat as graceful exit.
            pass
        else:
            raise


if __name__ == "__main__":
    main()
