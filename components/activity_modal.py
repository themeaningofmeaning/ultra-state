"""
components/activity_modal.py
────────────────────────────
ActivityModal — Phase 2 Step 3 extraction of the activity detail modal.

Owns:
  • _fetch_detail()   — FIT file parsing + LRU cache (was UltraStateApp.get_activity_detail)
  • open()            — async modal open with premium skeleton loader (was open_activity_detail_modal)

Does NOT own (injected via `callbacks` dict):
  • All compute helpers (hr_zones, gap_for_laps, decoupling, run_walk, terrain)
  • All render helpers (create_physiology_card, create_hr_zone_chart, etc.)
  • All navigation / action handlers (delete, recalculate, copy_splits)
  • The recursive open_modal_cb (for ◀/▶ navigation)
  • map_payload_cb, locate_fit_cb, normalize_bounds_cb (UltraStateApp private utils)

See implementation_plan.md for the full dependency boundary map.
"""
from __future__ import annotations

import asyncio
import copy
import json
import os
from datetime import timezone

from nicegui import run, ui

from constants import (
    SPLIT_BUCKET,
)
from analyzer import (
    MAP_PAYLOAD_VERSION,
    build_map_payload_from_streams,
    classify_split,
)

# HR Zone map legend gradient constant (duplicated from app.py global scope)
HR_ZONE_MAP_LEGEND_GRADIENT = (
    "linear-gradient(to right, #3b82f6, #10b981, #eab308, #f97316, #ef4444)"
)


class ActivityModal:
    """
    Self-contained activity detail modal component.

    Parameters
    ----------
    db : DatabaseManager
        Injected database handle — used by _fetch_detail for direct DB reads.
    state : AppState
        Injected application state (read-only for now; reserved for future
        reactive navigation wiring).
    callbacks : dict
        Dictionary of callable references passed down from UltraStateApp.
        Required keys:
          locate_fit_cb           — (activity) -> str|None
          map_payload_cb          — (activity) -> dict|None
          normalize_bounds_cb     — (bounds) -> nested list|None
          calc_distance_cb        — (speed_stream, timestamps) -> list
          hr_zones_cb             — (hr_stream, max_hr) -> dict
          gap_laps_cb             — (lap_data, elev, ts, cad, max_hr) -> list
          decoupling_cb           — (hr_stream, speed_stream) -> dict
          run_walk_cb             — (cad, speed, hr) -> dict|None
          terrain_cb              — (elev, hr, speed, ts) -> dict
          physiology_card_cb      — (session_data, activity) -> None
          hr_zone_chart_cb        — (hr_zones) -> figure
          lap_splits_cb           — (enhanced_laps) -> None
          decoupling_card_cb      — (decoupling, efficiency_factor) -> None
          dynamics_card_cb        — (session_data) -> None
          strategy_row_cb         — (run_walk_stats, terrain_stats) -> None
          terrain_graph_cb        — (detail_data, metric) -> figure
          copy_splits_cb          — (enhanced_laps) -> None
          open_modal_cb           — async (hash, from_feed, navigation_list) -> None
    """

    # ── LRU cache constants (previously on UltraStateApp) ─────────────────
    _CACHE_SIZE    = 24
    _CACHE_VERSION = 3    # bump to invalidate all cached payloads

    def __init__(self, db, state, callbacks: dict) -> None:
        self._db        = db
        self._state     = state
        self._cb        = callbacks

        # LRU-ish detail payload cache: {activity_hash: {file_path, file_mtime, cache_version, payload}}
        self._cache: dict = {}

    # ══════════════════════════════════════════════════════════════════════
    # Public API
    # ══════════════════════════════════════════════════════════════════════

    async def open(
        self,
        activity_hash: str,
        from_feed: bool = False,
        navigation_list: list | None = None,
    ) -> None:
        """
        Open the activity detail modal.

        Phase 1 (immediate, < 1 frame):
          • Builds a premium skeleton dialog with shimmer cards shaped like
            the real content — nav bar, map placeholder, metrics row, charts.

        Phase 2 (async, 0.3–1.5 s):
          • Runs _fetch_detail() in a background thread via run.io_bound.
          • On completion, hydrates each skeleton container with real content.
        """
        # ── Navigation metadata (no DB needed) ───────────────────────────
        _nav_guard = {'active': True, 'keyboard': None}
        _nav_idx   = -1
        _has_nav   = bool(navigation_list and len(navigation_list) > 1)
        if _has_nav:
            try:
                _nav_idx = navigation_list.index(activity_hash)
            except ValueError:
                _has_nav = False

        # ── Navigation callbacks (close + reopen) ─────────────────────────
        async def _nav_prev():
            if not _nav_guard['active'] or _nav_idx <= 0:
                return
            _nav_guard['active'] = False
            detail_dialog.close()
            await self._cb['open_modal_cb'](
                navigation_list[_nav_idx - 1],
                from_feed=from_feed,
                navigation_list=navigation_list,
            )

        async def _nav_next():
            if not _nav_guard['active'] or not _has_nav or _nav_idx >= len(navigation_list) - 1:
                return
            _nav_guard['active'] = False
            detail_dialog.close()
            await self._cb['open_modal_cb'](
                navigation_list[_nav_idx + 1],
                from_feed=from_feed,
                navigation_list=navigation_list,
            )

        def _on_dialog_close():
            _nav_guard['active'] = False
            kb = _nav_guard.get('keyboard')
            if getattr(kb, 'delete', None):
                try: kb.delete()
                except Exception: pass
            ui.run_javascript(
                'if(window._ultraNavKD){'
                'document.removeEventListener("keydown",window._ultraNavKD);'
                'window._ultraNavKD=null;}'
            )

        with ui.dialog() as detail_dialog:
            detail_dialog.props('transition-show=none transition-hide=none')
            detail_dialog.on('close', _on_dialog_close)

            with ui.card().classes('w-full max-w-[900px] p-0 bg-zinc-950 h-full border border-zinc-800 relative'):

                # ── Header nav bar (renders immediately — no data needed) ──
                with ui.row().classes('w-full justify-between items-center p-2 px-3'):
                    if _has_nav:
                        with ui.row().classes('items-center gap-1'):
                            prev_btn = ui.button(
                                icon='chevron_left', color=None, on_click=_nav_prev
                            ).props('flat round dense')
                            prev_btn.style('color: #9ca3af !important;')
                            if _nav_idx <= 0:
                                prev_btn.props('disable')
                                prev_btn.style('opacity: 0.3; color: #9ca3af !important;')
                            ui.label(f'{_nav_idx + 1} of {len(navigation_list)}').classes(
                                'text-xs text-zinc-500 font-mono tabular-nums tracking-wider'
                            )
                            next_btn = ui.button(
                                icon='chevron_right', color=None, on_click=_nav_next
                            ).props('flat round dense')
                            next_btn.style('color: #9ca3af !important;')
                            if _nav_idx >= len(navigation_list) - 1:
                                next_btn.props('disable')
                                next_btn.style('opacity: 0.3; color: #9ca3af !important;')
                    else:
                        ui.element('div')  # Spacer when no navigation
                    close_btn = ui.button(
                        icon='close', on_click=detail_dialog.close, color=None
                    ).props('flat round dense')
                    close_btn.style('color: #9ca3af !important;')

                # ── Content container (skeleton placeholders) ─────────────
                with ui.column().classes('w-full gap-4 px-4 pb-4'):

                    # Skeleton shimmer CSS (injected once — idempotent via browser cache)
                    ui.add_head_html('''
                    <style>
                    @keyframes _ultra_shimmer {
                      0%   { background-position: -600px 0; }
                      100% { background-position:  600px 0; }
                    }
                    .ultra-skeleton {
                      background: linear-gradient(
                        90deg,
                        rgba(39,39,42,0.8) 25%,
                        rgba(63,63,70,0.6) 50%,
                        rgba(39,39,42,0.8) 75%
                      );
                      background-size: 600px 100%;
                      animation: _ultra_shimmer 1.6s infinite linear;
                      border-radius: 8px;
                    }
                    </style>
                    ''')

                    # ── Skeleton shimmer elements ─────────────────────────
                    # All wrapped in a single column so one .delete() removes
                    # every shimmer card atomically when real content is ready.
                    with ui.column().classes('w-full gap-4') as skeleton_container:

                        # Map skeleton — matches real map h-80 (320 px)
                        ui.element('div').classes(
                            'ultra-skeleton w-full'
                        ).style('height: 320px; border-radius: 8px;')

                        # Header skeleton — title + metrics row
                        with ui.column().classes('w-full px-4 gap-2'):
                            ui.element('div').classes('ultra-skeleton').style('height: 28px; width: 55%; border-radius: 6px;')
                            ui.element('div').classes('ultra-skeleton').style('height: 16px; width: 40%; border-radius: 6px;')

                        ui.separator().classes('bg-zinc-800 my-2')

                        # Strategy row skeleton — 2 metric pills
                        with ui.row().classes('w-full gap-3'):
                            for _ in range(2):
                                ui.element('div').classes('ultra-skeleton flex-1').style('height: 64px; border-radius: 8px;')

                        # HR zones card skeleton
                        ui.element('div').classes('ultra-skeleton w-full').style('height: 180px; border-radius: 8px;')

                        # Body response row skeleton — 3 cards
                        with ui.row().classes('w-full gap-3'):
                            for _ in range(3):
                                ui.element('div').classes('ultra-skeleton flex-1').style('height: 120px; border-radius: 8px;')

                        # Terrain + splits skeleton
                        for h in ('180px', '160px'):
                            ui.element('div').classes('ultra-skeleton w-full').style(
                                f'height: {h}; border-radius: 8px;'
                            )

                    # Container that real content will be injected into once loaded.
                    # Lives OUTSIDE skeleton_container so skeleton.delete() doesn't remove it.
                    content_container = ui.column().classes('w-full gap-4')
                    content_container.set_visibility(False)

        detail_dialog.open()

        # ── Keyboard navigation (set up before async fetch) ────────────────
        if _has_nav:
            async def _handle_key(e):
                if not _nav_guard['active'] or getattr(e.action, 'keyup', False) or getattr(e.action, 'repeat', False):
                    return
                if e.key == 'ArrowLeft':
                    await _nav_prev()
                elif e.key == 'ArrowRight':
                    await _nav_next()

            _nav_guard['keyboard'] = ui.keyboard(on_key=_handle_key, active=True)
            ui.run_javascript(
                'if(window._ultraNavKD) document.removeEventListener("keydown",window._ultraNavKD);'
                'window._ultraNavKD = function(e) {'
                '  if (e.key === "ArrowLeft" || e.key === "ArrowRight") { e.preventDefault(); }'
                '};'
                'document.addEventListener("keydown", window._ultraNavKD);'
            )

        # ── Async data fetch + hydration (non-blocking) ───────────────────
        asyncio.create_task(
            self._load_and_hydrate(
                detail_dialog,
                skeleton_container,
                content_container,
                activity_hash,
                _has_nav,
                _nav_idx,
                navigation_list,
                from_feed,
                _nav_guard,
            )
        )

    # ══════════════════════════════════════════════════════════════════════
    # Private: data fetch
    # ══════════════════════════════════════════════════════════════════════

    async def _fetch_detail(self, activity_hash: str) -> dict | None:
        """
        Fetch and parse an activity's FIT file.

        Mirrors the previous UltraStateApp.get_activity_detail() exactly,
        with self.db → self._db and injected callbacks replacing private methods.

        Returns a detail_data dict or None / {error: ...} on failure.
        """
        # 1. DB metadata
        activity = self._db.get_activity_by_hash(activity_hash)
        if not activity:
            return None

        # Lightweight map payload prep (avoids stale legacy bounds in modal rendering)
        try:
            self._cb['map_payload_cb'](activity)
        except Exception as ex:
            print(f"Map payload prep warning for {activity_hash}: {ex}")

        # 2. Locate FIT file
        fit_file_path = self._cb['locate_fit_cb'](activity)
        if not fit_file_path:
            return {'error': 'file_not_found', 'activity': activity}

        file_mtime = os.path.getmtime(fit_file_path) if os.path.exists(fit_file_path) else None

        # 2.5 Fast-path: return cached payload when source file unchanged
        cached = self._cache.get(activity_hash)
        if (
            cached
            and cached.get('file_path') == fit_file_path
            and cached.get('file_mtime') == file_mtime
            and cached.get('cache_version') == self._CACHE_VERSION
        ):
            payload = copy.deepcopy(cached.get('payload', {}))
            if payload:
                payload['activity_metadata'] = activity
                return payload

        # 3. Parse FIT file in background thread (avoids blocking event loop)
        try:
            def _parse_fit():
                import fitparse
                fitfile = fitparse.FitFile(fit_file_path)

                hr_stream        = []
                elevation_stream = []
                speed_stream     = []
                cadence_stream   = []
                distance_stream  = []
                timestamps       = []

                vertical_oscillation = []
                stance_time          = []
                vertical_ratio       = []
                step_length          = []

                route_lats       = []
                route_lons       = []
                route_speeds     = []
                route_hrs        = []
                route_timestamps = []

                for record in fitfile.get_messages("record"):
                    vals = record.get_values()

                    ts = vals.get('timestamp')
                    if ts:
                        ts = ts.replace(tzinfo=timezone.utc).astimezone()
                    timestamps.append(ts)

                    hr_stream.append(vals.get('heart_rate'))
                    elevation_stream.append(
                        vals.get('enhanced_altitude') or vals.get('altitude')
                    )
                    speed_stream.append(
                        vals.get('enhanced_speed') or vals.get('speed')
                    )
                    cadence_stream.append(vals.get('cadence'))
                    distance_stream.append(vals.get('distance'))

                    vertical_oscillation.append(vals.get('vertical_oscillation'))
                    stance_time.append(
                        vals.get('stance_time') or vals.get('ground_contact_time')
                    )
                    vertical_ratio.append(vals.get('vertical_ratio'))
                    step_length.append(
                        vals.get('step_length') or vals.get('stride_length')
                    )

                    lat_raw = vals.get('position_lat')
                    lon_raw = vals.get('position_long')
                    if lat_raw is not None and lon_raw is not None:
                        try:
                            lat_raw = float(lat_raw)
                            lon_raw = float(lon_raw)
                            if abs(lat_raw) > 180 or abs(lon_raw) > 180:
                                point_lat = lat_raw * (180 / 2**31)
                                point_lon = lon_raw * (180 / 2**31)
                            else:
                                point_lat = lat_raw
                                point_lon = lon_raw

                            if (
                                -90 <= point_lat <= 90
                                and -180 <= point_lon <= 180
                                and not (abs(point_lat) < 1e-6 and abs(point_lon) < 1e-6)
                            ):
                                route_lats.append(point_lat)
                                route_lons.append(point_lon)
                                route_speeds.append(
                                    vals.get('enhanced_speed') or vals.get('speed') or 0
                                )
                                route_hrs.append(vals.get('heart_rate'))
                                route_timestamps.append(ts)
                        except (TypeError, ValueError):
                            pass

                # Lap data
                lap_data = []
                for lap_msg in fitfile.get_messages("lap"):
                    vals = lap_msg.get_values()
                    if not vals:
                        continue
                    if vals.get('start_time'):
                        vals['start_time'] = vals['start_time'].replace(
                            tzinfo=timezone.utc
                        ).astimezone()
                    if vals.get('timestamp'):
                        vals['timestamp'] = vals['timestamp'].replace(
                            tzinfo=timezone.utc
                        ).astimezone()

                    total_distance   = vals.get('total_distance')
                    total_timer_time = vals.get('total_timer_time')
                    if total_distance and total_timer_time and total_timer_time > 0:
                        avg_speed = total_distance / total_timer_time
                    else:
                        avg_speed = vals.get('enhanced_avg_speed') or vals.get('avg_speed')

                    lap_data.append({
                        'lap_number':        len(lap_data) + 1,
                        'distance':          total_distance,
                        'avg_speed':         avg_speed,
                        'avg_hr':            vals.get('avg_heart_rate'),
                        'avg_cadence':       vals.get('avg_cadence'),
                        'total_ascent':      vals.get('total_ascent'),
                        'total_descent':     vals.get('total_descent'),
                        'start_time':        vals.get('start_time'),
                        'timestamp':         vals.get('timestamp'),
                        'total_elapsed_time': vals.get('total_elapsed_time'),
                    })

                # Session-level metrics
                session_data = {}
                for session_msg in fitfile.get_messages("session"):
                    vals = session_msg.get_values()
                    session_data = {
                        'total_calories':                vals.get('total_calories'),
                        'avg_temperature':               vals.get('avg_temperature'),
                        'total_training_effect':         vals.get('total_training_effect'),
                        'total_anaerobic_training_effect': vals.get('total_anaerobic_training_effect'),
                        'avg_respiration_rate':          vals.get('avg_respiration_rate'),
                        'avg_vertical_oscillation':      vals.get('avg_vertical_oscillation'),
                        'avg_stance_time':               vals.get('avg_stance_time'),
                        'avg_step_length':               vals.get('avg_step_length'),
                    }
                    break  # Only first session

                # Fallback: calculate distance from speed if unavailable
                if all(d is None for d in distance_stream):
                    distance_stream = self._cb['calc_distance_cb'](speed_stream, timestamps)

                return {
                    'hr_stream':          hr_stream,
                    'elevation_stream':   elevation_stream,
                    'speed_stream':       speed_stream,
                    'cadence_stream':     cadence_stream,
                    'distance_stream':    distance_stream,
                    'lap_data':           lap_data,
                    'timestamps':         timestamps,
                    'session_data':       session_data,
                    'vertical_oscillation': vertical_oscillation,
                    'stance_time':        stance_time,
                    'vertical_ratio':     vertical_ratio,
                    'step_length':        step_length,
                    'route_lats':         route_lats,
                    'route_lons':         route_lons,
                    'route_speeds':       route_speeds,
                    'route_hrs':          route_hrs,
                    'route_timestamps':   route_timestamps,
                }

            result = await run.io_bound(_parse_fit)

            if result is None or not isinstance(result, dict):
                return {'error': 'parse_error', 'activity': activity, 'message': 'Failed to parse FIT file'}

            # Max HR (priority: user profile > session max > observed max)
            max_hr = activity.get('max_hr')
            if not max_hr and result.get('hr_stream'):
                valid_hrs = [hr for hr in result['hr_stream'] if hr is not None]
                max_hr = max(valid_hrs) if valid_hrs else 185
                result['max_hr_fallback'] = True
            else:
                max_hr = max_hr or 185
                result['max_hr_fallback'] = False
            result['max_hr'] = max_hr

            # Map payload version check / refresh
            current_payload = activity.get('map_payload')
            current_version = int(
                activity.get('map_payload_version')
                or (current_payload.get('v') if isinstance(current_payload, dict) else 1)
                or 1
            )
            current_bounds   = self._cb['normalize_bounds_cb'](
                current_payload.get('bounds') if isinstance(current_payload, dict) else None
            )
            current_segments = current_payload.get('segments') if isinstance(current_payload, dict) else None
            needs_refresh = (
                current_version < MAP_PAYLOAD_VERSION
                or not current_bounds
                or not isinstance(current_segments, list)
                or not current_segments
            )

            if needs_refresh and len(result.get('route_lats', [])) >= 2:
                refreshed_payload = build_map_payload_from_streams(
                    result.get('route_lats'),
                    result.get('route_lons'),
                    result.get('route_speeds'),
                    hr_stream=result.get('route_hrs'),
                    max_hr=max_hr,
                    timestamps=result.get('route_timestamps'),
                    target_segments=700,
                )
                if isinstance(refreshed_payload, dict) and refreshed_payload.get('segments'):
                    activity['map_payload']         = refreshed_payload
                    activity['map_payload_version'] = refreshed_payload.get('v', MAP_PAYLOAD_VERSION)
                    activity['route_segments']      = refreshed_payload.get('segments', [])
                    activity['bounds']              = refreshed_payload.get('bounds', [[0, 0], [0, 0]])
                    try:
                        self._db.update_activity_map_payload(
                            activity_hash,
                            refreshed_payload,
                            route_segments=refreshed_payload.get('segments'),
                            bounds=refreshed_payload.get('bounds'),
                            map_payload_version=refreshed_payload.get('v', MAP_PAYLOAD_VERSION),
                        )
                    except Exception as ex:
                        print(f"Map payload refresh warning for {activity_hash}: {ex}")

            # Strip transient route streams before caching
            for key in ('route_lats', 'route_lons', 'route_speeds', 'route_hrs', 'route_timestamps'):
                result.pop(key, None)

            result['activity_metadata'] = activity

            # Cache the result (LRU eviction)
            self._cache[activity_hash] = {
                'file_path':     fit_file_path,
                'file_mtime':    file_mtime,
                'cache_version': self._CACHE_VERSION,
                'payload':       copy.deepcopy(result),
            }
            if len(self._cache) > self._CACHE_SIZE:
                oldest = next(iter(self._cache))
                if oldest != activity_hash:
                    self._cache.pop(oldest, None)

            return result

        except Exception as e:
            import traceback
            print(f"Error parsing FIT file: {e}")
            traceback.print_exc()
            return {'error': 'parse_error', 'activity': activity, 'message': str(e)}

    # ══════════════════════════════════════════════════════════════════════
    # Private: async load + hydrate
    # ══════════════════════════════════════════════════════════════════════

    async def _load_and_hydrate(
        self,
        detail_dialog,
        skeleton_container,
        content_container,
        activity_hash: str,
        _has_nav: bool,
        _nav_idx: int,
        navigation_list,
        from_feed: bool,
        _nav_guard: dict,
    ) -> None:
        """
        Runs in the background after the skeleton dialog is open.
        Fetches detail data, then replaces skeleton with real content.
        """
        detail_data = await self._fetch_detail(activity_hash)

        if detail_data is None or detail_data.get('error'):
            # Dialog may already be closed by user navigating away
            if not _nav_guard['active']:
                return
            skeleton_container.delete()
            with content_container:
                ui.label('⚠️ Error loading activity').classes('text-red-400 p-4')
            content_container.set_visibility(True)
            ui.notify('Error loading activity', type='negative')
            return

        # ── Compute derived metrics ───────────────────────────────────────
        hr_zones      = self._cb['hr_zones_cb'](detail_data['hr_stream'], detail_data['max_hr'])
        enhanced_laps = self._cb['gap_laps_cb'](
            detail_data['lap_data'],
            detail_data['elevation_stream'],
            detail_data['timestamps'],
            detail_data.get('cadence_stream'),
            max_hr=detail_data.get('max_hr', 185),
        )
        decoupling   = self._cb['decoupling_cb'](detail_data['hr_stream'], detail_data['speed_stream'])
        run_walk_stats = None
        if detail_data.get('cadence_stream'):
            run_walk_stats = self._cb['run_walk_cb'](
                detail_data['cadence_stream'],
                detail_data['speed_stream'],
                detail_data['hr_stream'],
            )
        terrain_stats = self._cb['terrain_cb'](
            detail_data['elevation_stream'],
            detail_data['hr_stream'],
            detail_data['speed_stream'],
            detail_data['timestamps'],
        )

        activity = detail_data['activity_metadata']

        # ── Quality badge ─────────────────────────────────────────────────
        v_label = v_color = v_bg = None
        try:
            avg_cad  = activity.get('avg_cadence', 0)
            avg_hr   = activity.get('avg_hr', 0)
            max_hr   = detail_data.get('max_hr') or activity.get('max_hr', 185)
            dist_mi  = activity.get('distance_mi', 0)
            elev_ft  = activity.get('elevation_ft', 0)
            avg_grade = (elev_ft / (dist_mi * 5280)) * 100 if dist_mi > 0 else 0
            verdict  = classify_split(avg_cad, avg_hr, max_hr, avg_grade)
            if verdict == SPLIT_BUCKET.HIGH_QUALITY:
                v_label = 'High Quality Miles'
                v_color = 'text-emerald-400'
                v_bg    = 'bg-emerald-500/20 border-emerald-500/30'
            elif verdict == SPLIT_BUCKET.STRUCTURAL:
                v_label = 'Structural Miles'
                v_color = 'text-blue-400'
                v_bg    = 'bg-blue-500/20 border-blue-500/30'
            elif verdict == SPLIT_BUCKET.BROKEN:
                v_label = 'Broken Miles'
                v_color = 'text-red-400'
                v_bg    = 'bg-red-500/20 border-red-500/30'
        except Exception as e:
            print(f"Badge Calc Error: {e}")

        # ── Remove ALL skeleton shimmer cards, render real content ─────────
        if not _nav_guard['active']:
            return  # User navigated away while we were loading

        skeleton_container.delete()  # atomically removes map + header + all card shimmers

        # Map variables for bounds-fitting after dialog open
        modal_map        = None
        modal_fit_bounds = None
        modal_map_card   = None

        with content_container:
            # ── 1. Leaflet map ────────────────────────────────────────────
            activity_meta   = detail_data.get('activity_metadata', {})
            map_payload     = self._cb['map_payload_cb'](activity_meta) or {}
            route_segments  = map_payload.get('segments', []) if isinstance(map_payload, dict) else []
            final_bounds    = self._cb['normalize_bounds_cb'](
                map_payload.get('bounds')
            ) if isinstance(map_payload, dict) else None

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
                with ui.card().classes(
                    'w-full h-80 bg-zinc-950 p-0 border-none no-shadow overflow-hidden relative group'
                ).style('opacity: 0; transition: opacity 0.18s ease;') as map_card:
                    modal_map_card = map_card
                    m = ui.leaflet(center=map_center, zoom=13, options={
                        'zoomControl': False,
                        'attributionControl': False,
                    }).classes('w-full h-full leaflet-seam-fix')
                    m.tile_layer(
                        url_template=r'https://mt1.google.com/vt/lyrs=y&x={x}&y={y}&z={z}',
                        options={'maxZoom': 20, 'detectRetina': True},
                    )

                    map_color_mode    = {'value': 'pace'}
                    map_route_palette = {
                        'pace': 'linear-gradient(to right, #0000ff, #00ff00, #ffff00, #ffa500, #ff0000)',
                        'hr':   HR_ZONE_MAP_LEGEND_GRADIENT,
                    }
                    map_legend_left  = {'pace': 'Slower', 'hr': 'Zone 1'}
                    map_legend_right = {'pace': 'Faster', 'hr': 'Zone 5'}
                    route_weight     = 4

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
                        m.tile_layer(
                            url_template=r'https://mt1.google.com/vt/lyrs=y&x={x}&y={y}&z={z}',
                            options={'maxZoom': 20, 'detectRetina': True},
                        )
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
                                args=[latlngs, {
                                    'color': color, 'weight': route_weight,
                                    'opacity': 1.0, 'lineCap': 'round', 'lineJoin': 'round',
                                }],
                            )

                    _render_route('pace')

                    start_point = end_point = None
                    try:
                        if route_segments and len(route_segments[0]) >= 4:
                            start_point = [float(route_segments[0][0]),  float(route_segments[0][1])]
                        if route_segments and len(route_segments[-1]) >= 4:
                            end_point   = [float(route_segments[-1][2]), float(route_segments[-1][3])]
                    except Exception:
                        pass

                    def _valid_latlng(pt):
                        return (
                            isinstance(pt, (list, tuple)) and len(pt) == 2
                            and -90 <= pt[0] <= 90 and -180 <= pt[1] <= 180
                        )

                    if _valid_latlng(start_point) or _valid_latlng(end_point):
                        icon_start_svg = '<svg width="16" height="16" viewBox="0 0 24 24" fill="white" xmlns="http://www.w3.org/2000/svg"><path d="M8 5V19L19 12L8 5Z" /></svg>'
                        icon_end_svg   = '<svg width="14" height="14" viewBox="0 0 24 24" fill="white" xmlns="http://www.w3.org/2000/svg"><rect x="4" y="4" width="16" height="16" rx="2" /></svg>'

                        async def _add_pro_markers_once_ready(
                            _map=m, _start=start_point, _end=end_point,
                            _svg_start=icon_start_svg, _svg_end=icon_end_svg
                        ):
                            try:
                                await _map.initialized()
                                await asyncio.sleep(0.02)
                            except Exception as ex:
                                print(f"Pro marker init wait failed: {ex}")
                                return
                            for pt, svg, z_off in [(_start, _svg_start, 1000), (_end, _svg_end, 999)]:
                                if _valid_latlng(pt):
                                    try:
                                        js = (
                                            f'L.marker([{pt[0]}, {pt[1]}], {{'
                                            f'interactive:false,keyboard:false,zIndexOffset:{z_off},'
                                            f'icon:L.divIcon({{className:"pro-marker",'
                                            f'html:{json.dumps(svg)},'
                                            f'iconSize:[30,30],iconAnchor:[15,15]}})'
                                            f'}})'
                                        )
                                        _map.run_map_method(':addLayer', js)
                                    except Exception as ex:
                                        print(f"Pro marker add failed: {ex}")

                        asyncio.create_task(_add_pro_markers_once_ready())

                    with ui.element('div').classes(
                        'absolute bottom-1 right-2 z-[9999] pointer-events-none '
                        'text-[10px] text-zinc-400 font-mono mix-blend-plus-lighter'
                    ):
                        ui.html('&copy; Google 2026')

                    toggle_id = f'map_mode_toggle_{activity_hash}'

                    with ui.element('div').classes('absolute top-2 left-2 z-[9999] gap-2 flex flex-col'):
                        with ui.button(icon='add').props('round dense flat no-caps ripple=false').classes('map-zoom-btn no-ripple').style(
                            'background-color: rgba(9, 9, 11, 0.72) !important;'
                            'color: #ffffff !important; border: 1px solid rgba(255,255,255,0.14) !important;'
                        ).on('click', lambda: m.run_map_method("zoomIn")):
                            pass
                        with ui.button(icon='remove').props('round dense flat no-caps ripple=false').classes('map-zoom-btn no-ripple').style(
                            'background-color: rgba(9, 9, 11, 0.72) !important;'
                            'color: #ffffff !important; border: 1px solid rgba(255,255,255,0.14) !important;'
                        ).on('click', lambda: m.run_map_method("zoomOut")):
                            pass

                    with ui.column().classes(
                        'absolute top-2 right-2 z-[9999] p-3 rounded-xl '
                        'bg-black/60 backdrop-blur-md border border-white/10 shadow-lg gap-1'
                    ):
                        legend_bar   = ui.element('div').classes('w-32 h-2 rounded-full').style(
                            f"background: {map_route_palette['pace']}"
                        )
                        with ui.row().classes('w-full justify-between text-[10px] text-zinc-300 font-medium tracking-wide'):
                            legend_left  = ui.label(map_legend_left['pace'])
                            legend_right = ui.label(map_legend_right['pace'])

                    def _set_map_mode(mode):
                        if mode not in ('pace', 'hr'):
                            mode = 'pace'
                        if map_color_mode['value'] == mode:
                            return
                        map_color_mode['value'] = mode
                        _render_route(mode)
                        if _valid_latlng(start_point) or _valid_latlng(end_point):
                            try:
                                asyncio.create_task(_add_pro_markers_once_ready())
                            except Exception:
                                pass
                        legend_bar.style(f"background: {map_route_palette.get(mode, map_route_palette['pace'])}")
                        legend_left.set_text(map_legend_left.get(mode, map_legend_left['pace']))
                        legend_right.set_text(map_legend_right.get(mode, map_legend_right['pace']))

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
                                    checked = str(raw.get('value', '')).strip().lower() in {
                                        '1', 'true', 'on', 'checked', 'hr'
                                    }
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

                    if final_bounds:
                        modal_map        = m
                        modal_fit_bounds = final_bounds

            # ── 2. Activity header ────────────────────────────────────────
            with ui.column().classes('w-full px-4 gap-1'):
                date_str = activity.get('date', '')
                try:
                    from datetime import datetime
                    dt             = datetime.strptime(date_str, '%Y-%m-%d %H:%M')
                    formatted_date = dt.strftime('%A, %B %-d • %-I:%M %p')
                except Exception:
                    formatted_date = date_str
                ui.label(formatted_date).classes('text-2xl font-bold text-white tracking-tight')

                with ui.row().classes('items-center gap-3'):
                    distance     = activity.get('distance_mi', 0)
                    elevation    = activity.get('elevation_ft', 0)
                    pace         = activity.get('pace', '--:--')
                    calories     = activity.get('calories', 0)
                    metrics_str  = f"{distance:.1f} mi • {elevation} ft • {pace} /mi"
                    if calories:
                        metrics_str += f" • {calories} cal"
                    ui.label(metrics_str).classes(
                        'text-zinc-400 font-sans tabular-nums text-sm tracking-wide'
                    )
                    if v_label:
                        ui.label(v_label).classes(
                            f'text-[10px] font-bold px-2 py-0.5 rounded border '
                            f'{v_color} {v_bg} tracking-wider'
                        )

            ui.separator().classes('bg-zinc-800 my-4')

            # ── 3. Strategy row ───────────────────────────────────────────
            self._cb['strategy_row_cb'](run_walk_stats, terrain_stats)

            # ── 4. HR Zones chart ─────────────────────────────────────────
            with ui.card().classes('w-full bg-zinc-900 p-4 border border-zinc-800').style(
                'border-radius: 8px; box-shadow: 0px 4px 12px rgba(0,0,0,0.4);'
            ):
                if detail_data.get('max_hr_fallback'):
                    ui.label('⚠️ Zones based on Session Max HR').classes('text-xs text-yellow-500 mb-2')
                ui.label('TIME IN HEART RATE ZONES').classes('text-lg font-bold text-white mb-2')
                hr_chart = self._cb['hr_zone_chart_cb'](hr_zones)
                ui.plotly(hr_chart).classes('w-full')

            # ── 5. Body response row ──────────────────────────────────────
            session_data = detail_data.get('session_data', {})
            if not session_data.get('avg_cadence'):
                session_data['avg_cadence'] = activity.get('avg_cadence', 0)
            has_dynamics = session_data and any([
                session_data.get('avg_vertical_oscillation'),
                session_data.get('avg_stance_time'),
            ])

            with ui.row().classes('w-full gap-3 items-stretch'):
                if has_dynamics:
                    with ui.column().classes('flex-1 min-w-0'):
                        self._cb['dynamics_card_cb'](session_data)
                with ui.column().classes('flex-1 min-w-0'):
                    self._cb['decoupling_card_cb'](
                        decoupling,
                        efficiency_factor=activity.get('efficiency_factor', 0),
                    )
                if session_data:
                    with ui.column().classes('flex-1 min-w-0'):
                        self._cb['physiology_card_cb'](session_data, activity)

            # ── 6. Terrain context graph ──────────────────────────────────
            if detail_data.get('distance_stream') and detail_data.get('elevation_stream'):
                with ui.card().classes('w-full bg-zinc-900 p-4 border border-zinc-800').style(
                    'border-radius: 8px; box-shadow: 0px 4px 12px rgba(0,0,0,0.4);'
                ):
                    with ui.row().classes('w-full justify-between items-center mb-2'):
                        ui.label('ELEVATION ANALYSIS').classes('text-lg font-bold text-white')
                        terrain_toggle = ui.toggle(
                            ['Cadence', 'Heart Rate', 'Pace'], value='Cadence'
                        ).props('dense rounded toggle-color="grey-8" text-color="white" size="sm" no-caps').classes(
                            'terrain-metric-toggle'
                        ).style(
                            'background: rgba(44,44,46,0.95);'
                            ' border-radius: 20px; padding: 3px;'
                            ' box-shadow: inset 0 0 0 0.5px rgba(255,255,255,0.12), 0 2px 8px rgba(0,0,0,0.5);'
                        )
                    initial_fig    = self._cb['terrain_graph_cb'](detail_data, metric='Cadence')
                    terrain_plotly = ui.plotly(initial_fig).classes('w-full')

                    def _on_terrain_metric_change(e, _dd=detail_data, _chart=terrain_plotly):
                        new_fig       = self._cb['terrain_graph_cb'](_dd, metric=e.value)
                        _chart.figure = new_fig
                        _chart.update()

                    terrain_toggle.on_value_change(_on_terrain_metric_change)

                    ui.add_head_html('''
                    <style>
                    .terrain-metric-toggle { border-radius: 20px !important; padding: 3px !important; }
                    .terrain-metric-toggle .q-btn {
                        color: rgba(255,255,255,0.5) !important; font-size: 11.5px !important;
                        font-weight: 500 !important; padding: 3px 13px !important;
                        min-height: 28px !important; border-radius: 17px !important;
                        letter-spacing: 0.01em !important;
                        transition: color 0.18s ease, background 0.18s ease, box-shadow 0.18s ease !important;
                    }
                    .terrain-metric-toggle .q-btn--active {
                        background: #636366 !important; color: #ffffff !important;
                        font-weight: 600 !important;
                        box-shadow: 0 1px 5px rgba(0,0,0,0.55), 0 0.5px 1.5px rgba(0,0,0,0.35) !important;
                    }
                    </style>
                    ''')

            # ── 7. Lap splits ─────────────────────────────────────────────
            with ui.card().classes('w-full bg-zinc-900 p-4 border border-zinc-800').style(
                'border-radius: 8px; box-shadow: 0px 4px 12px rgba(0,0,0,0.4);'
            ):
                with ui.row().classes('w-full justify-between items-center mb-2'):
                    ui.label('Lap Splits').classes('text-lg font-bold text-white')
                    copy_icon = ui.icon('content_copy').classes(
                        'cursor-pointer text-zinc-500 hover:text-white transition-colors duration-200 text-sm'
                    )
                    copy_icon.on('click.stop', lambda: self._cb['copy_splits_cb'](enhanced_laps))
                self._cb['lap_splits_cb'](enhanced_laps)

        content_container.set_visibility(True)

        # ── Map fit-bounds lifecycle (after content is visible) ───────────
        if modal_map and modal_fit_bounds:
            fit_options = {'padding': [26, 26], 'maxZoom': 18, 'animate': False}

            def _bounds_contains(actual, target, tol=0.00002):
                if not actual or not target:
                    return False
                return (
                    actual[0][0] <= (target[0][0] + tol)
                    and actual[0][1] <= (target[0][1] + tol)
                    and actual[1][0] >= (target[1][0] - tol)
                    and actual[1][1] >= (target[1][1] - tol)
                )

            async def _read_current_map_bounds(_map):
                try:
                    expr = (
                        "((map) => {"
                        "const b = map.getBounds();"
                        "return [[b.getSouthWest().lat, b.getSouthWest().lng],"
                        "[b.getNorthEast().lat, b.getNorthEast().lng], map.getZoom()];"
                        "})"
                    )
                    raw = await _map.run_map_method(f':{expr}')
                    if isinstance(raw, (list, tuple)) and len(raw) >= 2:
                        return self._cb['normalize_bounds_cb'](raw[:2]), (raw[2] if len(raw) >= 3 else None)
                except Exception as ex:
                    print(f"Map bounds read warning: {ex}")
                return None, None

            async def _fit_modal_map_deterministic():
                # Bail silently if the user navigated away before map finished loading.
                # Without this, Leaflet JS calls on the destroyed widget time out and
                # spam the terminal with "bounds still off after retry" warnings.
                if not _nav_guard['active']:
                    return
                try:
                    await modal_map.initialized()
                    await asyncio.sleep(0)
                    if not _nav_guard['active']:
                        return
                    modal_map.run_map_method('invalidateSize')
                    modal_map.run_map_method('fitBounds', modal_fit_bounds, fit_options)
                    await asyncio.sleep(0.08)
                    if not _nav_guard['active']:
                        return
                    current_bounds, _ = await _read_current_map_bounds(modal_map)
                    if not _bounds_contains(current_bounds, modal_fit_bounds):
                        modal_map.run_map_method('invalidateSize')
                        modal_map.run_map_method('fitBounds', modal_fit_bounds, fit_options)
                        await asyncio.sleep(0.08)
                        if not _nav_guard['active']:
                            return
                        current_bounds, _ = await _read_current_map_bounds(modal_map)
                        if not _bounds_contains(current_bounds, modal_fit_bounds):
                            pass  # bounds discrepancy is cosmetic, skip noisy warning
                except Exception as ex:
                    if _nav_guard['active']:  # only log if modal is still open
                        print(f"Map fit lifecycle error: {ex}")
                finally:
                    if modal_map_card and not getattr(modal_map_card, 'is_deleted', False):
                        modal_map_card.style('opacity: 1; transition: opacity 0.18s ease;')

            asyncio.create_task(_fit_modal_map_deterministic())
