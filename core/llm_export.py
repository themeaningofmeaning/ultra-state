"""LLM export orchestration for clipboard-friendly coaching reports."""

from __future__ import annotations

import asyncio
import os
from datetime import timezone

import pandas as pd
from nicegui import run, ui

from analyzer import compute_training_load_and_zones


class LLMExporter:
    """Build and copy LLM-ready activity reports using injected app services."""

    def __init__(self, db, library_manager, layout):
        self.db = db
        self.library_manager = library_manager
        self.layout = layout

    async def _get_library_path(self):
        """Resolve configured library path using injected library manager APIs."""
        get_path_cb = getattr(self.library_manager, "get_library_path", None)
        if callable(get_path_cb):
            path = get_path_cb()
            if asyncio.iscoroutine(path):
                path = await path
            return path

        get_root_cb = getattr(self.library_manager, "get_library_root", None)
        if callable(get_root_cb):
            root = get_root_cb()
            if asyncio.iscoroutine(root):
                root = await root
            return root

        return None

    def _locate_file(self, activity, library_path=None):
        """Locate activity FIT file using DB path first, then configured library path."""
        db_path = activity.get("file_path")
        if db_path and os.path.exists(db_path):
            return db_path

        if library_path is None:
            get_path_cb = getattr(self.library_manager, "get_library_path", None)
            if callable(get_path_cb):
                maybe_path = get_path_cb()
                if not asyncio.iscoroutine(maybe_path):
                    library_path = maybe_path

        filename = activity.get("filename")
        if library_path and filename:
            candidate = os.path.join(library_path, filename)
            if os.path.exists(candidate):
                return candidate

        return None

    async def generate_export(self, activities_data, target_activity_id=None):
        """
        Copy detailed activity data to clipboard with LLM context.

        If target_activity_id is provided, exports the full report for that single activity.
        """
        if not activities_data:
            ui.notify("No data to copy", type="warning")
            return

        progress_dialog = getattr(self.layout, "copy_loading_dialog", None)
        progress_bar = getattr(self.layout, "copy_loading_progress", None)
        status_label = getattr(self.layout, "copy_loading_status_label", None)
        if progress_bar is not None:
            progress_bar.style("width: 0%")
        if status_label is not None:
            status_label.set_text("Initializing...")
        if progress_dialog is not None:
            progress_dialog.open()
        await asyncio.sleep(0.1)

        try:
            if target_activity_id:
                selected_activities = [
                    activity for activity in activities_data
                    if activity.get("db_hash") == target_activity_id
                ]
                if not selected_activities:
                    ui.notify("Selected activity is no longer available", type="warning")
                    return
            else:
                selected_activities = activities_data

            sorted_activities = sorted(
                selected_activities,
                key=lambda x: x.get("date", ""),
                reverse=True,
            )

            library_path = await self._get_library_path()
            files_to_process = []
            for activity in sorted_activities:
                activity_hash = activity.get("db_hash")
                if not activity_hash:
                    continue

                db_activity = self.db.get_activity_by_hash(activity_hash)
                if not db_activity:
                    continue

                fit_file_path = self._locate_file(db_activity, library_path=library_path)
                if fit_file_path:
                    files_to_process.append({
                        "path": fit_file_path,
                        "activity": activity,
                    })

            if not files_to_process:
                if target_activity_id:
                    ui.notify("No FIT file found for this activity", type="warning")
                else:
                    ui.notify("No FIT files available to copy", type="warning")
                return

            total_files = len(files_to_process)
            parsed_activity_payloads = []
            for i, info in enumerate(files_to_process):
                count = i + 1
                percent = int((count / total_files) * 100)
                if progress_bar is not None:
                    progress_bar.style(f"width: {percent}%")
                if status_label is not None:
                    status_label.set_text(f"Parsing activity {count} of {total_files}...")

                batch_result = await run.io_bound(_parse_fit_files_for_clipboard, [info])
                if batch_result:
                    parsed_activity_payloads.append(batch_result[0])
                else:
                    parsed_activity_payloads.append(None)

            if status_label is not None:
                status_label.set_text("Formatting report...")
            await asyncio.sleep(0.05)

            report_lines = []

            def _safe_float(value, default=0.0):
                try:
                    parsed = float(value)
                except (TypeError, ValueError):
                    return default
                if pd.isna(parsed):
                    return default
                return parsed

            def _fmt_optional_float(value, decimals=1):
                try:
                    parsed = float(value)
                except (TypeError, ValueError):
                    return "--"
                if pd.isna(parsed):
                    return "--"
                return f"{parsed:.{decimals}f}"

            for info, parsed_payload in zip(files_to_process, parsed_activity_payloads):
                activity = info["activity"]
                parsed_payload = parsed_payload or {}
                lap_splits = parsed_payload.get("lap_splits") or []

                date = activity.get("date", "")
                time_str = activity.get("time", "")
                icon = activity.get("context_emoji", "🏃")
                context = activity.get("context_name", "Run")
                dist = _safe_float(activity.get("distance_mi"), 0.0)
                pace = activity.get("pace", "--:--")
                elev = int(_safe_float(activity.get("elevation_ft"), 0))

                avg_hr = activity.get("avg_hr", "--")
                max_hr = activity.get("max_hr", "--")
                power = int(_safe_float(activity.get("avg_power"), 0))
                power_display = f"{power}W" if power > 0 else "--"
                ef = _safe_float(activity.get("efficiency_factor"), 0.0)
                decoupling = _safe_float(activity.get("decoupling"), 0.0)
                load_score = _safe_float(
                    activity.get("load_score"),
                    _safe_float(parsed_payload.get("load_score"), 0.0),
                )
                zone1_mins = _safe_float(
                    activity.get("zone1_mins"),
                    _safe_float(parsed_payload.get("zone1_mins"), 0.0),
                )
                zone2_mins = _safe_float(
                    activity.get("zone2_mins"),
                    _safe_float(parsed_payload.get("zone2_mins"), 0.0),
                )
                zone3_mins = _safe_float(
                    activity.get("zone3_mins"),
                    _safe_float(parsed_payload.get("zone3_mins"), 0.0),
                )
                zone4_mins = _safe_float(
                    activity.get("zone4_mins"),
                    _safe_float(parsed_payload.get("zone4_mins"), 0.0),
                )
                zone5_mins = _safe_float(
                    activity.get("zone5_mins"),
                    _safe_float(parsed_payload.get("zone5_mins"), 0.0),
                )
                zone_total_mins = zone1_mins + zone2_mins + zone3_mins + zone4_mins + zone5_mins
                zone_total_mins = zone_total_mins if zone_total_mins > 0 else 0.0
                zone1_pct = (zone1_mins / zone_total_mins * 100) if zone_total_mins > 0 else 0.0
                zone2_pct = (zone2_mins / zone_total_mins * 100) if zone_total_mins > 0 else 0.0
                zone3_pct = (zone3_mins / zone_total_mins * 100) if zone_total_mins > 0 else 0.0
                zone4_pct = (zone4_mins / zone_total_mins * 100) if zone_total_mins > 0 else 0.0
                zone5_pct = (zone5_mins / zone_total_mins * 100) if zone_total_mins > 0 else 0.0

                if decoupling < 5:
                    dec_label = "Excellent"
                elif decoupling < 10:
                    dec_label = "⚠️ Moderate"
                else:
                    dec_label = "🛑 High Fatigue"

                hrr_list = activity.get("hrr_list", [])
                hrr_str = "--"
                if hrr_list and len(hrr_list) > 0:
                    hrr_str = f"{hrr_list[0]}bpm (1min)"

                te_aerobic = _fmt_optional_float(
                    activity.get("training_effect", activity.get("aerobic_te")),
                    decimals=1,
                )
                te_anaerobic = _fmt_optional_float(activity.get("anaerobic_te"), decimals=1)

                cadence_value = _safe_float(activity.get("avg_cadence"), 0.0)
                cadence = int(cadence_value) if cadence_value > 0 else "--"
                form_tag = "[✅ ELITE FORM]" if cadence_value > 170 else "[👍 GOOD FORM]"

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
Avg Power:   {power_display}
EF:          {ef:.2f} (Efficiency Factor)
Decoupling:  {decoupling:.2f}% ({dec_label})
HRR:         {hrr_str}
Training Load: {load_score:.1f}
Training Effect: {te_aerobic} Aerobic, {te_anaerobic} Anaerobic

[TIME IN HEART RATE ZONES]
Zone 1 (Recovery):  {zone1_mins:.1f}m ({zone1_pct:.1f}%)
Zone 2 (Endurance): {zone2_mins:.1f}m ({zone2_pct:.1f}%)
Zone 3 (Steady):    {zone3_mins:.1f}m ({zone3_pct:.1f}%)
Zone 4 (Threshold): {zone4_mins:.1f}m ({zone4_pct:.1f}%)
Zone 5 (Max):       {zone5_mins:.1f}m ({zone5_pct:.1f}%)

MECHANICS (The Chassis)
Avg Cadence: {cadence} spm
Form Tag:    {form_tag}

[LAP SPLITS]"""
                report_lines.append(block)

                if lap_splits:
                    for lap in lap_splits:
                        lap_num = lap.get("lap_number", 0)
                        lap_dist = lap.get("distance", 0) * 0.000621371
                        lap_pace = lap.get("actual_pace", "--:--")
                        lap_hr = int(lap["avg_hr"]) if lap.get("avg_hr") else "--"
                        raw_cad = lap.get("avg_cadence")
                        lap_cad = int(raw_cad * 2) if raw_cad else "--"
                        lap_elev = lap.get("total_ascent", 0) * 3.28084 if lap.get("total_ascent") else 0
                        report_lines.append(
                            f"{lap_num} | {lap_dist:.2f}mi | {lap_pace}/mi | {lap_hr}bpm | {lap_cad}spm | +{int(lap_elev)}ft"
                        )

                report_lines.append("\n" + "=" * 50 + "\n")

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

TRAINING LOAD:
- Quantifies total physiological stress (duration x HR intensity)
- <75: Recovery / Easy
- 75-150: Base / Aerobic
- 150-300: Overload / Hard
- >300: Overreaching / Extreme

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
            full_report = "\n".join(report_lines).strip()
            full_content = f"{llm_context.strip()}\n\n{full_report}\n"

            import pyperclip
            pyperclip.copy(full_content)

            if target_activity_id:
                ui.notify("✅ Run Analysis Copied!", type="positive", close_button=True)
            else:
                ui.notify("✅ Analysis Data Copied!", type="positive", close_button=True)

        except Exception as e:
            ui.notify(f"Error: {str(e)}", type="negative")
            print(f"❌ Error copying to clipboard: {str(e)}")
            import traceback
            traceback.print_exc()
        finally:
            if progress_dialog is not None:
                progress_dialog.close()


def _parse_fit_files_for_clipboard(activity_info_list):
    """
    Parse FIT files to extract lap data + load/zone metrics (runs via run.io_bound).

    This runs in a ThreadPool to keep the WebSocket heartbeat alive during
    heavy file I/O operations.

    Args:
        activity_info_list: List of dicts with 'path' and 'activity' keys

    Returns:
        List of dict payloads with lap_splits and physiology metrics (or None per activity)
    """
    import fitparse
    from analyzer import minetti_cost_of_running

    results = []

    for info in activity_info_list:
        try:
            fit_file_path = info["path"]
            activity = info.get("activity") or {}

            fitfile = fitparse.FitFile(fit_file_path)

            lap_data = []
            for lap_msg in fitfile.get_messages("lap"):
                vals = lap_msg.get_values()

                total_distance = vals.get("total_distance")
                total_timer_time = vals.get("total_timer_time")

                if total_distance and total_timer_time and total_timer_time > 0:
                    avg_speed = total_distance / total_timer_time
                else:
                    avg_speed = vals.get("enhanced_avg_speed") or vals.get("avg_speed")

                lap_data.append({
                    "lap_number": len(lap_data) + 1,
                    "distance": total_distance,
                    "avg_speed": avg_speed,
                    "avg_hr": vals.get("avg_heart_rate"),
                    "avg_cadence": vals.get("avg_cadence"),
                    "total_ascent": vals.get("total_ascent"),
                    "start_time": vals.get("start_time"),
                    "total_elapsed_time": vals.get("total_elapsed_time"),
                })

            elevation_stream = []
            cadence_stream = []
            hr_stream = []
            timestamps = []
            for record in fitfile.get_messages("record"):
                vals = record.get_values()

                ts = vals.get("timestamp")
                if ts:
                    ts = ts.replace(tzinfo=timezone.utc).astimezone()
                timestamps.append(ts)

                elevation_stream.append(
                    vals.get("enhanced_altitude") or vals.get("altitude")
                )
                cadence_stream.append(vals.get("cadence"))
                hr_stream.append(vals.get("heart_rate"))

            enhanced_laps = []
            for lap in lap_data:
                if not lap.get("start_time") or not lap.get("total_elapsed_time"):
                    enhanced_laps.append({
                        **lap,
                        "gap_pace": "--:--",
                        "actual_pace": "--:--",
                        "is_steep": False,
                        "avg_gradient": 0,
                        "avg_cadence": None,
                    })
                    continue

                avg_speed = lap.get("avg_speed")
                if avg_speed is None or avg_speed == 0:
                    distance = lap.get("distance", 0)
                    elapsed_time = lap.get("total_elapsed_time", 0)
                    if distance > 0 and elapsed_time > 0:
                        avg_speed = distance / elapsed_time
                    else:
                        avg_speed = 0

                lap_start = lap["start_time"]
                if lap_start and lap_start.tzinfo is None:
                    lap_start = lap_start.replace(tzinfo=timezone.utc).astimezone()

                lap_end = lap_start + pd.Timedelta(seconds=lap["total_elapsed_time"])

                lap_elevations = []
                lap_cadences = []
                for i, ts in enumerate(timestamps):
                    if ts is None:
                        continue
                    if lap_start <= ts <= lap_end:
                        if elevation_stream[i] is not None:
                            if i > 0 and elevation_stream[i - 1] is not None:
                                elev_diff = elevation_stream[i] - elevation_stream[i - 1]
                                dist_diff = avg_speed

                                if dist_diff > 0:
                                    gradient = elev_diff / dist_diff
                                    lap_elevations.append(gradient)

                        if cadence_stream[i] is not None and cadence_stream[i] > 0:
                            lap_cadences.append(cadence_stream[i])

                avg_gradient = sum(lap_elevations) / len(lap_elevations) if lap_elevations else 0
                avg_lap_cadence = sum(lap_cadences) / len(lap_cadences) if lap_cadences else None

                flat_cost = 3.6
                terrain_cost = minetti_cost_of_running(avg_gradient)
                cost_multiplier = terrain_cost / flat_cost

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
                    "gap_pace": gap_pace_str,
                    "actual_pace": actual_pace_str,
                    "is_steep": is_steep,
                    "avg_gradient": avg_gradient,
                    "avg_cadence": avg_lap_cadence,
                })

            max_hr = activity.get("max_hr") or 185
            load_zone_metrics = compute_training_load_and_zones(
                hr_stream,
                max_hr=max_hr,
                timestamps=timestamps,
            )

            results.append({
                "lap_splits": enhanced_laps,
                **load_zone_metrics,
            })

        except Exception as e:
            print(f"Error parsing FIT file: {e}")
            results.append(None)

    return results
