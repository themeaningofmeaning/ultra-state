"""
components/charts.py
────────────────────
Phase 2 Step 4: Plotly chart builders extracted from UltraStateApp.

Standalone functions that return Plotly Figure objects.
No app state, no NiceGUI context assumptions.
"""
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from constants import SPLIT_BUCKET
from hr_zones import (
    HR_ZONE_COLORS,
    HR_ZONE_ORDER,
    HR_ZONE_RANGE_LABELS,
    HR_ZONE_MAP_LEGEND_GRADIENT,
    classify_hr_zone
)

def create_hr_zone_chart(zone_times):
    """
    Create horizontal bar chart showing time in each HR zone.

    Args:
        zone_times: Dictionary with zone names and time in minutes

    Returns:
        Plotly Figure object
    """
    import plotly.graph_objects as go

    y_labels = [HR_ZONE_RANGE_LABELS[zone] for zone in HR_ZONE_ORDER]
    x_values = [zone_times.get(label, 0) for label in y_labels]
    colors = [HR_ZONE_COLORS[zone] for zone in HR_ZONE_ORDER]

    # Create horizontal bar chart
    fig = go.Figure(data=[
        go.Bar(
            y=y_labels,
            x=x_values,
            orientation='h',
            marker=dict(color=colors),
            text=[f"{t:.1f} min" for t in x_values],
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



def build_terrain_graph(detail_data, metric='Cadence', use_miles=True):
    """
    Build Terrain Context Graph with selectable primary metric.

    The Elevation profile (grey filled area, secondary Y-axis) is always
    present.  The primary metric is one of 'Cadence', 'Heart Rate', or
    'Pace', each with its own coloring logic.

    Args:
        detail_data: Full dict returned by get_activity_detail()
        metric: 'Cadence' | 'Heart Rate' | 'Pace'
        use_miles: True for imperial, False for metric
    """
    from plotly.subplots import make_subplots
    import plotly.graph_objects as go
    import pandas as pd
    import numpy as np

    # ── 1. STREAMS & VALIDATION ──────────────────────────────────
    distance_stream = detail_data.get('distance_stream', [])
    elevation_stream = detail_data.get('elevation_stream', [])
    cadence_stream = detail_data.get('cadence_stream', [])
    hr_stream = detail_data.get('hr_stream', [])
    speed_stream = detail_data.get('speed_stream', [])
    timestamps = detail_data.get('timestamps')
    max_hr = detail_data.get('max_hr', 185)

    # Determine minimum length across always-required streams
    required_lens = [len(distance_stream), len(elevation_stream)]
    if metric == 'Cadence':
        required_lens.append(len(cadence_stream))
    elif metric == 'Heart Rate':
        required_lens.append(len(hr_stream))
    elif metric == 'Pace':
        required_lens.append(len(speed_stream))
    min_len = min(required_lens)
    if min_len < 2:
        # Not enough data – return empty figure
        fig = go.Figure()
        fig.update_layout(template='plotly_dark', paper_bgcolor='rgba(0,0,0,0)',
                          plot_bgcolor='rgba(0,0,0,0)', height=350)
        return fig

    dist = np.array(distance_stream[:min_len])
    elev = np.array(elevation_stream[:min_len])

    # ── helper for optional streams ───────────────────────────────
    def prepare_optional(stream, default_val=None):
        if stream is None or len(stream) == 0:
            return np.full(min_len, default_val)
        arr = np.array(stream[:min_len])
        if len(arr) < min_len:
            return np.pad(arr, (0, min_len - len(arr)), constant_values=default_val)
        return arr

    # ── Timestamps alignment ─────────────────────────────────────
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

    # ── 2. UNIT CONVERSION (shared) ──────────────────────────────
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

    # Marker sub-sampling for long activities
    step = 5 if min_len > 7200 else 1
    idx_sub = np.arange(0, min_len, step)

    # ══════════════════════════════════════════════════════════════
    # 3. METRIC-SPECIFIC DATA & TRACES
    # ══════════════════════════════════════════════════════════════

    if metric == 'Cadence':
        # ---------- CADENCE (original logic, untouched) ----------
        cad = np.array(cadence_stream[:min_len])

        vo = prepare_optional(detail_data.get('vertical_oscillation'))
        gct = prepare_optional(detail_data.get('stance_time'))
        vr = prepare_optional(detail_data.get('vertical_ratio'))
        sl = prepare_optional(detail_data.get('step_length'))

        cad_float = np.array(cad, dtype=float)
        cad_float[cad_float <= 0] = np.nan
        cad_float = cad_float * 2.0  # Convert to full steps per minute (SPM)
        cad_series = pd.Series(cad_float)
        cad_smoothed = cad_series.rolling(window=15, min_periods=1, center=True).mean()
        mask_gaps = (cad_float <= 0) | (np.isnan(cad_float))
        cad_smoothed[mask_gaps] = np.nan

        # Waterfall color logic
        marker_colors = []
        verdicts = []
        why_metrics = []
        C_GREEN = '#32D74B'
        C_YELLOW = '#FFD60A'
        C_RED = '#FF453A'
        C_GREY = '#8E8E93'

        for i in range(min_len):
            c_val = cad_smoothed.iloc[i] if not np.isnan(cad_smoothed.iloc[i]) else 0
            vr_val = vr[i]
            gct_val = gct[i]
            status = None

            if vr_val is not None and vr_val > 0:
                if vr_val < 8.0:
                    status = (C_GREEN, "✅ Elite Efficiency")
                elif vr_val <= 10.0:
                    status = (C_YELLOW, "⚖️ Good Form")
                else:
                    status = (C_RED, "⚠️ High Bounce")
            elif gct_val is not None and gct_val > 0:
                if gct_val < 250:
                    status = (C_GREEN, "🚀 Fast Reactivity")
                elif gct_val <= 300:
                    status = (C_YELLOW, "🏃 Balanced")
                else:
                    status = (C_RED, "🛑 Long Ground Contact")
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

            diag = []
            if sl[i] and sl[i] > 0:
                diag.append(f"📏 Stride: {sl[i]/1000:.2f}m")
            if gct_val and gct_val > 0:
                diag.append(f"⏱️ GCT: {int(gct_val)}ms")
            if vo[i] and vo[i] > 0:
                diag.append(f"↕️ Vert Osc: {vo[i]/10:.1f}cm")
            why_metrics.append(" | ".join(diag) if diag else "No advanced metrics")

        y_smoothed = cad_smoothed
        y_title = "Cadence (SPM)"
        
        y_extra_kwargs = {}
        valid_y = y_smoothed.dropna()
        if len(valid_y) > 0:
            q_low = valid_y.quantile(0.01)
            q_high = valid_y.quantile(0.99)
            span = max(q_high - q_low, 10.0)
            y_extra_kwargs['range'] = [max(0, q_low - span * 0.1), q_high + span * 0.1]

        # Hover template for cadence
        hover_template = (
            "<b>%{customdata[0]}</b><br>" +
            f"📏 Distance: %{{x:.2f}} {dist_units}<br>" +
            f"🏔️ Elevation: %{{customdata[3]:.0f}} {elev_units}<br>" +
            "👟 Cadence: <b>%{y:.0f}</b> spm<br>" +
            "<br>" +
            "<i>%{customdata[2]}</i><br>" +
            "<b>%{customdata[1]}</b>" +
            "<extra></extra>"
        )

    elif metric == 'Heart Rate':
        # ---------- HEART RATE ----------
        hr_raw = np.array(hr_stream[:min_len], dtype=float)
        hr_raw[hr_raw <= 0] = np.nan
        # Also NaN out None values
        for i in range(min_len):
            if hr_stream[i] is None:
                hr_raw[i] = np.nan

        hr_series = pd.Series(hr_raw)
        hr_smoothed = hr_series.rolling(window=15, min_periods=1, center=True).mean()
        mask_gaps = np.isnan(hr_raw)
        hr_smoothed[mask_gaps] = np.nan

        # Color by HR zone
        marker_colors = []
        verdicts = []
        why_metrics = []
        for i in range(min_len):
            hr_val = hr_raw[i]
            if np.isnan(hr_val):
                marker_colors.append('#8E8E93')
                verdicts.append('⏸️ No HR Data')
                why_metrics.append('')
            else:
                zone = classify_hr_zone(hr_val, max_hr)
                marker_colors.append(HR_ZONE_COLORS.get(zone, '#8E8E93'))
                verdicts.append(f"💗 {zone}")
                pct = (hr_val / max_hr) * 100 if max_hr > 0 else 0
                why_metrics.append(f"{pct:.0f}% of Max HR ({int(max_hr)} bpm)")

        y_smoothed = hr_smoothed
        y_title = "Heart Rate (BPM)"
        
        y_extra_kwargs = {}
        valid_y = y_smoothed.dropna()
        if len(valid_y) > 0:
            q_low = valid_y.quantile(0.01)
            q_high = valid_y.quantile(0.99)
            span = max(q_high - q_low, 10.0)
            y_extra_kwargs['range'] = [max(0, q_low - span * 0.1), q_high + span * 0.1]

        hover_template = (
            "<b>%{customdata[0]}</b><br>" +
            f"📏 Distance: %{{x:.2f}} {dist_units}<br>" +
            f"🏔️ Elevation: %{{customdata[3]:.0f}} {elev_units}<br>" +
            "❤️ HR: <b>%{y:.0f}</b> bpm<br>" +
            "<i>%{customdata[2]}</i><br>" +
            "<b>%{customdata[1]}</b>" +
            "<extra></extra>"
        )

    elif metric == 'Pace':
        # ---------- PACE ----------
        spd = np.array(speed_stream[:min_len], dtype=float)
        # Mask stopped/invalid points
        for i in range(min_len):
            if speed_stream[i] is None or spd[i] <= 0.3:
                spd[i] = np.nan

        # Convert m/s → min/mi: 26.8224 / speed
        pace_raw = np.where(np.isnan(spd), np.nan, 26.8224 / spd)

        pace_series = pd.Series(pace_raw)
        pace_smoothed = pace_series.rolling(window=15, min_periods=1, center=True).mean()
        mask_gaps = np.isnan(pace_raw)
        pace_smoothed[mask_gaps] = np.nan

        # Color by pace bands
        marker_colors = []
        verdicts = []
        why_metrics = []
        C_PACE_FAST = '#32D74B'    # Green  (< 8:00/mi)
        C_PACE_MOD  = '#FFD60A'    # Yellow (8:00 - 10:00)
        C_PACE_SLOW = '#FF9F0A'    # Orange (10:00 - 12:00)
        C_PACE_HIKE = '#FF453A'    # Red    (> 12:00/mi)
        C_GREY = '#8E8E93'

        for i in range(min_len):
            p = pace_raw[i]
            if np.isnan(p):
                marker_colors.append(C_GREY)
                verdicts.append('⏸️ Stopped')
                why_metrics.append('')
            else:
                # Format as MM:SS
                mins = int(p)
                secs = int((p - mins) * 60)
                pace_str = f"{mins}:{secs:02d} /mi"
                if p < 8.0:
                    marker_colors.append(C_PACE_FAST)
                    verdicts.append(f"🚀 Fast ({pace_str})")
                elif p < 10.0:
                    marker_colors.append(C_PACE_MOD)
                    verdicts.append(f"🏃 Moderate ({pace_str})")
                elif p < 12.0:
                    marker_colors.append(C_PACE_SLOW)
                    verdicts.append(f"🥾 Easy ({pace_str})")
                else:
                    marker_colors.append(C_PACE_HIKE)
                    verdicts.append(f"🚶 Hiking ({pace_str})")
                # Speed in mph for context
                mph = spd[i] * 2.23694
                why_metrics.append(f"Speed: {mph:.1f} mph")

        y_smoothed = pace_smoothed
        y_title = "Pace (min/mi)"
        
        y_extra_kwargs = {'autorange': 'reversed'}
        valid_y = y_smoothed.dropna()
        if len(valid_y) > 0:
            q_low = valid_y.quantile(0.02)
            q_high = valid_y.quantile(0.98)
            span = max(q_high - q_low, 1.0)
            y_extra_kwargs['range'] = [q_high + span * 0.1, max(0, q_low - span * 0.1)]

        # Build MM:SS tick text for hover using customdata
        pace_hover_labels = []
        for i in range(min_len):
            p = pace_smoothed.iloc[i] if not np.isnan(pace_smoothed.iloc[i]) else np.nan
            if np.isnan(p):
                pace_hover_labels.append('--:--')
            else:
                mins = int(p)
                secs = int((p - mins) * 60)
                pace_hover_labels.append(f"{mins}:{secs:02d}")

        hover_template = (
            "<b>%{customdata[0]}</b><br>" +
            f"📏 Distance: %{{x:.2f}} {dist_units}<br>" +
            f"🏔️ Elevation: %{{customdata[3]:.0f}} {elev_units}<br>" +
            "🏃 Pace: <b>%{customdata[4]}</b> /mi<br>" +
            "<i>%{customdata[2]}</i><br>" +
            "<b>%{customdata[1]}</b>" +
            "<extra></extra>"
        )
    else:
        # Fallback – treat as Cadence
        return build_terrain_graph(detail_data, metric='Cadence', use_miles=use_miles)

    # ══════════════════════════════════════════════════════════════
    # 4. BUILD FIGURE (shared across all metrics)
    # ══════════════════════════════════════════════════════════════
    fig = make_subplots(specs=[[{"secondary_y": True}]])

    # Trace 1: Elevation (Area – Background)
    fig.add_trace(
        go.Scatter(
            x=dist_conv,
            y=elev_conv,
            name="Elevation",
            fill='tozeroy',
            mode='lines',
            line=dict(color='#3f3f46', width=0),
            fillcolor='rgba(63, 63, 70, 0.3)',
            hoverinfo='skip'
        ),
        secondary_y=True
    )

    # Trace 2: Smoothed metric line
    fig.add_trace(
        go.Scatter(
            x=dist_conv,
            y=y_smoothed,
            name="Trend",
            mode='lines',
            line=dict(color='#525252', width=2),
            connectgaps=False,
            hoverinfo='skip'
        ),
        secondary_y=False
    )

    # Trace 3: Colored markers (subsampled)
    # Build customdata columns
    cd_cols = [
        [time_labels[i] for i in idx_sub],       # 0: Time
        [verdicts[i] for i in idx_sub],           # 1: Verdict
        [why_metrics[i] for i in idx_sub],        # 2: Diagnostic / context
        [str(elev_conv[i]) for i in idx_sub],     # 3: Elevation (as string for safe stacking)
    ]
    # Pace metric adds a 5th column for formatted pace string
    if metric == 'Pace':
        cd_cols.append([pace_hover_labels[i] for i in idx_sub])  # 4: Pace MM:SS

    customdata = np.stack(cd_cols, axis=-1)

    fig.add_trace(
        go.Scatter(
            x=dist_conv[idx_sub],
            y=y_smoothed.iloc[idx_sub] if hasattr(y_smoothed, 'iloc') else y_smoothed[idx_sub],
            name="Status",
            mode='markers',
            marker=dict(
                color=[marker_colors[i] for i in idx_sub],
                size=4,
                opacity=0.8
            ),
            customdata=customdata,
            hovertemplate=hover_template
        ),
        secondary_y=False
    )

    # ── Layout ────────────────────────────────────────────────────
    fig.update_layout(
        template='plotly_dark',
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        height=350,
        margin=dict(l=20, r=20, t=20, b=40),
        hovermode="x unified",
        showlegend=False,
        font=dict(family="Inter, sans-serif", size=12, color="#a1a1aa"),
        modebar={'remove': ['zoom', 'pan', 'select', 'lasso2d', 'zoomIn',
                            'zoomOut', 'autoScale', 'resetScale', 'toImage']}
    )

    # Primary Y-axis
    primary_y_kwargs = dict(
        title_text=y_title,
        title_font=dict(color='#A1A1AA'),
        tickfont=dict(color='#D4D4D8'),
        secondary_y=False,
        gridcolor='#27272a',
        zeroline=False,
    )
    primary_y_kwargs.update(y_extra_kwargs)
    fig.update_yaxes(**primary_y_kwargs)

    # Secondary Y-axis (Elevation – dynamically scaled for better variation)
    min_elev = np.nanmin(elev_conv) if len(elev_conv) > 0 else 0
    max_elev = np.nanmax(elev_conv) if len(elev_conv) > 0 else 100
    elev_range = max_elev - min_elev
    if elev_range < 50:
        elev_range = 50
    elev_y_range = [min_elev - (elev_range * 0.1), max_elev + (elev_range * 1.2)]

    fig.update_yaxes(
        title_text=f"Elevation ({elev_units})",
        title_font=dict(color='#A1A1AA'),
        tickfont=dict(color='#D4D4D8'),
        secondary_y=True,
        showgrid=False,
        zeroline=False,
        range=elev_y_range
    )

    # X-axis
    fig.update_xaxes(
        title_text=f"Distance ({dist_units})",
        title_font=dict(color='#A1A1AA'),
        tickfont=dict(color='#D4D4D8'),
        gridcolor='#27272a',
        zeroline=False
    )

    return fig

# Backwards-compatible wrapper for the original call signature


