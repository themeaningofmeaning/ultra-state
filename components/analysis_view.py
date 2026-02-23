"""
components/analysis_view.py
──────────────────────────
Phase 2 Step 7: Trends/Analysis dashboard extraction from UltraStateApp.

Encapsulates Trends tab rendering, chart generation, zoom behavior, and
volume-lens state management. App-level side effects stay injected via
callbacks to avoid circular imports.
"""
from __future__ import annotations

import inspect
import json

import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from nicegui import ui

from analyzer import analyze_form, classify_split
from constants import LOAD_CATEGORY, LOAD_CATEGORY_COLORS, LOAD_CATEGORY_DESCRIPTIONS
from hr_zones import (
    HR_ZONE_COLORS,
    HR_ZONE_DESCRIPTIONS,
    HR_ZONE_ORDER,
    classify_hr_zone_by_ratio,
)


class AnalysisView:
    """Owns Trends tab rendering, chart generation, and lens interactions."""

    def __init__(self, state, db, df, activities_data=None, callbacks=None):
        self.state = state
        self.db = db
        self.df = df
        self.activities_data = activities_data or []
        self.callbacks = callbacks or {}

        self.plotly_container = None
        self.volume_card_container = None
        self.volume_chart = None
        self.efficiency_chart = None
        self.cadence_chart = None

        self.volume_verdict_label = None
        self.volume_subtitle_label = None
        self.efficiency_verdict_label = None
        self.cadence_verdict_label = None
        self.ef_arrow_label = None
        self.ef_trend_value_label = None
        self.ef_consistency_label = None
        self.dec_arrow_label = None
        self.dec_trend_value_label = None
        self.dec_correlation_label = None
        self._lens_buttons = {}

        self.weekly_volume_data = None
        self.weekly_mix_data = None
        self.weekly_load_data = None
        self.weekly_hr_zones_data = None
        self.volume_week_starts = []

    def set_data(self, df, activities_data):
        """Update source data used by Trends charts."""
        self.df = df
        self.activities_data = activities_data or []

    def build(self):
        """Build the Trends tab root container."""
        self.build_trends_tab()
        return self.plotly_container

    def _invoke_callback(self, name, *args, **kwargs):
        cb = self.callbacks.get(name)
        if not callable(cb):
            return None
        return cb(*args, **kwargs)

    async def _invoke_callback_async(self, name, *args, **kwargs):
        result = self._invoke_callback(name, *args, **kwargs)
        if inspect.isawaitable(result):
            return await result
        return result

    def classify_run_type(self, run_data, long_run_threshold):
        """Delegate run-type classification to the injected app callback."""
        result = self._invoke_callback('classify_run_type', run_data, long_run_threshold)
        return result if isinstance(result, str) and result else '🟡 Base'

    def show_volume_info(self, highlight_verdict=None):
        self._invoke_callback('show_volume_info', highlight_verdict=highlight_verdict)

    def show_aerobic_efficiency_info(self, highlight_verdict=None, from_trends=False):
        self._invoke_callback(
            'show_aerobic_efficiency_info',
            highlight_verdict=highlight_verdict,
            from_trends=from_trends,
        )

    def show_form_info(self, highlight_verdict=None):
        self._invoke_callback('show_form_info', highlight_verdict=highlight_verdict)

    async def handle_bar_click(self, e):
        await self._invoke_callback_async('handle_bar_click', e)

    async def handle_efficiency_click(self, e):
        await self._invoke_callback_async('handle_efficiency_click', e)

    async def handle_cadence_click(self, e):
        await self._invoke_callback_async('handle_cadence_click', e)

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

    def generate_weekly_volume_chart(self):
        """
        Generate weekly volume chart. 
        Updates: 
        1. Passes 'Category Name' to click handler for better Modal Titles.
        2. Adds 'cursor-pointer' to the chart for better UX.
        """
        self.volume_week_starts = []
        if self.df is None or self.df.empty:
            self.weekly_volume_data = None
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
            self.weekly_volume_data = None
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
        self.volume_week_starts = [pd.Timestamp(week) for week in weeks]
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
        self.volume_week_starts = []
        self.weekly_mix_data = None
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
        self.volume_week_starts = [pd.Timestamp(week) for week in weeks]

        weekly_mix = df_mix.groupby(['week_start', 'category'])['distance'].sum().unstack(fill_value=0)
        for col in categories:
            if col not in weekly_mix.columns:
                weekly_mix[col] = 0
        self.weekly_mix_data = weekly_mix
        
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
        self.volume_week_starts = []
        self.weekly_load_data = None
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
            if strain < 75:   load_cat = LOAD_CATEGORY.RECOVERY
            elif strain < 150: load_cat = LOAD_CATEGORY.BASE
            elif strain < 300: load_cat = LOAD_CATEGORY.OVERLOAD
            else:              load_cat = LOAD_CATEGORY.OVERREACHING
            
            load_data.append({
                'week_start': week_start, 'distance': dist, 'category': load_cat,
                'hash': act_hash, 'date_str': act_date_str, 'strain': strain
            })
        
        if not load_data:
            return None
        
        df_load = pd.DataFrame(load_data)
        
        # Colors and descriptions pulled from constants for consistency
        categories = list(LOAD_CATEGORY.ALL)
        colors = LOAD_CATEGORY_COLORS
        descriptions = LOAD_CATEGORY_DESCRIPTIONS
        
        grouped = df_load.groupby(['week_start', 'category'])
        weeks = sorted(df_load['week_start'].unique())
        self.volume_week_starts = [pd.Timestamp(week) for week in weeks]

        weekly_load = df_load.groupby(['week_start', 'category'])['distance'].sum().unstack(fill_value=0)
        for col in categories:
            if col not in weekly_load.columns:
                weekly_load[col] = 0
        self.weekly_load_data = weekly_load
        
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

    def calculate_mix_verdict(self, df=None, start_index=None, end_index=None):
        """Calculate mix verdict from the currently visible stacked-bar mileage."""
        try:
            categories = ['Recovery', 'Base', 'Steady', 'Long Run', 'Tempo']
            data = self._slice_lens_weekly_data(self.weekly_mix_data, start_index, end_index)
            if data is None or data.empty:
                return 'N/A', '#71717a', 'bg-zinc-700'

            for col in categories:
                if col not in data.columns:
                    data[col] = 0

            total_miles = data[categories].sum().sum()
            if total_miles <= 0:
                return 'N/A', '#71717a', 'bg-zinc-700'

            recovery_miles = data['Recovery'].sum()
            base_miles = data['Base'].sum()
            steady_miles = data['Steady'].sum()
            long_miles = data['Long Run'].sum()
            tempo_miles = data['Tempo'].sum()

            easy_miles = recovery_miles + base_miles + long_miles
            intensity_miles = steady_miles + tempo_miles

            easy_pct = (easy_miles / total_miles) * 100
            intensity_pct = (intensity_miles / total_miles) * 100
            dominant_pct = max((data[c].sum() / total_miles) * 100 for c in categories)

            if intensity_pct >= 45 and intensity_pct > easy_pct:
                return 'TEMPO HEAVY', '#ef4444', 'bg-red-500/20'

            if easy_pct >= 65 and tempo_miles > 0 and intensity_pct <= 35:
                return 'POLARIZED', '#10b981', 'bg-emerald-500/20'

            if dominant_pct >= 85:
                return 'MONOTONE', '#f97316', 'bg-orange-500/20'

            return 'BALANCED', '#3b82f6', 'bg-blue-500/20'

        except:
            return 'N/A', '#71717a', 'bg-zinc-700'

    def calculate_load_verdict(self, df=None, start_index=None, end_index=None):
        """Calculate load verdict from visible stacked-bar mileage (distance-weighted)."""
        try:
            categories = list(LOAD_CATEGORY.ALL)
            data = self._slice_lens_weekly_data(self.weekly_load_data, start_index, end_index)
            if data is None or data.empty:
                return 'N/A', '#71717a', 'bg-zinc-700'

            for col in categories:
                if col not in data.columns:
                    data[col] = 0

            total_miles = data[categories].sum().sum()
            if total_miles <= 0:
                return 'N/A', '#71717a', 'bg-zinc-700'

            recovery    = data[LOAD_CATEGORY.RECOVERY].sum()
            base_load   = data[LOAD_CATEGORY.BASE].sum()
            overload    = data[LOAD_CATEGORY.OVERLOAD].sum()
            overreaching = data[LOAD_CATEGORY.OVERREACHING].sum()

            recovery_pct  = (recovery    / total_miles) * 100
            base_pct      = (base_load   / total_miles) * 100
            overload_pct  = (overload    / total_miles) * 100
            overreach_pct = (overreaching / total_miles) * 100
            easy_pct = recovery_pct + base_pct

            if (
                overreach_pct >= 20
                and overreach_pct >= overload_pct
                and overreach_pct >= base_pct
                and overreach_pct >= recovery_pct
            ):
                return 'OVERREACHING', '#ef4444', 'bg-red-500/20'

            if (
                overload_pct >= 20
                and overload_pct >= base_pct
                and overload_pct >= recovery_pct
                and overload_pct >= overreach_pct
            ):
                return 'OVERLOAD', '#f97316', 'bg-orange-500/20'

            if easy_pct >= 85 and overload_pct < 8 and overreach_pct < 5:
                return 'UNDERTRAINED', '#3b82f6', 'bg-blue-500/20'

            return 'BASE', '#10b981', 'bg-emerald-500/20'

        except:
            return 'N/A', '#71717a', 'bg-zinc-700'

    def generate_hr_zones_chart(self):
        """Generate weekly volume chart grouped by HEART RATE ZONE (HR Zones lens).
        Y-axis = minutes. Each activity assigned to dominant zone via avg_hr/max_hr ratio."""
        self.volume_week_starts = []
        self.weekly_hr_zones_data = None
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
            zone = classify_hr_zone_by_ratio(ratio)
            
            zone_data.append({
                'week_start': week_start, 'minutes': moving_time, 'zone': zone,
                'hash': act_hash, 'date_str': act_date_str
            })
        
        if not zone_data:
            return None
        
        df_zones = pd.DataFrame(zone_data)
        
        # Unified HR zone palette and labels (shared with modal map + charts)
        categories = list(HR_ZONE_ORDER)
        colors = HR_ZONE_COLORS
        descriptions = HR_ZONE_DESCRIPTIONS
        
        grouped = df_zones.groupby(['week_start', 'zone'])
        weeks = sorted(df_zones['week_start'].unique())
        self.volume_week_starts = [pd.Timestamp(week) for week in weeks]

        weekly_zones = df_zones.groupby(['week_start', 'zone'])['minutes'].sum().unstack(fill_value=0)
        for col in categories:
            if col not in weekly_zones.columns:
                weekly_zones[col] = 0
        self.weekly_hr_zones_data = weekly_zones
        
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

    def calculate_hr_zones_verdict(self, df=None, start_index=None, end_index=None):
        """Calculate HR zones verdict from visible stacked-bar minutes."""
        try:
            categories = ['Zone 1', 'Zone 2', 'Zone 3', 'Zone 4', 'Zone 5']
            data = self._slice_lens_weekly_data(self.weekly_hr_zones_data, start_index, end_index)
            if data is None or data.empty:
                return 'N/A', '#71717a', 'bg-zinc-700'

            for col in categories:
                if col not in data.columns:
                    data[col] = 0

            total_time = data[categories].sum().sum()
            if total_time <= 0:
                return 'N/A', '#71717a', 'bg-zinc-700'

            z1 = data['Zone 1'].sum()
            z2 = data['Zone 2'].sum()
            z3 = data['Zone 3'].sum()
            z4 = data['Zone 4'].sum()
            z5 = data['Zone 5'].sum()

            easy_pct = ((z1 + z2) / total_time) * 100
            z3_pct = (z3 / total_time) * 100
            z4_pct = (z4 / total_time) * 100
            z5_pct = (z5 / total_time) * 100
            hard_pct = z4_pct + z5_pct

            if easy_pct >= 85:
                return 'ZONE 2 BASE', '#3b82f6', 'bg-blue-500/20'

            if 70 <= easy_pct <= 90 and 10 <= hard_pct <= 30 and z3_pct < 25:
                return '80/20 BALANCED', '#10b981', 'bg-emerald-500/20'

            bucket_values = {
                'easy': easy_pct,
                'z3': z3_pct,
                'z4': z4_pct,
                'z5': z5_pct,
            }
            dominant_bucket = max(bucket_values, key=bucket_values.get)

            if dominant_bucket == 'z5' and z5_pct >= 12:
                return 'ZONE 5 REDLINING', '#ef4444', 'bg-red-500/20'

            if dominant_bucket == 'z4' and z4_pct >= 20:
                return 'ZONE 4 THRESHOLD ADDICT', '#f97316', 'bg-orange-500/20'

            if dominant_bucket == 'z3' and z3_pct >= 25:
                return 'ZONE 3 JUNK', '#64748b', 'bg-slate-500/20'

            if dominant_bucket == 'easy':
                return 'ZONE 2 BASE', '#3b82f6', 'bg-blue-500/20'

            return '80/20 BALANCED', '#10b981', 'bg-emerald-500/20'

        except:
            return 'N/A', '#71717a', 'bg-zinc-700'

    def _get_volume_zoom_indices(self, relayout_args):
        """Extract categorical x-axis zoom indices from a Plotly relayout payload."""
        if not relayout_args:
            return None, None

        if 'xaxis.range[0]' in relayout_args and 'xaxis.range[1]' in relayout_args:
            return relayout_args.get('xaxis.range[0]'), relayout_args.get('xaxis.range[1]')

        x_range = relayout_args.get('xaxis.range')
        if isinstance(x_range, (list, tuple)) and len(x_range) >= 2:
            return x_range[0], x_range[1]

        return None, None

    def _slice_lens_weekly_data(self, weekly_data, start_index=None, end_index=None):
        """Slice a lens weekly dataframe to the currently zoomed categorical week window."""
        if weekly_data is None or weekly_data.empty:
            return None

        data = weekly_data.copy()

        if self.volume_week_starts:
            data = data.reindex(self.volume_week_starts, fill_value=0)

        if start_index is None or end_index is None:
            return data

        try:
            start = max(0, int(round(float(start_index))))
            end = min(len(data), int(round(float(end_index))) + 1)
        except (TypeError, ValueError):
            return data

        if end <= start:
            return data.iloc[0:0]

        return data.iloc[start:end]

    def refresh_volume_card(self):
        """Surgically refresh only the volume card content based on current lens."""
        if self.volume_card_container is None:
            return
        
        self.volume_card_container.clear()
        with self.volume_card_container:
            fig, verdict, v_color, v_bg, subtitle = self.get_active_volume_lens_state()
            
            # Update verdict badge
            if hasattr(self, 'volume_verdict_label'):
                self.volume_verdict_label.set_text(f'{verdict}')
                self.volume_verdict_label.classes(
                    f'text-sm font-bold px-3 py-1 rounded {v_bg}',
                    remove='bg-emerald-500/20 bg-blue-500/20 bg-red-500/20 bg-orange-500/20 bg-amber-500/20 bg-slate-500/20 bg-zinc-700 bg-zinc-800 text-zinc-500'
                )
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

    def get_active_volume_lens_state(self):
        """Return chart + metadata for the currently selected Training Volume lens."""
        if self.state.volume_lens == 'mix':
            fig = self.generate_training_mix_chart()
            verdict, v_color, v_bg = self.calculate_mix_verdict(self.df)
            subtitle = 'Weekly distribution by run type'
        elif self.state.volume_lens == 'load':
            fig = self.generate_load_chart()
            verdict, v_color, v_bg = self.calculate_load_verdict(self.df)
            subtitle = 'Weekly distribution by training stress'
        elif self.state.volume_lens == 'zones':
            fig = self.generate_hr_zones_chart()
            verdict, v_color, v_bg = self.calculate_hr_zones_verdict(self.df)
            subtitle = 'Weekly time in each heart rate zone'
        else:
            fig = self.generate_weekly_volume_chart()
            verdict, v_color, v_bg = self.calculate_volume_verdict(self.df)
            subtitle = 'Breakdown in quality of miles (click any section to inspect runs)'

        return fig, verdict, v_color, v_bg, subtitle

    def get_volume_lens_label(self):
        """Return UI label for the active Training Volume lens."""
        lens_labels = {
            'quality': 'Quality',
            'zones': 'HR Zones',
            'load': 'Load',
            'mix': 'Training Mix',
        }
        return lens_labels.get(self.state.volume_lens, 'Quality')

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
                return 'EFFICIENT', '#10b981', 'bg-emerald-500/20'
            
            # 2. Building Speed: EF rising fast, some drift is acceptable (progressive overload)
            if slope_ef_per_week > 1.0 and slope_dec_per_week > 0.3:
                return 'PUSHING', '#f97316', 'bg-orange-500/20'
            
            # 3. Fitness Loss: EF clearly declining
            if slope_ef_per_week < -0.5:
                # Renamed DETRAINING -> FATIGUED to match single-run terms
                return 'FATIGUED', '#ef4444', 'bg-red-500/20'
            
            # 4. Pure Drift: EF flat but decoupling increasing significantly
            if abs(slope_ef_per_week) <= 0.5 and slope_dec_per_week > 1.0:
                # Renamed DRIFTING -> FATIGUED to match single-run terms
                return 'FATIGUED', '#ef4444', 'bg-red-500/20'
            
            # 5. Default: modest gains or flat, manageable drift
            # Renamed STABLE -> BASE to match single-run terms
            return 'BASE', '#3b82f6', 'bg-blue-500/20'
            
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
                volume_fig, vol_verdict, vol_color, vol_bg, vol_subtitle = self.get_active_volume_lens_state()
                
                if volume_fig:
                    with ui.card().classes('w-full bg-zinc-900 border border-zinc-800 p-6 mb-8').style('border-radius: 12px; box-shadow: 0 10px 15px -3px rgba(0, 0, 0, 0.3);'):
                        
                        # Header Row: Title + ❓ Icon + Verdict Badge (INLINE)
                        with ui.row().classes('w-full items-center gap-2 mb-1'):
                            ui.label('Training Volume').classes('text-xl font-bold text-white')
                            
                            # Verdict Badge
                            self.volume_verdict_label = ui.label(f'{vol_verdict}').classes(f'text-sm font-bold px-3 py-1 rounded {vol_bg}').style(f'color: {vol_color};')
                            
                            # Info Icon (inline)
                            ui.icon('help_outline').classes('text-zinc-500 hover:text-white cursor-pointer text-lg transition-colors').on(
                                'click', lambda: self.show_volume_info(
                                    highlight_verdict=self.volume_verdict_label.text if hasattr(self, 'volume_verdict_label') else None
                                )
                            )
                        
                        # Subtitle (lens-adaptive)
                        self.volume_subtitle_label = ui.label(vol_subtitle).classes('text-sm text-zinc-400 mb-3')
                        
                        # === SEGMENTED TOGGLE (Pill Group) ===
                        def switch_lens(lens):
                            self.state.volume_lens = lens
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
                            for lens_key, lens_label in [('quality', 'Quality'), ('zones', 'HR Zones'), ('load', 'Load'), ('mix', 'Training Mix')]:
                                is_active = self.state.volume_lens == lens_key
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
                            self.efficiency_verdict_label = ui.label(f'{eff_verdict}').classes(f'text-sm font-bold px-3 py-1 rounded {eff_bg}').style(f'color: {eff_color}; cursor: pointer;')
                            self.efficiency_verdict_label.on(
                                'click',
                                lambda: self.show_aerobic_efficiency_info(
                                    highlight_verdict=self.efficiency_verdict_label.text if hasattr(self, 'efficiency_verdict_label') else None,
                                    from_trends=True
                                )
                            )
                            ae_info_icon = ui.icon('help_outline').classes('text-zinc-500 hover:text-white cursor-pointer text-lg transition-colors')
                            ae_info_icon.on(
                                'click',
                                lambda: self.show_aerobic_efficiency_info(
                                    highlight_verdict=self.efficiency_verdict_label.text if hasattr(self, 'efficiency_verdict_label') else None,
                                    from_trends=True
                                )
                            )
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
                            self.cadence_verdict_label = ui.label(f'{cad_verdict}').classes(f'text-sm font-bold px-3 py-1 rounded {cad_bg}').style(f'color: {cad_color}; cursor: pointer;')
                            self.cadence_verdict_label.on(
                                'click',
                                lambda: self.show_form_info(
                                    highlight_verdict=self.cadence_verdict_label.text if hasattr(self, 'cadence_verdict_label') else None
                                )
                            )
                            form_info_icon = ui.icon('help_outline').classes('text-zinc-500 hover:text-white cursor-pointer text-lg transition-colors')
                            form_info_icon.on(
                                'click',
                                lambda: self.show_form_info(
                                    highlight_verdict=self.cadence_verdict_label.text if hasattr(self, 'cadence_verdict_label') else None
                                )
                            )
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
            self.efficiency_verdict_label.set_text(eff_verdict)
            self.efficiency_verdict_label.classes(remove='bg-emerald-500/20 bg-orange-500/20 bg-red-500/20 bg-blue-500/20 bg-zinc-700', add=eff_bg)
            self.efficiency_verdict_label.style(f'color: {eff_color}; cursor: pointer;')
            self.efficiency_verdict_label.on('click', lambda: self.show_aerobic_efficiency_info(highlight_verdict=eff_verdict, from_trends=True), replace=True)
                
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
            idx_start, idx_end = self._get_volume_zoom_indices(e.args)
            has_zoom_range = idx_start is not None and idx_end is not None

            # Determine the correct verdict calculator based on current lens
            if self.state.volume_lens == 'mix':
                if has_zoom_range:
                    vol_verdict, vol_color, vol_bg = self.calculate_mix_verdict(start_index=idx_start, end_index=idx_end)
                else:
                    vol_verdict, vol_color, vol_bg = self.calculate_mix_verdict()
            elif self.state.volume_lens == 'load':
                if has_zoom_range:
                    vol_verdict, vol_color, vol_bg = self.calculate_load_verdict(start_index=idx_start, end_index=idx_end)
                else:
                    vol_verdict, vol_color, vol_bg = self.calculate_load_verdict()
            elif self.state.volume_lens == 'zones':
                if has_zoom_range:
                    vol_verdict, vol_color, vol_bg = self.calculate_hr_zones_verdict(start_index=idx_start, end_index=idx_end)
                else:
                    vol_verdict, vol_color, vol_bg = self.calculate_hr_zones_verdict()
            elif has_zoom_range:
                # Quality lens with zoom — recalculate with slice
                vol_verdict, vol_color, vol_bg = self.calculate_volume_verdict(
                    start_index=idx_start, 
                    end_index=idx_end
                )
            else:
                # Quality lens — reset / autorange
                vol_verdict, vol_color, vol_bg = self.calculate_volume_verdict()
            
            # Update UI
            self.volume_verdict_label.set_text(f'{vol_verdict}')
            self.volume_verdict_label.classes(f'text-sm font-bold px-3 py-1 rounded {vol_bg}', remove='bg-emerald-500/20 bg-blue-500/20 bg-red-500/20 bg-orange-500/20 bg-amber-500/20 bg-slate-500/20 bg-zinc-700 bg-zinc-800 text-zinc-500')
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
            self.cadence_verdict_label.classes(f'text-sm font-bold px-3 py-1 rounded {cad_bg}', remove='bg-emerald-500/20 bg-blue-500/20 bg-yellow-500/20 bg-orange-500/20 bg-red-500/20 bg-zinc-700')
            self.cadence_verdict_label.style(f'color: {cad_color}; cursor: pointer;')
            self.cadence_verdict_label.on('click', lambda: self.show_form_info(highlight_verdict=cad_verdict), replace=True)
                
        except Exception as ex:
            # Silently catch errors
            pass
