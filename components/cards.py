"""
components/cards.py
───────────────────
Phase 2 Step 4: Render helpers extracted from UltraStateApp.

Standalone NiceGUI card-building functions. Each function receives
pure data arguments and renders into the current NiceGUI context.
No app state, no self references.
"""
from nicegui import ui
from analyzer import analyze_form, classify_split

def create_lap_splits_table(lap_data):
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



def create_decoupling_card(decoupling_data, efficiency_factor=None,
                           aerobic_verdict_cb=None, aerobic_info_cb=None):
    """
    Create card displaying aerobic decoupling metrics.

    Args:
        decoupling_data: Dictionary with decoupling metrics
        efficiency_factor: Optional EF value from the activity
        aerobic_verdict_cb: Optional callable(run_ef, run_decoupling) -> (verdict, bg, border, text, icon)
        aerobic_info_cb: Optional callable(highlight_verdict=...) that opens the info dialog

    Returns:
        NiceGUI card component
    """
    run_ef = efficiency_factor or 0
    run_decoupling = decoupling_data.get('decoupling_pct', 0)
    if aerobic_verdict_cb:
        aero_verdict, aero_bg, aero_border, aero_text, aero_icon = aerobic_verdict_cb(
        run_ef,
        run_decoupling,
    )
    else:
        aero_verdict, aero_bg, aero_border, aero_text, aero_icon = ('—', 'bg-zinc-800', 'border-zinc-700', 'text-zinc-400', '📊')

    with ui.card().classes('bg-zinc-900 p-4 border border-zinc-800 h-full') as card:
        card.style('border-radius: 8px; box-shadow: 0px 4px 12px rgba(0, 0, 0, 0.4);')

        # Title with info icon
        with ui.row().classes('items-center gap-2 mb-2'):
            ui.label('AEROBIC EFFICIENCY').classes('text-lg font-bold text-white')
            ae_info_icon = ui.icon('help_outline').classes('text-zinc-500 hover:text-white cursor-pointer text-base transition-colors ml-auto')
            if aerobic_info_cb:
                ae_info_icon.on('click', lambda av=aero_verdict: aerobic_info_cb(highlight_verdict=av))

        # Verdict pill (visual indicator only; modal opens from header info icon)
        aero_pill = ui.row().classes(
            f'items-center gap-2 px-2.5 py-1 rounded border {aero_bg} {aero_border} mb-3 w-fit'
        )
        with aero_pill:
            ui.label(aero_icon).classes('text-sm')
            ui.label(aero_verdict).classes(f'text-xs font-bold {aero_text}')

        # Compact two-block layout to reduce total card height
        with ui.row().classes('w-full gap-2'):
            if efficiency_factor and efficiency_factor > 0:
                with ui.column().classes('flex-1 min-w-0 gap-0 bg-zinc-950/70 border border-zinc-800 rounded px-3 py-2'):
                    ui.label('EFFICIENCY FACTOR').classes('text-[10px] text-zinc-500 uppercase tracking-wider font-semibold')
                    with ui.row().classes('items-baseline gap-2'):
                        ui.label(f'{efficiency_factor:.2f}').classes('text-2xl font-bold text-white leading-none')
                        ui.label('speed/HR').classes('text-[11px] text-zinc-500 font-bold')

            with ui.column().classes('flex-1 min-w-0 gap-0 bg-zinc-950/70 border border-zinc-800 rounded px-3 py-2'):
                ui.label('AEROBIC DECOUPLING').classes('text-[10px] text-zinc-500 uppercase tracking-wider font-semibold')
                with ui.row().classes('items-baseline gap-2'):
                    ui.label(f"{decoupling_data['decoupling_pct']:.1f}%").classes('text-2xl font-bold leading-none').style(
                        f"color: {decoupling_data['color']};"
                    )
                    ui.label(decoupling_data['status']).classes('text-base font-bold').style(
                        f"color: {decoupling_data['color']};"
                    )

        # Detail Metrics (compact single line)
        ui.label(
            f"1st Half EF: {decoupling_data['ef_first_half']:.4f}  |  2nd Half EF: {decoupling_data['ef_second_half']:.4f}"
        ).classes('text-[10px] text-zinc-600 mt-2')

    return card



def create_physiology_card(session_data, activity):
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
                        
                        with ui.row().classes('items-baseline gap-2'):
                            ui.label(f"{score}").classes('text-3xl font-bold font-mono leading-none').style(f'color: {hrr_color};')
                            ui.label('bpm').classes('text-xs text-zinc-500 font-bold')
        
    return card



def create_running_dynamics_card(session_data, form_info_cb=None):
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
            if form_info_cb:
                rm_info_icon.on('click', lambda: form_info_cb())
        
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




def create_strategy_row(run_walk_stats, terrain_stats):
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





