"""
components/run_card.py
──────────────────────
Phase 2 Step 6: Feed run-card extraction from UltraStateApp.

Standalone NiceGUI run-card renderer with callback injection.
No app-state imports and no circular dependency on app.py.
"""
from datetime import datetime

from nicegui import ui

from analyzer import analyze_form
from constants import LOAD_CATEGORY, LOAD_CATEGORY_COLORS


def _hex_to_rgb(hex_color):
    """Convert #RRGGBB to 'R, G, B' for CSS variables."""
    hex_color = str(hex_color or '#000000').lstrip('#')
    if len(hex_color) != 6:
        return '0, 0, 0'
    return f"{int(hex_color[0:2], 16)}, {int(hex_color[2:4], 16)}, {int(hex_color[4:6], 16)}"


def create_run_card(
    activity,
    *,
    avg_ef,
    long_run_threshold,
    navigation_list,
    classify_run_type_cb,
    classify_aerobic_verdict_cb,
    callbacks,
):
    """
    Render a single activity card in the feed.

    Required callbacks in `callbacks`:
      - on_click(activity_hash, navigation_list)
      - on_copy(activity_hash)
      - on_te_info(te_label)
      - on_eff_info(verdict)

    Optional callbacks in `callbacks`:
      - on_load_info()
      - on_form_info(form_verdict)
    """
    callbacks = callbacks or {}
    on_click = callbacks.get('on_click')
    on_copy = callbacks.get('on_copy')
    on_te_info = callbacks.get('on_te_info')
    on_eff_info = callbacks.get('on_eff_info')
    on_load_info = callbacks.get('on_load_info')
    on_form_info = callbacks.get('on_form_info')

    date_str = activity.get('date', '')
    try:
        dt = datetime.strptime(date_str, '%Y-%m-%d %H:%M')
        formatted_date = dt.strftime('%a, %b %-d')
        formatted_time = dt.strftime('%-I:%M %p')
    except Exception:
        formatted_date = date_str
        formatted_time = ''

    run_type_tag = classify_run_type_cb(activity, long_run_threshold)

    moving_time_min = activity.get('moving_time_min', 0)
    avg_hr = activity.get('avg_hr', 0)
    max_hr = activity.get('max_hr', 185)

    intensity = avg_hr / max_hr if max_hr > 0 else 0
    if intensity < 0.65:
        factor = 1.0
    elif intensity < 0.75:
        factor = 1.5
    elif intensity < 0.85:
        factor = 3.0
    elif intensity < 0.92:
        factor = 6.0
    else:
        factor = 10.0

    strain = int(moving_time_min * factor)
    if strain < 75:
        strain_label = LOAD_CATEGORY.RECOVERY
        strain_color = LOAD_CATEGORY_COLORS[LOAD_CATEGORY.RECOVERY]
        strain_text_color = LOAD_CATEGORY_COLORS[LOAD_CATEGORY.RECOVERY]
    elif strain < 150:
        strain_label = LOAD_CATEGORY.BASE
        strain_color = LOAD_CATEGORY_COLORS[LOAD_CATEGORY.BASE]
        strain_text_color = LOAD_CATEGORY_COLORS[LOAD_CATEGORY.BASE]
    elif strain < 300:
        strain_label = LOAD_CATEGORY.OVERLOAD
        strain_color = LOAD_CATEGORY_COLORS[LOAD_CATEGORY.OVERLOAD]
        strain_text_color = LOAD_CATEGORY_COLORS[LOAD_CATEGORY.OVERLOAD]
    else:
        strain_label = LOAD_CATEGORY.OVERREACHING
        strain_color = LOAD_CATEGORY_COLORS[LOAD_CATEGORY.OVERREACHING]
        strain_text_color = LOAD_CATEGORY_COLORS[LOAD_CATEGORY.OVERREACHING]

    strain_bg = f"{strain_color}26"
    strain_border = f"{strain_color}4D"
    strain_border_hover = f"{strain_color}80"
    strain_shadow = f"{strain_color}1A"
    strain_shadow_hover = f"{strain_color}33"
    strain_rgb = _hex_to_rgb(strain_color)

    activity_hash = activity.get('db_hash')

    card = ui.card().classes(
        'w-full p-4 glass-card interactive-card cursor-pointer relative overflow-hidden group'
    ).style(
        f'max-width: 720px; margin: 0 auto; '
        f'--theme-color-rgb: {strain_rgb}; '
        f'--strain-bg: {strain_bg}; '
        f'--strain-border: {strain_border}; '
        f'--strain-border-hover: {strain_border_hover}; '
        f'--strain-shadow: {strain_shadow}; '
        f'--strain-shadow-hover: {strain_shadow_hover};'
    )

    if activity_hash and on_click:
        card.on('click', lambda h=activity_hash, nl=navigation_list: on_click(h, nl))

    with card:
        with ui.element('div').classes(
            'absolute top-3 right-3 z-30 '
            'opacity-0 group-hover:opacity-50 '
            'hover:!opacity-100 hover:scale-110 hover:rotate-12 '
            'transition-all duration-200 cursor-pointer bg-transparent rounded-full'
        ).on('click.stop', lambda e, h=activity_hash: on_copy and on_copy(h)):
            ui.tooltip('Copy Analysis to AI').classes('bg-zinc-800 text-xs font-bold shadow-lg')
            ui.icon('auto_awesome', size='28px').classes('text-white')

        with ui.column().classes('w-full gap-1'):
            with ui.row().classes('w-full items-start pr-10'):
                with ui.column().classes('gap-0'):
                    ui.label(formatted_date).classes('font-bold text-zinc-200 text-sm group-hover:text-white transition-colors')
                    ui.label(formatted_time).classes('text-xs text-zinc-500 group-hover:text-zinc-400 transition-colors')

            with ui.row().classes('w-full items-center gap-2 mt-2'):
                for tag in run_type_tag.split(' | '):
                    ui.label(tag).classes(
                        'text-[10px] font-bold px-2 py-0.5 rounded bg-zinc-800 text-zinc-400 border border-zinc-700 tracking-wide '
                        'cursor-pointer hover:brightness-125 hover:-translate-y-px hover:shadow-lg transition-all duration-200'
                    )

                te_label = activity.get('te_label')
                if te_label and str(te_label) != 'None':
                    te_color = activity.get('te_label_color', 'text-zinc-400')

                    if 'text-purple-400' in te_color:
                        bg_color, border_color = 'bg-purple-500/10', 'border-purple-500/30'
                    elif 'text-red-400' in te_color:
                        bg_color, border_color = 'bg-red-500/10', 'border-red-500/30'
                    elif 'text-orange-400' in te_color:
                        bg_color, border_color = 'bg-orange-500/10', 'border-orange-500/30'
                    elif 'text-emerald-400' in te_color:
                        bg_color, border_color = 'bg-emerald-500/10', 'border-emerald-500/30'
                    elif 'text-blue-400' in te_color:
                        bg_color, border_color = 'bg-blue-500/10', 'border-blue-500/30'
                    else:
                        bg_color, border_color = 'bg-zinc-800', 'border-zinc-700'

                    text_color = te_color.split()[0] if ' ' in te_color else te_color
                    physio_tag = ui.label(te_label).classes(
                        f'text-[10px] font-bold px-2 py-0.5 rounded {bg_color} border {border_color} {text_color} '
                        'tracking-wide cursor-pointer hover:brightness-125 transition-all'
                    )
                    physio_tag.on('click.stop', lambda t=te_label: on_te_info and on_te_info(t))

            with ui.row().classes('w-full gap-4 mb-1 items-center mt-3'):
                with ui.column().classes('flex-1'):
                    with ui.grid(columns=3).classes('w-full gap-3'):
                        with ui.column().classes('gap-0').style('line-height: 1.1;'):
                            with ui.row().classes('items-center gap-1'):
                                ui.icon('straighten').classes('text-blue-400 text-xs')
                                ui.label('DISTANCE').classes('text-[10px] text-gray-500 font-bold tracking-wider')
                            ui.label(f"{activity.get('distance_mi', 0):.1f} mi").classes('text-lg font-bold').style('line-height: 1;')

                        with ui.column().classes('gap-0').style('line-height: 1.1;'):
                            with ui.row().classes('items-center gap-1'):
                                ui.icon('terrain').classes('text-green-400 text-xs')
                                ui.label('ELEVATION').classes('text-[10px] text-gray-500 font-bold tracking-wider')
                            ui.label(f"{activity.get('elevation_ft', 0)} ft").classes('text-lg font-bold').style('line-height: 1;')

                        with ui.column().classes('gap-0').style('line-height: 1.1;'):
                            with ui.row().classes('items-center gap-1'):
                                ui.icon('speed').classes('text-purple-400 text-xs')
                                ui.label('PACE').classes('text-[10px] text-gray-500 font-bold tracking-wider')
                            ui.label(activity.get('pace', '--')).classes('text-lg font-bold').style('line-height: 1;')

                ui.element('div').classes('h-full').style('width: 1px; background-color: #27272a; margin: 0 8px;')

                with ui.column().classes('items-center justify-center gap-1 mr-2'):
                    with ui.element('div').classes('relative'):
                        ui.circular_progress(value=min(strain / 500, 1.0), size='80px', color=strain_color, show_value=False)
                        with ui.element('div').classes('absolute inset-0 flex items-center justify-center'):
                            ui.label(str(strain)).classes('text-xl font-bold')
                    with ui.row().classes('items-center gap-1'):
                        load_info_icon = ui.icon('help_outline').classes('text-zinc-600 hover:text-white text-[10px] cursor-pointer')
                        if on_load_info:
                            load_info_icon.on('click.stop', lambda: on_load_info())
                        ui.label('LOAD:').classes('text-xs text-zinc-300 font-bold uppercase tracking-widest')
                        ui.label(strain_label).classes('text-sm font-bold').style(f'color: {strain_text_color};')

            ui.separator().classes('my-1').style('background-color: #52525b; height: 1px;')
            with ui.row().classes('w-full justify-between items-center'):
                with ui.row().classes('gap-6'):
                    with ui.column().classes('items-center gap-0'):
                        ui.label('CALORIES').classes('text-xs text-gray-500 font-bold tracking-wider mb-0.5')
                        with ui.row().classes('items-center gap-1'):
                            ui.label(f"🔥 {int(activity.get('calories') or 0)} cal").classes('text-sm font-bold text-white')

                    with ui.column().classes('items-center gap-0'):
                        ui.label('AVG HR').classes('text-xs text-gray-500 font-bold tracking-wider mb-0.5')
                        with ui.row().classes('items-center gap-1'):
                            ui.icon('favorite').classes('text-pink-400 text-sm')
                            ui.label(f"{activity.get('avg_hr', 0)}").classes('text-sm font-bold text-white')

                    with ui.column().classes('items-center gap-0'):
                        ui.label('CADENCE').classes('text-xs text-gray-500 font-bold tracking-wider mb-0.5')
                        with ui.row().classes('items-center gap-1'):
                            ui.icon('directions_run').classes('text-blue-400 text-sm')
                            ui.label(f"{activity.get('avg_cadence', 0)}").classes('text-sm font-bold text-white')

                with ui.row().classes('items-center gap-2 flex-wrap'):
                    run_ef = activity.get('efficiency_factor', 0)
                    run_cost = activity.get('decoupling', 0)
                    aero_verdict, aero_bg, aero_border, aero_text, aero_icon = classify_aerobic_verdict_cb(
                        run_ef,
                        run_cost,
                        avg_ef=avg_ef,
                    )

                    hover_bg = aero_bg.replace('/10', '/30')
                    aero_pill = ui.row().classes(
                        f'items-center gap-2 px-3 py-1.5 rounded border {aero_bg} {aero_border} cursor-pointer hover:{hover_bg} transition-all'
                    )
                    with aero_pill:
                        ui.label(aero_icon).classes('text-sm')
                        ui.label('Efficiency:').classes(f'text-xs font-bold {aero_text}')
                        ui.label(aero_verdict).classes(f'text-xs font-bold {aero_text}')
                    aero_pill.on('click.stop', lambda av=aero_verdict: on_eff_info and on_eff_info(av))

                    form = analyze_form(
                        activity.get('avg_cadence'),
                        activity.get('avg_stance_time'),
                        activity.get('avg_step_length'),
                        activity.get('avg_vertical_oscillation'),
                    )
                    if form['verdict'] != 'ANALYZING':
                        if form['verdict'] == 'ELITE FORM':
                            pill_bg, pill_border, pill_text = 'bg-emerald-500/10', 'border-emerald-700/30', 'text-emerald-400'
                        elif form['verdict'] == 'GOOD FORM':
                            pill_bg, pill_border, pill_text = 'bg-blue-500/10', 'border-blue-700/30', 'text-blue-400'
                        elif form['verdict'] == 'HEAVY FEET':
                            pill_bg, pill_border, pill_text = 'bg-orange-500/10', 'border-orange-700/30', 'text-orange-400'
                        elif form['verdict'] == 'PLODDING':
                            pill_bg, pill_border, pill_text = 'bg-yellow-500/10', 'border-yellow-700/30', 'text-yellow-400'
                        elif form['verdict'] == 'HIKING / REST':
                            pill_bg, pill_border, pill_text = 'bg-blue-500/10', 'border-blue-700/30', 'text-blue-400'
                        else:
                            pill_bg, pill_border, pill_text = 'bg-slate-500/10', 'border-slate-700/30', 'text-slate-400'

                        form_hover_bg = pill_bg.replace('/10', '/30')
                        form_pill = ui.row().classes(
                            f'items-center gap-2 px-3 py-1.5 rounded border {pill_bg} {pill_border} cursor-pointer hover:{form_hover_bg} transition-all'
                        )
                        with form_pill:
                            ui.label('🦶').classes('text-sm')
                            ui.label('Form:').classes(f'text-xs font-bold {pill_text}')
                            ui.label(form['verdict'].title()).classes(f'text-xs font-bold {pill_text}')
                        if on_form_info:
                            form_pill.on('click.stop', lambda fv=form['verdict']: on_form_info(fv))

    return card
