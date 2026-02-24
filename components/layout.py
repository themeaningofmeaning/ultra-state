"""
components/layout.py
────────────────────
Phase 2 final step: application shell extraction from UltraStateApp.

Owns:
  • Sidebar scaffolding (branding, timeframe, actions, library entry)
  • Header/tabs scaffolding (tab row + save-chart button)
  • Activities filter-bar rendering and filter toggling
  • Floating action bar (FAB) rendering

Does not own:
  • Domain/business handlers (injected callbacks)
  • Data loading or dataframe computation (injected via set_data)
"""
from __future__ import annotations

from nicegui import ui

from constants import DEFAULT_TIMEFRAME, TIMEFRAME_OPTIONS


class AppShell:
    """Encapsulates app-level shell scaffolding and shell-owned UI state."""

    def __init__(self, state, library_modal, callbacks=None, tag_config=None):
        self.state = state
        self.library_modal = library_modal
        self.callbacks = callbacks or {}
        self.tag_config = tag_config or {}

        self.df = None
        self.activities_data = []

        # Sidebar refs
        self.timeframe_select = None
        self.focus_token = None
        self.focus_token_label = None
        self.export_btn = None
        self.copy_btn = None
        self.copy_btn_label = None
        self.copy_loading_dialog = None
        self.copy_loading_progress = None
        self.copy_loading_status_label = None

        # Main-content refs
        self.save_chart_btn = None
        self.feed_container = None
        self.filter_container = None
        self.grid_container = None

        # FAB ref
        self.fab_container = None

    def set_data(self, df, activities_data):
        """Update data used for tag filter derivation in the shell."""
        self.df = df
        self.activities_data = activities_data or []

    def _invoke_callback(self, name, *args, **kwargs):
        cb = self.callbacks.get(name)
        if not callable(cb):
            return None
        return cb(*args, **kwargs)

    def build(self):
        """Build the full shell (sidebar + header/tabs + FAB)."""
        with ui.row().classes('w-full h-screen m-0 p-0 gap-0 no-wrap overflow-hidden'):
            self.build_sidebar()
            self.build_main_content()

        self.fab_container = ui.row().classes(
            'fixed bottom-8 left-1/2 transform -translate-x-1/2 '
            'bg-zinc-900/90 backdrop-blur-lg text-white px-6 py-2 rounded-full '
            'shadow-2xl border border-zinc-800 border-t-white/10 z-50 items-center gap-4 '
            'transition-all duration-300 translate-y-[150%] opacity-0 pointer-events-none'
        )
        return self

    def build_sidebar(self):
        """Create fixed left sidebar with controls."""
        on_timeframe_change = self.callbacks.get('on_timeframe_change')
        on_exit_focus_mode = self.callbacks.get('on_exit_focus_mode')
        on_export_csv = self.callbacks.get('on_export_csv')
        on_copy_to_llm = self.callbacks.get('on_copy_to_llm')

        with ui.column().classes('w-56 bg-zinc-900 p-4 h-screen sticky top-0 flex-shrink-0'):
            ui.label('🏃‍♂️ Ultra State').classes('text-2xl font-black tracking-tight text-white mb-8')

            ui.label('TIMEFRAME').classes('text-xs text-gray-400 font-bold text-center mb-1')
            self.timeframe_select = ui.select(
                options=TIMEFRAME_OPTIONS,
                value=DEFAULT_TIMEFRAME,
                on_change=on_timeframe_change,
            ).classes('w-full mb-4 bg-zinc-900').style('color: white;').props(
                'outlined dense dark behavior="menu"'
            )

            self.focus_token = ui.button(
                color=None,
                on_click=on_exit_focus_mode,
            ).props('flat no-caps no-ripple').classes(
                'w-full mb-4 rounded flex items-center justify-between px-3 py-1.5 '
                'border border-zinc-700 bg-zinc-800 hover:bg-zinc-700 '
                'transition-all duration-200 hidden group'
            )
            with self.focus_token:
                self.focus_token_label = ui.label('🎯 Focus (0)').classes('text-white font-bold tracking-wide')
                ui.icon('clear').classes(
                    'text-zinc-400 group-hover:text-white transition-colors duration-200 text-lg'
                )

            ui.label('ACTIONS').classes('text-[10px] text-zinc-500 font-semibold tracking-[0.10em] mt-1 mb-1')
            with ui.column().classes('w-full gap-2'):
                self.export_btn = ui.button(
                    'EXPORT CSV',
                    on_click=on_export_csv,
                    icon='download',
                ).classes('w-full bg-zinc-800 text-white hover:bg-zinc-700').props('flat disable')

                self.copy_btn = ui.element('button').classes(
                    'ai-gradient-btn w-full text-white font-bold tracking-wide '
                    'transform transition-transform duration-300 hover:scale-[1.01] '
                    'flex items-center justify-center gap-2 py-2 px-4 rounded shadow-lg'
                ).style('cursor: pointer; opacity: 0.5; pointer-events: none;')
                if callable(on_copy_to_llm):
                    self.copy_btn.on('click.stop', on_copy_to_llm)

                with self.copy_btn:
                    ui.icon('auto_awesome').classes('text-white')
                    self.copy_btn_label = ui.label('COPY FOR AI').classes('text-white font-bold')

            ui.separator().classes('my-3 bg-zinc-800')

            self.copy_loading_dialog = ui.dialog().props('persistent')
            with self.copy_loading_dialog, ui.card().classes(
                'bg-zinc-900/95 backdrop-blur-xl border border-white/10 rounded-2xl '
                'shadow-2xl shadow-emerald-500/20 p-6 items-start'
            ).style('min-width: 360px; box-shadow: none;'):
                with ui.column().classes('w-full gap-4'):
                    ui.label('Constructing Analysis...').classes('text-white font-medium tracking-wide text-lg')
                    with ui.element('div').classes('w-full h-1.5 bg-zinc-800 rounded-full overflow-hidden'):
                        self.copy_loading_progress = ui.element('div').classes(
                            'h-full bg-gradient-to-r from-emerald-500 to-teal-400 transition-all duration-300'
                        ).style('width: 0%')
                    self.copy_loading_status_label = ui.label('Initializing...').classes(
                        'font-mono text-xs text-zinc-400'
                    )

            ui.element('div').classes('flex-grow')

            ui.separator().classes('my-3 border-zinc-800')
            if self.library_modal:
                self.library_modal.render_status_row()
                self.library_modal.build()

    def build_main_content(self):
        """Create tabbed main content area with scrolling."""
        on_save_chart = self.callbacks.get('on_save_chart')

        with ui.column().classes('flex-1 h-screen overflow-hidden p-0 gap-0'):
            with ui.column().classes('w-full min-h-full pt-6 px-6 pb-0 gap-4'):
                with ui.row().classes('w-full items-center mb-0 relative pb-3'):
                    with ui.tabs(on_change=lambda e: self.toggle_save_chart_button(e.value)).classes(
                        'ember-tabs w-full justify-center'
                    ).props('active-color="white" align="center" content-class="text-zinc-500"') as tabs:
                        trends_tab = ui.tab('Trends')
                        report_tab = ui.tab('FEED')
                        activities_tab = ui.tab('ACTIVITIES')

                    self.save_chart_btn = ui.button(
                        icon='download',
                        on_click=on_save_chart,
                        color=None,
                    ).classes('text-white absolute right-0 top-0 z-10').style(
                        'background-color: #27272a; border-radius: 6px; border: none; '
                        'padding: 8px; min-width: 40px; '
                        'transition: opacity 0.3s ease-in-out, background-color 0.2s ease;'
                    ).props('flat dense').tooltip('Save Chart')

                with ui.tab_panels(tabs, value=trends_tab).classes('w-full flex-1').props('transparent'):
                    with ui.tab_panel(trends_tab).classes('p-0'):
                        self._invoke_callback('on_build_trends_tab')

                    with ui.tab_panel(report_tab).classes('p-0'):
                        self.build_report_tab()

                    with ui.tab_panel(activities_tab).classes('p-0 h-full flex flex-col'):
                        self.build_activities_tab()

    def build_report_tab(self):
        """Create report tab container used by feed renderers."""
        with ui.scroll_area().classes('w-full h-full'):
            self.feed_container = ui.column().classes('w-full max-w-4xl mx-auto gap-4 p-4 pb-8')

    def build_activities_tab(self):
        """Create activities tab containers and initial filter/grid render."""
        with ui.column().classes('w-full flex-1 overflow-hidden p-8 gap-2'):
            self.filter_container = ui.row().classes('w-full mb-0 gap-2')
            self.grid_container = ui.column().classes('w-full flex-1 overflow-hidden')

        self.update_filter_bar()

    def toggle_save_chart_button(self, tab_name):
        """Show/hide save-chart button based on active tab."""
        if not self.save_chart_btn:
            return
        if tab_name == 'Trends':
            self.save_chart_btn.style('opacity: 1; pointer-events: auto;')
        else:
            self.save_chart_btn.style('opacity: 0; pointer-events: none;')

    def show_floating_action_bar(self, selected_rows):
        """Show FAB when rows are selected."""
        if not self.fab_container:
            return
        if not selected_rows:
            self.hide_floating_action_bar()
            return

        count = len(selected_rows)
        self.fab_container.clear()
        self.fab_container.classes(
            remove='translate-y-[150%] opacity-0 pointer-events-none',
            add='translate-y-0 opacity-100 pointer-events-auto',
        )

        with self.fab_container:
            with ui.element('div').classes(
                'flex items-center justify-center rounded-full px-3 py-0.5 mr-1 bg-zinc-800'
            ).style('border: 1px solid rgba(255, 255, 255, 0.1);'):
                ui.label(f'{count}').classes('font-bold text-base text-white')
            ui.label('Selected').classes('text-sm text-zinc-400 mr-3 font-medium')

            with ui.element('button').classes(
                'flex items-center justify-center gap-1.5 px-3 py-1 cursor-pointer rounded-md '
                'font-bold text-sm transition-transform duration-300 hover:scale-[1.05] '
                'text-white drop-shadow-[0_0_8px_rgba(255,255,255,0.4)] '
                'hover:drop-shadow-[0_0_12px_rgba(255,255,255,0.8)]'
            ).on('click.stop', lambda: self._invoke_callback('on_focus_selected', selected_rows)):
                ui.icon('center_focus_strong').classes('text-lg')
                ui.label('Focus').classes('tracking-wide')

            ui.element('div').classes('w-px h-5 mx-1').style('background: rgba(255,255,255,0.1);')

            ui.button(
                icon='download',
                color=None,
                on_click=lambda: self._invoke_callback('on_bulk_download', selected_rows),
            ).props('flat round dense').classes('text-zinc-400 hover:text-white')

            ui.button(
                icon='delete_outline',
                color=None,
                on_click=lambda: self._invoke_callback('on_bulk_delete', selected_rows),
            ).props('flat round dense').classes('text-zinc-400 hover:text-red-400')

    def hide_floating_action_bar(self):
        """Hide FAB."""
        if not self.fab_container:
            return
        self.fab_container.classes(
            remove='translate-y-0 opacity-100 pointer-events-auto',
            add='translate-y-[150%] opacity-0 pointer-events-none',
        )

    def _get_unique_tags_from_current_data(self):
        """Find available context + physio tags from currently loaded data."""
        context_tags = set()
        physio_tags = set()

        if self.df is not None and len(self.df) >= 5:
            long_run_threshold = self.df['distance_mi'].quantile(0.8)
        else:
            long_run_threshold = 10.0

        classify_run_type = self.callbacks.get('classify_run_type')

        for activity in self.activities_data:
            if callable(classify_run_type):
                context = classify_run_type(activity, long_run_threshold)
            else:
                context = None

            if context:
                for tag in str(context).split(' | '):
                    clean_tag = (
                        tag.replace('⛰️', '')
                        .replace('🔥', '')
                        .replace('🦅', '')
                        .replace('⚡', '')
                        .replace('🏃', '')
                        .replace('🧘', '')
                        .replace('🔷', '')
                        .strip()
                    )
                    if clean_tag:
                        context_tags.add(clean_tag)

            physio = activity.get('te_label')
            if physio and str(physio) != 'None':
                physio_tags.add(physio)

        return sorted(list(context_tags)), sorted(list(physio_tags))

    def update_filter_bar(self):
        """Render stacked activities filter controls (distance + tag groups)."""
        if self.filter_container is None:
            return

        self.filter_container.clear()
        available_context, available_physio = self._get_unique_tags_from_current_data()

        distance_filters = [
            {'id': 'all', 'label': 'All'},
            {'id': 'short', 'label': 'Short'},
            {'id': 'med', 'label': 'Medium'},
            {'id': 'long_dist', 'label': 'Long'},
        ]

        with self.filter_container:
            with ui.column().classes('w-full gap-3 mb-2'):
                with ui.row().classes('w-full items-center'):
                    with ui.row().classes('bg-zinc-900 border border-zinc-800 p-1 rounded-lg gap-1'):
                        for distance_filter in distance_filters:
                            if distance_filter['id'] == 'all':
                                is_active = not any(
                                    key in self.state.active_filters
                                    for key in ['short', 'med', 'long_dist']
                                )
                            else:
                                is_active = distance_filter['id'] in self.state.active_filters

                            if is_active:
                                classes = (
                                    'bg-zinc-800 text-white shadow-lg shadow-zinc-900/50 '
                                    'border border-zinc-700 font-bold'
                                )
                            else:
                                classes = 'bg-transparent text-zinc-500 hover:text-zinc-300 font-medium'

                            ui.button(
                                distance_filter['label'],
                                on_click=lambda filter_id=distance_filter['id']: self.toggle_filter(filter_id),
                                color=None,
                            ).props('flat dense no-caps ripple=False').classes(
                                f'rounded-md px-4 py-1 text-xs transition-all duration-200 no-ripple {classes}'
                            )

                with ui.row().classes('w-full gap-2 wrap items-center'):
                    for tag_name in available_context:
                        config = self.tag_config.get(tag_name, {'icon': '🏷️', 'color': 'zinc'})
                        color = config['color']
                        label = (
                            tag_name
                            if any(c in tag_name for c in ['🏃', '🔥', '⚡', '⛰️', '🧘', '🔷'])
                            else f"{config['icon']} {tag_name}"
                        )
                        is_active = tag_name in self.state.active_filters

                        if is_active:
                            classes = (
                                f'filter-active bg-{color}-500 text-white shadow-md '
                                f'border border-{color}-600/20 transform scale-105'
                            )
                        else:
                            classes = (
                                f'bg-zinc-800/20 text-zinc-300 border border-zinc-700 '
                                f'hover:bg-white/10 hover:border-{color}-500 hover:text-{color}-400 '
                                'hover:shadow-[0_0_15px_-3px_currentColor] hover:-translate-y-px '
                                'transition-all duration-300'
                            )

                        ui.button(
                            label.replace('🏷️', '🏔️'),
                            on_click=lambda tag=tag_name: self.toggle_filter(tag),
                            color=None,
                        ).props('flat dense no-caps ripple=False').classes(
                            f'rounded-full px-4 py-1 text-xs font-bold transition-all duration-200 no-ripple {classes}'
                        )

                    for tag_name in available_physio:
                        color = 'emerald'
                        if 'MAX' in tag_name or 'VO2' in tag_name:
                            color = 'fuchsia'
                        elif 'ANAEROBIC' in tag_name:
                            color = 'orange'
                        elif 'THRESHOLD' in tag_name:
                            color = 'amber'

                        is_active = tag_name in self.state.active_filters
                        if is_active:
                            classes = (
                                f'filter-active bg-{color}-500 text-white shadow-md '
                                f'border border-{color}-600/20'
                            )
                        else:
                            classes = (
                                f'bg-zinc-900 text-{color}-400 border border-{color}-500/50 '
                                f'hover:border-{color}-400 hover:shadow-[0_0_10px_currentColor] '
                                'hover:-translate-y-px transition-all duration-300'
                            )

                        ui.button(
                            tag_name,
                            on_click=lambda tag=tag_name: self.toggle_filter(tag),
                            color=None,
                        ).props('flat dense no-caps ripple=False').classes(
                            f'rounded-md px-3 py-1 text-[10px] font-bold tracking-wider '
                            f'uppercase transition-all duration-200 no-ripple {classes}'
                        )

    def toggle_filter(self, filter_id):
        """Toggle distance/tag filters and request a grid refresh."""
        distance_keys = {'short', 'med', 'long_dist'}

        if filter_id == 'all':
            self.state.active_filters -= distance_keys
        elif filter_id in distance_keys:
            if filter_id in self.state.active_filters:
                self.state.active_filters.remove(filter_id)
            else:
                self.state.active_filters -= distance_keys
                self.state.active_filters.add(filter_id)
        else:
            if filter_id in self.state.active_filters:
                self.state.active_filters.remove(filter_id)
            else:
                self.state.active_filters.add(filter_id)

        self.update_filter_bar()
        self._invoke_callback('on_update_activities_grid')
