"""
components/library_modal.py
──────────────────────────
Phase 2 Step 5: Library settings extraction from UltraStateApp.

Owns:
  • Sidebar library status row
  • Library settings dialog
  • Library status polling + sync side effects
  • Library folder/file pickers and manual sync/import handlers

Does NOT own:
  • App startup/shutdown orchestration (UltraStateApp.start_library_services/stop_library_services)
  • Database or LibraryManager lifecycle creation (injected)
  • Main app data refresh implementation (injected callback)
"""
from __future__ import annotations

import asyncio
import inspect
import os
import subprocess
import sys
from datetime import datetime

from nicegui import ui


class LibraryModal:
    """Encapsulates the library sidebar row + settings dialog UI and handlers."""

    def __init__(self, db, library_manager, state, on_library_changed_cb, disable_controls_cb=None):
        """
        Parameters
        ----------
        db : DatabaseManager
            Injected DB handle.
        library_manager : LibraryManager
            Injected library sync manager.
        state : AppState
            Injected app state for session id updates.
        on_library_changed_cb : callable
            Callback triggered after successful sync side effects
            (UltraStateApp.refresh_data_view).
        disable_controls_cb : callable | None
            Optional callback(bool) used to disable/enable app-level controls
            (e.g., timeframe select) while heavy library operations run.
        """
        self.db = db
        self.library_manager = library_manager
        self.state = state
        self.on_library_changed_cb = on_library_changed_cb
        self.disable_controls_cb = disable_controls_cb

        # Polling / dedupe state
        self._last_sync_status = None
        self._library_last_report_key = None
        self._library_widget_ready = False
        self._library_status_poll_in_progress = False

        # Sidebar row refs
        self.library_status_row = None
        self.library_status_row_tooltip = None
        self.library_status_row_label = None
        self.library_status_row_subtitle = None
        self.library_status_dot = None

        # Modal refs
        self.library_settings_dialog = None
        self.library_modal_status_label = None
        self.library_modal_status_dot = None
        self.library_modal_last_synced_label = None
        self.library_modal_summary_label = None
        self.library_modal_path_icon = None
        self.library_modal_path_label = None
        self.library_modal_path_tooltip = None
        self.library_modal_error_label = None
        self.library_modal_change_button = None
        self.library_modal_resync_button = None
        self.library_modal_import_button = None
        self.runs_count_label = None

    def render_status_row(self):
        """Render compact sidebar status row that opens Library settings."""
        self.library_status_row = ui.button(
            on_click=self.open,
        ).classes(
            'w-full px-1 py-1.5 rounded-md text-zinc-200 transition-colors library-status-row'
        ).props('flat no-caps no-ripple')
        with self.library_status_row:
            self.library_status_row_tooltip = ui.tooltip('Not configured').classes(
                'bg-zinc-900 text-white text-xs shadow-lg max-w-[24rem] break-all'
            )
            with ui.row().classes('w-full items-center gap-2 min-w-0'):
                ui.icon('folder').classes('text-zinc-400 text-sm shrink-0')
                self.library_status_row_label = ui.label('Library').classes('text-sm text-zinc-300 shrink-0')
                with ui.row().classes('items-center gap-1 min-w-0'):
                    self.library_status_dot = ui.label('●').classes('text-[10px] text-zinc-500')
                    self.library_status_row_subtitle = ui.label('Not configured').classes(
                        'text-sm text-zinc-300 truncate'
                    )

    def build(self):
        """Build the library settings modal UI."""
        self.library_settings_dialog = ui.dialog()
        with self.library_settings_dialog, ui.card().classes(
            'relative bg-zinc-900/98 border border-zinc-700/90 rounded-2xl p-6 w-[520px] max-w-[92vw] shadow-2xl backdrop-blur-md'
        ):
            ui.button(
                icon='close',
                on_click=self.library_settings_dialog.close,
                color=None,
            ).props('flat round dense no-ripple').style(
                'color: #9ca3af !important; position: absolute; top: 12px; right: 12px; z-index: 10;'
            )

            with ui.column().classes('w-full gap-1 mb-4'):
                ui.label('Library Settings').classes('text-xl font-bold text-white tracking-tight')

                with ui.row().classes('items-center gap-2'):
                    ui.label('Status:').classes('text-sm text-zinc-500')
                    self.library_modal_status_dot = ui.label('●').classes('text-[10px] text-zinc-500')
                    self.library_modal_status_label = ui.label('Not configured').classes('text-sm text-zinc-400')
                    self.library_modal_resync_button = ui.button(
                        icon='sync',
                        on_click=self.handle_library_resync,
                        color=None,
                    ).props('flat round dense no-ripple').classes('text-zinc-500 hover:text-white transition-colors')
                    self.library_modal_resync_button.tooltip('Resync now')
                self.library_modal_last_synced_label = ui.label('Last synced: Never').classes('text-xs text-zinc-500')

            with ui.row().classes('w-full items-center justify-between mb-6 bg-zinc-800/50 rounded-xl p-3 border border-zinc-700/30'):
                with ui.row().classes('items-center gap-3'):
                    with ui.column().classes('gap-0'):
                        ui.label('TOTAL ACTIVITIES').classes('text-[10px] font-bold tracking-wider text-zinc-500')
                        self.runs_count_label = ui.label(f'{self.db.get_count()}').classes('text-lg font-bold text-white leading-none')

                self.library_modal_summary_label = ui.label('').classes('text-xs text-zinc-400 text-right')

            with ui.column().classes('w-full mb-6'):
                ui.label('LIBRARY FOLDER').classes('text-[10px] font-bold tracking-wider text-zinc-500 mb-2')
                with ui.row().classes('w-full bg-zinc-950/50 border border-zinc-700/50 rounded-lg p-3 items-center'):
                    self.library_modal_path_icon = ui.icon('folder_open').classes('text-zinc-500 text-sm mr-2')
                    self.library_modal_path_label = ui.label('⚠ No folder selected').classes(
                        'flex-1 truncate font-mono text-xs text-zinc-300'
                    )
                    self.library_modal_path_label.style(
                        'display: block; overflow: hidden; text-overflow: ellipsis; white-space: nowrap;'
                    )
                    self.library_modal_path_tooltip = self.library_modal_path_label.tooltip('⚠ No folder selected')

            self.library_modal_error_label = ui.label('').classes(
                'text-xs text-red-400 mb-4 hidden bg-red-500/10 p-2 rounded-lg w-full border border-red-500/20'
            )

            with ui.row().classes('w-full gap-3 pt-2'):
                self.library_modal_change_button = ui.button(
                    'Change Folder',
                    on_click=self.handle_set_library_folder,
                ).props('flat no-caps no-ripple').classes(
                    'flex-1 bg-zinc-800 hover:bg-zinc-700 text-white rounded-xl border border-zinc-700/50 shadow-sm transition-all duration-200 font-medium py-2'
                )

                self.library_modal_import_button = ui.button(
                    'Import .FIT',
                    on_click=self.handle_library_manual_import,
                    icon='upload_file',
                    color=None,
                ).props('flat no-caps no-ripple').classes(
                    'flex-1 bg-white hover:bg-zinc-200 text-zinc-900 rounded-xl border border-zinc-200 shadow-sm transition-all duration-200 font-bold py-2'
                )

        self._library_widget_ready = True

    async def open(self):
        """Open modal and ensure status values are fresh."""
        await self.refresh_status()
        if self.library_settings_dialog:
            self.library_settings_dialog.open()

    @staticmethod
    def _status_dot_classes(status_name):
        if status_name == 'setup':
            return 'text-[10px] text-amber-500'
        if status_name == 'syncing':
            return 'text-[10px] text-amber-300 animate-pulse'
        if status_name == 'error':
            return 'text-[10px] text-red-400'
        if status_name == 'synced':
            return 'text-[10px] text-emerald-400'
        return 'text-[10px] text-zinc-500'

    @staticmethod
    def _format_sync_timestamp(finished_at):
        if not finished_at:
            return 'Never'
        try:
            parsed = datetime.fromisoformat(finished_at.replace('Z', '+00:00')).astimezone()
        except Exception:
            return 'Unknown'

        hour_12 = parsed.strftime('%I').lstrip('0') or '12'
        return f"{parsed.strftime('%a, %b')} {parsed.day} at {hour_12}:{parsed.strftime('%M %p')}"

    @staticmethod
    def _sync_report_key(report):
        if not report:
            return None
        reason = report.reason.value if hasattr(report.reason, 'value') else str(report.reason)
        reprocessed = getattr(report, 'reprocessed_upgraded', 0)
        reprocess_failed = getattr(report, 'reprocess_failed', 0)
        reprocess_missing = getattr(report, 'reprocess_missing_source', 0)
        return (
            f'{reason}|{report.finished_at}|{report.imported_new}|{reprocessed}|'
            f'{report.failed}|{reprocess_failed}|{reprocess_missing}|{len(report.errors)}'
        )

    async def refresh_status(self):
        """Poll library manager status and keep sidebar row + modal in sync."""
        if not self._library_widget_ready:
            return
        if self._library_status_poll_in_progress:
            return

        self._library_status_poll_in_progress = True
        try:
            status = await self.library_manager.get_status()
            report = status.last_report
            has_root = bool(status.library_root)
            root_exists = bool(status.library_root and os.path.isdir(status.library_root))
            root_readable = bool(root_exists and os.access(status.library_root, os.R_OK))
            root_error = bool(has_root and (not root_exists or not root_readable))
            report_failed = bool(
                report and (
                    int(getattr(report, 'failed', 0) or 0) > 0
                    or int(getattr(report, 'reprocess_failed', 0) or 0) > 0
                    or bool(getattr(report, 'errors', None))
                )
            )
            # skipped_unsupported is intentional behavior, not an error.
            has_error = bool(root_error or report_failed)

            status_name = 'idle'
            if not has_root:
                status_name = 'setup'
            elif has_error:
                status_name = 'error'
            elif status.sync_in_progress:
                status_name = 'syncing'
            elif has_root:
                status_name = 'synced'

            status_value = '⚠ Setup Library'
            if status_name == 'error':
                status_value = 'Sync Error'
            elif status_name == 'syncing':
                status_value = 'Syncing...'
            elif status_name == 'synced':
                status_value = 'Synced'

            row_subtitle = '⚠ Setup Library'
            if status_name == 'error':
                row_subtitle = '⚠ Sync Error'
            elif status_name == 'syncing':
                row_subtitle = 'Syncing...'
            elif status_name == 'synced':
                row_subtitle = 'Synced'

            summary_text = ''
            if has_root and status.sync_in_progress:
                summary_text = 'Upgrading older runs...'
            elif has_root and report and not status.sync_in_progress:
                if report.imported_new > 0:
                    summary_text = f'+{report.imported_new} new activities'
                elif getattr(report, 'reprocessed_upgraded', 0) > 0:
                    summary_text = f'Upgraded {report.reprocessed_upgraded} older run(s)'
                elif report.failed > 0 or getattr(report, 'reprocess_failed', 0) > 0:
                    total_failures = report.failed + getattr(report, 'reprocess_failed', 0)
                    summary_text = f'{total_failures} failed'
                elif getattr(report, 'skipped_unsupported', 0) > 0:
                    skipped = getattr(report, 'skipped_unsupported', 0)
                    summary_text = f'Skipped {skipped} unsupported file(s)'
                elif (
                    getattr(report, 'missing_files', 0) > 0
                    or getattr(report, 'reprocess_missing_source', 0) > 0
                ):
                    total_missing = (
                        getattr(report, 'missing_files', 0)
                        + getattr(report, 'reprocess_missing_source', 0)
                    )
                    summary_text = f'{total_missing} missing on disk'
                elif report.finished_at:
                    summary_text = ''

            last_synced_text = 'Last synced: Never'
            if report and report.finished_at:
                last_synced_text = f'Last synced: {self._format_sync_timestamp(report.finished_at)}'

            if self.library_status_row_label:
                self.library_status_row_label.text = 'Library'

            if self.library_status_row_subtitle:
                self.library_status_row_subtitle.text = row_subtitle
                self.library_status_row_subtitle.classes(
                    remove='text-zinc-500 text-zinc-300 text-amber-300 text-amber-400 text-red-300 text-emerald-400'
                )
                if status_name == 'error':
                    self.library_status_row_subtitle.classes(add='text-red-300')
                elif status_name == 'setup':
                    self.library_status_row_subtitle.classes(add='text-amber-400')
                else:
                    self.library_status_row_subtitle.classes(add='text-zinc-300')

            if self.library_status_dot:
                self.library_status_dot.classes(
                    remove='text-zinc-500 text-amber-300 text-amber-400 text-amber-500 text-red-400 text-emerald-400 animate-pulse'
                )
                self.library_status_dot.classes(add=self._status_dot_classes(status_name))

            if self.library_modal_status_dot:
                self.library_modal_status_dot.classes(
                    remove='text-zinc-500 text-amber-300 text-amber-400 text-amber-500 text-red-400 text-emerald-400 animate-pulse'
                )
                self.library_modal_status_dot.classes(add=self._status_dot_classes(status_name))

            if self.library_status_row_tooltip:
                if root_error:
                    self.library_status_row_tooltip.text = (
                        f'Library folder is missing or unreadable: {status.library_root}'
                    )
                else:
                    self.library_status_row_tooltip.text = status.library_root or 'Not configured'

            if self.library_modal_status_label:
                self.library_modal_status_label.text = status_value

            if self.library_modal_last_synced_label:
                self.library_modal_last_synced_label.text = last_synced_text

            if self.library_modal_summary_label:
                self.library_modal_summary_label.text = summary_text or ' '

            if self.library_modal_path_label:
                display_path = status.library_root or '⚠ No folder selected'
                self.library_modal_path_label.text = display_path
                self.library_modal_path_label.classes(remove='text-zinc-300 text-amber-500')
                if not has_root:
                    self.library_modal_path_label.classes(add='text-amber-500')
                else:
                    self.library_modal_path_label.classes(add='text-zinc-300')

                if self.library_modal_path_tooltip:
                    self.library_modal_path_tooltip.text = display_path

            if self.library_modal_path_icon:
                self.library_modal_path_icon.classes(remove='text-zinc-500 text-amber-500')
                if not has_root:
                    self.library_modal_path_icon.classes(add='text-amber-500')
                else:
                    self.library_modal_path_icon.classes(add='text-zinc-500')

            if self.library_modal_error_label:
                if has_error and status_name not in ('setup', 'syncing'):
                    if root_error:
                        error_text = 'Configured library folder is missing or unreadable. Choose a valid folder.'
                    elif report and report.errors:
                        error_text = report.errors[0]
                    else:
                        error_text = 'Last sync failed.'
                    if len(error_text) > 240:
                        error_text = f'{error_text[:237]}...'
                    self.library_modal_error_label.text = error_text
                    self.library_modal_error_label.classes(remove='hidden')
                else:
                    self.library_modal_error_label.text = ''
                    self.library_modal_error_label.classes(add='hidden')

            current_btn_state = (status.sync_in_progress, has_root)
            if current_btn_state != self._last_sync_status:
                self._last_sync_status = current_btn_state

                if self.library_modal_change_button:
                    if status.sync_in_progress:
                        self.library_modal_change_button.props(add='disable')
                    else:
                        self.library_modal_change_button.props(remove='disable')
                    self.library_modal_change_button.text = 'Change Folder' if has_root else 'Select Folder'

                if self.library_modal_resync_button:
                    self.library_modal_resync_button.props(remove='loading')
                    if has_root and not status.sync_in_progress:
                        self.library_modal_resync_button.props(remove='disable')
                    else:
                        self.library_modal_resync_button.props(add='disable')
                    if status.sync_in_progress:
                        self.library_modal_resync_button.props(add='loading')

                if self.library_modal_import_button:
                    if status.sync_in_progress:
                        self.library_modal_import_button.props(add='disable')
                    else:
                        self.library_modal_import_button.props(remove='disable')

                    self.library_modal_import_button.classes(
                        remove='bg-white bg-zinc-800 hover:bg-zinc-200 hover:bg-zinc-700 text-zinc-900 text-white border-zinc-200 border-zinc-700/50 opacity-50'
                    )
                    if has_root:
                        self.library_modal_import_button.classes(
                            add='bg-white hover:bg-zinc-200 text-zinc-900 border-zinc-200'
                        )
                    else:
                        self.library_modal_import_button.classes(
                            add='bg-zinc-800 hover:bg-zinc-700 text-white border-zinc-700/50 opacity-50'
                        )

            report_key = self._sync_report_key(report)
            if report_key and report_key != self._library_last_report_key:
                self._library_last_report_key = report_key
                await self._apply_library_sync_side_effects(report, notify_user=False)
        finally:
            self._library_status_poll_in_progress = False

    async def _apply_library_sync_side_effects(self, report, notify_user=False):
        """Apply data/UI updates after a completed sync report."""
        if not report:
            return

        imported_new = int(getattr(report, 'imported_new', 0) or 0)
        reprocessed_upgraded = int(getattr(report, 'reprocessed_upgraded', 0) or 0)
        reprocess_failed = int(getattr(report, 'reprocess_failed', 0) or 0)
        reprocess_missing = int(getattr(report, 'reprocess_missing_source', 0) or 0)
        missing_files = int(getattr(report, 'missing_files', 0) or 0)
        skipped_unsupported = int(getattr(report, 'skipped_unsupported', 0) or 0)

        self.state.session_id = self.db.get_last_session_id()
        self.update_run_count()

        if imported_new > 0 or reprocessed_upgraded > 0:
            maybe = self.on_library_changed_cb()
            if inspect.isawaitable(maybe):
                await maybe

        if not notify_user:
            return

        if report.failed > 0 or reprocess_failed > 0:
            total_failures = int(report.failed) + reprocess_failed
            ui.notify(
                f'Sync finished with {total_failures} error(s). Imported {imported_new} new file(s).',
                type='warning',
            )
        elif imported_new > 0 or reprocessed_upgraded > 0:
            fragments = []
            if imported_new > 0:
                fragments.append(f'imported {imported_new} new file(s)')
            if reprocessed_upgraded > 0:
                fragments.append(f'upgraded {reprocessed_upgraded} older run(s)')
            if skipped_unsupported > 0:
                fragments.append(f'skipped {skipped_unsupported} unsupported file(s)')
            ui.notify(f"Sync complete: {'; '.join(fragments)}.", type='positive')
        elif skipped_unsupported > 0:
            ui.notify(
                f'Skipped {skipped_unsupported} file(s) (unsupported activities like cycling or swimming).',
                type='info',
            )
        elif missing_files > 0 or reprocess_missing > 0:
            total_missing = missing_files + reprocess_missing
            ui.notify(
                f'Sync complete: {total_missing} file(s) missing on disk. Activities were kept.',
                type='warning',
            )
        else:
            ui.notify('Sync complete: no new files found.', type='info')

    def update_run_count(self):
        """Refresh total activities label in the library modal."""
        if self.runs_count_label:
            self.runs_count_label.text = f'{self.db.get_count()}'

    async def _set_controls_disabled(self, disabled: bool):
        if not self.disable_controls_cb:
            return
        maybe = self.disable_controls_cb(disabled)
        if inspect.isawaitable(maybe):
            await maybe

    def _set_modal_action_buttons_disabled(self, disabled: bool):
        action = 'add' if disabled else 'remove'
        if self.library_modal_change_button:
            self.library_modal_change_button.props(**{action: 'disable'})
        if self.library_modal_resync_button:
            self.library_modal_resync_button.props(**{action: 'disable'})
        if self.library_modal_import_button:
            self.library_modal_import_button.props(**{action: 'disable'})

    async def handle_set_library_folder(self):
        """Choose and persist library root, then trigger immediate sync."""
        folder_path = await self._choose_folder_path()
        if not folder_path:
            return

        await self._set_controls_disabled(True)
        self._set_modal_action_buttons_disabled(True)
        try:
            await self.library_manager.set_library_root(folder_path)
            status = await self.library_manager.get_status()
            report = status.last_report
            self._library_last_report_key = self._sync_report_key(report)
            await self._apply_library_sync_side_effects(report, notify_user=True)
            await self.refresh_status()

            if report and not getattr(report, 'errors', None) and self.library_settings_dialog:
                self.library_settings_dialog.close()
        except Exception as e:
            ui.notify(f'Failed to set library folder: {e}', type='negative')
        finally:
            await self._set_controls_disabled(False)
            self._set_modal_action_buttons_disabled(False)

    async def handle_library_resync(self):
        """Run a manual sync and refresh views if new files were ingested."""
        await self._set_controls_disabled(True)
        self._set_modal_action_buttons_disabled(True)
        try:
            report = await self.library_manager.resync_now()
            self._library_last_report_key = self._sync_report_key(report)
            await self._apply_library_sync_side_effects(report, notify_user=True)
            await self.refresh_status()

            if report and not getattr(report, 'errors', None) and self.library_settings_dialog:
                self.library_settings_dialog.close()
        except Exception as e:
            ui.notify(f'Resync failed: {e}', type='negative')
        finally:
            await self._set_controls_disabled(False)
            self._set_modal_action_buttons_disabled(False)

    async def handle_library_manual_import(self):
        """Import one-off local FIT files through the same library ingest pipeline."""
        selected_paths = await self._choose_fit_file_paths()
        if not selected_paths:
            return

        await self._set_controls_disabled(True)
        self._set_modal_action_buttons_disabled(True)
        try:
            report = await self.library_manager.ingest_files(selected_paths)
            self._library_last_report_key = self._sync_report_key(report)
            await self._apply_library_sync_side_effects(report, notify_user=True)
            await self.refresh_status()
        except Exception as exc:
            ui.notify(f'Import failed: {exc}', type='negative')
        finally:
            await self._set_controls_disabled(False)
            self._set_modal_action_buttons_disabled(False)

    async def _choose_folder_path(self):
        """Open platform-specific folder picker and return selected path (or None)."""
        if sys.platform == 'darwin':
            try:
                cmd = (
                    "osascript -e 'tell application \"System Events\" to activate' "
                    "-e 'POSIX path of (choose folder with prompt \"Set Library Folder\")'"
                )
                process = await asyncio.create_subprocess_shell(
                    cmd,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                )
                stdout, _stderr = await process.communicate()
                selected = stdout.decode().strip()
                return selected or None
            except Exception as e:
                ui.notify(f'Folder picker failed: {e}', type='negative')
                return None

        script = """
import tkinter as tk
from tkinter import filedialog

root = tk.Tk()
root.withdraw()
root.wm_attributes('-topmost', 1)
folder_path = filedialog.askdirectory(title='Set Library Folder')
if folder_path:
    print(folder_path, end='')
root.destroy()
"""

        try:
            process = await asyncio.create_subprocess_exec(
                sys.executable,
                '-c',
                script,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
            )
            stdout, _stderr = await process.communicate()
            selected = stdout.decode().strip()
            return selected or None
        except Exception as e:
            ui.notify(f'Folder picker failed: {e}', type='negative')
            return None

    async def _choose_fit_file_paths(self):
        """Open platform-specific file picker for one or more FIT files."""
        if sys.platform == 'darwin':
            script_lines = [
                'tell application "System Events" to activate',
                'set selectedFiles to choose file with prompt "Import .FIT file(s)" '
                'of type {"fit"} with multiple selections allowed',
                'set output to ""',
                'repeat with selectedFile in selectedFiles',
                'set output to output & POSIX path of selectedFile & linefeed',
                'end repeat',
                'return output',
            ]
            try:
                proc = await asyncio.create_subprocess_exec(
                    'osascript',
                    *sum([['-e', line] for line in script_lines], []),
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                )
                stdout, stderr = await proc.communicate()
                if proc.returncode != 0:
                    error_text = stderr.decode().strip()
                    if 'User canceled' in error_text:
                        return []
                    ui.notify(f'File picker failed: {error_text}', type='negative')
                    return []

                selected = [line.strip() for line in stdout.decode().splitlines() if line.strip()]
                return [path for path in selected if path.lower().endswith('.fit')]
            except Exception as exc:
                ui.notify(f'File picker failed: {exc}', type='negative')
                return []

        script = """
import tkinter as tk
from tkinter import filedialog

root = tk.Tk()
root.withdraw()
root.wm_attributes('-topmost', 1)
paths = filedialog.askopenfilenames(
    title='Import .FIT file(s)',
    filetypes=[('FIT files', '*.fit'), ('All files', '*.*')],
)
for p in paths:
    print(p)
root.destroy()
"""
        try:
            proc = await asyncio.create_subprocess_exec(
                sys.executable,
                '-c',
                script,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
            )
            stdout, _stderr = await proc.communicate()
            selected = [line.strip() for line in stdout.decode().splitlines() if line.strip()]
            return [path for path in selected if path.lower().endswith('.fit')]
        except Exception as exc:
            ui.notify(f'File picker failed: {exc}', type='negative')
            return []
