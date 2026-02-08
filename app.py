"""
Garmin FIT Analyzer
"""

# Standard library imports
import os
import time
import json
import hashlib
import sqlite3
import asyncio
from datetime import datetime

# Third-party imports
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from nicegui import ui

# Local imports
from analyzer import FitAnalyzer


# --- DATABASE MANAGER (Preserved from gui.py) ---
class DatabaseManager:
    def __init__(self, db_path='runner_stats.db'):
        self.db_path = db_path
        self.create_tables()

    def get_connection(self):
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        return conn

    def create_tables(self):
        with self.get_connection() as conn:
            conn.execute('''
                CREATE TABLE IF NOT EXISTS activities (
                    hash TEXT PRIMARY KEY,
                    filename TEXT,
                    date TEXT,
                    json_data TEXT,
                    session_id INTEGER
                )
            ''')

    def activity_exists(self, file_hash):
        with self.get_connection() as conn:
            cursor = conn.execute("SELECT 1 FROM activities WHERE hash = ?", (file_hash,))
            return cursor.fetchone() is not None

    def insert_activity(self, activity_data, file_hash, session_id):
        json_str = json.dumps(activity_data, default=str)
        with self.get_connection() as conn:
            conn.execute(
                "INSERT OR REPLACE INTO activities (hash, filename, date, json_data, session_id) VALUES (?, ?, ?, ?, ?)",
                (file_hash, activity_data.get('filename'), activity_data.get('date'), json_str, session_id)
            )
            
    def delete_activity(self, file_hash):
        with self.get_connection() as conn:
            conn.execute("DELETE FROM activities WHERE hash = ?", (file_hash,))

    def get_count(self):
        with self.get_connection() as conn:
            return conn.execute("SELECT COUNT(*) FROM activities").fetchone()[0]

    def get_activities(self, timeframe="All Time", current_session_id=None):
        query = "SELECT json_data, hash FROM activities"
        params = []
        
        if timeframe == "Last Import" and current_session_id:
            query += " WHERE session_id = ?"
            params.append(current_session_id)
        elif timeframe == "Last 30 Days":
            date_Limit = (datetime.now() - pd.Timedelta(days=30)).strftime("%Y-%m-%d")
            query += " WHERE date >= ?"
            params.append(date_Limit)
        elif timeframe == "Last 90 Days":
            date_Limit = (datetime.now() - pd.Timedelta(days=90)).strftime("%Y-%m-%d")
            query += " WHERE date >= ?"
            params.append(date_Limit)
        elif timeframe == "This Year":
            current_year = datetime.now().year
            query += " WHERE date >= ?"
            params.append(f"{current_year}-01-01")
            
        query += " ORDER BY date DESC"
        
        with self.get_connection() as conn:
            rows = conn.execute(query, params).fetchall()
            results = []
            for row in rows:
                d = json.loads(row[0])
                d['db_hash'] = row[1]
                results.append(d)
            return results


# --- HELPER: FILE HASHING (Preserved from gui.py) ---
def calculate_file_hash(filepath):
    """Calculate SHA-256 hash of a file for deduplication."""
    hasher = hashlib.sha256()
    with open(filepath, 'rb') as f:
        buf = f.read(65536)
        while len(buf) > 0:
            hasher.update(buf)
            buf = f.read(65536)
    return hasher.hexdigest()


# --- MAIN APPLICATION CLASS ---
class GarminAnalyzerApp:
    """Main application class for the Garmin FIT Analyzer."""
    
    def __init__(self):
        """Initialize the application with database and state."""
        self.db = DatabaseManager()
        self.current_session_id = None
        self.current_timeframe = "Last 30 Days"
        self.activities_data = []
        self.df = None
        self.import_in_progress = False
        
        # Build UI
        self.build_ui()
        
        # Schedule initial data load using NiceGUI's timer (runs after event loop starts)
        ui.timer(0.1, self.refresh_data_view, once=True)
        
        # Show Save Chart button initially with fade (Trends tab is default)
        ui.timer(0.05, lambda: self.save_chart_btn.style('opacity: 1; pointer-events: auto;'), once=True)
        
    def build_ui(self):
        """Construct the complete UI layout with fixed sidebar."""
        # 1. CSS HACK: Hide Plotly Logo and notifications globally
        ui.add_head_html('''
        <style>
        .modebar-btn--logo { display: none !important; }
        /* Hide Plotly's "Double-click to zoom back out" notification */
        .plotly .notifier { display: none !important; }
        .js-plotly-plot .notifier { display: none !important; }
        .notifier { display: none !important; }
        
        /* Remove default page padding/margin to eliminate grey strips */
        body, .q-page, .nicegui-content {
            margin: 0 !important;
            padding: 0 !important;
        }
        </style>
        ''')
        # Set dark background for entire page
        ui.colors(primary='#2CC985', secondary='#1F1F1F', accent='#ff9900', 
                  dark='#0a0a0a', positive='#2CC985', negative='#ff4d4d', 
                  info='#3b82f6', warning='#ff9900')
        ui.query('body').classes('bg-zinc-900')
        
        # Add custom CSS for dropdown styling
        ui.add_head_html('''
        <style>
        /* Style dropdown options to be visible */
        .q-menu .q-item {
            color: white !important;
            background-color: #1F1F1F !important;
        }
        .q-menu .q-item:hover {
            background-color: #2CC985 !important;
        }
        /* Style the select input text and icon */
        .q-field__native, .q-field__input {
            color: white !important;
        }
        /* Style the dropdown arrow icon */
        .q-field__append .q-icon {
            color: white !important;
        }
        /* Style the select border */
        .q-field--outlined .q-field__control:before {
            border-color: rgba(255, 255, 255, 0.3) !important;
        }
        .q-field--outlined .q-field__control:hover:before {
            border-color: rgba(255, 255, 255, 0.6) !important;
        }
        </style>
        ''')
        
        with ui.row().classes('w-full h-screen m-0 p-0 gap-0 no-wrap overflow-hidden'):
            # Fixed sidebar - stays in place
            self.build_sidebar()
            # Scrollable main content area
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
            ).classes('w-full mb-4').style('color: white;').props('outlined dense')
            
            # Action buttons
            ui.button('Import Folder', icon='create_new_folder', on_click=self.select_folder, color=None).classes(
                'w-full mb-2 bg-zinc-800 text-white border border-zinc-700 hover:bg-zinc-700 shadow-md'
            )
            
            self.export_btn = ui.button('Export CSV', icon='file_download',
                                         on_click=self.export_csv, color=None).classes(
                'w-full mb-2 bg-zinc-800 text-white border border-zinc-700 hover:bg-zinc-700 shadow-md'
            ).props('disable')
            
            self.copy_btn = ui.button('Copy for LLM', icon='content_copy',
                                       on_click=self.copy_to_llm).classes(
                'w-full mb-2 bg-green-600'
            ).props('disable')
            
            # Separator
            ui.separator().classes('my-4 border-zinc-700')
            
            # Stats section
            self.runs_label = ui.label(f'Runs Stored: {self.db.get_count()}').classes(
                'text-sm font-bold text-green-500 text-center w-full'
            )
            self.status_label = ui.label('Ready').classes(
                'text-xs text-gray-400 text-center mt-2 w-full'
            )
    
    def build_main_content(self):
        """Create tabbed main content area with scrolling."""
        # Outer wrapper: No padding, flush to edges (removes grey strip)
        with ui.column().classes('flex-1 bg-zinc-950 h-screen overflow-y-auto p-0 gap-0'):
            # Inner container: Adds padding for content breathing room (no bottom padding)
            with ui.column().classes('w-full min-h-full pt-6 px-6 pb-0 gap-4'):
                # Create tabs row with absolute positioned Save Chart button
                with ui.row().classes('w-full items-center mb-0 relative'):
                    # Tabs centered, taking full width
                    with ui.tabs().classes('w-full text-white justify-center') as tabs:
                        trends_tab = ui.tab('Trends').classes('text-white')
                        report_tab = ui.tab('Report').classes('text-white')
                        activities_tab = ui.tab('Activities').classes('text-white')
                    
                    # Save Chart button absolutely positioned on the right (floats, doesn't affect layout)
                    # Add transition for smooth fade effect
                    self.save_chart_btn = ui.button('📸 Save Chart', on_click=self.save_chart_to_downloads, color=None).classes('bg-zinc-800 text-white border border-zinc-700 hover:bg-zinc-700 absolute right-0 top-0 shadow-sm z-10').style('transition: opacity 0.3s ease-in-out;')
                
                # Create tab panels with transparent background
                with ui.tab_panels(tabs, value=trends_tab).classes('w-full flex-1').props('transparent'):
                    # Trends tab panel
                    with ui.tab_panel(trends_tab).classes('p-0'):
                        self.build_trends_tab()
                    
                    # Report tab panel
                    with ui.tab_panel(report_tab).classes('p-0'):
                        self.build_report_tab()
                    
                    # Activities tab panel
                    with ui.tab_panel(activities_tab).classes('p-0'):
                        self.build_activities_tab()
                
                # Show/hide Save Chart button based on active tab
                tabs.on('update:model-value', lambda e: self.toggle_save_chart_button(e.args))
    
    def build_trends_tab(self):
        """Create trends tab with embedded Plotly chart."""
        # Store the plotly_container as an instance variable so it can be updated later
        self.plotly_container = ui.column().classes('w-full bg-zinc-900').style('min-height: 900px;')
        
        with self.plotly_container:
            # Show placeholder message when no data is available
            ui.label('No data available. Import activities to view trends.').classes(
                'text-center text-gray-400 mt-20'
            )
    
    def build_report_tab(self):
        """Create report tab with Card View."""
        with ui.tab_panel('report').classes('w-full p-0 bg-transparent'):
            # The container for our cards. 
            # 'gap-2' puts space between cards. 'bg-transparent' ensures no white box.
            self.report_container = ui.column().classes('w-full gap-2 bg-transparent p-4')

    
    def build_activities_tab(self):
        """Create activities tab with AG Grid table."""
        # Store the grid_container as an instance variable so it can be updated later
        # by update_activities_grid() method (task 9.1)
        self.grid_container = ui.column().classes('w-full bg-zinc-900').style('min-height: 800px;')
        
        # Initialize with placeholder
        with self.grid_container:
            ui.label('No activities found. Import activities to view data.').classes(
                'text-center text-gray-400 mt-20'
            )
    
    async def refresh_data_view(self):
        """
        Refresh all views with current filter.
        
        This method:
        - Queries database using DatabaseManager.get_activities()
        - Converts results to Pandas DataFrame
        - Parses dates and sorts by date
        - Updates state variables (activities_data, df)
        - Enables/disables export and copy buttons based on data availability
        
        Requirements: 6.7, 6.8, 11.1, 11.7
        """
        # Query database with current timeframe filter
        self.activities_data = self.db.get_activities(
            self.current_timeframe, 
            self.current_session_id
        )
        
        # Convert to DataFrame if we have data
        if self.activities_data:
            self.df = pd.DataFrame(self.activities_data)
            # Parse dates and sort by date
            self.df['date_obj'] = pd.to_datetime(self.df['date'])
            self.df = self.df.sort_values('date_obj')
        else:
            self.df = None
        
        # Update all tabs
        self.update_report_text()  # Task 8.1 - implemented
        self.update_activities_grid()  # Task 9.1 - implemented
        self.update_trends_chart()  # Task 12.1 - implemented
        
        # Enable/disable export and copy buttons based on data availability
        has_data = bool(self.activities_data)
        if has_data:
            self.export_btn.props(remove='disable')
            self.copy_btn.props(remove='disable')
        else:
            self.export_btn.props(add='disable')
            self.copy_btn.props(add='disable')
    
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
            
            # Show progress dialog
            with ui.dialog() as progress_dialog, ui.card():
                ui.label('ANALYZING FIT FILES...').classes('text-lg font-bold mb-4')
                progress_bar = ui.linear_progress(value=0).classes('w-96')
                progress_label = ui.label('Starting...').classes('mt-2')
            
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
                        # Analyze and import the file
                        result = analyzer.analyze_file(filepath)
                        if result:
                            self.db.insert_activity(result, f_hash, self.current_session_id)
                            new_count += 1
                        else:
                            error_count += 1
                    
                    # Update progress bar after each file
                    progress = (i + 1) / len(fit_files)
                    progress_bar.value = progress
                    progress_label.text = f'Processing {i + 1}/{len(fit_files)} - {os.path.basename(filepath)}'
                    
                    # Allow UI to update
                    await asyncio.sleep(0)
                    
                except Exception as e:
                    error_count += 1
                    print(f"Error processing {filepath}: {e}")
            
            # Close dialog
            progress_dialog.close()
            
            # Update runs counter
            self.runs_label.text = f'Runs Stored: {self.db.get_count()}'
            
            # Update status and switch timeframe if new activities imported
            if new_count > 0:
                self.status_label.text = f'Imported {new_count} new'
                self.status_label.classes(remove='text-gray-400', add='text-green-500')
                
                # Switch to "Last Import" timeframe
                self.current_timeframe = 'Last Import'
                self.timeframe_select.value = 'Last Import'
                
                # Show success notification
                ui.notify(f'Import complete: {new_count} new activities', type='positive')
            else:
                # No new files imported
                self.status_label.text = 'No new runs'
                self.status_label.classes(remove='text-gray-400', add='text-orange-500')
                
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
    
    def update_report_text(self):
        """Update report with Beautiful Cards."""
        self.report_container.clear()
        
        if not self.activities_data:
            with self.report_container:
                ui.label('No runs found for this timeframe.').classes('text-gray-500 italic')
            return
        
        # Calculate Average EF for context
        avg_ef = self.df['efficiency_factor'].mean() if self.df is not None else 0
        
        with self.report_container:
            for d in sorted(self.activities_data, key=lambda x: x.get('date', ''), reverse=True):
                
                # 1. Determine Border Color based on Cost
                cost = d.get('decoupling', 0)
                border_color = 'border-green-500' if cost <= 5 else 'border-red-500'
                
                # 2. Parse date string to human-readable format
                # Input: "2026-02-05 21:10" -> Output: "Thu, Feb 5 • 9:10 PM"
                date_str = d.get('date', '')
                try:
                    from datetime import datetime
                    dt = datetime.strptime(date_str, '%Y-%m-%d %H:%M')
                    formatted_date = dt.strftime('%a, %b %-d • %-I:%M %p')
                except:
                    # Fallback if parsing fails
                    formatted_date = date_str
                
                # 3. Create the Card with better contrast and defined edges
                with ui.card().classes(f'w-full bg-zinc-900 border border-zinc-800 border-l-4 {border_color} p-3 shadow-none text-white'):
                    
                    # Row 1: Date & Filename
                    with ui.row().classes('w-full justify-between items-center mb-2'):
                        ui.label(formatted_date).classes('font-bold text-base')
                        ui.label(d.get('filename', '')[:40]).classes('text-xs text-gray-500')
                    
                    # Row 2: Metrics Grid (4 Columns) - Compact
                    with ui.grid(columns=4).classes('w-full gap-3'):
                        # Distance
                        with ui.column().classes('gap-0'):
                            ui.label('DIST').classes('text-xs text-gray-500')
                            ui.label(f"{d.get('distance_mi', 0):.1f} mi").classes('text-lg font-bold')
                        
                        # Pace
                        with ui.column().classes('gap-0'):
                            ui.label('PACE').classes('text-xs text-gray-500')
                            ui.label(d.get('pace', '--')).classes('text-lg font-bold')
                            
                        # EF
                        with ui.column().classes('gap-0'):
                            with ui.row().classes('items-center gap-1'):
                                ui.label('EF').classes('text-xs text-gray-500')
                                ui.icon('help_outline').classes('text-zinc-500 hover:text-white text-[10px] cursor-pointer').on('click', lambda: self.show_ef_info())
                            ui.label(f"{d.get('efficiency_factor', 0):.2f}").classes('text-lg font-bold text-green-400')
                            
                        # Cost
                        with ui.column().classes('gap-0'):
                            with ui.row().classes('items-center gap-1'):
                                ui.label('COST').classes('text-xs text-gray-500')
                                ui.icon('help_outline').classes('text-zinc-500 hover:text-white text-[10px] cursor-pointer').on('click', lambda: self.show_cost_info())
                            cost_class = 'text-green-400' if cost <= 5 else 'text-red-400'
                            ui.label(f"{cost:.1f}%").classes(f'text-lg font-bold {cost_class}')

                    # Row 3: Professional HRR Display with Color Coding
                    hrr_list = d.get('hrr_list')
                    if hrr_list:
                        # Parse HRR list and get first value (1-minute drop)
                        try:
                            score = hrr_list[0] if isinstance(hrr_list, list) else int(hrr_list)
                        except:
                            score = 0
                        
                        # Determine color based on HRR score
                        if score > 30:
                            color_class = 'text-green-400'
                        elif score >= 20:
                            color_class = 'text-yellow-400'
                        else:
                            color_class = 'text-red-400'
                        
                        ui.separator().classes('my-2 border-zinc-700')
                        with ui.row().classes('items-center justify-between mt-2 pt-2 border-t border-zinc-800'):
                            # Left Side: Heart Icon + Label (Clean)
                            with ui.row().classes('items-center'):
                                ui.icon('favorite', color='red-500').classes('mr-1')
                                ui.label('HR Recovery').classes('text-xs text-gray-400 font-bold')
                            
                            # Right Side: Value + Target + Info Icon
                            with ui.row().classes('items-center gap-2'):
                                ui.label(f'{score} bpm').classes(f'text-sm font-bold {color_class}')
                                ui.label('(Target: >30)').classes('text-[10px] text-zinc-600')
                                # Info icon moved to far right - brightened for visibility
                                info_icon = ui.icon('help_outline').classes('text-zinc-400 hover:text-white text-xs cursor-pointer')
                                info_icon.on('click', lambda: self.show_hrr_info())
    
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
    
    def show_ef_info(self):
        """
        Show informational modal about Efficiency Factor.
        Explains what EF means and how to interpret it.
        """
        with ui.dialog() as dialog, ui.card().classes('bg-zinc-900 text-white p-6 max-w-2xl'):
            # Title
            ui.label('Efficiency Factor (EF)').classes('text-xl font-bold mb-4')
            
            # Body explanation
            ui.label("EF is your 'Gas Mileage'. It measures Speed divided by Heart Rate.").classes('text-sm text-gray-300 mb-4')
            
            # Scale
            ui.label('Interpretation:').classes('text-sm font-bold mb-2')
            ui.markdown('''
📈 **Higher is Better**  
It means you are running faster at the same heart rate.

**How to Use It:**  
Compare this number to runs of similar intensity. If your EF is improving over time, your aerobic engine is getting stronger.

**Typical Ranges:**
- 0.8-1.5: Recreational runners
- 1.5-2.5+: Elite runners
            ''').classes('text-sm text-gray-300 mb-4')
            
            # Close button
            ui.button('Got it!', on_click=dialog.close).classes('w-full bg-green-600')
        
        dialog.open()
    
    def show_cost_info(self):
        """
        Show informational modal about Aerobic Decoupling (Cost).
        Explains what decoupling means and how to interpret it.
        """
        with ui.dialog() as dialog, ui.card().classes('bg-zinc-900 text-white p-6 max-w-2xl'):
            # Title
            ui.label('Aerobic Decoupling (Cost)').classes('text-xl font-bold mb-4')
            
            # Body explanation
            ui.label("Cost measures 'Cardiac Drift'—how much your heart rate rises to hold the same pace over time.").classes('text-sm text-gray-300 mb-4')
            
            # Scale with color coding
            ui.label('Interpretation Scale:').classes('text-sm font-bold mb-2')
            with ui.column().classes('gap-2 mb-4'):
                with ui.row().classes('items-center gap-2'):
                    ui.label('✅').classes('text-lg')
                    ui.label('< 5%: Excellent (Durable)').classes('text-sm text-green-400')
                
                with ui.row().classes('items-center gap-2'):
                    ui.label('⚠️').classes('text-lg')
                    ui.label('5-10%: Moderate Drift').classes('text-sm text-yellow-400')
                
                with ui.row().classes('items-center gap-2'):
                    ui.label('🛑').classes('text-lg')
                    ui.label('> 10%: High Fatigue / Undeveloped Base').classes('text-sm text-red-400')
            
            ui.markdown('''
**What It Means:**  
Lower decoupling = better aerobic durability. Your cardiovascular system can maintain efficiency throughout the run.

**High Decoupling Indicates:**
- Insufficient aerobic base
- Running too fast for current fitness
- Accumulated fatigue from training
            ''').classes('text-sm text-gray-300 mb-4')
            
            # Close button
            ui.button('Got it!', on_click=dialog.close).classes('w-full bg-green-600')
        
        dialog.open()
    
    def format_run_data(self, d, folder_avg_ef=0):
        """
        Format single activity data (preserved logic from gui.py).
        
        This method formats an activity dictionary into a text block with:
        - Distance, pace, EF, decoupling, HRR
        - Status indicators (✅ Excellent, ⚠️ Moderate, 🛑 High Fatigue)
        
        Requirements: 4.4, 4.5, 13.7
        """
        decoupling = d.get('decoupling', 0)
        d_status = ""
        if decoupling < 5: 
            d_status = " (✅ Excellent)"
        elif decoupling <= 10: 
            d_status = " (⚠️ Moderate)"
        else: 
            d_status = " (🛑 High Fatigue)"
        
        ef = d.get('efficiency_factor', 0)
        hrr_list = d.get('hrr_list', [])
        hrr_str = str(hrr_list) if hrr_list else "--"
        
        return f"""
RUN: {d.get('date')} ({d.get('filename')})
--------------------------------------------------
Distance:   {d.get('distance_mi')} mi
Pace:       {d.get('pace')} /mi
EF:         {ef:.2f}
Decoupling: {decoupling}%{d_status}
HRR:        {hrr_str}
"""
    
    def update_activities_grid(self):
        """
        Update activities table with native NiceGUI table component.
        
        This method:
        - Clears grid container
        - Checks if activities_data is empty
        - Creates responsive table with inline delete buttons
        - Uses flex layout for responsive columns
        
        Requirements: 5.2, 5.3, 12.6
        """
        # Clear grid container
        self.grid_container.clear()
        
        # Check if activities_data is empty
        if not self.activities_data:
            with self.grid_container:
                ui.label('No activities found. Import activities to view data.').classes(
                    'text-center text-gray-400 mt-20'
                )
            return
        
        # Define columns with responsive flex layout
        columns = [
            {'name': 'date', 'label': 'Date', 'field': 'date', 'align': 'left', 'sortable': True},
            {'name': 'filename', 'label': 'Filename', 'field': 'filename', 'align': 'left', 'sortable': True},
            {'name': 'distance', 'label': 'Dist', 'field': 'distance', 'align': 'left', 'sortable': True},
            {'name': 'elevation', 'label': 'Elev', 'field': 'elevation', 'align': 'left', 'sortable': True},
            {'name': 'ef', 'label': 'EF', 'field': 'ef', 'align': 'left', 'sortable': True},
            {'name': 'cost', 'label': 'Cost', 'field': 'cost', 'align': 'left', 'sortable': True},
            {'name': 'cadence', 'label': 'Cadence', 'field': 'cadence', 'align': 'left', 'sortable': True},
            {'name': 'actions', 'label': '', 'field': 'actions', 'align': 'center'},
        ]
        
        # Transform activities into row data
        rows = []
        for act in self.activities_data:
            cost = act.get('decoupling', 0)
            rows.append({
                'date': act.get('date', '')[:10],
                'filename': act.get('filename', '')[:30],
                'distance': f"{act.get('distance_mi', 0):.1f} mi",
                'elevation': f"{act.get('elevation_ft', 0)} ft",
                'ef': f"{act.get('efficiency_factor', 0):.2f}",
                'cost': f"{cost:.1f}%",
                'cost_value': cost,
                'cadence': f"{act.get('avg_cadence', 0)} spm",
                'hash': act.get('db_hash', ''),
                'full_filename': act.get('filename', ''),
            })
        
        # Create table with custom styling
        with self.grid_container:
            table = ui.table(
                columns=columns,
                rows=rows,
                row_key='hash'
            ).classes('w-full h-full')
            
            # Add custom slot for the actions column with inline delete button
            table.add_slot('body-cell-actions', '''
                <q-td :props="props">
                    <q-btn 
                        flat 
                        dense 
                        round 
                        icon="delete" 
                        size="sm"
                        style="opacity: 0.4; transition: all 0.2s;"
                        @mouseover="$event.target.style.opacity='1'; $event.target.style.color='#ff4d4d'"
                        @mouseout="$event.target.style.opacity='0.4'; $event.target.style.color=''"
                        @click="$parent.$emit('delete-row', props.row)"
                    />
                </q-td>
            ''')
            
            # Add custom slot for cost column with color coding
            table.add_slot('body-cell-cost', '''
                <q-td :props="props">
                    <span :style="props.row.cost_value > 5 ? 'color: #ff4d4d; font-weight: 600;' : 'color: #2CC985; font-weight: 600;'">
                        {{ props.value }}
                    </span>
                </q-td>
            ''')
            
            # Handle delete button clicks
            table.on('delete-row', lambda e: self.delete_activity_inline(e.args['hash'], e.args['full_filename']))
            
            # Apply dark theme styling
            table.props('flat bordered dense dark')
            table.classes('bg-zinc-900 text-gray-200')
            
            # Add custom CSS for better styling
            ui.add_head_html('''
            <style>
            .q-table thead tr, .q-table tbody td {
                height: 48px;
            }
            .q-table th {
                background-color: #2a2a2a !important;
                color: #aaa !important;
                font-weight: 600 !important;
                border-bottom: 2px solid #444 !important;
            }
            .q-table td {
                color: #ddd !important;
                border-bottom: 1px solid #333 !important;
            }
            .q-table tbody tr:hover {
                background-color: #252525 !important;
            }
            </style>
            ''')
    
    async def delete_activity_inline(self, activity_hash, filename):
        """
        Delete activity from inline button click.
        
        This method:
        - Shows confirmation dialog
        - Deletes activity from database
        - Updates runs counter
        - Refreshes data view
        - Shows success notification
        
        Requirements: 5.4, 5.5, 5.6, 5.7, 5.8
        """
        # Show confirmation dialog
        result = await ui.run_javascript(
            f'confirm("Delete activity: {filename}?")', 
            timeout=10
        )
        
        # If confirmed, delete the activity
        if result:
            try:
                # Delete the activity
                self.db.delete_activity(activity_hash)
                
                # Update runs counter
                self.runs_label.text = f'Runs Stored: {self.db.get_count()}'
                
                # Refresh all data views
                await self.refresh_data_view()
                
                # Show success notification
                ui.notify(f'Deleted: {filename[:30]}', type='positive')
                
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
            self.runs_label.text = f'Runs Stored: {self.db.get_count()}'
            
            # Call refresh_data_view() to update all views
            await self.refresh_data_view()
            
            # Show success notification
            ui.notify('Activity deleted successfully', type='positive')
    
    def calculate_trend_stats(self, df_subset):
        """
        Calculate trend statistics for a DataFrame subset.
        
        Args:
            df_subset: DataFrame with 'date_obj' and 'efficiency_factor' columns
            
        Returns:
            tuple: (trend_msg, trend_color) - message string and hex color
        """
        if df_subset is None or df_subset.empty:
            return ("Trend: No Data", "silver")
        
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
        
        # --- 2. Calculate mean EF and split data into 4 performance groups ---
        ef_mean = self.df['efficiency_factor'].mean()
        
        # Split data into 4 groups based on EF and decoupling
        df_green = self.df[(self.df['efficiency_factor'] >= ef_mean) & (self.df['decoupling'] <= 5)]  # Race Ready
        df_yellow = self.df[(self.df['efficiency_factor'] < ef_mean) & (self.df['decoupling'] <= 5)]  # Base Maintenance
        df_orange = self.df[(self.df['efficiency_factor'] >= ef_mean) & (self.df['decoupling'] > 5)]  # Expensive Speed
        df_red = self.df[(self.df['efficiency_factor'] < ef_mean) & (self.df['decoupling'] > 5)]  # Struggling
        
        # --- 3. Calculate cadence colors based on cadence thresholds ---
        cad_colors = []
        for c in self.df['avg_cadence']:
            if c >= 170: 
                cad_colors.append('#2CC985')  # Green - Efficient
            elif c >= 160: 
                cad_colors.append('#e6e600')  # Yellow - OK
            else: 
                cad_colors.append('#ff4d4d')  # Red - Sloppy
        
        # --- 4. Create subplots with make_subplots (2 rows, shared x-axis) ---
        fig = make_subplots(
            rows=2, cols=1, 
            shared_xaxes=True,
            vertical_spacing=0.15,
            subplot_titles=("Aerobic Durability (Engine vs. Cost)", "Mechanics (Cadence Trend)"),
            specs=[[{"secondary_y": True}], [{"secondary_y": False}]]
        )
        
        # === TOP GRAPH: DURABILITY ===
        
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
            row=1, col=1, secondary_y=True
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
            row=1, col=1, secondary_y=True
        )
        
        # Add horizontal line at 5% decoupling threshold
        fig.add_hline(
            y=5, 
            line_dash="dot", 
            line_color="rgba(255, 77, 77, 0.5)", 
            secondary_y=True, 
            row=1, col=1
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
            row=1, col=1, secondary_y=False
        )
        
        # --- 7. Add 4 separate marker traces for each performance category ---
        performance_groups = [
            (df_green, '#2CC985', 'Race Ready (Fast & Stable)', 'Race Ready 🟢'),
            (df_yellow, '#e6e600', 'Base Maintenance (Slow & Stable)', 'Base Maintenance 🟡'),
            (df_orange, '#ff9900', 'Expensive Speed (Fast but Drifted)', 'Expensive Speed 🟠'),
            (df_red, '#ff4d4d', 'Struggling (Slow & Drifted)', 'Struggling 🔴')
        ]
        
        for df_group, color, legend_name, verdict in performance_groups:
            if not df_group.empty:
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
                            [verdict] * len(df_group),
                            df_group['decoupling'], 
                            df_group['pace'], 
                            df_group['distance_mi'], 
                            df_group['avg_hr']
                        )),
                        hovertemplate=(
                            "<b>%{customdata[0]}</b><br>"
                            "EF: <b>%{y:.2f}</b> | Cost: <b>%{customdata[1]:.1f}%</b><br>"
                            "<span style='color:#666'>──────────</span><br>"
                            "Dist: %{customdata[3]} mi @ %{customdata[2]}<br>"
                            "Avg HR: %{customdata[4]} bpm<extra></extra>"
                        )
                    ), 
                    row=1, col=1, secondary_y=False
                )
        
        # === BOTTOM GRAPH: CADENCE ===
        
        # --- 8. Add cadence scatter trace ---
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
                hovertemplate="Cadence: <b>%{y} spm</b><extra></extra>",
                showlegend=False  # Hide from legend (bottom subplot is self-explanatory)
            ), 
            row=2, col=1
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
            ),
            template="plotly_dark",
            height=900, 
            showlegend=True,
            legend=dict(
                orientation="v",  # Vertical layout
                yanchor="top",
                y=0.98,  # Position at top of plot area
                xanchor="right",
                x=0.99,  # Position at right edge
                bgcolor="rgba(26, 26, 26, 0.85)",  # Semi-transparent dark background
                bordercolor="rgba(255, 255, 255, 0.1)",  # Subtle border
                borderwidth=1,
                font=dict(size=11)
            ),
            margin=dict(l=60, r=60, t=100, b=40),  # Reduced top margin since legend is in plot area
            hoverlabel=dict(
                bgcolor="#1F1F1F", 
                bordercolor="#444", 
                font_color="white"
            ),
            modebar={
                'remove': ['toImage', 'select2d', 'lasso2d', 'autoScale2d', 'resetScale2d', 'zoom', 'pan', 'zoomIn', 'zoomOut', 'autoScale', 'resetScale']
            },
            dragmode='zoom',
            hovermode='closest'
        )
        
        # Axis Styling
        fig.update_yaxes(
            title_text="[Gains] Efficiency Factor", 
            color="#2CC985", 
            row=1, col=1, secondary_y=False
        )
        fig.update_yaxes(
            title_text="[Cost] Decoupling %", 
            color="#ff4d4d", 
            row=1, col=1, secondary_y=True, 
            range=[-5, max(20, self.df['decoupling'].max()+2)]
        )
        fig.update_yaxes(
            title_text="Cadence (spm)", 
            color="white", 
            row=2, col=1
        )
        
        # --- 10. Return figure object ---
        return fig
    
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
                # If data exists, call generate_plotly_figure()
                fig = self.generate_plotly_figure()
                if fig:
                    # Create ui.plotly() with figure (modebar config is in generate_plotly_figure)
                    self.chart = ui.plotly(fig).style('width: 100%; height: 900px;')
                    
                    # Bind zoom event for live trend updates
                    self.chart.on('plotly_relayout', self.handle_chart_zoom)
                else:
                    # Show error message if figure generation failed
                    ui.label('Error generating chart. Please check your data.').classes(
                        'text-center text-red-500 mt-20'
                    )
            else:
                # If no data, show placeholder message
                ui.label('No data available. Import activities to view trends.').classes(
                    'text-center text-gray-400 mt-20'
                )
    
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
                
                # Recalculate trend for zoomed data
                trend_msg, trend_color = self.calculate_trend_stats(df_zoomed)
                
            elif 'xaxis.autorange' in e.args:
                # User double-clicked to reset zoom - use full dataset
                trend_msg, trend_color = self.calculate_trend_stats(self.df)
            else:
                return
            
            # Update the chart title subtitle using Plotly's relayout
            new_title = (
                f"<b>Training Trends</b><br>"
                f"<span style='font-size: 14px; color: {trend_color};'>{trend_msg}</span>"
            )
            
            await ui.run_javascript(f'''
                const plotDiv = document.querySelector('.js-plotly-plot');
                if (plotDiv) {{
                    Plotly.relayout(plotDiv, {{'title.text': `{new_title}`}});
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

AEROBIC DECOUPLING (Cost):
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
- Green (Race Ready): High EF + Low Decoupling = Peak fitness
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
        Save Plotly chart as PNG to Downloads folder.
        
        This method:
        - Uses JavaScript to extract the chart as base64 PNG
        - Decodes the base64 data
        - Saves to ~/Downloads with timestamp
        - Shows success notification
        """
        try:
            # Check if we have chart data
            if self.df is None or self.df.empty:
                ui.notify('No chart data to save', type='warning')
                return
            
            # Execute JavaScript to get the chart as base64 PNG
            js_code = """
            return await Plotly.toImage(
                document.querySelector('.js-plotly-plot'), 
                {format: 'png', height: 900, width: 1200, scale: 2}
            )
            """
            
            img_data = await ui.run_javascript(js_code, timeout=10.0)
            
            # Clean the data (strip the data:image/png;base64, header)
            if img_data and img_data.startswith('data:image/png;base64,'):
                img_data = img_data.replace('data:image/png;base64,', '')
            
            # Decode base64 to bytes
            import base64
            img_bytes = base64.b64decode(img_data)
            
            # Generate filename with timestamp
            from datetime import datetime
            timestamp = datetime.now().strftime("%Y-%m-%d_%H%M%S")
            filename = f"Garmin_Trends_{timestamp}.png"
            
            # Save to Downloads folder
            downloads_path = os.path.expanduser("~/Downloads")
            file_path = os.path.join(downloads_path, filename)
            
            with open(file_path, 'wb') as f:
                f.write(img_bytes)
            
            # Show success notification
            ui.notify(f'Chart saved to Downloads! ({filename})', type='positive', timeout=5000)
            print(f"Chart saved to: {file_path}")
            
        except Exception as e:
            # Handle save failure
            ui.notify(f'Error saving chart: {str(e)}', type='negative')
            print(f"Error in save_chart_to_downloads: {e}")
            import traceback
            traceback.print_exc()
    
    async def copy_to_llm(self):
        """
        Copy activity data to clipboard with LLM context.
        
        This method:
        - Checks if more than 20 activities (summary mode)
        - If summary mode, groups by month and aggregates
        - Otherwise, generates report text from activities_data on the fly
        - Appends LLM context dictionary
        - Copies to clipboard using pyperclip
        - Updates button text and color temporarily
        - Resets button after 2 seconds
        
        Requirements: 9.1, 9.2, 9.3, 9.4, 9.5, 9.6, 13.6
        """
        # Check if we have data
        if not self.activities_data:
            ui.notify('No data to copy', type='warning')
            return
        
        try:
            # Check if more than 20 activities (summary mode)
            if len(self.activities_data) > 20:
                # Summary mode: group by month and aggregate
                self.df['month'] = pd.to_datetime(self.df['date']).dt.to_period('M')
                monthly = self.df.groupby('month').agg({
                    'distance_mi': 'sum',
                    'efficiency_factor': 'mean',
                    'decoupling': 'mean',
                    'avg_cadence': 'mean',
                    'filename': 'count'  # Count of runs
                }).round(2)
                
                # Format monthly summary
                report_text = "=== MONTHLY TRAINING SUMMARY ===\n\n"
                for month, row in monthly.iterrows():
                    report_text += f"{month}:\n"
                    report_text += f"  Runs: {row['filename']}\n"
                    report_text += f"  Total Distance: {row['distance_mi']:.1f} mi\n"
                    report_text += f"  Avg EF: {row['efficiency_factor']:.2f}\n"
                    report_text += f"  Avg Decoupling: {row['decoupling']:.1f}%\n"
                    report_text += f"  Avg Cadence: {row['avg_cadence']:.0f} spm\n\n"
            else:
                # Generate report text from activities_data on the fly
                avg_ef = self.df['efficiency_factor'].mean() if self.df is not None else 0
                report_lines = []
                for activity in sorted(self.activities_data, 
                                      key=lambda x: x.get('date', ''), 
                                      reverse=True):
                    # Use format_run_data to generate text
                    report_lines.append(self.format_run_data(activity, avg_ef))
                    report_lines.append('=' * 50)
                report_text = '\n'.join(report_lines)
            
            # Append LLM context dictionary
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

AEROBIC DECOUPLING (Cost):
- Measures aerobic durability (% efficiency loss over run)
- Lower = better endurance
- <5%: Excellent, 5-10%: Moderate, >10%: High fatigue
- High decoupling = cardiovascular drift, need more base training

HEART RATE RECOVERY (HRR):
- BPM drop 60 seconds after peak efforts
- Higher = better cardiovascular fitness
- >30: Excellent, 20-30: Good, <20: Poor/fatigued

TRAINING ZONES:
- 🟢 Green (Race Ready): High EF + Low Decoupling = Peak fitness
- 🟡 Yellow (Base Maintenance): Low EF + Low Decoupling = Building base
- 🟠 Orange (Expensive Speed): High EF + High Decoupling = Fast but unsustainable
- 🔴 Red (Struggling): Low EF + High Decoupling = Fatigue/overtraining

FORM METRICS:
- Cadence: Target 170-180 spm for efficiency
- Vertical Ratio: Lower = less wasted vertical motion
- GCT Balance: Target 50/50 left/right symmetry
"""
            
            # Combine report and context
            clipboard_content = report_text + llm_context
            
            # Use pyperclip for reliable clipboard access
            try:
                import pyperclip
                pyperclip.copy(clipboard_content)
                
                # Update button text and color temporarily
                original_text = self.copy_btn.text
                self.copy_btn.text = '✅ Copied!'
                self.copy_btn.classes(remove='bg-green-600', add='bg-white text-black')
                
                # Show success notification
                ui.notify('Data copied to clipboard!', type='positive')
                
                # Reset button after 2 seconds
                await asyncio.sleep(2)
                self.copy_btn.text = original_text
                self.copy_btn.classes(remove='bg-white text-black', add='bg-green-600')
                
            except ImportError:
                # Show error if pyperclip not installed
                ui.notify('Please install pyperclip: pip3 install pyperclip', type='negative')
                print("Error: pyperclip not installed. Run: pip3 install pyperclip")
            
        except Exception as e:
            # Handle clipboard access failure
            ui.notify(f'Error copying to clipboard: {str(e)}', type='negative')
            print(f"Error in copy_to_llm: {e}")
            import traceback
            traceback.print_exc()



def main():
    """Application entry point."""
    # Instantiate the application
    app = GarminAnalyzerApp()
    
    # Run in native mode with specified window configuration
    ui.run(
        native=True,
        window_size=(1200, 900),
        title="Garmin Analyzer Pro",
        reload=False
    )


if __name__ == "__main__":
    main()
