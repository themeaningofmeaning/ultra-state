# Design Document: NiceGUI Refactor

## Overview

This design document outlines the architecture and implementation approach for refactoring the Garmin FIT Analyzer from CustomTkinter to NiceGUI. The refactor maintains all existing functionality while modernizing the user interface with a web-based framework running in native desktop mode.

### Key Design Decisions

**Framework Choice**: NiceGUI was selected for its ability to create modern web-based UIs using pure Python while running as a native desktop application. It provides:
- Native desktop window mode via system webview
- Built-in Tailwind CSS support for modern styling
- Reactive data binding for automatic UI updates
- Integration with Plotly for interactive charts
- AG Grid support for professional data tables

**Architecture Pattern**: The application follows a single-page application (SPA) pattern with:
- Persistent sidebar for navigation and controls
- Tabbed main content area for different views
- Reactive state management for data updates
- Asynchronous file processing to prevent UI blocking

**Preservation Strategy**: All business logic from the original CustomTkinter implementation is preserved:
- DatabaseManager class remains unchanged
- FitAnalyzer class from analyzer.py remains unchanged
- File hashing and deduplication logic preserved
- Metric calculations and color-coding logic preserved
- Plotly chart generation logic preserved

## Architecture

### Application Structure

```
app.py (NiceGUI Application)
├── DatabaseManager (Preserved from gui.py)
├── calculate_file_hash() (Preserved from gui.py)
├── GarminAnalyzerApp (Refactored to NiceGUI)
│   ├── __init__() - Initialize UI components
│   ├── build_sidebar() - Create left sidebar
│   ├── build_main_content() - Create tabbed interface
│   ├── build_trends_tab() - Plotly chart view
│   ├── build_report_tab() - Text/markdown view
│   ├── build_activities_tab() - AG Grid table
│   ├── select_folder() - File import handler
│   ├── process_folder_async() - Background import
│   ├── on_filter_change() - Timeframe handler
│   ├── refresh_data_view() - Update all views
│   ├── generate_plotly_figure() - Create charts
│   ├── delete_activity() - Remove activity
│   ├── export_csv() - CSV export
│   └── copy_to_llm() - Clipboard copy
└── main() - Entry point
```

### Component Hierarchy

```
Native Window (1200x900)
├── Left Sidebar (Fixed, 220px width)
│   ├── Logo/Title
│   ├── Timeframe Dropdown
│   ├── Import Folder Button
│   ├── Export CSV Button
│   ├── Copy for LLM Button
│   └── Stats Section (Runs Counter, Status)
└── Main Content Area (Tabbed)
    ├── Tab 1: Trends (ui.plotly)
    ├── Tab 2: Report (ui.markdown or ui.textarea)
    └── Tab 3: Activities (ui.aggrid)
```

### State Management

The application maintains the following state:
- `current_session_id`: Unix timestamp for current import batch
- `current_timeframe`: Selected filter (Last Import, Last 30 Days, etc.)
- `activities_data`: List of activity dictionaries from database
- `df`: Pandas DataFrame for chart generation
- `import_in_progress`: Boolean flag for import state

State updates trigger reactive UI updates through NiceGUI's binding system.

## Components and Interfaces

### Main Application Class

```python
class GarminAnalyzerApp:
    def __init__(self):
        """Initialize application with database and state."""
        self.db = DatabaseManager()
        self.current_session_id = None
        self.current_timeframe = "Last 30 Days"
        self.activities_data = []
        self.df = None
        self.import_in_progress = False
        
        # Build UI
        self.build_ui()
        
    def build_ui(self):
        """Construct the complete UI layout."""
        with ui.row().classes('w-full h-screen'):
            self.build_sidebar()
            self.build_main_content()
```

### Sidebar Component

```python
def build_sidebar(self):
    """Create fixed left sidebar with controls."""
    with ui.column().classes('w-56 bg-zinc-900 p-4 h-screen'):
        # Logo
        ui.label('Garmin\nAnalyzer 🏃‍♂️').classes(
            'text-2xl font-bold text-center mb-4'
        )
        
        # Timeframe filter
        ui.label('TIMEFRAME').classes('text-xs text-gray-500 font-bold mb-1')
        self.timeframe_select = ui.select(
            options=['Last Import', 'Last 30 Days', 'Last 90 Days', 
                     'This Year', 'All Time'],
            value='Last 30 Days',
            on_change=self.on_filter_change
        ).classes('w-full mb-4')
        
        # Action buttons
        ui.button('📂 Import Folder', on_click=self.select_folder).classes(
            'w-full mb-2'
        )
        self.export_btn = ui.button('💾 Export CSV', 
                                     on_click=self.export_csv).classes(
            'w-full mb-2'
        ).props('disable')
        self.copy_btn = ui.button('📋 Copy for LLM', 
                                   on_click=self.copy_to_llm).classes(
            'w-full mb-2 bg-green-600'
        ).props('disable')
        
        # Separator
        ui.separator().classes('my-4')
        
        # Stats
        self.runs_label = ui.label(f'Runs Stored: {self.db.get_count()}').classes(
            'text-sm font-bold text-green-500 text-center'
        )
        self.status_label = ui.label('Ready').classes(
            'text-xs text-gray-500 text-center mt-2'
        )
```

### Trends Tab (Plotly Integration)

```python
def build_trends_tab(self):
    """Create trends tab with embedded Plotly chart."""
    with ui.tab_panel('trends'):
        self.plotly_container = ui.column().classes('w-full h-full')
        with self.plotly_container:
            if self.df is not None and not self.df.empty:
                fig = self.generate_plotly_figure()
                ui.plotly(fig).classes('w-full h-full')
            else:
                ui.label('No data available. Import activities to view trends.').classes(
                    'text-center text-gray-500 mt-20'
                )

def generate_plotly_figure(self):
    """Generate Plotly figure with stacked graphs (preserved logic)."""
    # This function preserves the exact logic from gui.py generate_dashboard()
    # Including:
    # - Linear regression trend calculation
    # - Color-coded markers (green/yellow/orange/red)
    # - Dual-axis stacked subplots
    # - Red/teal filled zones for decoupling
    # - Custom hover tooltips
    # - Static HTML annotation legend at bottom
    
    # [Full implementation preserved from original]
    pass
```

### Report Tab

```python
def build_report_tab(self):
    """Create report tab with formatted text."""
    with ui.tab_panel('report'):
        self.report_area = ui.markdown('').classes(
            'w-full h-full p-4 overflow-auto font-mono text-sm'
        )

def update_report_text(self):
    """Update report with formatted activity data."""
    if not self.activities_data:
        self.report_area.content = '\n\nNo runs found for this timeframe.'
        return
    
    avg_ef = self.df['efficiency_factor'].mean() if self.df is not None else 0
    report_lines = []
    
    for activity in sorted(self.activities_data, 
                          key=lambda x: x.get('date', ''), 
                          reverse=True):
        report_lines.append(self.format_run_data(activity, avg_ef))
        report_lines.append('=' * 40)
    
    self.report_area.content = '\n'.join(report_lines)

def format_run_data(self, d, folder_avg_ef=0):
    """Format single activity data (preserved logic)."""
    # Preserved from gui.py
    decoupling = d.get('decoupling', 0)
    d_status = ""
    if decoupling < 5: d_status = " (✅ Excellent)"
    elif decoupling <= 10: d_status = " (⚠️ Moderate)"
    else: d_status = " (🛑 High Fatigue)"
    
    return f"""
RUN: {d.get('date')} ({d.get('filename')})
--------------------------------------------------
Distance:   {d.get('distance_mi')} mi
Pace:       {d.get('pace')} /mi
EF:         {d.get('efficiency_factor', 0):.2f}
Decoupling: {decoupling}%{d_status}
HRR:        {d.get('hrr_list', [])}
"""
```

### Activities Tab (AG Grid)

```python
def build_activities_tab(self):
    """Create activities tab with AG Grid table."""
    with ui.tab_panel('activities'):
        self.grid_container = ui.column().classes('w-full h-full')
        self.update_activities_grid()

def update_activities_grid(self):
    """Update AG Grid with current activities."""
    self.grid_container.clear()
    
    if not self.activities_data:
        with self.grid_container:
            ui.label('No activities found.').classes('text-center text-gray-500 mt-20')
        return
    
    # Prepare grid data
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
            'cadence': f"{act.get('avg_cadence', 0)} spm",
            'hash': act.get('db_hash', '')
        })
    
    # Column definitions
    columns = [
        {'field': 'date', 'headerName': 'Date', 'width': 110},
        {'field': 'filename', 'headerName': 'Filename', 'width': 220},
        {'field': 'distance', 'headerName': 'Dist', 'width': 90},
        {'field': 'elevation', 'headerName': 'Elev', 'width': 90},
        {'field': 'ef', 'headerName': 'EF', 'width': 70},
        {'field': 'cost', 'headerName': 'Cost', 'width': 80,
         'cellStyle': {'params': 'value.cost > 5 ? {color: "#ff4d4d"} : {color: "#2CC985"}'}},
        {'field': 'cadence', 'headerName': 'Cadence', 'width': 100},
    ]
    
    with self.grid_container:
        self.activities_grid = ui.aggrid({
            'columnDefs': columns,
            'rowData': rows,
            'theme': 'balham-dark',
            'rowSelection': 'single',
        }).classes('w-full h-full')
        
        # Delete button (ghost style)
        ui.button('Delete Selected', 
                  on_click=self.delete_selected_activity).classes(
            'mt-2 bg-transparent hover:bg-red-900 text-gray-600 hover:text-red-500'
        )

async def delete_selected_activity(self):
    """Delete selected activity from grid."""
    selected = await self.activities_grid.get_selected_rows()
    if not selected:
        ui.notify('No activity selected', type='warning')
        return
    
    activity_hash = selected[0]['hash']
    result = await ui.run_javascript(
        'confirm("Delete this activity?")', 
        timeout=10
    )
    
    if result:
        self.db.delete_activity(activity_hash)
        self.runs_label.text = f'Runs Stored: {self.db.get_count()}'
        await self.refresh_data_view()
        ui.notify('Activity deleted', type='positive')
```

### File Import Handler

```python
async def select_folder(self):
    """Handle folder selection and import."""
    # Use NiceGUI's file picker or native dialog
    folder_path = await ui.run_javascript('''
        new Promise((resolve) => {
            const input = document.createElement('input');
            input.type = 'file';
            input.webkitdirectory = true;
            input.onchange = (e) => resolve(e.target.files[0].path.split('/').slice(0, -1).join('/'));
            input.click();
        })
    ''', timeout=60)
    
    if folder_path:
        self.timeframe_select.disable()
        await self.process_folder_async(folder_path)

async def process_folder_async(self, folder_path):
    """Process FIT files in background with progress."""
    self.current_session_id = int(time.time())
    
    # Show progress dialog
    with ui.dialog() as progress_dialog, ui.card():
        ui.label('ANALYZING...').classes('text-lg font-bold')
        progress_bar = ui.linear_progress(value=0).classes('w-64')
        progress_label = ui.label('Starting...')
    
    progress_dialog.open()
    
    # Get FIT files
    fit_files = [
        os.path.join(folder_path, f) 
        for f in os.listdir(folder_path) 
        if f.lower().endswith('.fit')
    ]
    
    if not fit_files:
        progress_dialog.close()
        ui.notify('No .FIT files found!', type='warning')
        self.timeframe_select.enable()
        return
    
    # Process files
    analyzer = FitAnalyzer()
    new_count = 0
    
    for i, filepath in enumerate(fit_files):
        try:
            f_hash = calculate_file_hash(filepath)
            if not self.db.activity_exists(f_hash):
                result = analyzer.analyze_file(filepath)
                if result:
                    self.db.insert_activity(result, f_hash, self.current_session_id)
                    new_count += 1
            
            # Update progress
            progress = (i + 1) / len(fit_files)
            progress_bar.value = progress
            progress_label.text = f'Processing {i + 1}/{len(fit_files)}'
            await asyncio.sleep(0)  # Allow UI update
            
        except Exception as e:
            print(f"Error processing {filepath}: {e}")
    
    # Finish import
    progress_dialog.close()
    self.runs_label.text = f'Runs Stored: {self.db.get_count()}'
    
    if new_count > 0:
        self.status_label.text = f'Imported {new_count} new'
        self.status_label.classes('text-green-500')
        self.current_timeframe = 'Last Import'
        self.timeframe_select.value = 'Last Import'
    else:
        self.status_label.text = 'No new runs'
        self.status_label.classes('text-orange-500')
    
    self.timeframe_select.enable()
    await self.refresh_data_view()
    ui.notify(f'Import complete: {new_count} new activities', type='positive')
```

### Data Refresh Logic

```python
async def refresh_data_view(self):
    """Refresh all views with current filter."""
    # Query database
    self.activities_data = self.db.get_activities(
        self.current_timeframe, 
        self.current_session_id
    )
    
    # Update DataFrame
    if self.activities_data:
        self.df = pd.DataFrame(self.activities_data)
        self.df['date_obj'] = pd.to_datetime(self.df['date'])
        self.df = self.df.sort_values('date_obj')
    else:
        self.df = None
    
    # Update all tabs
    self.update_report_text()
    self.update_activities_grid()
    self.update_trends_chart()
    
    # Enable/disable buttons
    has_data = bool(self.activities_data)
    if has_data:
        self.export_btn.props(remove='disable')
        self.copy_btn.props(remove='disable')
    else:
        self.export_btn.props(add='disable')
        self.copy_btn.props(add='disable')

def update_trends_chart(self):
    """Update Plotly chart in trends tab."""
    self.plotly_container.clear()
    
    with self.plotly_container:
        if self.df is not None and not self.df.empty:
            fig = self.generate_plotly_figure()
            ui.plotly(fig).classes('w-full h-full')
        else:
            ui.label('No data available. Import activities to view trends.').classes(
                'text-center text-gray-500 mt-20'
            )
```

## Data Models

### Activity Data Structure

Activities are stored as JSON in SQLite with the following structure:

```python
{
    'filename': str,           # Base filename of .FIT file
    'date': str,              # ISO format: 'YYYY-MM-DD HH:MM'
    'distance_mi': float,     # Distance in miles
    'pace': str,              # Format: 'MM:SS'
    'gap_pace': str,          # Grade-adjusted pace: 'MM:SS'
    'avg_hr': int,            # Average heart rate (bpm)
    'avg_power': int,         # Average power (watts)
    'avg_cadence': int,       # Average cadence (spm)
    'efficiency_factor': float,  # Normalized speed / HR
    'decoupling': float,      # Percentage drift
    'avg_temp': float,        # Average temperature (°C)
    'avg_resp': float,        # Average respiration rate
    'hrr_list': List[int],    # Heart rate recovery values
    'elevation_ft': int,      # Total elevation gain (feet)
    'moving_time_min': float, # Active time (minutes)
    'rest_time_min': float,   # Rest time (minutes)
    'gct_change': float,      # Ground contact time change
    'v_ratio': float,         # Vertical ratio
    'gct_balance': float,     # Ground contact time balance
    'db_hash': str            # SHA-256 file hash (added by DB)
}
```

### Database Schema

```sql
CREATE TABLE activities (
    hash TEXT PRIMARY KEY,
    filename TEXT,
    date TEXT,
    json_data TEXT,
    session_id INTEGER
)
```

### Chart Data Structure

For Plotly chart generation, data is transformed into:

```python
{
    'dates': List[datetime],
    'ef_values': List[float],
    'decoupling_values': List[float],
    'cadence_values': List[int],
    'marker_colors': List[str],  # ['#2CC985', '#ff9900', ...]
    'verdicts': List[str],       # ['Race Ready 🟢', ...]
    'hover_data': List[Dict]     # Custom tooltip data
}
```


## Correctness Properties

A property is a characteristic or behavior that should hold true across all valid executions of a system—essentially, a formal statement about what the system should do. Properties serve as the bridge between human-readable specifications and machine-verifiable correctness guarantees.

### Property 1: Decoupling Color Mapping

*For any* activity with aerobic decoupling value, when rendered in the Plotly chart, positive decoupling values should be displayed as red filled areas and negative decoupling values should be displayed as teal filled areas.

**Validates: Requirements 3.6, 3.7**

### Property 2: File Hash Deduplication

*For any* FIT file, if its SHA-256 hash already exists in the database, then importing that file should skip the import and not create a duplicate database entry.

**Validates: Requirements 6.2, 6.3, 6.4**

### Property 3: Session ID Assignment

*For any* import operation, all files imported in that batch should be assigned the same session ID based on the Unix timestamp at the start of the import.

**Validates: Requirements 6.5**

### Property 4: Timeframe Filtering

*For any* timeframe selection and database state, the filtered activities should only include activities that match the timeframe criteria (date range for date-based filters, session ID for "Last Import").

**Validates: Requirements 6.7, 6.8, 11.2, 11.3, 11.4, 11.5, 11.6**

### Property 5: Activity Deletion Propagation

*For any* activity in the database, when that activity is deleted, it should be removed from the database and all UI views (report, activities table, trends chart) should refresh to exclude the deleted activity.

**Validates: Requirements 5.8**

### Property 6: Trend Slope Message Mapping

*For any* dataset with calculable linear regression, the trend message should be "Engine Improving" (green) when slope > 0.0000001, "Fitness Declining" (red) when slope < -0.0000001, and "Fitness Stable" (silver) otherwise.

**Validates: Requirements 10.3, 10.4, 10.5**

### Property 7: Activity Marker Color Logic

*For any* activity with efficiency factor and decoupling values, the marker color should be determined as: green when EF >= mean AND decoupling <= 5%, orange when EF >= mean AND decoupling > 5%, yellow when EF < mean AND decoupling <= 5%, and red when EF < mean AND decoupling > 5%.

**Validates: Requirements 10.6, 10.7, 10.8, 10.9, 10.10**

### Property 8: Cadence Color Coding

*For any* activity with cadence value, the cadence dot color should be green when cadence >= 170 spm, yellow when cadence >= 160 spm, and red when cadence < 160 spm.

**Validates: Requirements 10.11**

### Property 9: Linear Regression Calculation

*For any* dataset with at least 2 activities, calculating the linear regression trend for Efficiency Factor over time should produce a valid slope value that can be used for trend analysis.

**Validates: Requirements 10.2**

## Error Handling

### File Import Errors

**Empty Folder**: When a selected folder contains no .FIT files, display a notification and return to ready state without changing the database.

**Corrupted Files**: When a .FIT file cannot be parsed by FitAnalyzer, log the error, skip that file, and continue processing remaining files.

**Permission Errors**: When file access is denied, display an error notification with the filename and continue processing other files.

**Database Errors**: When database operations fail, display an error notification and rollback any partial changes to maintain data integrity.

### UI State Errors

**No Data State**: When no activities match the current timeframe filter, display appropriate messages in each tab:
- Trends: "No data available. Import activities to view trends."
- Report: "No runs found for this timeframe."
- Activities: "No activities found."

**Empty Database**: When the database is empty on startup, disable export and copy buttons until data is imported.

**Concurrent Operations**: Prevent multiple simultaneous import operations by disabling the import button during processing.

### Chart Generation Errors

**Insufficient Data**: When fewer than 2 activities exist, display a message instead of attempting to generate charts with insufficient data points.

**Missing Fields**: When activity data is missing required fields (date, EF, decoupling), use default values (0 for numeric, empty string for text) and log a warning.

**Regression Failure**: When linear regression calculation fails (e.g., all EF values identical), catch the exception and display "Trend: Insufficient Data" message.

### Export Errors

**File Write Failure**: When CSV export fails due to permissions or disk space, display an error notification with the specific error message.

**Clipboard Access Failure**: When clipboard operations fail (e.g., in some sandboxed environments), display an error notification suggesting manual copy.

## Testing Strategy

### Dual Testing Approach

The testing strategy employs both unit tests and property-based tests to ensure comprehensive coverage:

**Unit Tests**: Focus on specific examples, edge cases, and integration points:
- UI component initialization and structure
- Specific timeframe filter examples (e.g., "Last 30 Days" with known dates)
- File import workflow with sample .FIT files
- Database operations with known data
- Chart generation with sample datasets
- Error handling scenarios

**Property-Based Tests**: Verify universal properties across randomized inputs:
- File hash deduplication with random file contents
- Timeframe filtering with random dates and activities
- Color mapping logic with random EF and decoupling values
- Session ID assignment with random import batches
- Activity deletion with random activities
- Trend calculation with random datasets

### Property-Based Testing Configuration

**Library**: Use Hypothesis for Python property-based testing

**Test Configuration**:
- Minimum 100 iterations per property test
- Each test tagged with format: `# Feature: nicegui-refactor, Property N: [property text]`
- Use custom strategies for generating valid activity data
- Use database fixtures for isolation between tests

**Example Property Test Structure**:

```python
from hypothesis import given, strategies as st
import pytest

# Feature: nicegui-refactor, Property 2: File Hash Deduplication
@given(file_content=st.binary(min_size=100, max_size=10000))
def test_file_hash_deduplication(file_content, tmp_path, db_manager):
    """For any FIT file, duplicate hashes should skip import."""
    # Create temporary file with content
    file_path = tmp_path / "test.fit"
    file_path.write_bytes(file_content)
    
    # Calculate hash
    hash1 = calculate_file_hash(file_path)
    
    # Import once
    db_manager.insert_activity({'filename': 'test.fit', 'date': '2024-01-01'}, hash1, 1)
    
    # Check exists
    assert db_manager.activity_exists(hash1) == True
    
    # Attempt duplicate import should be skipped
    initial_count = db_manager.get_count()
    # (import logic would skip here)
    assert db_manager.get_count() == initial_count
```

### Unit Test Coverage Areas

**Component Tests**:
- Sidebar renders with all required buttons and controls
- Tabs render with correct structure (Trends, Report, Activities)
- AG Grid initializes with correct column definitions
- Plotly chart renders with correct subplot structure

**Integration Tests**:
- Import workflow: folder selection → processing → database update → UI refresh
- Filter change: dropdown selection → database query → all tabs update
- Delete workflow: button click → confirmation → database removal → UI refresh
- Export workflow: button click → file dialog → CSV generation → success notification

**Edge Case Tests**:
- Empty database on startup
- Import folder with no .FIT files
- Import folder with all duplicate files
- Timeframe filter with no matching activities
- Chart generation with single activity
- Delete last remaining activity

### Testing Constraints

**Avoid Excessive Unit Tests**: Since property-based tests handle comprehensive input coverage, unit tests should focus on:
- Specific examples that demonstrate correct behavior
- Integration points between components
- Edge cases that are difficult to generate randomly
- Error conditions and exception handling

**Balance**: Aim for ~20-30 unit tests covering specific scenarios and ~10 property tests covering universal behaviors.

## Implementation Notes

### NiceGUI-Specific Patterns

**Async/Await**: File import and database operations use async/await to prevent UI blocking:
```python
async def process_folder_async(self, folder_path):
    # Long-running operation
    for file in files:
        # Process file
        await asyncio.sleep(0)  # Yield to event loop
```

**Reactive Updates**: Use NiceGUI's binding system for automatic UI updates:
```python
self.runs_label.bind_text_from(self.db, 'count', 
                                lambda c: f'Runs Stored: {c}')
```

**Component Clearing**: When updating dynamic content, clear containers first:
```python
self.grid_container.clear()
with self.grid_container:
    # Rebuild content
```

### Tailwind CSS Classes

**Color Palette**:
- Background: `bg-zinc-900` (sidebar), `bg-zinc-800` (main)
- Text: `text-gray-500` (muted), `text-white` (primary)
- Accents: `text-green-500` (success), `text-red-500` (error), `text-orange-500` (warning)

**Layout**:
- Sidebar: `w-56 h-screen p-4`
- Buttons: `w-full mb-2`
- Spacing: `mb-4` (medium), `mb-2` (small), `my-4` (vertical)

### Plotly Configuration

**Dark Theme**: Use `template="plotly_dark"` for consistent dark mode appearance

**Responsive Sizing**: Set `height=900` for desktop window, use `classes('w-full h-full')` for container

**Interactive Config**: Disable unnecessary toolbar buttons:
```python
config = {
    'displayModeBar': True,
    'displaylogo': False,
    'modeBarButtonsToRemove': ['zoom2d', 'pan2d', 'select2d', 'lasso2d']
}
```

### AG Grid Configuration

**Dark Theme**: Use `'theme': 'balham-dark'` for professional dark appearance

**Column Configuration**:
```python
{
    'field': 'cost',
    'headerName': 'Cost',
    'cellStyle': {
        'params': 'value.cost > 5 ? {color: "#ff4d4d"} : {color: "#2CC985"}'
    }
}
```

**Row Selection**: Enable `'rowSelection': 'single'` for delete functionality

### Database Preservation

The DatabaseManager class is used as-is from the original implementation:
- No modifications to SQL queries
- No changes to JSON serialization
- No alterations to hash-based deduplication logic

This ensures data compatibility and reduces refactoring risk.

### Entry Point

```python
def main():
    """Application entry point."""
    ui.run(
        native=True,
        window_size=(1200, 900),
        title="Garmin Analyzer Pro",
        reload=False  # Disable auto-reload in production
    )

if __name__ == "__main__":
    main()
```

## Migration Checklist

- [ ] Install NiceGUI: `pip install nicegui`
- [ ] Create app.py with NiceGUI imports
- [ ] Copy DatabaseManager class unchanged
- [ ] Copy calculate_file_hash function unchanged
- [ ] Implement GarminAnalyzerApp class with NiceGUI components
- [ ] Implement sidebar with Tailwind CSS styling
- [ ] Implement tabbed interface (Trends, Report, Activities)
- [ ] Implement Plotly chart generation (preserve logic from gui.py)
- [ ] Implement AG Grid table with dark theme
- [ ] Implement async file import with progress dialog
- [ ] Implement timeframe filtering
- [ ] Implement CSV export with LLM context
- [ ] Implement clipboard copy functionality
- [ ] Test all functionality with sample .FIT files
- [ ] Verify dark theme appearance
- [ ] Verify native window mode operation
- [ ] Remove gui.py after successful migration
