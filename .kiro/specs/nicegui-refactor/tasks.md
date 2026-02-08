# Implementation Plan: NiceGUI Refactor

## Overview

This implementation plan converts the CustomTkinter-based Garmin FIT Analyzer to NiceGUI while preserving all existing functionality. The approach is incremental, building the UI structure first, then adding data operations, and finally integrating the complex chart generation logic.

## Tasks

- [x] 1. Set up project structure and dependencies
  - Install NiceGUI package: `pip install nicegui`
  - Create app.py file
  - Import required libraries (nicegui, pandas, plotly, sqlite3, hashlib, etc.)
  - Copy DatabaseManager class unchanged from gui.py
  - Copy calculate_file_hash function unchanged from gui.py
  - _Requirements: 1.1, 13.1, 13.3_

- [ ] 2. Implement basic application structure and entry point
  - [x] 2.1 Create GarminAnalyzerApp class with initialization
    - Initialize DatabaseManager instance
    - Set up state variables (current_session_id, current_timeframe, activities_data, df)
    - _Requirements: 1.1, 13.1_
  
  - [x] 2.2 Implement main() entry point with native mode
    - Call ui.run() with native=True, window_size=(1200, 900), title="Garmin Analyzer Pro"
    - _Requirements: 1.2, 1.3_
  
  - [ ]* 2.3 Write unit test for application initialization
    - Test that DatabaseManager is created
    - Test that state variables are initialized correctly
    - _Requirements: 1.1_

- [ ] 3. Build sidebar UI component
  - [x] 3.1 Implement build_sidebar() method
    - Create left column with Tailwind classes: w-56 bg-zinc-900 p-4 h-screen
    - Add logo/title label with emoji
    - Add timeframe dropdown with 5 options
    - Add "Import Folder" button
    - Add "Export CSV" button (initially disabled)
    - Add "Copy for LLM" button (initially disabled, green background)
    - Add separator
    - Add "Runs Stored" counter label
    - Add status label
    - _Requirements: 2.1, 2.2, 2.3, 2.4, 2.5, 2.6, 2.7, 12.1, 12.2, 12.3_
  
  - [ ]* 3.2 Write unit test for sidebar structure
    - Test that all required UI elements exist
    - Test that buttons have correct initial states
    - _Requirements: 2.2, 2.3, 2.4, 2.5, 2.6, 2.7_

- [ ] 4. Build tabbed main content area
  - [x] 4.1 Implement build_main_content() method
    - Create main column for content area
    - Create ui.tabs() with three tabs: Trends, Report, Activities
    - Set Trends as default active tab
    - _Requirements: 2.8, 3.1, 3.10_
  
  - [x] 4.2 Implement build_trends_tab() method
    - Create tab panel for Trends
    - Add container for Plotly chart
    - Add placeholder message when no data available
    - _Requirements: 3.1, 3.2_
  
  - [x] 4.3 Implement build_report_tab() method
    - Create tab panel for Report
    - Add ui.markdown() or ui.textarea() for formatted text
    - Style with monospace font
    - _Requirements: 4.1, 4.2_
  
  - [x] 4.4 Implement build_activities_tab() method
    - Create tab panel for Activities
    - Add container for AG Grid
    - Add placeholder message when no data available
    - _Requirements: 5.1_
  
  - [ ]* 4.5 Write unit test for tab structure
    - Test that three tabs exist
    - Test that Trends is default active tab
    - _Requirements: 2.8, 3.1, 3.10_

- [x] 5. Checkpoint - Ensure UI structure renders correctly
  - Ensure all tests pass, ask the user if questions arise.

- [ ] 6. Implement database query and data refresh logic
  - [x] 6.1 Implement refresh_data_view() method
    - Query database using DatabaseManager.get_activities()
    - Convert results to Pandas DataFrame
    - Parse dates and sort by date
    - Update state variables (activities_data, df)
    - Enable/disable export and copy buttons based on data availability
    - _Requirements: 6.7, 6.8, 11.1, 11.7_
  
  - [x] 6.2 Implement on_filter_change() handler
    - Update current_timeframe state
    - Call refresh_data_view()
    - _Requirements: 11.1, 11.7_
  
  - [ ]* 6.3 Write property test for timeframe filtering
    - **Property 4: Timeframe Filtering**
    - **Validates: Requirements 6.7, 6.8, 11.2, 11.3, 11.4, 11.5, 11.6**
    - Generate random activities with various dates
    - Test each timeframe filter returns correct subset
    - _Requirements: 6.7, 6.8, 11.2, 11.3, 11.4, 11.5, 11.6_

- [ ] 7. Implement file import workflow
  - [x] 7.1 Implement select_folder() async method
    - Use JavaScript to trigger folder selection dialog (webkitdirectory attribute)
    - If JavaScript folder picker proves flaky in Native Mode, fallback to using tkinter.filedialog.askdirectory() (headless) as a robust alternative
    - Disable timeframe dropdown during import
    - Call process_folder_async() with selected path
    - _Requirements: 7.1, 7.2_
  
  - [x] 7.2 Implement process_folder_async() method
    - Generate session ID from Unix timestamp
    - Show progress dialog with progress bar
    - Get list of .FIT files from folder
    - Loop through files with FitAnalyzer
    - Calculate file hash for each file
    - Check if hash exists in database
    - Skip if exists, otherwise import
    - Update progress bar after each file
    - Close dialog and update UI when complete
    - Switch to "Last Import" timeframe
    - Call refresh_data_view()
    - _Requirements: 7.3, 7.4, 7.5, 7.6, 7.7, 7.8, 7.9, 13.2, 13.3, 13.4_
  
  - [ ]* 7.3 Write property test for file hash deduplication
    - **Property 2: File Hash Deduplication**
    - **Validates: Requirements 6.2, 6.3, 6.4**
    - Generate random file contents
    - Test that duplicate hashes skip import
    - _Requirements: 6.2, 6.3, 6.4_
  
  - [ ]* 7.4 Write property test for session ID assignment
    - **Property 3: Session ID Assignment**
    - **Validates: Requirements 6.5**
    - Test that all files in one import batch get same session ID
    - _Requirements: 6.5_
  
  - [ ]* 7.5 Write unit test for empty folder handling
    - Test that empty folder shows notification
    - Test that UI returns to ready state
    - _Requirements: 7.9_

- [ ] 8. Implement Report tab data display
  - [x] 8.1 Implement update_report_text() method
    - Check if activities_data is empty
    - Calculate average EF from DataFrame
    - Sort activities by date descending
    - Format each activity using format_run_data()
    - Join with separator lines
    - Update markdown/textarea content
    - _Requirements: 4.2, 4.3, 4.4, 13.7_
  
  - [x] 8.2 Implement format_run_data() method (preserve from gui.py)
    - Format activity dictionary into text block
    - Include distance, pace, EF, decoupling, HRR
    - Add status indicators (✅ Excellent, ⚠️ Moderate, 🛑 High Fatigue)
    - _Requirements: 4.4, 4.5, 13.7_
  
  - [ ]* 8.3 Write unit test for report formatting
    - Test format_run_data() with sample activity
    - Test that status indicators appear correctly
    - _Requirements: 4.4, 4.5_

- [ ] 9. Implement Activities tab with AG Grid
  - [x] 9.1 Implement update_activities_grid() method
    - Clear grid container
    - Check if activities_data is empty
    - Transform activities into row data format
    - Define column definitions with headers and widths
    - Add cellStyle for cost column (red if > 5%, green otherwise)
    - Create ui.aggrid with balham-dark theme
    - Enable single row selection
    - Add delete button below grid
    - _Requirements: 5.2, 5.3, 12.6_
  
  - [x] 9.2 Implement delete_selected_activity() async method
    - Get selected row from AG Grid
    - Show confirmation dialog
    - If confirmed, call DatabaseManager.delete_activity()
    - Update runs counter
    - Call refresh_data_view()
    - Show success notification
    - _Requirements: 5.4, 5.5, 5.6, 5.7, 5.8_
  
  - [ ]* 9.3 Write property test for activity deletion
    - **Property 5: Activity Deletion Propagation**
    - **Validates: Requirements 5.8**
    - Generate random activities
    - Test that deletion removes from database and triggers refresh
    - _Requirements: 5.8_
  
  - [ ]* 9.4 Write unit test for AG Grid structure
    - Test that grid has correct columns
    - Test that cost column has color styling
    - _Requirements: 5.2, 5.3_

- [x] 10. Checkpoint - Ensure data operations work correctly
  - Ensure all tests pass, ask the user if questions arise.

- [ ] 11. Implement Plotly chart generation
  - [x] 11.1 Implement generate_plotly_figure() method (preserve logic from gui.py)
    - Calculate linear regression trend for EF over time
    - Determine trend message and color based on slope
    - Calculate marker colors based on EF and decoupling logic
    - Calculate cadence colors based on cadence thresholds
    - Create subplots with make_subplots (2 rows, shared x-axis)
    - Add decoupling filled areas (red for positive, teal for negative)
    - Add EF line trace with colored markers
    - Add cadence scatter trace
    - Configure layout with dark theme
    - Add static HTML annotation legend at bottom (y=-0.15)
    - Return figure object
    - _Requirements: 3.3, 3.4, 3.5, 3.6, 3.7, 3.8, 10.1, 10.12, 10.13, 13.5, 13.7_
  
  - [ ]* 11.2 Write property test for decoupling color mapping
    - **Property 1: Decoupling Color Mapping**
    - **Validates: Requirements 3.6, 3.7**
    - Generate random decoupling values
    - Test that positive values map to red, negative to teal
    - _Requirements: 3.6, 3.7_
  
  - [ ]* 11.3 Write property test for trend slope message
    - **Property 6: Trend Slope Message Mapping**
    - **Validates: Requirements 10.3, 10.4, 10.5**
    - Generate random datasets with known slopes
    - Test that message matches slope thresholds
    - _Requirements: 10.3, 10.4, 10.5_
  
  - [ ]* 11.4 Write property test for activity marker colors
    - **Property 7: Activity Marker Color Logic**
    - **Validates: Requirements 10.6, 10.7, 10.8, 10.9, 10.10**
    - Generate random EF and decoupling values
    - Test that colors match the four-quadrant logic
    - _Requirements: 10.6, 10.7, 10.8, 10.9, 10.10_
  
  - [ ]* 11.5 Write property test for cadence color coding
    - **Property 8: Cadence Color Coding**
    - **Validates: Requirements 10.11**
    - Generate random cadence values
    - Test that colors match thresholds (170, 160)
    - _Requirements: 10.11_
  
  - [ ]* 11.6 Write property test for linear regression
    - **Property 9: Linear Regression Calculation**
    - **Validates: Requirements 10.2**
    - Generate random datasets with >= 2 activities
    - Test that regression produces valid slope
    - _Requirements: 10.2_
  
  - [ ]* 11.7 Write unit test for chart structure
    - Test that figure has 2 subplots
    - Test that legend annotation exists at y=-0.15
    - Test that hover tooltips contain required fields
    - _Requirements: 3.3, 3.4, 3.5, 10.1, 10.12, 10.13_

- [ ] 12. Implement chart display in Trends tab
  - [x] 12.1 Implement update_trends_chart() method
    - Clear plotly_container
    - Check if df is not None and not empty
    - If data exists, call generate_plotly_figure()
    - Create ui.plotly() with figure
    - If no data, show placeholder message
    - _Requirements: 3.1, 3.2_
  
  - [x] 12.2 Update refresh_data_view() to call update_trends_chart()
    - Add call to update_trends_chart() after updating report and grid
    - _Requirements: 3.1, 3.2_

- [ ] 13. Implement CSV export functionality
  - [x] 13.1 Implement export_csv() async method
    - Check if df is None
    - Show file save dialog
    - Export DataFrame to CSV with specified columns
    - Append data dictionary section to file
    - Show success notification
    - _Requirements: 8.1, 8.2, 8.3, 8.4, 8.5, 8.6, 13.6_
  
  - [ ]* 13.2 Write unit test for CSV export
    - Test that CSV contains correct columns
    - Test that data dictionary is appended
    - _Requirements: 8.3, 8.4, 8.5_

- [ ] 14. Implement LLM context copy functionality
  - [x] 14.1 Implement copy_to_llm() async method
    - Check if more than 20 activities (summary mode)
    - If summary mode, group by month and aggregate
    - Otherwise, get report text from report_area
    - Append LLM context dictionary
    - Copy to clipboard using JavaScript
    - Update button text and color temporarily
    - Reset button after 2 seconds
    - _Requirements: 9.1, 9.2, 9.3, 9.4, 9.5, 9.6, 13.6_
  
  - [ ]* 14.2 Write unit test for LLM context copy
    - Test that context dictionary is included
    - Test summary mode for > 20 activities
    - _Requirements: 9.2, 9.3, 9.4_

- [ ] 15. Implement error handling
  - [x] 15.1 Add error handling to file import
    - Handle empty folder (show notification)
    - Handle corrupted files (skip and continue)
    - Handle permission errors (show error and continue)
    - Handle database errors (rollback and notify)
    - _Requirements: 7.9_
  
  - [x] 15.2 Add error handling to chart generation
    - Handle insufficient data (< 2 activities)
    - Handle missing fields (use defaults)
    - Handle regression failure (show "Insufficient Data")
    - _Requirements: 10.2_
  
  - [x] 15.3 Add error handling to export operations
    - Handle file write failure (show error)
    - Handle clipboard access failure (show error)
    - _Requirements: 8.6_
  
  - [ ]* 15.4 Write unit tests for error scenarios
    - Test empty folder handling
    - Test insufficient data for charts
    - Test export failures
    - _Requirements: 7.9, 10.2, 8.6_

- [ ] 16. Apply Tailwind CSS styling and theming
  - [x] 16.1 Apply dark theme colors throughout
    - Sidebar: bg-zinc-900
    - Main content: bg-zinc-800
    - Text colors: text-gray-500 (muted), text-white (primary)
    - Accent colors: text-green-500, text-red-500, text-orange-500
    - _Requirements: 12.1, 12.2, 12.3_
  
  - [x] 16.2 Style buttons with consistent classes
    - Primary buttons: w-full mb-2
    - Ghost delete button: bg-transparent hover:bg-red-900 text-gray-600 hover:text-red-500
    - Green LLM button: bg-green-600
    - _Requirements: 5.5, 12.3, 12.4, 12.5_
  
  - [x] 16.3 Apply spacing and layout classes
    - Sidebar padding: p-4
    - Button spacing: mb-2, mb-4
    - Separator: my-4
    - _Requirements: 12.3, 12.7_

- [ ] 17. Final integration and testing
  - [x] 17.1 Wire all components together
    - Ensure refresh_data_view() updates all three tabs
    - Ensure filter changes trigger refresh
    - Ensure import completion triggers refresh
    - Ensure deletion triggers refresh
    - _Requirements: 6.7, 7.8, 11.7_
  
  - [x] 17.2 Test complete workflow end-to-end
    - Launch application in native mode
    - Import folder of FIT files
    - Verify all tabs display data correctly
    - Test timeframe filtering
    - Test activity deletion
    - Test CSV export
    - Test LLM copy
    - _Requirements: 1.2, 1.3, 7.1-7.9, 11.1-11.8_
  
  - [ ]* 17.3 Write integration tests
    - Test complete import workflow
    - Test filter change workflow
    - Test delete workflow
    - Test export workflow
    - _Requirements: 7.1-7.9, 11.1-11.8, 8.1-8.6, 9.1-9.6_

- [x] 18. Final checkpoint - Ensure all tests pass
  - Ensure all tests pass, ask the user if questions arise.

## Notes

- Tasks marked with `*` are optional property-based and unit tests that can be skipped for faster MVP
- Each task references specific requirements for traceability
- Checkpoints ensure incremental validation
- Property tests validate universal correctness properties with minimum 100 iterations
- Unit tests validate specific examples and edge cases
- The DatabaseManager and FitAnalyzer classes are preserved unchanged to maintain data compatibility
- All Plotly chart generation logic is preserved from the original gui.py implementation
- Tailwind CSS classes are used throughout for modern, consistent styling
