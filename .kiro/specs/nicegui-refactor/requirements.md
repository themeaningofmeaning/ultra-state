# Requirements Document: NiceGUI Refactor

## Introduction

This specification defines the requirements for refactoring a Python desktop application from CustomTkinter to NiceGUI. The application is a Garmin FIT file analyzer that processes running activity data, calculates performance metrics, and displays interactive visualizations. The refactor aims to modernize the user interface with a "Silicon Valley" style dark-themed design while preserving all existing functionality.

## Glossary

- **System**: The Garmin FIT Analyzer application
- **FIT_File**: A binary file format used by Garmin devices to store activity data
- **Activity**: A single running session recorded by a Garmin device
- **Efficiency_Factor**: Normalized graded speed (m/min) divided by heart rate, measuring aerobic efficiency
- **Aerobic_Decoupling**: Percentage loss of efficiency from first half to second half of a run
- **HRR**: Heart Rate Recovery, measured as BPM drop 60 seconds after peak efforts
- **Database**: SQLite database storing activity records with hash-based deduplication
- **Timeframe**: User-selected filter for viewing activities (Last Import, Last 30 Days, Last 90 Days, This Year, All Time)
- **Native_Mode**: NiceGUI's desktop window mode using system webview
- **Plotly_Chart**: Interactive JavaScript-based data visualization
- **Session_ID**: Unix timestamp identifying a batch import operation

## Requirements

### Requirement 1: Framework Migration

**User Story:** As a developer, I want to migrate from CustomTkinter to NiceGUI, so that the application uses a modern web-based UI framework with native desktop capabilities.

#### Acceptance Criteria

1. THE System SHALL use NiceGUI framework instead of CustomTkinter
2. WHEN the application starts, THE System SHALL run in native mode with window size 1200x900 pixels
3. THE System SHALL set the window title to "Garmin Analyzer Pro"
4. THE System SHALL use Tailwind CSS for styling with Slate/Zinc color palette
5. THE System SHALL implement a dark theme as the default appearance

### Requirement 2: Layout Structure

**User Story:** As a user, I want a modern sidebar layout with main content area, so that I can easily navigate between different views and access key functions.

#### Acceptance Criteria

1. THE System SHALL display a fixed left sidebar with darker grey background
2. THE Sidebar SHALL contain a logo/title section at the top
3. THE Sidebar SHALL display a timeframe dropdown with options: Last Import, Last 30 Days, Last 90 Days, This Year, All Time
4. THE Sidebar SHALL provide an "Import Folder" button for loading FIT files
5. THE Sidebar SHALL provide an "Export CSV" button for data export
6. THE Sidebar SHALL provide a "Copy for LLM" button for clipboard operations
7. THE Sidebar SHALL display a "Runs Stored" counter showing total activities in database
8. THE System SHALL display a main content area with tabbed interface containing 3 tabs

### Requirement 3: Trends Tab Implementation

**User Story:** As a user, I want to view interactive training trends directly in the application, so that I can analyze my performance without opening external browser windows.

#### Acceptance Criteria

1. THE System SHALL display a "Trends" tab as the default active tab
2. WHEN the Trends tab is active, THE System SHALL render interactive Plotly charts using ui.plotly component
3. THE Plotly_Chart SHALL display two stacked graphs with shared x-axis
4. THE First_Graph SHALL show Efficiency Factor as a green line with dual y-axes
5. THE First_Graph SHALL show Aerobic Decoupling as filled zones on secondary y-axis
6. WHEN Aerobic Decoupling is positive, THE System SHALL render it as a red filled area
7. WHEN Aerobic Decoupling is negative, THE System SHALL render it as a teal filled area
8. THE Second_Graph SHALL show cadence as dots with color coding based on efficiency
9. THE System SHALL NOT include a "Launch Trends" button (trends are embedded in tab)
10. WHEN the application launches, THE System SHALL default to the Trends Tab active, so the user immediately sees their data without clicking anything

### Requirement 4: Report Tab Implementation

**User Story:** As a user, I want to view textual analysis of my runs, so that I can read detailed metrics for each activity.

#### Acceptance Criteria

1. THE System SHALL display a "Report" tab in the main content area
2. WHEN the Report tab is active, THE System SHALL display formatted run data in a text area or markdown view
3. THE Report_View SHALL show activities sorted by date in descending order
4. THE Report_View SHALL display metrics including distance, pace, Efficiency Factor, Aerobic Decoupling, and HRR for each activity
5. THE Report_View SHALL include visual indicators for decoupling status (✅ Excellent, ⚠️ Moderate, 🛑 High Fatigue)

### Requirement 5: Activities Tab Implementation

**User Story:** As a user, I want to view and manage my activities in a data table, so that I can see all runs at a glance and delete unwanted entries.

#### Acceptance Criteria

1. THE System SHALL display an "Activities" tab in the main content area
2. THE System SHALL render the data table using ui.aggrid (not ui.table) with a dark theme, allowing for column sorting and resizing, to maintain the 'Pro SaaS' look
3. THE Activities_Table SHALL display columns: Date, Filename, Distance, Elevation, EF, Cost (Decoupling), Cadence
4. THE Activities_Table SHALL provide a delete button for each row
5. THE Delete_Button SHALL use the flat and round props in NiceGUI. It must be invisible (opacity 0 or transparent) by default and only reveal a red background/text when hovered (hover:bg-red-900 class)
6. WHEN the user hovers over the delete button, THE System SHALL change its color to red
7. WHEN the user clicks a delete button, THE System SHALL prompt for confirmation before deletion
8. WHEN an activity is deleted, THE System SHALL remove it from the database and refresh all views

### Requirement 6: Database Operations

**User Story:** As a user, I want my activity data persisted in a database with automatic deduplication, so that I don't accidentally import the same files multiple times.

#### Acceptance Criteria

1. THE System SHALL use SQLite database (runner_stats.db) for storing activities
2. THE System SHALL calculate SHA-256 hash for each FIT file before import
3. WHEN importing a file, THE System SHALL check if the hash exists in the database
4. IF a file hash already exists, THEN THE System SHALL skip importing that file
5. WHEN importing files, THE System SHALL assign a session ID based on Unix timestamp
6. THE System SHALL store activity data as JSON in the database
7. WHEN a timeframe filter changes, THE System SHALL query the database with appropriate date constraints
8. WHEN "Last Import" timeframe is selected, THE System SHALL filter activities by current session ID

### Requirement 7: File Import Workflow

**User Story:** As a user, I want to import folders of FIT files with visual progress feedback, so that I know the import status and can see results immediately.

#### Acceptance Criteria

1. WHEN the user clicks "Import Folder", THE System SHALL open a folder selection dialog
2. WHEN a folder is selected, THE System SHALL disable the import button and timeframe dropdown
3. THE System SHALL display a progress indicator showing current file and total files
4. THE System SHALL process FIT files in a background thread to prevent UI blocking
5. WHEN import completes, THE System SHALL display count of new activities imported
6. WHEN import completes, THE System SHALL automatically switch timeframe to "Last Import"
7. WHEN import completes, THE System SHALL automatically switch to Report tab
8. WHEN import completes, THE System SHALL refresh all data views
9. WHEN no new files are imported, THE System SHALL display a message indicating all files were duplicates

### Requirement 8: Data Export Functionality

**User Story:** As a user, I want to export my activity data to CSV with AI context, so that I can analyze data in external tools or share with AI assistants.

#### Acceptance Criteria

1. THE System SHALL enable the "Export CSV" button when activities are loaded
2. WHEN the user clicks "Export CSV", THE System SHALL open a file save dialog
3. THE System SHALL export activities with columns: date, filename, distance_mi, pace, gap_pace, efficiency_factor, decoupling, avg_hr, avg_resp, avg_temp, avg_power, avg_cadence, hrr_list, v_ratio, gct_balance, gct_change, elevation_ft, moving_time_min, rest_time_min
4. THE System SHALL append a data dictionary section to the CSV file explaining metrics for AI analysis
5. THE Data_Dictionary SHALL include definitions for HRR, Efficiency Factor, Aerobic Decoupling, and form metrics
6. WHEN export completes, THE System SHALL display a success message

### Requirement 9: LLM Context Copy

**User Story:** As a user, I want to copy my run data with explanatory context to clipboard, so that I can paste it into AI chat interfaces for coaching advice.

#### Acceptance Criteria

1. THE System SHALL enable the "Copy for LLM" button when activities are loaded
2. WHEN the user clicks "Copy for LLM", THE System SHALL copy report text and context dictionary to clipboard
3. WHEN more than 20 activities are loaded, THE System SHALL copy monthly summary instead of individual runs
4. THE Clipboard_Content SHALL include explanations of Efficiency Factor, Aerobic Decoupling, HRR, and form metrics
5. WHEN copy completes, THE System SHALL change button text to "✅ Copied w/ Context!" for 2 seconds
6. WHEN copy completes, THE System SHALL change button color to white temporarily

### Requirement 10: Plotly Chart Generation

**User Story:** As a user, I want to see interactive charts with trend analysis and color-coded performance indicators, so that I can quickly understand my training progression.

#### Acceptance Criteria

1. THE System SHALL generate Plotly charts with two stacked subplots
2. THE System SHALL calculate linear regression trend for Efficiency Factor over time
3. WHEN EF trend slope is positive, THE System SHALL display "📈 Trend: Engine Improving" message in green
4. WHEN EF trend slope is negative, THE System SHALL display "📉 Trend: Fitness Declining" message in red
5. WHEN EF trend slope is near zero, THE System SHALL display "➡️ Trend: Fitness Stable" message in silver
6. THE System SHALL color-code activity markers based on EF and decoupling values
7. WHEN EF >= mean AND decoupling <= 5%, THE System SHALL use green marker (Race Ready)
8. WHEN EF >= mean AND decoupling > 5%, THE System SHALL use orange marker (Expensive Speed)
9. WHEN EF < mean AND decoupling <= 5%, THE System SHALL use yellow marker (Base Maintenance)
10. WHEN EF < mean AND decoupling > 5%, THE System SHALL use red marker (Struggling)
11. THE System SHALL color-code cadence dots: green for >= 170 spm, yellow for >= 160 spm, red for < 160 spm
12. THE Chart SHALL include custom hover tooltips showing verdict, EF, cost, pace, distance, and average HR
13. THE Chart SHALL include the static HTML annotation/legend at the bottom. This must be implemented using fig.add_annotation with xref='paper', yref='paper', y=-0.15 to ensure it sits outside the graph area but inside the widget, preserving the exact layout of the previous version

### Requirement 11: Timeframe Filtering

**User Story:** As a user, I want to filter activities by different time periods, so that I can focus on recent training or analyze long-term trends.

#### Acceptance Criteria

1. THE System SHALL provide a timeframe dropdown with 5 options
2. WHEN "Last Import" is selected, THE System SHALL display only activities from the most recent import session
3. WHEN "Last 30 Days" is selected, THE System SHALL display activities from the past 30 days
4. WHEN "Last 90 Days" is selected, THE System SHALL display activities from the past 90 days
5. WHEN "This Year" is selected, THE System SHALL display activities from January 1st of current year
6. WHEN "All Time" is selected, THE System SHALL display all activities in database
7. WHEN timeframe changes, THE System SHALL refresh all tabs with filtered data
8. WHEN no activities match the timeframe, THE System SHALL display an appropriate message

### Requirement 12: UI Styling and Theming

**User Story:** As a user, I want a modern, professional dark-themed interface, so that the application looks like a premium SaaS product.

#### Acceptance Criteria

1. THE System SHALL use Tailwind CSS Slate/Zinc color palette for dark mode
2. THE Sidebar SHALL have a darker grey background than the main content area
3. THE System SHALL apply consistent button styling across all components
4. THE System SHALL use smooth transitions for hover effects
5. THE Delete_Button SHALL transition from transparent to red on hover
6. THE System SHALL use modern typography with appropriate font sizes and weights
7. THE System SHALL maintain visual hierarchy with proper spacing and padding

### Requirement 13: Data Preservation

**User Story:** As a developer, I want to preserve all existing business logic from the CustomTkinter version, so that the refactor maintains feature parity.

#### Acceptance Criteria

1. THE System SHALL use the existing DatabaseManager class without modifications
2. THE System SHALL use the existing FitAnalyzer class from analyzer.py without modifications
3. THE System SHALL preserve the file hash calculation logic for deduplication
4. THE System SHALL preserve the color logic for performance indicators
5. THE System SHALL preserve the Plotly chart generation logic with dual-axis stacked graphs
6. THE System SHALL preserve the "Copy to LLM" context dictionary content
7. THE System SHALL preserve all metric calculations (EF, decoupling, HRR, etc.)
8. THE System SHALL preserve the monthly summary logic for datasets with > 20 activities
