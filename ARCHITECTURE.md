# Ultra State - Architecture & Design Rules

Status: **v1.0.1 (Architecture Complete)**

## Core Philosophy
Ultra State is a local-first, premium desktop analytics app for performance runners who care about their training data, but don't want to swim through an endless sea of metrics and graphs with no tangible takeaways.  

The architecture favors clear ownership boundaries:
- `core/` owns business/data workflows.
- `components/` owns UI rendering and view behavior.
- `app.py` coordinates lifecycle, dependency wiring, and cross-component orchestration.

## System Architecture
### Hub-and-Spoke Pattern
`/Users/meaning/Documents/ultra-state/app.py` is the lifecycle hub:
- Initializes long-lived services (`DatabaseManager`, `LibraryManager`, `AppState`, `DataManager`, `MapPayloadBuilder`, `LLMExporter`).
- Wires components and injects callbacks/dependencies.
- Coordinates refresh flows and navigation-level events.

Architecture rule for all new code:
- `app.py` must not directly parse FIT files.
- `app.py` must not own DataFrame aggregation/business calculations.
- `app.py` must not introduce new large, component-like UI layouts.

Note on current reality:
- `app.py` still contains some legacy controller/UI methods (table rendering and several info dialogs). These are acceptable in v1.0.1 but should be extracted when touched.

### Directory Map
`/Users/meaning/Documents/ultra-state/core/` (logic services):
- `data_manager.py`: owns `df`, activity list cache, filtering, run classification, CSV export generation.
- `map_payload.py`: owns map payload geometry/bounds normalization, segment color migration, and payload backfill.
- `llm_export.py`: owns "Copy for AI" export orchestration, FIT parsing pipeline handoff, and progress reporting.

`/Users/meaning/Documents/ultra-state/components/` (UI views):
- `layout.py`: app shell (sidebar, header/tabs, FAB).
- `analysis_view.py`: trends dashboard and chart interactions.
- `activity_modal.py`: activity detail modal and modal hydration/fetch flow.
- `library_modal.py`: library settings, sync status, folder/import actions.
- `run_card.py`: feed run-card renderer.
- `cards.py` and `charts.py`: reusable UI/Plotly rendering primitives.

Core analytics:
- `analyzer.py`: canonical analysis primitives (HR zones, gap pace helpers, decoupling, terrain/run-walk metrics, form analysis), plus shared FIT-analysis helpers consumed by core/components.

### Key Patterns
Dependency Injection:
- Components and services receive only what they need (`db`, `state`, `library_manager`, callbacks).
- `components/*` and `core/*` must never import `app.py` or `UltraStateApp`.

State:
- `/Users/meaning/Documents/ultra-state/state.py` (`AppState`) is the single source of truth for session-level UI state.
- `DataManager` owns the activity data cache (`df`, `activities_data`) derived from DB.

Events:
- UI components emit intent through injected callbacks.
- Data mutation and persistence live in services (`core/`, DB layer), not view components.

## Data Ingestion Rules
- **Manual FIT Files Only:** The app relies on local `.fit` imports via library folder scanning/watching. It does not use Garmin cloud API scraping.
- **Strict Sport Guard:** Only `running` and `trail_running` activities are processed. Other sports are rejected in parser flow and treated as clean skips.

## UI & Visual Design Language
- **Strict Color Decoupling:**
  - **UI/Navigation Actions:** Mint/Emerald is reserved for primary actions (for example, "Copy for AI").
  - **Data States:** Blue = Recovery, Green = Base, Orange = Overload, Red = Overreaching.
  - **Monochrome Hardware:** Utility controls (for example, Focus mode token) should stay zinc/neutral with high-contrast text.

## Color Gradient — Single Source of Truth
The canonical Garmin 5-color speed gradient (Blue -> Green -> Yellow -> Orange -> Red) lives in `/Users/meaning/Documents/ultra-state/analyzer.py`:
- `_get_speed_color(speed_mps, min_speed, max_speed)` for raw speed values.
- `gradient_color_from_t(t)` for pre-normalized values (`0.0` to `1.0`).

Do not re-implement this gradient anywhere else. Import from `analyzer.py`.

For map payload migration/colorization, `/Users/meaning/Documents/ultra-state/core/map_payload.py` must call `gradient_color_from_t(...)` rather than defining another gradient implementation.

## Auto-Update Strategy
Version checks use the GitHub Releases API (notification-only flow in `/Users/meaning/Documents/ultra-state/updater.py`).

The local app version is defined as `APP_VERSION` in `/Users/meaning/Documents/ultra-state/updater.py`. Bump that value for each release.
