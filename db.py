import sqlite3
import json
from datetime import datetime
import pandas as pd
import hashlib

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
            # Enhanced schema with specific columns for sorting/filtering
            conn.execute('''
                CREATE TABLE IF NOT EXISTS activities (
                    hash TEXT PRIMARY KEY,
                    filename TEXT,
                    date TEXT,
                    timestamp_utc INTEGER,  -- Source of truth for sorting
                    distance_mi REAL,
                    duration_min REAL,
                    pace_min_mi REAL,
                    avg_hr INTEGER,
                    max_hr INTEGER,
                    elevation_ft INTEGER,
                    efficiency_factor REAL,
                    decoupling REAL,
                    avg_cadence INTEGER,
                    json_data TEXT,
                    session_id INTEGER,
                    import_session_id INTEGER,
                    file_path TEXT,
                    analyzer_version INTEGER NOT NULL DEFAULT 0,
                    analysis_status TEXT NOT NULL DEFAULT 'stale',
                    analysis_attempts INTEGER NOT NULL DEFAULT 0,
                    analysis_last_error TEXT,
                    analysis_next_retry_at TEXT,
                    analyzed_at TEXT,
                    last_processed_at TEXT
                )
            ''')
            
            # Migration: Check for new columns and add them if missing
            # This allows existing users (you) to upgrade without losing data
            cursor = conn.execute("PRAGMA table_info(activities)")
            columns = [info[1] for info in cursor.fetchall()]
            
            migrations = {
                'timestamp_utc': 'INTEGER',
                'distance_mi': 'REAL',
                'duration_min': 'REAL',
                'pace_min_mi': 'REAL',
                'avg_hr': 'INTEGER',
                'max_hr': 'INTEGER',
                'elevation_ft': 'INTEGER',
                'efficiency_factor': 'REAL',
                'decoupling': 'REAL',
                'avg_cadence': 'INTEGER',
                'import_session_id': 'INTEGER',
                'analyzer_version': 'INTEGER NOT NULL DEFAULT 0',
                'analysis_status': "TEXT NOT NULL DEFAULT 'stale'",
                'analysis_attempts': 'INTEGER NOT NULL DEFAULT 0',
                'analysis_last_error': 'TEXT',
                'analysis_next_retry_at': 'TEXT',
                'analyzed_at': 'TEXT',
                'last_processed_at': 'TEXT',
            }
            
            for col, dtype in migrations.items():
                if col not in columns:
                    print(f"Migrating database: Adding {col} column...")
                    conn.execute(f"ALTER TABLE activities ADD COLUMN {col} {dtype}")

            # Backfill for existing databases.
            conn.execute(
                "UPDATE activities SET import_session_id = session_id "
                "WHERE import_session_id IS NULL"
            )
            conn.execute(
                "UPDATE activities SET analyzer_version = 0 "
                "WHERE analyzer_version IS NULL"
            )
            conn.execute(
                "UPDATE activities SET analysis_status = 'stale' "
                "WHERE analysis_status IS NULL OR analysis_status = ''"
            )
            conn.execute(
                "UPDATE activities SET analysis_attempts = 0 "
                "WHERE analysis_attempts IS NULL"
            )
            conn.execute(
                "UPDATE activities SET analysis_status = 'stale' "
                "WHERE analysis_status = 'reprocessing'"
            )

            # Indexes for stale scan + timeline queries.
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_activities_analysis_scan "
                "ON activities(analysis_status, analyzer_version, analysis_next_retry_at)"
            )
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_activities_import_session "
                "ON activities(import_session_id)"
            )

    def insert_activity(self, activity_data, file_hash, session_id, file_path=None):
        json_str = json.dumps(activity_data, default=str)
        
        # Parse pace string "8:30" to float 8.5 for sorting
        pace_str = activity_data.get('pace', '0:00')
        try:
            m, s = map(int, pace_str.split(':'))
            pace_val = m + s/60.0
        except:
            pace_val = 0.0

        now_iso = datetime.utcnow().isoformat(timespec='milliseconds') + "Z"
        analyzer_version = int(activity_data.get('analyzer_version') or 0)
        analysis_status = activity_data.get('analysis_status') or 'fresh'
        analysis_attempts = int(activity_data.get('analysis_attempts') or 0)
        analysis_last_error = activity_data.get('analysis_last_error')
        analysis_next_retry_at = activity_data.get('analysis_next_retry_at')
        analyzed_at = activity_data.get('analyzed_at') or now_iso
        last_processed_at = activity_data.get('last_processed_at') or now_iso
        import_session_id = activity_data.get('import_session_id')
        if import_session_id is None:
            import_session_id = session_id

        with self.get_connection() as conn:
            conn.execute('''
                INSERT INTO activities (
                    hash, filename, date, timestamp_utc,
                    distance_mi, duration_min, pace_min_mi,
                    avg_hr, max_hr, elevation_ft,
                    efficiency_factor, decoupling, avg_cadence,
                    json_data, session_id, import_session_id, file_path,
                    analyzer_version, analysis_status, analysis_attempts,
                    analysis_last_error, analysis_next_retry_at,
                    analyzed_at, last_processed_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(hash) DO UPDATE SET
                    filename = excluded.filename,
                    date = excluded.date,
                    timestamp_utc = excluded.timestamp_utc,
                    distance_mi = excluded.distance_mi,
                    duration_min = excluded.duration_min,
                    pace_min_mi = excluded.pace_min_mi,
                    avg_hr = excluded.avg_hr,
                    max_hr = excluded.max_hr,
                    elevation_ft = excluded.elevation_ft,
                    efficiency_factor = excluded.efficiency_factor,
                    decoupling = excluded.decoupling,
                    avg_cadence = excluded.avg_cadence,
                    json_data = excluded.json_data,
                    file_path = COALESCE(excluded.file_path, activities.file_path),
                    analyzer_version = excluded.analyzer_version,
                    analysis_status = excluded.analysis_status,
                    analysis_attempts = excluded.analysis_attempts,
                    analysis_last_error = excluded.analysis_last_error,
                    analysis_next_retry_at = excluded.analysis_next_retry_at,
                    analyzed_at = excluded.analyzed_at,
                    last_processed_at = excluded.last_processed_at,
                    import_session_id = COALESCE(activities.import_session_id, excluded.import_session_id),
                    session_id = COALESCE(activities.session_id, excluded.session_id)
            ''', (
                file_hash,
                activity_data.get('filename'),
                activity_data.get('date'),
                activity_data.get('timestamp_utc', 0),  # New UTC field
                activity_data.get('distance_mi', 0),
                activity_data.get('moving_time_min', 0),
                pace_val,
                activity_data.get('avg_hr', 0),
                activity_data.get('max_hr', 0),
                activity_data.get('elevation_ft', 0),
                activity_data.get('efficiency_factor', 0),
                activity_data.get('decoupling', 0),
                activity_data.get('avg_cadence', 0),
                json_str,
                session_id,
                import_session_id,
                file_path,
                analyzer_version,
                analysis_status,
                analysis_attempts,
                analysis_last_error,
                analysis_next_retry_at,
                analyzed_at,
                last_processed_at,
            ))

    def update_activity_map_payload(self, file_hash, map_payload, route_segments=None, bounds=None, map_payload_version=None):
        """Patch map payload fields inside json_data for an existing activity row."""
        with self.get_connection() as conn:
            row = conn.execute(
                "SELECT json_data FROM activities WHERE hash = ?",
                (file_hash,),
            ).fetchone()
            if not row:
                return False

            try:
                activity_data = json.loads(row[0]) if row[0] else {}
            except Exception:
                activity_data = {}

            activity_data['map_payload'] = map_payload or {}

            if map_payload_version is None:
                if isinstance(map_payload, dict):
                    map_payload_version = map_payload.get('v', 1)
                else:
                    map_payload_version = 1
            activity_data['map_payload_version'] = int(map_payload_version)

            if route_segments is not None:
                activity_data['route_segments'] = route_segments
            elif isinstance(map_payload, dict) and 'segments' in map_payload:
                activity_data['route_segments'] = map_payload.get('segments', [])

            if bounds is not None:
                activity_data['bounds'] = bounds
            elif isinstance(map_payload, dict) and 'bounds' in map_payload:
                activity_data['bounds'] = map_payload.get('bounds', [[0, 0], [0, 0]])

            conn.execute(
                "UPDATE activities SET json_data = ? WHERE hash = ?",
                (json.dumps(activity_data, default=str), file_hash),
            )
            return True

    def activity_exists(self, file_hash):
        with self.get_connection() as conn:
            cursor = conn.execute("SELECT 1 FROM activities WHERE hash = ?", (file_hash,))
            return cursor.fetchone() is not None

    def delete_activity(self, file_hash):
        with self.get_connection() as conn:
            conn.execute("DELETE FROM activities WHERE hash = ?", (file_hash,))

    def delete_library_file(self, file_hash):
        with self.get_connection() as conn:
            conn.execute("DELETE FROM library_files WHERE content_hash = ?", (file_hash,))

    def get_activity_file_path(self, file_hash):
        with self.get_connection() as conn:
            row = conn.execute(
                "SELECT file_path FROM activities WHERE hash = ?",
                (file_hash,),
            ).fetchone()
            if row:
                return row[0]
            return None

    def get_last_session_id(self):
        """Retrieve the session_id from the most recent import."""
        with self.get_connection() as conn:
            # Use immutable import session identity when available.
            row = conn.execute(
                "SELECT MAX(COALESCE(import_session_id, session_id)) FROM activities"
            ).fetchone()
            return row[0] if row[0] is not None else None
            
    def get_count(self):
        with self.get_connection() as conn:
            return conn.execute("SELECT COUNT(*) FROM activities").fetchone()[0]

    def get_activities(self, timeframe="All Time", current_session_id=None, sort_by='date', sort_order='desc'):
        """
        Fetch activities with server-side sorting.
        sort_by options: 'date', 'distance', 'duration', 'elevation', 'efficiency', 'decoupling'
        """
        # Map UI sort keys to DB columns
        sort_map = {
            'date': 'timestamp_utc',     # Sort by the raw integer timestamp
            'distance': 'distance_mi',
            'duration': 'duration_min',
            'elevation': 'elevation_ft',
            'efficiency': 'efficiency_factor',
            'decoupling': 'decoupling',
            'cadence': 'avg_cadence'
        }
        
        # Default to timestamp_utc if key not found
        db_sort_col = sort_map.get(sort_by, 'timestamp_utc')
        
        # Validate order
        order_sql = "ASC" if sort_order == "asc" else "DESC"

        query = f"SELECT json_data, hash, file_path FROM activities"
        params = []
        
        # --- Timeframe Filtering ---
        where_clauses = []
        if timeframe == "Last Import" and current_session_id:
            where_clauses.append("COALESCE(import_session_id, session_id) = ?")
            params.append(current_session_id)
        elif timeframe == "Last 30 Days":
            date_limit = (datetime.now() - pd.Timedelta(days=30)).strftime("%Y-%m-%d")
            where_clauses.append("date >= ?")
            params.append(date_limit)
        elif timeframe == "Last 90 Days":
            date_limit = (datetime.now() - pd.Timedelta(days=90)).strftime("%Y-%m-%d")
            where_clauses.append("date >= ?")
            params.append(date_limit)
        elif timeframe == "This Year":
            current_year = datetime.now().year
            where_clauses.append("date >= ?")
            params.append(f"{current_year}-01-01")
            
        if where_clauses:
            query += " WHERE " + " AND ".join(where_clauses)
            
        # --- The Magic: Server-Side Sorting ---
        query += f" ORDER BY {db_sort_col} {order_sql}"
        
        with self.get_connection() as conn:
            rows = conn.execute(query, params).fetchall()
            results = []
            for row in rows:
                d = json.loads(row[0])
                d['db_hash'] = row[1]
                # Explicitly capture the file_path from the query result
                d['file_path'] = row[2] 
                results.append(d)
            return results

    def get_activity_by_hash(self, file_hash):
        with self.get_connection() as conn:
            row = conn.execute(
                "SELECT json_data, file_path FROM activities WHERE hash = ?",
                (file_hash,)
            ).fetchone()
            if row:
                activity = json.loads(row[0])
                activity['file_path'] = row[1]
                activity['hash'] = file_hash
                return activity
            return None

    def get_stale_activity_candidates(self, target_version, now_iso, limit, manual_mode=False):
        manual_flag = 1 if manual_mode else 0
        with self.get_connection() as conn:
            try:
                rows = conn.execute(
                    """
                    SELECT
                        a.hash,
                        a.file_path AS activity_file_path,
                        lf.file_path AS library_file_path,
                        COALESCE(a.analyzer_version, 0) AS analyzer_version,
                        COALESCE(a.analysis_status, 'stale') AS analysis_status,
                        COALESCE(a.analysis_attempts, 0) AS analysis_attempts,
                        a.analysis_next_retry_at,
                        COALESCE(lf.is_missing, 0) AS is_missing
                    FROM activities a
                    LEFT JOIN library_files lf ON lf.content_hash = a.hash
                    WHERE
                        (
                            COALESCE(a.analyzer_version, 0) < ?
                            OR COALESCE(a.analysis_status, 'stale') IN ('stale', 'failed', 'stale_missing_source')
                        )
                        AND
                        (
                            COALESCE(a.analysis_status, 'stale') != 'failed'
                            OR ? = 1
                            OR (a.analysis_next_retry_at IS NOT NULL AND a.analysis_next_retry_at <= ?)
                        )
                    ORDER BY COALESCE(a.last_processed_at, a.analyzed_at, '1970-01-01T00:00:00Z') ASC
                    LIMIT ?
                    """,
                    (target_version, manual_flag, now_iso, int(limit)),
                ).fetchall()
            except sqlite3.OperationalError:
                # Fallback for environments where library_files table is unavailable.
                rows = conn.execute(
                    """
                    SELECT
                        a.hash,
                        a.file_path AS activity_file_path,
                        NULL AS library_file_path,
                        COALESCE(a.analyzer_version, 0) AS analyzer_version,
                        COALESCE(a.analysis_status, 'stale') AS analysis_status,
                        COALESCE(a.analysis_attempts, 0) AS analysis_attempts,
                        a.analysis_next_retry_at,
                        0 AS is_missing
                    FROM activities a
                    WHERE
                        (
                            COALESCE(a.analyzer_version, 0) < ?
                            OR COALESCE(a.analysis_status, 'stale') IN ('stale', 'failed', 'stale_missing_source')
                        )
                        AND
                        (
                            COALESCE(a.analysis_status, 'stale') != 'failed'
                            OR ? = 1
                            OR (a.analysis_next_retry_at IS NOT NULL AND a.analysis_next_retry_at <= ?)
                        )
                    ORDER BY COALESCE(a.last_processed_at, a.analyzed_at, '1970-01-01T00:00:00Z') ASC
                    LIMIT ?
                    """,
                    (target_version, manual_flag, now_iso, int(limit)),
                ).fetchall()
        return rows

    def mark_activity_reprocessing(self, file_hash, now_iso):
        with self.get_connection() as conn:
            cursor = conn.execute(
                """
                UPDATE activities
                SET
                    analysis_status = 'reprocessing',
                    analysis_last_error = NULL,
                    last_processed_at = ?
                WHERE hash = ?
                  AND COALESCE(analysis_status, 'stale') != 'reprocessing'
                """,
                (now_iso, file_hash),
            )
            return cursor.rowcount > 0

    def mark_activity_reprocess_success(self, file_hash, now_iso, analyzer_version):
        with self.get_connection() as conn:
            conn.execute(
                """
                UPDATE activities
                SET
                    analyzer_version = ?,
                    analysis_status = 'fresh',
                    analysis_attempts = 0,
                    analysis_last_error = NULL,
                    analysis_next_retry_at = NULL,
                    analyzed_at = ?,
                    last_processed_at = ?
                WHERE hash = ?
                """,
                (int(analyzer_version), now_iso, now_iso, file_hash),
            )

    def mark_activity_reprocess_failure(self, file_hash, now_iso, error_text, next_retry_at):
        with self.get_connection() as conn:
            conn.execute(
                """
                UPDATE activities
                SET
                    analysis_status = 'failed',
                    analysis_attempts = COALESCE(analysis_attempts, 0) + 1,
                    analysis_last_error = ?,
                    analysis_next_retry_at = ?,
                    last_processed_at = ?
                WHERE hash = ?
                """,
                (error_text, next_retry_at, now_iso, file_hash),
            )

    def mark_activity_missing_source(self, file_hash, now_iso, error_text='FIT source file missing on disk.'):
        with self.get_connection() as conn:
            conn.execute(
                """
                UPDATE activities
                SET
                    analysis_status = 'stale_missing_source',
                    analysis_last_error = ?,
                    analysis_next_retry_at = NULL,
                    last_processed_at = ?
                WHERE hash = ?
                """,
                (error_text, now_iso, file_hash),
            )

# --- IMPORTANT: THIS FUNCTION IS NOW OUTSIDE THE CLASS ---
def calculate_file_hash(filepath):
    """Calculate SHA-256 hash of a file for deduplication."""
    hasher = hashlib.sha256()
    with open(filepath, 'rb') as f:
        buf = f.read(65536)
        while len(buf) > 0:
            hasher.update(buf)
            buf = f.read(65536)
    return hasher.hexdigest()
