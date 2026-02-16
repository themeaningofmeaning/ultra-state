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
                    file_path TEXT
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
                'avg_cadence': 'INTEGER'
            }
            
            for col, dtype in migrations.items():
                if col not in columns:
                    print(f"Migrating database: Adding {col} column...")
                    conn.execute(f"ALTER TABLE activities ADD COLUMN {col} {dtype}")

    def insert_activity(self, activity_data, file_hash, session_id, file_path=None):
        json_str = json.dumps(activity_data, default=str)
        
        # Parse pace string "8:30" to float 8.5 for sorting
        pace_str = activity_data.get('pace', '0:00')
        try:
            m, s = map(int, pace_str.split(':'))
            pace_val = m + s/60.0
        except:
            pace_val = 0.0

        with self.get_connection() as conn:
            conn.execute('''
                INSERT OR REPLACE INTO activities (
                    hash, filename, date, timestamp_utc,
                    distance_mi, duration_min, pace_min_mi,
                    avg_hr, max_hr, elevation_ft,
                    efficiency_factor, decoupling, avg_cadence,
                    json_data, session_id, file_path
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                file_hash, 
                activity_data.get('filename'), 
                activity_data.get('date'),
                activity_data.get('timestamp_utc', 0), # New UTC field
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
                file_path
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

    def get_last_session_id(self):
        """Retrieve the session_id from the most recent import."""
        with self.get_connection() as conn:
            # Get the maximum session_id present in the table
            row = conn.execute("SELECT MAX(session_id) FROM activities").fetchone()
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
            where_clauses.append("session_id = ?")
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
