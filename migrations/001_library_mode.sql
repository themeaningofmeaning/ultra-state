-- Library Mode phase 1 schema additions.
-- Safe to run repeatedly.

CREATE TABLE IF NOT EXISTS settings (
    key TEXT PRIMARY KEY,
    value TEXT NOT NULL,
    updated_at TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ', 'now'))
);

CREATE TABLE IF NOT EXISTS library_files (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    content_hash TEXT NOT NULL UNIQUE,
    file_path TEXT NOT NULL,
    file_size_bytes INTEGER NOT NULL,
    file_mtime_ns INTEGER NOT NULL,
    import_status TEXT NOT NULL CHECK (import_status IN ('pending', 'imported', 'failed')),
    is_missing INTEGER NOT NULL DEFAULT 0,
    missing_since_at TEXT,
    last_error TEXT,
    first_seen_at TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ', 'now')),
    last_seen_at TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ', 'now')),
    imported_at TEXT,
    source_type TEXT NOT NULL DEFAULT 'library' CHECK (source_type IN ('library', 'drop')),
    last_sync_reason TEXT NOT NULL CHECK (last_sync_reason IN ('startup', 'interval', 'manual', 'drop'))
);

CREATE INDEX IF NOT EXISTS idx_library_files_path
ON library_files(file_path);

CREATE INDEX IF NOT EXISTS idx_library_files_last_seen_at
ON library_files(last_seen_at);

CREATE INDEX IF NOT EXISTS idx_library_files_import_status
ON library_files(import_status);
