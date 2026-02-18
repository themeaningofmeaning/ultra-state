from __future__ import annotations

import asyncio
import logging
import os
import tempfile
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import List, Optional, Sequence, Tuple

from analyzer import FitAnalyzer
from db import DatabaseManager, calculate_file_hash


logger = logging.getLogger(__name__)


SCHEMA_FALLBACK_SQL = """
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
"""


class SyncReason(str, Enum):
    STARTUP = "startup"
    INTERVAL = "interval"
    MANUAL = "manual"
    DROP = "drop"


@dataclass
class SyncReport:
    reason: SyncReason
    started_at: str
    finished_at: str
    scanned_files: int = 0
    hashed_files: int = 0
    imported_new: int = 0
    duplicates: int = 0
    moved_or_renamed: int = 0
    unchanged: int = 0
    missing_files: int = 0
    failed: int = 0
    skipped_busy: bool = False
    errors: List[str] = field(default_factory=list)


@dataclass
class LibraryStatus:
    library_root: Optional[str]
    auto_sync_interval_sec: int
    sync_in_progress: bool
    last_report: Optional[SyncReport]


class LibraryManager:
    """Backend orchestration for library mode sync and one-off drop imports."""

    def __init__(
        self,
        db: Optional[DatabaseManager] = None,
        analyzer: Optional[FitAnalyzer] = None,
        auto_sync_interval_sec: int = 60,
        migration_sql_path: Optional[str] = None,
        drop_cache_dir: Optional[str] = None,
        drop_cache_max_age_seconds: int = 24 * 60 * 60,
    ) -> None:
        self.db = db or DatabaseManager()
        self.analyzer = analyzer or FitAnalyzer()
        self._sync_lock = asyncio.Lock()
        self._timer_task: Optional[asyncio.Task] = None
        self._running = False
        self._last_report: Optional[SyncReport] = None
        self._default_interval_sec = max(10, int(auto_sync_interval_sec))
        self._migration_sql_path = Path(migration_sql_path) if migration_sql_path else (
            Path(__file__).resolve().parent / "migrations" / "001_library_mode.sql"
        )
        self.drop_cache_dir = Path(drop_cache_dir) if drop_cache_dir else (
            Path.home() / ".ultrastate" / "drop_cache"
        )
        self.drop_cache_max_age_seconds = max(3600, int(drop_cache_max_age_seconds))

        self._ensure_schema()
        self._initialize_default_settings()
        self._ensure_drop_cache_directory()

    async def start(self) -> None:
        if self._running:
            return

        self._running = True
        # Boot-safe drop-cache retention to prevent unbounded growth.
        await asyncio.to_thread(self._cleanup_drop_cache)
        await self.sync_library(SyncReason.STARTUP)
        self._timer_task = asyncio.create_task(self._interval_loop())

    async def stop(self) -> None:
        self._running = False
        if self._timer_task is None:
            return

        task = self._timer_task
        self._timer_task = None
        task.cancel()
        try:
            await task
        except asyncio.CancelledError:
            pass

    async def set_library_root(self, path: str) -> None:
        normalized = self._normalize_path(path)
        if not normalized:
            raise ValueError("Library path is empty or invalid.")
        if not os.path.isdir(normalized):
            raise ValueError("Library path does not exist or is not a directory.")
        if not os.access(normalized, os.R_OK):
            raise ValueError("Library path is not readable.")

        self._set_setting("library_root", normalized)
        await self.sync_library(SyncReason.MANUAL)

    async def get_library_root(self) -> Optional[str]:
        return self._get_setting("library_root")

    async def sync_library(self, reason: SyncReason) -> SyncReport:
        return await self._run_sync(reason=reason, override_paths=None, source_type="library")

    async def resync_now(self) -> SyncReport:
        return await self.sync_library(SyncReason.MANUAL)

    async def ingest_drop_files(self, paths: Sequence[str]) -> SyncReport:
        normalized_paths: List[str] = []
        for raw_path in paths:
            normalized = self._normalize_path(raw_path)
            if not normalized:
                continue
            if not os.path.isfile(normalized):
                continue
            if normalized.lower().endswith(".fit"):
                normalized_paths.append(normalized)

        normalized_paths = sorted(set(normalized_paths))
        if not normalized_paths:
            now_iso = self._utc_now_iso()
            report = SyncReport(
                reason=SyncReason.DROP,
                started_at=now_iso,
                finished_at=now_iso,
                failed=1,
                errors=["No valid FIT files were provided to drag-and-drop ingest."],
            )
            self._last_report = report
            return report

        return await self._run_sync(
            reason=SyncReason.DROP,
            override_paths=normalized_paths,
            source_type="drop",
        )

    async def get_status(self) -> LibraryStatus:
        return LibraryStatus(
            library_root=self._get_setting("library_root"),
            auto_sync_interval_sec=self._get_interval_seconds(),
            sync_in_progress=self._sync_lock.locked(),
            last_report=self._last_report,
        )

    async def _interval_loop(self) -> None:
        while self._running:
            try:
                await asyncio.sleep(self._get_interval_seconds())
                if not self._running:
                    break
                await self.sync_library(SyncReason.INTERVAL)
            except asyncio.CancelledError:
                break
            except Exception:
                now_iso = self._utc_now_iso()
                self._set_setting("library_last_sync_at", now_iso)
                self._set_setting("library_last_sync_status", "error")

    async def _run_sync(
        self,
        reason: SyncReason,
        override_paths: Optional[Sequence[str]],
        source_type: str,
    ) -> SyncReport:
        now_iso = self._utc_now_iso()
        if self._sync_lock.locked():
            report = SyncReport(
                reason=reason,
                started_at=now_iso,
                finished_at=now_iso,
                skipped_busy=True,
            )
            self._last_report = report
            return report

        async with self._sync_lock:
            started_at = self._utc_now_iso()
            report = SyncReport(reason=reason, started_at=started_at, finished_at=started_at)
            self._set_setting("library_last_sync_status", "running")

            try:
                if override_paths is None:
                    library_root = self._get_setting("library_root")
                    if not library_root:
                        report.errors.append("Library root is not configured.")
                        report.finished_at = self._utc_now_iso()
                        self._finalize_sync(report, final_status="error")
                        return report
                    if not os.path.isdir(library_root):
                        report.errors.append(
                            "Configured library_root does not exist: {0}".format(library_root)
                        )
                        report.finished_at = self._utc_now_iso()
                        self._finalize_sync(report, final_status="error")
                        return report

                    candidate_paths, scan_errors = await asyncio.to_thread(
                        self._scan_fit_files, library_root
                    )
                    report.errors.extend(scan_errors)
                else:
                    candidate_paths = sorted(set(override_paths))

                session_id = int(time.time())
                for path in candidate_paths:
                    await self._process_candidate_file(
                        path=path,
                        report=report,
                        session_id=session_id,
                        reason=reason,
                        source_type=source_type,
                    )

                if override_paths is None and source_type == "library":
                    report.missing_files = self._reconcile_missing_library_files(
                        scan_started_at=started_at,
                        reason=reason,
                    )

                report.finished_at = self._utc_now_iso()
                final_status = "error" if (report.failed > 0 or report.errors) else "ok"
                self._finalize_sync(report, final_status=final_status)
                return report
            except Exception as exc:
                report.failed += 1
                report.errors.append("Unexpected sync error: {0}".format(exc))
                report.finished_at = self._utc_now_iso()
                self._finalize_sync(report, final_status="error")
                return report

    async def _process_candidate_file(
        self,
        path: str,
        report: SyncReport,
        session_id: int,
        reason: SyncReason,
        source_type: str,
    ) -> None:
        normalized_path = self._normalize_path(path)
        if not normalized_path:
            report.failed += 1
            report.errors.append("Invalid file path encountered: {0}".format(path))
            return

        report.scanned_files += 1
        try:
            stat_result = os.stat(normalized_path)
        except OSError as exc:
            report.failed += 1
            report.errors.append("{0}: {1}".format(normalized_path, exc))
            return

        existing_by_path = self._get_library_row_by_path(normalized_path)
        if (
            existing_by_path
            and existing_by_path["file_size_bytes"] == stat_result.st_size
            and existing_by_path["file_mtime_ns"] == stat_result.st_mtime_ns
            and existing_by_path["import_status"] == "imported"
        ):
            self._touch_seen(
                content_hash=existing_by_path["content_hash"],
                file_path=normalized_path,
                file_size_bytes=stat_result.st_size,
                file_mtime_ns=stat_result.st_mtime_ns,
                reason=reason,
                source_type=source_type,
            )
            report.unchanged += 1
            return

        try:
            content_hash = await asyncio.to_thread(calculate_file_hash, normalized_path)
            report.hashed_files += 1
        except Exception as exc:
            report.failed += 1
            report.errors.append("Hashing failed for {0}: {1}".format(normalized_path, exc))
            return

        existing_by_hash = self._get_library_row_by_hash(content_hash)
        if existing_by_hash:
            path_changed = existing_by_hash["file_path"] != normalized_path
            self._touch_seen(
                content_hash=content_hash,
                file_path=normalized_path,
                file_size_bytes=stat_result.st_size,
                file_mtime_ns=stat_result.st_mtime_ns,
                reason=reason,
                source_type=source_type,
            )

            if existing_by_hash["import_status"] == "imported":
                # Recovery path: if activities row is missing, re-ingest even though hash is known.
                if self.db.activity_exists(content_hash):
                    self._update_activity_file_path(content_hash, normalized_path)
                    if path_changed:
                        report.moved_or_renamed += 1
                    else:
                        report.duplicates += 1
                    return

                ingest_ok, ingest_error = await self._ingest_file(
                    file_path=normalized_path,
                    content_hash=content_hash,
                    session_id=session_id,
                )
                if ingest_ok:
                    self._mark_imported(
                        content_hash=content_hash,
                        file_path=normalized_path,
                        file_size_bytes=stat_result.st_size,
                        file_mtime_ns=stat_result.st_mtime_ns,
                        reason=reason,
                        source_type=source_type,
                    )
                    self._update_activity_file_path(content_hash, normalized_path)
                    report.imported_new += 1
                    return

                report.failed += 1
                error_text = ingest_error or "Unknown ingest error."
                self._mark_failed(
                    content_hash=content_hash,
                    file_path=normalized_path,
                    file_size_bytes=stat_result.st_size,
                    file_mtime_ns=stat_result.st_mtime_ns,
                    reason=reason,
                    source_type=source_type,
                    error_text=error_text,
                )
                report.errors.append("{0}: {1}".format(normalized_path, error_text))
                return

            ingest_ok, ingest_error = await self._ingest_file(
                file_path=normalized_path,
                content_hash=content_hash,
                session_id=session_id,
            )
            if ingest_ok:
                self._mark_imported(
                    content_hash=content_hash,
                    file_path=normalized_path,
                    file_size_bytes=stat_result.st_size,
                    file_mtime_ns=stat_result.st_mtime_ns,
                    reason=reason,
                    source_type=source_type,
                )
                self._update_activity_file_path(content_hash, normalized_path)
                report.imported_new += 1
                return

            report.failed += 1
            error_text = ingest_error or "Unknown ingest error."
            self._mark_failed(
                content_hash=content_hash,
                file_path=normalized_path,
                file_size_bytes=stat_result.st_size,
                file_mtime_ns=stat_result.st_mtime_ns,
                reason=reason,
                source_type=source_type,
                error_text=error_text,
            )
            report.errors.append("{0}: {1}".format(normalized_path, error_text))
            return

        self._insert_pending(
            content_hash=content_hash,
            file_path=normalized_path,
            file_size_bytes=stat_result.st_size,
            file_mtime_ns=stat_result.st_mtime_ns,
            reason=reason,
            source_type=source_type,
        )

        ingest_ok, ingest_error = await self._ingest_file(
            file_path=normalized_path,
            content_hash=content_hash,
            session_id=session_id,
        )
        if ingest_ok:
            self._mark_imported(
                content_hash=content_hash,
                file_path=normalized_path,
                file_size_bytes=stat_result.st_size,
                file_mtime_ns=stat_result.st_mtime_ns,
                reason=reason,
                source_type=source_type,
            )
            report.imported_new += 1
            return

        report.failed += 1
        error_text = ingest_error or "Unknown ingest error."
        self._mark_failed(
            content_hash=content_hash,
            file_path=normalized_path,
            file_size_bytes=stat_result.st_size,
            file_mtime_ns=stat_result.st_mtime_ns,
            reason=reason,
            source_type=source_type,
            error_text=error_text,
        )
        report.errors.append("{0}: {1}".format(normalized_path, error_text))

    async def _ingest_file(self, file_path: str, content_hash: str, session_id: int) -> Tuple[bool, str]:
        try:
            result = await asyncio.to_thread(self.analyzer.analyze_file, file_path)
            if not result:
                return False, "Analyzer returned no activity data."
            self.db.insert_activity(result, content_hash, session_id, file_path)
            return True, ""
        except Exception as exc:
            return False, str(exc)

    def _initialize_default_settings(self) -> None:
        if self._get_setting("library_auto_sync_interval_sec") is None:
            self._set_setting(
                "library_auto_sync_interval_sec",
                str(self._default_interval_sec),
            )
        self._set_setting("library_last_sync_status", self._get_setting("library_last_sync_status") or "idle")

    def _ensure_schema(self) -> None:
        schema_sql = SCHEMA_FALLBACK_SQL
        if self._migration_sql_path.exists():
            schema_sql = self._migration_sql_path.read_text(encoding="utf-8")

        with self.db.get_connection() as conn:
            conn.executescript(schema_sql)
            columns = {
                row[1] for row in conn.execute("PRAGMA table_info(library_files)").fetchall()
            }
            if "is_missing" not in columns:
                conn.execute(
                    "ALTER TABLE library_files "
                    "ADD COLUMN is_missing INTEGER NOT NULL DEFAULT 0"
                )
            if "missing_since_at" not in columns:
                conn.execute("ALTER TABLE library_files ADD COLUMN missing_since_at TEXT")

    def _scan_fit_files(self, root: str) -> Tuple[List[str], List[str]]:
        fit_files: List[str] = []
        errors: List[str] = []

        def _on_error(err: OSError) -> None:
            filename = getattr(err, "filename", root)
            errors.append("{0}: {1}".format(filename, err))

        for dir_path, _dir_names, file_names in os.walk(root, onerror=_on_error, followlinks=False):
            for file_name in file_names:
                if file_name.lower().endswith(".fit"):
                    normalized = self._normalize_path(os.path.join(dir_path, file_name))
                    if normalized:
                        fit_files.append(normalized)

        return sorted(set(fit_files)), errors

    def _ensure_drop_cache_directory(self) -> None:
        try:
            self.drop_cache_dir.mkdir(parents=True, exist_ok=True)
        except OSError as exc:
            fallback_dir = Path(tempfile.gettempdir()) / "ultrastate_drop_cache"
            logger.warning(
                "Unable to create drop cache directory '%s': %s. Falling back to '%s'.",
                self.drop_cache_dir,
                exc,
                fallback_dir,
            )
            try:
                fallback_dir.mkdir(parents=True, exist_ok=True)
                self.drop_cache_dir = fallback_dir
            except OSError as fallback_exc:
                logger.warning(
                    "Unable to create fallback drop cache directory '%s': %s",
                    fallback_dir,
                    fallback_exc,
                )

    def _cleanup_drop_cache(self) -> None:
        """Delete cached drop files older than configured retention window."""
        self._ensure_drop_cache_directory()

        cutoff_ts = time.time() - self.drop_cache_max_age_seconds
        try:
            entries = list(self.drop_cache_dir.iterdir())
        except OSError as exc:
            logger.warning("Unable to scan drop cache directory '%s': %s", self.drop_cache_dir, exc)
            return

        for entry in entries:
            try:
                if not entry.is_file():
                    continue
                if entry.stat().st_mtime >= cutoff_ts:
                    continue
                entry.unlink()
            except OSError as exc:
                # Best effort only: never fail app boot for cleanup issues.
                logger.warning("Unable to delete stale drop cache file '%s': %s", entry, exc)

    def _finalize_sync(self, report: SyncReport, final_status: str) -> None:
        self._set_setting("library_last_sync_at", report.finished_at)
        self._set_setting("library_last_sync_status", final_status)
        self._last_report = report

    def _get_setting(self, key: str) -> Optional[str]:
        with self.db.get_connection() as conn:
            row = conn.execute("SELECT value FROM settings WHERE key = ?", (key,)).fetchone()
            if row is None:
                return None
            return row[0]

    def _set_setting(self, key: str, value: str) -> None:
        now_iso = self._utc_now_iso()
        with self.db.get_connection() as conn:
            conn.execute(
                """
                INSERT INTO settings (key, value, updated_at)
                VALUES (?, ?, ?)
                ON CONFLICT(key) DO UPDATE SET
                    value = excluded.value,
                    updated_at = excluded.updated_at
                """,
                (key, value, now_iso),
            )

    def _get_interval_seconds(self) -> int:
        setting_value = self._get_setting("library_auto_sync_interval_sec")
        if setting_value is None:
            return self._default_interval_sec
        try:
            parsed = int(setting_value)
            return max(10, parsed)
        except (TypeError, ValueError):
            return self._default_interval_sec

    def _get_library_row_by_path(self, file_path: str):
        with self.db.get_connection() as conn:
            row = conn.execute(
                """
                SELECT content_hash, file_path, file_size_bytes, file_mtime_ns, import_status
                FROM library_files
                WHERE file_path = ?
                ORDER BY last_seen_at DESC
                LIMIT 1
                """,
                (file_path,),
            ).fetchone()
            return row

    def _get_library_row_by_hash(self, content_hash: str):
        with self.db.get_connection() as conn:
            row = conn.execute(
                """
                SELECT content_hash, file_path, file_size_bytes, file_mtime_ns, import_status
                FROM library_files
                WHERE content_hash = ?
                LIMIT 1
                """,
                (content_hash,),
            ).fetchone()
            return row

    def _insert_pending(
        self,
        content_hash: str,
        file_path: str,
        file_size_bytes: int,
        file_mtime_ns: int,
        reason: SyncReason,
        source_type: str,
    ) -> None:
        now_iso = self._utc_now_iso()
        with self.db.get_connection() as conn:
            conn.execute(
                """
                INSERT INTO library_files (
                    content_hash, file_path, file_size_bytes, file_mtime_ns,
                    import_status, last_error, first_seen_at, last_seen_at,
                    source_type, last_sync_reason
                )
                VALUES (?, ?, ?, ?, 'pending', NULL, ?, ?, ?, ?)
                ON CONFLICT(content_hash) DO UPDATE SET
                    file_path = excluded.file_path,
                    file_size_bytes = excluded.file_size_bytes,
                    file_mtime_ns = excluded.file_mtime_ns,
                    import_status = 'pending',
                    is_missing = 0,
                    missing_since_at = NULL,
                    last_error = NULL,
                    last_seen_at = excluded.last_seen_at,
                    source_type = excluded.source_type,
                    last_sync_reason = excluded.last_sync_reason
                """,
                (
                    content_hash,
                    file_path,
                    file_size_bytes,
                    file_mtime_ns,
                    now_iso,
                    now_iso,
                    source_type,
                    reason.value,
                ),
            )

    def _touch_seen(
        self,
        content_hash: str,
        file_path: str,
        file_size_bytes: int,
        file_mtime_ns: int,
        reason: SyncReason,
        source_type: str,
    ) -> None:
        now_iso = self._utc_now_iso()
        with self.db.get_connection() as conn:
            conn.execute(
                """
                UPDATE library_files
                SET
                    file_path = ?,
                    file_size_bytes = ?,
                    file_mtime_ns = ?,
                    is_missing = 0,
                    missing_since_at = NULL,
                    last_seen_at = ?,
                    source_type = ?,
                    last_sync_reason = ?
                WHERE content_hash = ?
                """,
                (
                    file_path,
                    file_size_bytes,
                    file_mtime_ns,
                    now_iso,
                    source_type,
                    reason.value,
                    content_hash,
                ),
            )

    def _mark_imported(
        self,
        content_hash: str,
        file_path: str,
        file_size_bytes: int,
        file_mtime_ns: int,
        reason: SyncReason,
        source_type: str,
    ) -> None:
        now_iso = self._utc_now_iso()
        with self.db.get_connection() as conn:
            conn.execute(
                """
                UPDATE library_files
                SET
                    file_path = ?,
                    file_size_bytes = ?,
                    file_mtime_ns = ?,
                    import_status = 'imported',
                    is_missing = 0,
                    missing_since_at = NULL,
                    last_error = NULL,
                    last_seen_at = ?,
                    imported_at = COALESCE(imported_at, ?),
                    source_type = ?,
                    last_sync_reason = ?
                WHERE content_hash = ?
                """,
                (
                    file_path,
                    file_size_bytes,
                    file_mtime_ns,
                    now_iso,
                    now_iso,
                    source_type,
                    reason.value,
                    content_hash,
                ),
            )

    def _mark_failed(
        self,
        content_hash: str,
        file_path: str,
        file_size_bytes: int,
        file_mtime_ns: int,
        reason: SyncReason,
        source_type: str,
        error_text: str,
    ) -> None:
        now_iso = self._utc_now_iso()
        with self.db.get_connection() as conn:
            conn.execute(
                """
                UPDATE library_files
                SET
                    file_path = ?,
                    file_size_bytes = ?,
                    file_mtime_ns = ?,
                    import_status = 'failed',
                    is_missing = 0,
                    missing_since_at = NULL,
                    last_error = ?,
                    last_seen_at = ?,
                    source_type = ?,
                    last_sync_reason = ?
                WHERE content_hash = ?
                """,
                (
                    file_path,
                    file_size_bytes,
                    file_mtime_ns,
                    error_text,
                    now_iso,
                    source_type,
                    reason.value,
                    content_hash,
                ),
            )

    def _reconcile_missing_library_files(self, scan_started_at: str, reason: SyncReason) -> int:
        """Mark previously seen library files as missing when they are no longer on disk."""
        now_iso = self._utc_now_iso()
        marked_missing = 0

        with self.db.get_connection() as conn:
            rows = conn.execute(
                """
                SELECT content_hash, file_path
                FROM library_files
                WHERE source_type = 'library'
                  AND is_missing = 0
                  AND last_seen_at < ?
                """,
                (scan_started_at,),
            ).fetchall()

            for row in rows:
                file_path = row["file_path"]
                if file_path and os.path.exists(file_path):
                    continue

                conn.execute(
                    """
                    UPDATE library_files
                    SET
                        is_missing = 1,
                        missing_since_at = COALESCE(missing_since_at, ?),
                        last_sync_reason = ?
                    WHERE content_hash = ?
                    """,
                    (now_iso, reason.value, row["content_hash"]),
                )
                marked_missing += 1

        return marked_missing

    def _update_activity_file_path(self, content_hash: str, file_path: str) -> None:
        with self.db.get_connection() as conn:
            conn.execute(
                "UPDATE activities SET file_path = ? WHERE hash = ?",
                (file_path, content_hash),
            )

    @staticmethod
    def _normalize_path(path: str) -> Optional[str]:
        if not path:
            return None
        try:
            return str(Path(path).expanduser().resolve())
        except Exception:
            try:
                return os.path.abspath(os.path.expanduser(path))
            except Exception:
                return None

    @staticmethod
    def _utc_now_iso() -> str:
        return (
            datetime.now(timezone.utc)
            .isoformat(timespec="milliseconds")
            .replace("+00:00", "Z")
        )
