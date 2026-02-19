from __future__ import annotations

import asyncio
import logging
import os
import tempfile
import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from enum import Enum
from pathlib import Path
from typing import List, Optional, Sequence, Tuple

from analyzer import ANALYZER_DATA_VERSION, FitAnalyzer
from db import DatabaseManager, calculate_file_hash


logger = logging.getLogger(__name__)


REPROCESS_BATCH_STARTUP = 40
REPROCESS_BATCH_INTERVAL = 5
REPROCESS_BATCH_MANUAL = 200
RETRY_BASE_SEC = 300
RETRY_MAX_SEC = 21600
MAX_AUTO_RETRY_ATTEMPTS = 8

try:
    from watchdog.events import FileSystemEventHandler
    from watchdog.observers import Observer

    WATCHDOG_AVAILABLE = True
except Exception:  # pragma: no cover - optional dependency
    FileSystemEventHandler = object  # type: ignore[assignment]
    Observer = None  # type: ignore[assignment]
    WATCHDOG_AVAILABLE = False


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
    reprocessed_upgraded: int = 0
    reprocess_failed: int = 0
    reprocess_missing_source: int = 0
    skipped_busy: bool = False
    errors: List[str] = field(default_factory=list)


@dataclass
class LibraryStatus:
    library_root: Optional[str]
    auto_sync_interval_sec: int
    sync_in_progress: bool
    last_report: Optional[SyncReport]


class _FitFileEventHandler(FileSystemEventHandler):
    """Watchdog event handler that forwards FIT file changes to async loop."""

    def __init__(self, notify_callback):
        super().__init__()
        self._notify_callback = notify_callback

    def on_created(self, event):  # pragma: no cover - exercised when watchdog installed
        self._handle(event)

    def on_modified(self, event):  # pragma: no cover - exercised when watchdog installed
        self._handle(event)

    def on_moved(self, event):  # pragma: no cover - exercised when watchdog installed
        self._handle(event)

    def on_deleted(self, event):  # pragma: no cover - exercised when watchdog installed
        self._handle(event)

    def _handle(self, event):  # pragma: no cover - exercised when watchdog installed
        if getattr(event, "is_directory", False):
            return
        src_path = getattr(event, "src_path", "") or ""
        dest_path = getattr(event, "dest_path", "") or ""
        path = dest_path or src_path
        if path.lower().endswith(".fit"):
            self._notify_callback()


class LibraryManager:
    """Backend orchestration for library mode sync and one-off manual imports."""

    def __init__(
        self,
        db: Optional[DatabaseManager] = None,
        analyzer: Optional[FitAnalyzer] = None,
        auto_sync_interval_sec: int = 60,
        live_check_interval_sec: float = 5.0,
        event_debounce_sec: float = 1.2,
        migration_sql_path: Optional[str] = None,
        import_cache_dir: Optional[str] = None,
        import_cache_max_age_seconds: int = 24 * 60 * 60,
    ) -> None:
        self.db = db or DatabaseManager()
        self.analyzer = analyzer or FitAnalyzer()
        self._sync_lock = asyncio.Lock()
        self._timer_task: Optional[asyncio.Task] = None
        self._event_task: Optional[asyncio.Task] = None
        self._poll_task: Optional[asyncio.Task] = None
        self._event_trigger = asyncio.Event()
        self._loop: Optional[asyncio.AbstractEventLoop] = None
        self._observer = None
        self._running = False
        self._last_report: Optional[SyncReport] = None
        self._default_interval_sec = max(10, int(auto_sync_interval_sec))
        self._live_check_interval_sec = max(1.0, float(live_check_interval_sec))
        self._event_debounce_sec = max(0.25, float(event_debounce_sec))
        self._last_poll_signature: Optional[Tuple[int, int, int, int]] = None
        self._migration_sql_path = Path(migration_sql_path) if migration_sql_path else (
            Path(__file__).resolve().parent / "migrations" / "001_library_mode.sql"
        )
        self.import_cache_dir = Path(import_cache_dir) if import_cache_dir else (
            Path.home() / ".ultrastate" / "import_cache"
        )
        self.import_cache_max_age_seconds = max(3600, int(import_cache_max_age_seconds))

        self._ensure_schema()
        self._initialize_default_settings()
        self._ensure_import_cache_directory()

    async def start(self) -> None:
        if self._running:
            return

        self._running = True
        self._loop = asyncio.get_running_loop()
        # Boot-safe import-cache retention to prevent unbounded growth.
        await asyncio.to_thread(self._cleanup_import_cache)
        await self.sync_library(SyncReason.STARTUP)
        self._timer_task = asyncio.create_task(self._interval_loop())
        await self._start_live_monitoring()

    async def stop(self) -> None:
        self._running = False
        await self._stop_live_monitoring()
        self._event_trigger.set()

        tasks: List[asyncio.Task] = []
        for task in (self._timer_task, self._event_task, self._poll_task):
            if task is not None:
                task.cancel()
                tasks.append(task)

        self._timer_task = None
        self._event_task = None
        self._poll_task = None

        for task in tasks:
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
        self._last_poll_signature = await asyncio.to_thread(
            self._compute_fit_tree_signature, normalized
        )
        if self._running:
            await self._restart_live_monitoring()

    async def get_library_root(self) -> Optional[str]:
        return self._get_setting("library_root")

    async def sync_library(self, reason: SyncReason) -> SyncReport:
        return await self._run_sync(reason=reason, override_paths=None, source_type="library")

    async def resync_now(self) -> SyncReport:
        return await self.sync_library(SyncReason.MANUAL)

    async def ingest_files(self, paths: Sequence[str]) -> SyncReport:
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
                errors=["No valid FIT files were provided to manual import ingest."],
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

    async def _start_live_monitoring(self) -> None:
        """Start low-latency change detection (watchdog if available, polling fallback)."""
        if self._event_task is None:
            self._event_task = asyncio.create_task(self._event_sync_loop())

        library_root = self._get_setting("library_root")
        if not library_root:
            return

        self._last_poll_signature = await asyncio.to_thread(
            self._compute_fit_tree_signature, library_root
        )

        if WATCHDOG_AVAILABLE:
            if self._observer is not None:
                return
            try:
                observer = Observer()
                observer.schedule(
                    _FitFileEventHandler(self._notify_live_file_event),
                    library_root,
                    recursive=True,
                )
                observer.start()
                self._observer = observer
                return
            except Exception as exc:
                logger.warning("Watchdog live monitoring unavailable; falling back to polling: %s", exc)
                self._observer = None

        if self._poll_task is None:
            self._poll_task = asyncio.create_task(self._poll_live_changes_loop())

    async def _stop_live_monitoring(self) -> None:
        if self._observer is not None:
            try:
                self._observer.stop()
                self._observer.join(timeout=1.0)
            except Exception:
                pass
            self._observer = None

    async def _restart_live_monitoring(self) -> None:
        await self._stop_live_monitoring()
        if self._poll_task is not None:
            self._poll_task.cancel()
            try:
                await self._poll_task
            except asyncio.CancelledError:
                pass
            self._poll_task = None
        await self._start_live_monitoring()

    def _notify_live_file_event(self) -> None:
        if not self._running or self._loop is None or self._loop.is_closed():
            return
        self._loop.call_soon_threadsafe(self._event_trigger.set)

    async def _event_sync_loop(self) -> None:
        """Debounced sync trigger loop for live file events."""
        while self._running:
            try:
                await self._event_trigger.wait()
                self._event_trigger.clear()
                await asyncio.sleep(self._event_debounce_sec)
                # Coalesce event bursts into one sync.
                while self._event_trigger.is_set():
                    self._event_trigger.clear()
                    await asyncio.sleep(self._event_debounce_sec)
                if not self._running:
                    break
                await self.sync_library(SyncReason.INTERVAL)
            except asyncio.CancelledError:
                break
            except Exception as exc:
                logger.warning("Live sync trigger loop error: %s", exc)

    async def _poll_live_changes_loop(self) -> None:
        """Fallback live detector when watchdog is unavailable."""
        while self._running:
            try:
                await asyncio.sleep(self._live_check_interval_sec)
                if not self._running:
                    break
                library_root = self._get_setting("library_root")
                if not library_root or not os.path.isdir(library_root):
                    continue

                signature = await asyncio.to_thread(
                    self._compute_fit_tree_signature, library_root
                )
                if self._last_poll_signature is None:
                    self._last_poll_signature = signature
                    continue
                if signature != self._last_poll_signature:
                    self._last_poll_signature = signature
                    self._event_trigger.set()
            except asyncio.CancelledError:
                break
            except Exception as exc:
                logger.warning("Live polling loop error: %s", exc)

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
                # Only skip hashes that were freshly ingested this sync pass;
                # unchanged library rows still need stale-version reprocessing.
                processed_hashes: set[str] = set()
                for path in candidate_paths:
                    processed_hash = await self._process_candidate_file(
                        path=path,
                        report=report,
                        session_id=session_id,
                        reason=reason,
                        source_type=source_type,
                    )
                    if processed_hash:
                        processed_hashes.add(processed_hash)

                if override_paths is None and source_type == "library":
                    report.missing_files = self._reconcile_missing_library_files(
                        scan_started_at=started_at,
                        reason=reason,
                    )
                    (
                        report.reprocessed_upgraded,
                        report.reprocess_failed,
                        report.reprocess_missing_source,
                    ) = await self._run_reprocess_pass(reason=reason, processed_hashes=processed_hashes)

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
    ) -> Optional[str]:
        normalized_path = self._normalize_path(path)
        if not normalized_path:
            report.failed += 1
            report.errors.append("Invalid file path encountered: {0}".format(path))
            return None

        report.scanned_files += 1
        try:
            stat_result = os.stat(normalized_path)
        except OSError as exc:
            report.failed += 1
            report.errors.append("{0}: {1}".format(normalized_path, exc))
            return None

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
            return None

        try:
            content_hash = await asyncio.to_thread(calculate_file_hash, normalized_path)
            report.hashed_files += 1
        except Exception as exc:
            report.failed += 1
            report.errors.append("Hashing failed for {0}: {1}".format(normalized_path, exc))
            return None

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
                    return None

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
                    return content_hash

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
                return None

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
                return content_hash

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
            return None

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
            return content_hash

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
        return None

    async def _ingest_file(self, file_path: str, content_hash: str, session_id: int) -> Tuple[bool, str]:
        try:
            result = await asyncio.to_thread(self.analyzer.analyze_file, file_path)
            if not result:
                return False, "Analyzer returned no activity data."
            self.db.insert_activity(result, content_hash, session_id, file_path)
            return True, ""
        except Exception as exc:
            return False, str(exc)

    async def _run_reprocess_pass(
        self,
        reason: SyncReason,
        processed_hashes: set[str],
    ) -> Tuple[int, int, int]:
        batch_size = self._compute_reprocess_batch_size(reason)
        if batch_size <= 0:
            return 0, 0, 0

        now_iso = self._utc_now_iso()
        manual_mode = reason == SyncReason.MANUAL
        candidates = self.db.get_stale_activity_candidates(
            target_version=ANALYZER_DATA_VERSION,
            now_iso=now_iso,
            limit=batch_size,
            manual_mode=manual_mode,
        )

        upgraded = 0
        failed = 0
        missing_source = 0

        for row in candidates:
            content_hash = row["hash"]
            if not content_hash or content_hash in processed_hashes:
                continue
            processed_hashes.add(content_hash)

            fit_path = row["activity_file_path"] or row["library_file_path"]
            is_marked_missing = bool(row["is_missing"])

            if is_marked_missing or not fit_path or not os.path.exists(fit_path):
                self.db.mark_activity_missing_source(content_hash, self._utc_now_iso())
                missing_source += 1
                continue

            if not self.db.mark_activity_reprocessing(content_hash, self._utc_now_iso()):
                continue

            error_text = ""
            result = None
            try:
                result = await asyncio.to_thread(self.analyzer.analyze_file, fit_path)
            except Exception as exc:
                error_text = str(exc)

            if result:
                try:
                    processed_at = self._utc_now_iso()
                    result["analyzer_version"] = int(
                        result.get("analyzer_version") or ANALYZER_DATA_VERSION
                    )
                    result["analysis_status"] = "fresh"
                    result["analysis_attempts"] = 0
                    result["analysis_last_error"] = None
                    result["analysis_next_retry_at"] = None
                    result["analyzed_at"] = processed_at
                    result["last_processed_at"] = processed_at

                    # session_id/import_session_id remain immutable via COALESCE in UPSERT.
                    self.db.insert_activity(result, content_hash, session_id=0, file_path=fit_path)
                    self.db.mark_activity_reprocess_success(
                        content_hash,
                        now_iso=processed_at,
                        analyzer_version=result["analyzer_version"],
                    )
                    upgraded += 1
                    continue
                except Exception as exc:
                    error_text = str(exc)

            if not error_text:
                error_text = "Analyzer returned no activity data."
            failed_at = self._utc_now_iso()
            attempts_after_failure = int(row["analysis_attempts"] or 0) + 1
            retry_at = self._compute_retry_at(attempts_after_failure, failed_at)
            self.db.mark_activity_reprocess_failure(
                file_hash=content_hash,
                now_iso=failed_at,
                error_text=error_text,
                next_retry_at=retry_at,
            )
            failed += 1

        return upgraded, failed, missing_source

    @staticmethod
    def _compute_reprocess_batch_size(reason: SyncReason) -> int:
        if reason == SyncReason.STARTUP:
            return REPROCESS_BATCH_STARTUP
        if reason == SyncReason.MANUAL:
            return REPROCESS_BATCH_MANUAL
        return REPROCESS_BATCH_INTERVAL

    @staticmethod
    def _compute_retry_at(attempts: int, now_iso: str) -> Optional[str]:
        if attempts >= MAX_AUTO_RETRY_ATTEMPTS:
            return None

        backoff_sec = RETRY_BASE_SEC * (2 ** max(0, attempts - 1))
        delay_sec = min(RETRY_MAX_SEC, backoff_sec)

        try:
            base_dt = datetime.fromisoformat(now_iso.replace("Z", "+00:00"))
        except ValueError:
            base_dt = datetime.now(timezone.utc)

        retry_dt = base_dt + timedelta(seconds=delay_sec)
        return retry_dt.isoformat(timespec="milliseconds").replace("+00:00", "Z")

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

    def _compute_fit_tree_signature(self, root: str) -> Tuple[int, int, int, int]:
        """Fast signature for FIT tree change detection."""
        file_count = 0
        total_size = 0
        newest_mtime_ns = 0
        path_fingerprint = 0

        for dir_path, _dir_names, file_names in os.walk(root, followlinks=False):
            for file_name in file_names:
                if not file_name.lower().endswith(".fit"):
                    continue
                full_path = os.path.join(dir_path, file_name)
                try:
                    stat_result = os.stat(full_path)
                except OSError:
                    continue

                normalized = self._normalize_path(full_path) or full_path
                file_count += 1
                total_size += int(stat_result.st_size)
                newest_mtime_ns = max(newest_mtime_ns, int(stat_result.st_mtime_ns))
                path_fingerprint ^= hash(normalized.lower())
                path_fingerprint ^= int(stat_result.st_mtime_ns)

        return (file_count, total_size, newest_mtime_ns, path_fingerprint)

    def _ensure_import_cache_directory(self) -> None:
        try:
            self.import_cache_dir.mkdir(parents=True, exist_ok=True)
        except OSError as exc:
            fallback_dir = Path(tempfile.gettempdir()) / "ultrastate_import_cache"
            logger.warning(
                "Unable to create import cache directory '%s': %s. Falling back to '%s'.",
                self.import_cache_dir,
                exc,
                fallback_dir,
            )
            try:
                fallback_dir.mkdir(parents=True, exist_ok=True)
                self.import_cache_dir = fallback_dir
            except OSError as fallback_exc:
                logger.warning(
                    "Unable to create fallback import cache directory '%s': %s",
                    fallback_dir,
                    fallback_exc,
                )

    def _cleanup_import_cache(self) -> None:
        """Delete cached import files older than configured retention window."""
        self._ensure_import_cache_directory()

        cutoff_ts = time.time() - self.import_cache_max_age_seconds
        try:
            entries = list(self.import_cache_dir.iterdir())
        except OSError as exc:
            logger.warning("Unable to scan import cache directory '%s': %s", self.import_cache_dir, exc)
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
                logger.warning("Unable to delete stale import cache file '%s': %s", entry, exc)

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
