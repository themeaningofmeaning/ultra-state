import os
import tempfile
import time
import asyncio
import unittest
from pathlib import Path

from db import DatabaseManager, calculate_file_hash
from library_manager import LibraryManager, SyncReason


class StubAnalyzer:
    """Deterministic analyzer stub for LibraryManager tests."""

    def analyze_file(self, file_path):
        return {
            "filename": os.path.basename(file_path),
            "date": "2026-02-01",
            "timestamp_utc": 1700000000,
            "distance_mi": 6.2,
            "moving_time_min": 50.0,
            "pace": "8:03",
            "avg_hr": 145,
            "max_hr": 166,
            "elevation_ft": 230,
            "efficiency_factor": 1.2,
            "decoupling": 0.8,
            "avg_cadence": 176,
        }


class LibraryManagerTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.root = Path(self.temp_dir.name)
        self.db_path = self.root / "test.db"
        self.library_root = self.root / "library"
        self.import_cache = self.root / "import_cache"
        self.library_root.mkdir(parents=True, exist_ok=True)
        self.import_cache.mkdir(parents=True, exist_ok=True)
        self.db = DatabaseManager(str(self.db_path))
        self.analyzer = StubAnalyzer()

    def tearDown(self):
        self.temp_dir.cleanup()

    def _create_manager(
        self,
        *,
        interval=3600,
        retention_seconds=24 * 60 * 60,
        live_check_interval=5.0,
        event_debounce=1.2,
    ):
        return LibraryManager(
            db=self.db,
            analyzer=self.analyzer,
            auto_sync_interval_sec=interval,
            live_check_interval_sec=live_check_interval,
            event_debounce_sec=event_debounce,
            import_cache_dir=str(self.import_cache),
            import_cache_max_age_seconds=retention_seconds,
        )

    def _write_fit(self, path: Path, payload: bytes = b"fit-data-1"):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(payload)
        return path

    async def test_startup_auto_sync_imports_new_file(self):
        fit_path = self._write_fit(self.library_root / "run_001.fit")
        normalized_fit_path = os.path.realpath(fit_path)
        manager = self._create_manager()
        manager._set_setting("library_root", str(self.library_root))

        try:
            await manager.start()
            status = await manager.get_status()
            self.assertIsNotNone(status.last_report)
            self.assertEqual(status.last_report.reason, SyncReason.STARTUP)
            self.assertEqual(status.last_report.imported_new, 1)
            self.assertEqual(status.last_report.failed, 0)
            self.assertEqual(self.db.get_count(), 1)

            expected_hash = calculate_file_hash(str(fit_path))
            row = self.db.get_activity_by_hash(expected_hash)
            self.assertIsNotNone(row)
            self.assertEqual(os.path.realpath(row["file_path"]), normalized_fit_path)
        finally:
            await manager.stop()

    async def test_manual_import_flows_through_hash_dedupe(self):
        fit_path = self._write_fit(self.root / "one_off.fit", payload=b"same-content")
        manager = self._create_manager()

        report_1 = await manager.ingest_files([str(fit_path)])
        report_2 = await manager.ingest_files([str(fit_path)])

        self.assertEqual(report_1.reason, SyncReason.DROP)
        self.assertEqual(report_1.imported_new, 1)
        self.assertEqual(report_1.failed, 0)
        self.assertEqual(report_2.imported_new, 0)
        self.assertEqual(report_2.unchanged, 1)
        self.assertEqual(self.db.get_count(), 1)

    async def test_rename_move_updates_path_without_reimport(self):
        original = self._write_fit(self.library_root / "run_original.fit", payload=b"stable-data")
        moved = self.library_root / "moved" / "run_renamed.fit"

        manager = self._create_manager()
        initial_report = await manager.set_library_root(str(self.library_root))
        self.assertIsNone(initial_report)
        self.assertEqual(self.db.get_count(), 1)

        original_hash = calculate_file_hash(str(original))
        moved.parent.mkdir(parents=True, exist_ok=True)
        original.rename(moved)

        report = await manager.resync_now()
        self.assertEqual(report.imported_new, 0)
        self.assertEqual(report.moved_or_renamed, 1)
        self.assertEqual(report.failed, 0)

        activity = self.db.get_activity_by_hash(original_hash)
        self.assertIsNotNone(activity)
        self.assertEqual(os.path.realpath(activity["file_path"]), os.path.realpath(str(moved)))

        with self.db.get_connection() as conn:
            row = conn.execute(
                "SELECT file_path, import_status FROM library_files WHERE content_hash = ?",
                (original_hash,),
            ).fetchone()
        self.assertIsNotNone(row)
        self.assertEqual(os.path.realpath(row["file_path"]), os.path.realpath(str(moved)))
        self.assertEqual(row["import_status"], "imported")

    async def test_startup_cleanup_removes_stale_import_cache_files(self):
        stale_file = self._write_fit(self.import_cache / "old_import.fit", payload=b"old")
        fresh_file = self._write_fit(self.import_cache / "fresh_import.fit", payload=b"fresh")

        now = time.time()
        stale_ts = now - (48 * 60 * 60)
        os.utime(stale_file, (stale_ts, stale_ts))
        os.utime(fresh_file, (now, now))

        manager = self._create_manager(retention_seconds=24 * 60 * 60)
        try:
            await manager.start()
            self.assertFalse(stale_file.exists())
            self.assertTrue(fresh_file.exists())
        finally:
            await manager.stop()

    async def test_reconcile_marks_deleted_library_file_missing_without_deleting_activity(self):
        fit_path = self._write_fit(self.library_root / "run_for_delete.fit", payload=b"delete-me")
        file_hash = calculate_file_hash(str(fit_path))
        manager = self._create_manager()

        await manager.set_library_root(str(self.library_root))
        self.assertEqual(self.db.get_count(), 1)

        fit_path.unlink()
        report = await manager.resync_now()

        self.assertEqual(report.missing_files, 1)
        self.assertEqual(self.db.get_count(), 1)
        self.assertIsNotNone(self.db.get_activity_by_hash(file_hash))

        with self.db.get_connection() as conn:
            row = conn.execute(
                "SELECT is_missing, missing_since_at FROM library_files WHERE content_hash = ?",
                (file_hash,),
            ).fetchone()
        self.assertIsNotNone(row)
        self.assertEqual(row["is_missing"], 1)
        self.assertIsNotNone(row["missing_since_at"])

    async def test_reconcile_clears_missing_flag_when_file_reappears(self):
        fit_path = self._write_fit(self.library_root / "run_reappear.fit", payload=b"reappear-me")
        file_hash = calculate_file_hash(str(fit_path))
        manager = self._create_manager()

        await manager.set_library_root(str(self.library_root))
        fit_path.unlink()
        first_report = await manager.resync_now()
        self.assertEqual(first_report.missing_files, 1)

        self._write_fit(fit_path, payload=b"reappear-me")
        second_report = await manager.resync_now()
        self.assertEqual(second_report.missing_files, 0)

        with self.db.get_connection() as conn:
            row = conn.execute(
                "SELECT is_missing, missing_since_at FROM library_files WHERE content_hash = ?",
                (file_hash,),
            ).fetchone()
        self.assertIsNotNone(row)
        self.assertEqual(row["is_missing"], 0)
        self.assertIsNone(row["missing_since_at"])

    async def test_live_monitor_imports_new_file_without_manual_resync(self):
        manager = self._create_manager(
            interval=3600,
            live_check_interval=0.25,
            event_debounce=0.2,
        )
        manager._set_setting("library_root", str(self.library_root))

        try:
            await manager.start()
            self.assertEqual(self.db.get_count(), 0)

            fit_path = self._write_fit(self.library_root / "live_added.fit", payload=b"live-add")
            expected_hash = calculate_file_hash(str(fit_path))

            deadline = time.monotonic() + 5.0
            imported = False
            while time.monotonic() < deadline:
                if self.db.activity_exists(expected_hash):
                    imported = True
                    break
                await asyncio.sleep(0.15)

            self.assertTrue(imported, "live monitor did not ingest newly added FIT file")
            self.assertEqual(self.db.get_count(), 1)
        finally:
            await manager.stop()


if __name__ == "__main__":
    unittest.main()
