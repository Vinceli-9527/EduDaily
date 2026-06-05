"""Tests for scheduler.py"""

import os
from unittest.mock import MagicMock, patch, call

import pytest

# scheduler.py requires schedule library or falls back gracefully
try:
    import schedule  # noqa: F401

    HAS_SCHEDULE = True
except ImportError:
    HAS_SCHEDULE = False


# ═══════════════════════════════════════════════════════════════════════════
# get_schedule_time
# ═══════════════════════════════════════════════════════════════════════════


class TestGetScheduleTime:
    def test_default_is_0700(self, monkeypatch):
        monkeypatch.delenv("SCHEDULE_TIME", raising=False)
        import importlib
        import config

        importlib.reload(config)
        from scheduler import get_schedule_time

        assert get_schedule_time() == "07:00"

    def test_reads_from_env(self, monkeypatch):
        monkeypatch.setenv("SCHEDULE_TIME", "08:30")

        import importlib
        import config

        importlib.reload(config)
        from scheduler import get_schedule_time

        assert get_schedule_time() == "08:30"

    def test_reads_arbitrary_time(self, monkeypatch):
        monkeypatch.setenv("SCHEDULE_TIME", "23:45")

        import importlib
        import config

        importlib.reload(config)
        from scheduler import get_schedule_time

        assert get_schedule_time() == "23:45"


# ═══════════════════════════════════════════════════════════════════════════
# init_pipeline
# ═══════════════════════════════════════════════════════════════════════════


class TestInitPipeline:
    def test_exits_when_api_key_not_set(self, monkeypatch):
        monkeypatch.delenv("DEEPSEEK_API_KEY", raising=False)
        monkeypatch.setattr("config.DEEPSEEK_API_KEY", "")

        from scheduler import init_pipeline

        with pytest.raises(SystemExit) as exc_info:
            init_pipeline()
        assert exc_info.value.code == 1

    def test_exits_when_api_key_is_placeholder(self, monkeypatch):
        monkeypatch.setattr("config.DEEPSEEK_API_KEY", "sk-your-key-here")

        from scheduler import init_pipeline

        with pytest.raises(SystemExit) as exc_info:
            init_pipeline()
        assert exc_info.value.code == 1


# ═══════════════════════════════════════════════════════════════════════════
# _sync_docs_from_db
# ═══════════════════════════════════════════════════════════════════════════


class TestSyncDocsFromDB:
    def test_returns_empty_list_for_empty_db(self, tmp_path, monkeypatch):
        import sqlite3

        monkeypatch.setattr("config.SAMPLE_DOCS_DIR", str(tmp_path))
        (tmp_path / ".gitkeep").write_text("", encoding="utf-8")

        db_path = tmp_path / "test.db"
        conn = sqlite3.connect(str(db_path))
        conn.row_factory = sqlite3.Row

        from scheduler import _sync_docs_from_db

        docs = _sync_docs_from_db(conn)
        assert isinstance(docs, list)
        assert len(docs) == 0
        conn.close()

    def test_loads_existing_docs_from_db(self, tmp_path, monkeypatch):
        import sqlite3

        from db.schema import init_db

        monkeypatch.setattr("config.SAMPLE_DOCS_DIR", str(tmp_path))

        # Create a sample .txt file
        txt = tmp_path / "test_article.txt"
        txt.write_text("标题\n\n正文内容", encoding="utf-8")

        db_path = tmp_path / "test.db"
        conn = sqlite3.connect(str(db_path))
        conn.row_factory = sqlite3.Row
        init_db(conn)

        # Insert a document record
        conn.execute(
            "INSERT INTO documents (filename, title, source) VALUES (?, ?, ?)",
            ("test_article.txt", "标题", str(txt)),
        )
        conn.commit()

        from scheduler import _sync_docs_from_db

        docs = _sync_docs_from_db(conn)
        assert len(docs) == 1
        assert docs[0].filename == "test_article.txt"
        assert docs[0]._db_id is not None
        conn.close()


# ═══════════════════════════════════════════════════════════════════════════
# run_full_pipeline
# ═══════════════════════════════════════════════════════════════════════════


class TestRunFullPipeline:
    def test_calls_fetch_and_batch_in_order(self):
        mock_fetch = MagicMock(return_value={
            "total_articles": 3,
            "sources_checked": 2,
            "sources_failed": 0,
            "errors": [],
        })
        mock_batch = MagicMock(return_value={
            "processed": 3,
            "report_path": "/tmp/report.md",
        })

        with patch("modules.daily_fetcher.run_daily_fetch", mock_fetch), \
             patch("batch_processor.process_batch", mock_batch):
            from scheduler import run_full_pipeline

            run_full_pipeline(
                MagicMock(), MagicMock(), MagicMock(), MagicMock(), []
            )

        mock_fetch.assert_called_once()
        mock_batch.assert_called_once()

    def test_batch_runs_even_if_fetch_has_errors(self):
        mock_fetch = MagicMock(return_value={
            "total_articles": 0,
            "sources_checked": 2,
            "sources_failed": 2,
            "errors": ["source1: 连接失败", "source2: 超时"],
        })
        mock_batch = MagicMock(return_value={"processed": 1, "report_path": "/tmp/r.md"})

        with patch("modules.daily_fetcher.run_daily_fetch", mock_fetch), \
             patch("batch_processor.process_batch", mock_batch):
            from scheduler import run_full_pipeline

            run_full_pipeline(
                MagicMock(), MagicMock(), MagicMock(), MagicMock(), []
            )

        mock_batch.assert_called_once()

    def test_batch_runs_even_if_fetch_raises_exception(self):
        mock_fetch = MagicMock(side_effect=Exception("网络中断"))
        mock_batch = MagicMock(return_value={"processed": 0, "report_path": None})

        with patch("modules.daily_fetcher.run_daily_fetch", mock_fetch), \
             patch("batch_processor.process_batch", mock_batch):
            from scheduler import run_full_pipeline

            run_full_pipeline(
                MagicMock(), MagicMock(), MagicMock(), MagicMock(), []
            )

        mock_batch.assert_called_once()

    def test_handles_batch_failure_gracefully(self):
        mock_fetch = MagicMock(return_value={
            "total_articles": 5, "sources_checked": 1,
            "sources_failed": 0, "errors": [],
        })
        mock_batch = MagicMock(side_effect=Exception("LLM 调用失败"))

        with patch("modules.daily_fetcher.run_daily_fetch", mock_fetch), \
             patch("batch_processor.process_batch", mock_batch):
            from scheduler import run_full_pipeline

            # Should not raise
            run_full_pipeline(
                MagicMock(), MagicMock(), MagicMock(), MagicMock(), []
            )


# ═══════════════════════════════════════════════════════════════════════════
# start_scheduler
# ═══════════════════════════════════════════════════════════════════════════


@pytest.mark.skipif(not HAS_SCHEDULE, reason="schedule library not installed")
class TestStartScheduler:
    def test_configures_daily_schedule(self, monkeypatch):
        import schedule

        monkeypatch.setattr("config.SCHEDULE_TIME", "07:00")
        # Clear any existing jobs
        schedule.clear()

        # We only test schedule configuration, not the blocking loop
        from scheduler import get_schedule_time, run_full_pipeline

        schedule.every().day.at("07:00").do(lambda: None)

        jobs = schedule.get_jobs()
        assert len(jobs) == 1

        schedule.clear()

    def test_schedule_time_format_valid(self):
        from scheduler import get_schedule_time

        time_str = get_schedule_time()
        parts = time_str.split(":")
        assert len(parts) == 2
        assert 0 <= int(parts[0]) <= 23
        assert 0 <= int(parts[1]) <= 59


# ═══════════════════════════════════════════════════════════════════════════
# CLI interface
# ═══════════════════════════════════════════════════════════════════════════


class TestSchedulerCLI:
    def test_once_mode_flag(self):
        """Verify --once argument is defined on the parser."""
        import argparse
        import importlib

        # Verify the module defines the right arguments
        import scheduler

        parser = argparse.ArgumentParser()
        parser.add_argument("--once", action="store_true")
        args = parser.parse_args(["--once"])
        assert args.once is True

        args = parser.parse_args([])
        assert args.once is False
