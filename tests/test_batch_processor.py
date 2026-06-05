"""Tests for batch_processor.py"""

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from batch_processor import (
    generate_summary,
    find_unprocessed,
    process_batch,
    generate_daily_report,
    load_processed_log,
    save_processed_log,
    PROCESSED_LOG_PATH,
)


# ═══════════════════════════════════════════════════════════════════════════
# generate_summary
# ═══════════════════════════════════════════════════════════════════════════


class TestGenerateSummary:
    def test_returns_string_for_valid_text(self, mock_client, sample_article_body, sample_article_title):
        result = generate_summary(mock_client, sample_article_body, sample_article_title)
        assert isinstance(result, str)
        assert len(result) > 0
        mock_client.chat.completions.create.assert_called_once()

    def test_sends_correct_model_and_temperature(self, mock_client, sample_article_body, sample_article_title):
        generate_summary(mock_client, sample_article_body, sample_article_title)

        call_kwargs = mock_client.chat.completions.create.call_args.kwargs
        assert "model" in call_kwargs
        assert call_kwargs["temperature"] == 0.3
        assert call_kwargs["max_tokens"] == 800

    def test_returns_placeholder_for_short_text(self, mock_client):
        result = generate_summary(mock_client, "太短", "标题")
        assert "过短" in result
        mock_client.chat.completions.create.assert_not_called()

    def test_returns_placeholder_for_empty_text(self, mock_client):
        result = generate_summary(mock_client, "", "标题")
        assert "过短" in result
        mock_client.chat.completions.create.assert_not_called()

    def test_truncates_long_article_to_5000_chars(self, mock_client):
        long_text = "教育" * 6000
        generate_summary(mock_client, long_text, "标题")

        sent_text = mock_client.chat.completions.create.call_args.kwargs["messages"][1]["content"]
        assert len(sent_text) <= 5200  # 5000 + prompt overhead

    def test_includes_title_in_prompt(self, mock_client, sample_article_body):
        generate_summary(mock_client, sample_article_body, "测试标题ABC")

        user_msg = mock_client.chat.completions.create.call_args.kwargs["messages"][1]["content"]
        assert "测试标题ABC" in user_msg

    def test_strips_response_whitespace(self, sample_article_body, mock_openai_response):
        client = MagicMock()
        client.chat.completions.create.return_value = mock_openai_response("  摘要内容含空格  \n\n")
        result = generate_summary(client, sample_article_body, "标题")
        assert result == "摘要内容含空格"


# ═══════════════════════════════════════════════════════════════════════════
# Processed log helpers
# ═══════════════════════════════════════════════════════════════════════════


class TestProcessedLog:
    def test_load_returns_empty_dict_when_no_file(self, tmp_path, monkeypatch):
        log_path = tmp_path / "nonexistent.json"
        monkeypatch.setattr("batch_processor.PROCESSED_LOG_PATH", log_path)
        result = load_processed_log()
        assert result == {}

    def test_save_and_load_roundtrip(self, tmp_path, monkeypatch):
        log_path = tmp_path / "test_processed.json"
        monkeypatch.setattr("batch_processor.PROCESSED_LOG_PATH", log_path)

        data = {"file1.txt": {"status": "processed", "date": "2026-06-02"}}
        save_processed_log(data)

        assert log_path.exists()
        loaded = load_processed_log()
        assert loaded == data

    def test_save_creates_parent_dirs(self, tmp_path, monkeypatch):
        log_path = tmp_path / "deep" / "nested" / "processed.json"
        monkeypatch.setattr("batch_processor.PROCESSED_LOG_PATH", log_path)

        save_processed_log({"key": "value"})
        assert log_path.exists()


# ═══════════════════════════════════════════════════════════════════════════
# find_unprocessed
# ═══════════════════════════════════════════════════════════════════════════


class TestFindUnprocessed:
    def test_returns_empty_when_no_files_exist(self, temp_data_dir, monkeypatch):
        # temp_data_dir only has .gitkeep
        monkeypatch.setattr("batch_processor.PROCESSED_LOG_PATH",
                            temp_data_dir.parent / "processed.json")
        result = find_unprocessed(str(temp_data_dir))
        assert result == []

    def test_skips_gitkeep(self, temp_data_dir, monkeypatch):
        monkeypatch.setattr("batch_processor.PROCESSED_LOG_PATH",
                            temp_data_dir.parent / "processed.json")
        result = find_unprocessed(str(temp_data_dir))
        gitkeeps = [f for f in result if f.name == ".gitkeep"]
        assert len(gitkeeps) == 0

    def test_finds_new_text_files(self, temp_data_dir, temp_article_files, monkeypatch):
        monkeypatch.setattr("batch_processor.PROCESSED_LOG_PATH",
                            temp_data_dir.parent / "processed.json")
        result = find_unprocessed(str(temp_data_dir))
        assert len(result) == len(temp_article_files)

    def test_excludes_already_processed_files(self, temp_data_dir, temp_article_files, monkeypatch):
        log_path = temp_data_dir.parent / "processed.json"
        monkeypatch.setattr("batch_processor.PROCESSED_LOG_PATH", log_path)

        # Mark first file as processed
        processed = {temp_article_files[0].name: {"status": "processed"}}
        log_path.write_text(json.dumps(processed, ensure_ascii=False), encoding="utf-8")

        result = find_unprocessed(str(temp_data_dir))
        assert len(result) == len(temp_article_files) - 1
        assert temp_article_files[0] not in result

    def test_returns_sorted_by_name(self, temp_data_dir, monkeypatch):
        monkeypatch.setattr("batch_processor.PROCESSED_LOG_PATH",
                            temp_data_dir.parent / "processed.json")
        # Create files out of order
        (temp_data_dir / "z_file.txt").write_text("z content", encoding="utf-8")
        (temp_data_dir / "a_file.txt").write_text("a content", encoding="utf-8")

        result = find_unprocessed(str(temp_data_dir))
        names = [f.name for f in result]
        assert names == sorted(names)

    def test_force_all_returns_all_files(self, temp_data_dir, temp_article_files, monkeypatch):
        log_path = temp_data_dir.parent / "processed.json"
        monkeypatch.setattr("batch_processor.PROCESSED_LOG_PATH", log_path)

        # Mark ALL files as processed
        processed = {f.name: {"status": "processed"} for f in temp_article_files}
        log_path.write_text(json.dumps(processed, ensure_ascii=False), encoding="utf-8")

        result = find_unprocessed(str(temp_data_dir), force_all=True)
        assert len(result) == len(temp_article_files)


# ═══════════════════════════════════════════════════════════════════════════
# generate_daily_report
# ═══════════════════════════════════════════════════════════════════════════


class TestGenerateDailyReport:
    def test_creates_markdown_file(self, temp_data_dir):
        output_dir = temp_data_dir.parent / "output"
        output_dir.mkdir(parents=True, exist_ok=True)

        summaries = [
            {"title": "测试文章1", "filename": "f1.txt", "summary": "摘要1", "elapsed_ms": 100},
            {"title": "测试文章2", "filename": "f2.txt", "summary": "摘要2", "elapsed_ms": 200},
        ]

        path = generate_daily_report("2026-06-02", summaries, str(output_dir))
        assert Path(path).exists()
        assert "daily_summary_2026-06-02.md" in path

    def test_includes_all_summary_titles(self, temp_data_dir):
        output_dir = temp_data_dir.parent / "output"
        output_dir.mkdir(parents=True, exist_ok=True)

        summaries = [
            {"title": "文章A", "filename": "a.txt", "summary": "摘要A", "elapsed_ms": 100},
            {"title": "文章B", "filename": "b.txt", "summary": "摘要B", "elapsed_ms": 100},
            {"title": "文章C", "filename": "c.txt", "summary": "摘要C", "elapsed_ms": 100},
        ]

        path = generate_daily_report("2026-06-02", summaries, str(output_dir))
        content = Path(path).read_text(encoding="utf-8")

        for s in summaries:
            assert s["title"] in content
            assert s["summary"] in content

    def test_has_table_of_contents(self, temp_data_dir):
        output_dir = temp_data_dir.parent / "output"
        output_dir.mkdir(parents=True, exist_ok=True)

        summaries = [{"title": "测试", "filename": "f.txt", "summary": "摘要", "elapsed_ms": 100}]

        path = generate_daily_report("2026-06-02", summaries, str(output_dir))
        content = Path(path).read_text(encoding="utf-8")

        assert "## 目录" in content
        assert "[测试](#article-1)" in content

    def test_has_anchor_ids(self, temp_data_dir):
        output_dir = temp_data_dir.parent / "output"
        output_dir.mkdir(parents=True, exist_ok=True)

        summaries = [{"title": "文章X", "filename": "x.txt", "summary": "摘要X", "elapsed_ms": 100}]

        path = generate_daily_report("2026-06-02", summaries, str(output_dir))
        content = Path(path).read_text(encoding="utf-8")

        assert 'id="article-1"' in content

    def test_includes_date_in_header(self, temp_data_dir):
        output_dir = temp_data_dir.parent / "output"
        output_dir.mkdir(parents=True, exist_ok=True)

        summaries = [{"title": "测试", "filename": "f.txt", "summary": "摘要", "elapsed_ms": 100}]

        path = generate_daily_report("2026-06-02", summaries, str(output_dir))
        content = Path(path).read_text(encoding="utf-8")
        assert "2026-06-02" in content

    def test_handles_empty_summaries(self, temp_data_dir):
        output_dir = temp_data_dir.parent / "output"
        output_dir.mkdir(parents=True, exist_ok=True)

        path = generate_daily_report("2026-06-02", [], str(output_dir))
        content = Path(path).read_text(encoding="utf-8")
        assert "共 0 篇" in content


# ═══════════════════════════════════════════════════════════════════════════
# process_batch (integration)
# ═══════════════════════════════════════════════════════════════════════════


class TestProcessBatch:
    def test_returns_zero_when_no_files(self, mock_client, temp_data_dir):
        result = process_batch(
            client=mock_client,
            data_dir=str(temp_data_dir),
            output_dir=str(temp_data_dir.parent / "output"),
        )
        assert result["processed"] == 0
        assert result["total"] == 0
        assert result["report_path"] is None

    def test_processes_all_unprocessed_files(self, mock_client, temp_article_files, temp_data_dir):
        result = process_batch(
            client=mock_client,
            data_dir=str(temp_data_dir),
            output_dir=str(temp_data_dir.parent / "output"),
        )
        assert result["processed"] == len(temp_article_files)
        assert result["total"] == len(temp_article_files)
        assert result["report_path"] is not None

    def test_generates_report_file(self, mock_client, temp_article_files, temp_data_dir):
        result = process_batch(
            client=mock_client,
            data_dir=str(temp_data_dir),
            output_dir=str(temp_data_dir.parent / "output"),
        )
        assert Path(result["report_path"]).exists()

    def test_updates_processed_log(self, mock_client, temp_article_files, temp_data_dir):
        process_batch(
            client=mock_client,
            data_dir=str(temp_data_dir),
            output_dir=str(temp_data_dir.parent / "output"),
        )

        log = load_processed_log()
        for f in temp_article_files:
            assert f.name in log
            assert log[f.name]["status"] == "processed"

    def test_skips_already_processed(self, mock_client, temp_article_files, temp_data_dir):
        # First pass
        process_batch(
            client=mock_client,
            data_dir=str(temp_data_dir),
            output_dir=str(temp_data_dir.parent / "output"),
        )
        call_count_first = mock_client.chat.completions.create.call_count

        # Second pass — should skip all
        result = process_batch(
            client=mock_client,
            data_dir=str(temp_data_dir),
            output_dir=str(temp_data_dir.parent / "output"),
        )
        assert result["processed"] == 0
        # No additional LLM calls
        assert mock_client.chat.completions.create.call_count == call_count_first

    def test_force_all_reprocesses(self, mock_client, temp_article_files, temp_data_dir):
        # First pass
        process_batch(client=mock_client, data_dir=str(temp_data_dir),
                      output_dir=str(temp_data_dir.parent / "output"))
        call_count_first = mock_client.chat.completions.create.call_count

        # Second pass with force_all
        result = process_batch(
            client=mock_client,
            data_dir=str(temp_data_dir),
            output_dir=str(temp_data_dir.parent / "output"),
            force_all=True,
        )
        assert result["processed"] == len(temp_article_files)
        assert mock_client.chat.completions.create.call_count > call_count_first

    def test_skips_empty_files(self, mock_client, temp_data_dir):
        (temp_data_dir / "empty.txt").write_text("", encoding="utf-8")

        result = process_batch(
            client=mock_client,
            data_dir=str(temp_data_dir),
            output_dir=str(temp_data_dir.parent / "output"),
        )
        assert result["processed"] == 0
        assert result["total"] == 1  # found but skipped

    def test_prints_progress_messages(self, mock_client, temp_article_files, temp_data_dir, capsys):
        process_batch(
            client=mock_client,
            data_dir=str(temp_data_dir),
            output_dir=str(temp_data_dir.parent / "output"),
        )
        captured = capsys.readouterr()
        assert "找到" in captured.out
        assert "正在生成摘要" in captured.out
        assert "批量分析完成" in captured.out

    def test_handles_auth_error_gracefully(self, temp_article_files, temp_data_dir):
        import openai

        fake_response = MagicMock()
        fake_response.request = MagicMock()

        client = MagicMock()
        client.chat.completions.create.side_effect = openai.AuthenticationError(
            "auth failed", response=fake_response, body=None
        )

        result = process_batch(
            client=client,
            data_dir=str(temp_data_dir),
            output_dir=str(temp_data_dir.parent / "output"),
        )
        assert result["processed"] == 0  # auth error stops batch immediately

    def test_handles_rate_limit_with_retry(self, temp_article_files, temp_data_dir, mock_openai_response,
                                           sample_summary_response):
        import openai

        fake_response = MagicMock()
        fake_response.request = MagicMock()

        client = MagicMock()
        # First call fails with rate limit, subsequent succeed
        client.chat.completions.create.side_effect = [
            openai.RateLimitError("rate limited", response=fake_response, body=None),
            mock_openai_response(sample_summary_response),
            mock_openai_response(sample_summary_response),
            mock_openai_response(sample_summary_response),
        ]

        result = process_batch(
            client=client,
            data_dir=str(temp_data_dir),
            output_dir=str(temp_data_dir.parent / "output"),
        )
        # 3 files, first file needs retry with 5s sleep, all should succeed
        assert result["processed"] == len(temp_article_files)

    def test_report_markdown_is_valid(self, mock_client, temp_article_files, temp_data_dir):
        result = process_batch(
            client=mock_client,
            data_dir=str(temp_data_dir),
            output_dir=str(temp_data_dir.parent / "output"),
        )
        content = Path(result["report_path"]).read_text(encoding="utf-8")
        # Basic markdown structure checks
        assert content.startswith("# EduDaily")
        assert "## 目录" in content
        assert "---" in content
