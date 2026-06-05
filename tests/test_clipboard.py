"""Tests for clipboard.py"""

import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from clipboard import (
    copy_to_clipboard,
    copy_report_file,
    copy_latest_report,
    find_latest_report,
    _pyperclip_copy,
    _native_copy,
)


class TestPyperclipCopy:
    def test_returns_true_when_pyperclip_available(self, monkeypatch):
        mock_pyperclip = MagicMock()
        monkeypatch.setitem(sys.modules, "pyperclip", mock_pyperclip)

        result = _pyperclip_copy("hello")
        assert result is True
        mock_pyperclip.copy.assert_called_once_with("hello")

    def test_returns_false_when_pyperclip_not_installed(self, monkeypatch):
        # Simulate ImportError
        with patch("clipboard._pyperclip_copy", return_value=False):
            result = _pyperclip_copy("hello")
            assert result is False


class TestNativeCopy:
    def test_macos_uses_pbcopy(self):
        with patch("sys.platform", "darwin"), \
             patch("subprocess.run") as mock_run:
            mock_run.return_value.returncode = 0
            result = _native_copy("test")
            assert result is True
            mock_run.assert_called_once()
            args = mock_run.call_args[0][0]
            assert args[0] == "pbcopy"

    def test_windows_uses_powershell(self):
        with patch("sys.platform", "win32"), \
             patch("subprocess.run") as mock_run:
            mock_run.return_value.returncode = 0
            result = _native_copy("test")
            assert result is True
            args = mock_run.call_args[0][0]
            assert "powershell" in args[0]


class TestCopyToClipboard:
    def test_returns_false_for_empty_string(self):
        result = copy_to_clipboard("")
        assert result is False

    def test_returns_false_for_whitespace_only(self):
        result = copy_to_clipboard("   \n  ")
        assert result is False

    def test_tries_pyperclip_first(self, monkeypatch):
        mock = MagicMock()
        monkeypatch.setattr("clipboard._pyperclip_copy", mock)
        monkeypatch.setattr("clipboard._native_copy", MagicMock())
        mock.return_value = True

        result = copy_to_clipboard("content")
        assert result is True

    def test_falls_back_to_native(self, monkeypatch):
        pyper = MagicMock(return_value=False)
        native = MagicMock(return_value=True)
        monkeypatch.setattr("clipboard._pyperclip_copy", pyper)
        monkeypatch.setattr("clipboard._native_copy", native)

        result = copy_to_clipboard("content")
        assert result is True
        native.assert_called_once()

    def test_returns_false_when_all_methods_fail(self, monkeypatch):
        monkeypatch.setattr("clipboard._pyperclip_copy", MagicMock(return_value=False))
        monkeypatch.setattr("clipboard._native_copy", MagicMock(return_value=False))

        result = copy_to_clipboard("content")
        assert result is False


class TestFindLatestReport:
    def test_returns_none_for_missing_dir(self, tmp_path):
        nonexistent = str(tmp_path / "no_such_dir")
        result = find_latest_report(nonexistent)
        assert result is None

    def test_finds_default_daily_summary(self, tmp_path):
        report = tmp_path / "daily_summary_2026-06-02.md"
        report.write_text("# 日报", encoding="utf-8")

        result = find_latest_report(str(tmp_path), date_str="2026-06-02")
        assert result == report

    def test_finds_platform_specific_report(self, tmp_path, monkeypatch):
        # Register a mock platform for testing
        from template_engine import PLATFORMS

        report = tmp_path / "daily_2026-06-02_wechat.md"
        report.write_text("# 微信版", encoding="utf-8")

        result = find_latest_report(str(tmp_path), platform="wechat", date_str="2026-06-02")
        assert result == report

    def test_finds_counter_suffixed_report(self, tmp_path, monkeypatch):
        # Create a file with counter suffix
        (tmp_path / "daily_2026-06-02_xhs.txt").write_text("first")
        (tmp_path / "daily_2026-06-02_xhs_1.txt").write_text("second")
        (tmp_path / "daily_2026-06-02_xhs_2.txt").write_text("third")

        result = find_latest_report(str(tmp_path), platform="xhs", date_str="2026-06-02")
        assert result is not None
        assert "_2" in result.name  # highest counter

    def test_returns_none_when_no_match(self, tmp_path):
        result = find_latest_report(str(tmp_path), platform="wechat", date_str="2026-06-02")
        assert result is None


class TestCopyReportFile:
    def test_returns_false_for_missing_file(self, tmp_path):
        ok, chars = copy_report_file(tmp_path / "nonexistent.md")
        assert ok is False
        assert chars == 0

    def test_copies_file_content(self, tmp_path, monkeypatch):
        # Mock copy_to_clipboard to avoid actual clipboard access
        mock_copy = MagicMock(return_value=True)
        monkeypatch.setattr("clipboard.copy_to_clipboard", mock_copy)

        report = tmp_path / "report.md"
        report.write_text("# Hello World\n\nContent here.", encoding="utf-8")

        ok, chars = copy_report_file(str(report))
        assert ok is True
        assert chars > 0
        mock_copy.assert_called_once_with("# Hello World\n\nContent here.")


class TestCopyLatestReport:
    def test_returns_false_when_no_report_found(self, tmp_path, monkeypatch):
        monkeypatch.setattr("clipboard.copy_to_clipboard", MagicMock())
        result = copy_latest_report(str(tmp_path), date_str="2026-06-02")
        assert result is False

    def test_copies_when_report_found(self, tmp_path, monkeypatch):
        mock_copy = MagicMock(return_value=True)
        monkeypatch.setattr("clipboard.copy_to_clipboard", mock_copy)

        report = tmp_path / "daily_summary_2026-06-02.md"
        report.write_text("# 日报内容", encoding="utf-8")

        result = copy_latest_report(str(tmp_path), date_str="2026-06-02")
        assert result is True
        mock_copy.assert_called_once()
