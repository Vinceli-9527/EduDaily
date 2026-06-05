"""Tests for publisher.py"""

import json
from unittest.mock import MagicMock, patch

import pytest

from publisher import (
    PublishResult,
    load_credentials,
    save_credentials,
    get_credentials_for,
    BasePublisher,
    WeChatPublisher,
    ZhihuPublisher,
    WeiboPublisher,
    PUBLISHERS,
    list_publishers,
    publish_draft,
    CREDENTIALS_PATH,
)


class TestCredentialManagement:
    def test_load_returns_empty_when_no_file(self, monkeypatch, tmp_path):
        monkeypatch.setattr("publisher.CREDENTIALS_PATH", tmp_path / "nonexistent.json")
        assert load_credentials() == {}

    def test_save_and_load_roundtrip(self, monkeypatch, tmp_path):
        creds_path = tmp_path / "credentials.json"
        monkeypatch.setattr("publisher.CREDENTIALS_PATH", creds_path)

        data = {"wechat": {"app_id": "wx123", "app_secret": "secret"}}
        save_credentials(data)
        assert creds_path.exists()
        loaded = load_credentials()
        assert loaded == data

    def test_get_credentials_for_platform(self, monkeypatch, tmp_path):
        creds_path = tmp_path / "credentials.json"
        monkeypatch.setattr("publisher.CREDENTIALS_PATH", creds_path)

        save_credentials({"wechat": {"app_id": "wx123"}, "zhihu": {"z_c0": "abc"}})
        assert get_credentials_for("wechat") == {"app_id": "wx123"}
        assert get_credentials_for("unknown") == {}


class TestPublishDraft:
    def test_returns_error_for_unknown_platform(self):
        result = publish_draft("unknown", "content", "title")
        assert result.success is False
        assert "未知平台" in result.error

    def test_returns_error_when_no_credentials(self, monkeypatch, tmp_path):
        monkeypatch.setattr("publisher.CREDENTIALS_PATH", tmp_path / "nonexistent.json")
        monkeypatch.setattr("publisher.get_credentials_for", lambda p: {})

        result = publish_draft("wechat", "content", "title")
        assert result.success is False
        assert "凭据" in result.error or "credentials" in result.error.lower()

    def test_zhihu_returns_not_available(self, monkeypatch, tmp_path):
        creds_path = tmp_path / "credentials.json"
        creds_path.write_text(json.dumps({"zhihu": {"z_c0": "test123"}}))
        monkeypatch.setattr("publisher.CREDENTIALS_PATH", creds_path)

        result = publish_draft("zhihu", "content", "title")
        assert result.success is False
        assert "草稿箱" in result.message or "API 不可用" in result.error


class TestListPublishers:
    def test_returns_three_entries(self):
        publishers = list_publishers()
        assert len(publishers) == 3

    def test_each_has_required_fields(self):
        for p in list_publishers():
            assert "key" in p
            assert "name" in p
            assert "requires" in p


class TestPublishResult:
    def test_has_expected_fields(self):
        result = PublishResult(platform="test", success=True, message="ok")
        assert result.platform == "test"
        assert result.success is True
        assert result.message == "ok"
        assert result.error == ""


class TestBasePublisher:
    def test_validate_detects_missing_credentials(self):
        with pytest.raises(ValueError, match="缺少凭据"):
            WeChatPublisher(credentials={})

    def test_validate_passes_with_all_required(self):
        pub = WeChatPublisher(credentials={
            "app_id": "wx123", "app_secret": "sec", "access_token": "tok",
        })
        assert pub.platform_key == "wechat"

    def test_weibo_publisher_attempts_http_call(self, monkeypatch):
        mock_post = MagicMock()
        mock_post.return_value.json.return_value = {"id": "12345"}
        monkeypatch.setattr("requests.post", mock_post)

        pub = WeiboPublisher(credentials={
            "access_token": "tok123",
            "uid": "111",
        })
        result = pub.publish("content", "title")
        assert result.success is True
        assert result.draft_id == "12345"

    def test_weibo_handles_api_error(self, monkeypatch):
        mock_post = MagicMock()
        mock_post.return_value.json.return_value = {"error": "invalid token"}
        monkeypatch.setattr("requests.post", mock_post)

        pub = WeiboPublisher(credentials={"access_token": "bad"})
        result = pub.publish("content", "title")
        assert result.success is False
        assert "错误" in result.error
