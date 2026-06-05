"""Tests for template_engine.py"""

from unittest.mock import MagicMock, patch

import pytest

from template_engine import (
    PLATFORMS,
    list_platforms,
    load_prompt,
    rewrite_for_platform,
    generate_platform_content,
    generate_all_platforms,
)


# ── Helper: create a clean mock client (no side_effect from conftest) ────


def _make_mock_client(response_content: str = "默认回复"):
    """Create a fresh mock OpenAI client that returns the given content."""
    choice = MagicMock()
    choice.message.content = response_content
    resp = MagicMock()
    resp.choices = [choice]

    client = MagicMock()
    client.chat.completions.create.return_value = resp
    return client


# ═══════════════════════════════════════════════════════════════════════════
# Platform registry
# ═══════════════════════════════════════════════════════════════════════════


class TestPlatformRegistry:
    def test_four_platforms_registered(self):
        assert len(PLATFORMS) == 4
        assert "wechat" in PLATFORMS
        assert "xhs" in PLATFORMS
        assert "douyin" in PLATFORMS
        assert "podcast" in PLATFORMS

    def test_each_platform_has_required_fields(self):
        required = {"name", "template", "extension", "description"}
        for key, cfg in PLATFORMS.items():
            missing = required - set(cfg.keys())
            assert not missing, f"{key} missing fields: {missing}"

    def test_wechat_is_markdown(self):
        assert PLATFORMS["wechat"]["extension"] == ".md"

    def test_xhs_is_txt(self):
        assert PLATFORMS["xhs"]["extension"] == ".txt"

    def test_douyin_is_txt(self):
        assert PLATFORMS["douyin"]["extension"] == ".txt"


class TestListPlatforms:
    def test_returns_four_items(self):
        result = list_platforms()
        assert len(result) == 4

    def test_each_has_key_name_description_extension(self):
        for item in list_platforms():
            assert "key" in item
            assert "name" in item
            assert "description" in item
            assert "extension" in item


# ═══════════════════════════════════════════════════════════════════════════
# load_prompt
# ═══════════════════════════════════════════════════════════════════════════


class TestLoadPrompt:
    def test_raises_for_unknown_platform(self):
        with pytest.raises(ValueError, match="Unknown platform"):
            load_prompt("nonexistent", "summary", "title")

    def test_injects_title(self):
        prompt = load_prompt("xhs", "摘要内容", "测试标题ABC")
        assert "测试标题ABC" in prompt

    def test_injects_summary(self):
        prompt = load_prompt("xhs", "摘要内容XYZ", "标题")
        assert "摘要内容XYZ" in prompt

    def test_injects_date(self):
        prompt = load_prompt("wechat", "摘要", "标题", date_str="2026-06-15")
        assert "2026-06-15" in prompt

    def test_wechat_template_loads(self):
        prompt = load_prompt("wechat", "摘要", "标题")
        assert len(prompt) > 100
        assert "微信" in prompt

    def test_douyin_template_loads(self):
        prompt = load_prompt("douyin", "摘要", "标题")
        assert len(prompt) > 100
        assert "抖音" in prompt or "douyin" in prompt.lower()

    def test_podcast_template_loads(self):
        prompt = load_prompt("podcast", "摘要", "标题")
        assert len(prompt) > 100


# ═══════════════════════════════════════════════════════════════════════════
# rewrite_for_platform
# ═══════════════════════════════════════════════════════════════════════════


class TestRewriteForPlatform:
    def test_returns_string(self):
        client = _make_mock_client("# 改写后的内容")
        result = rewrite_for_platform(client, "测试摘要", "xhs", "测试标题")
        assert isinstance(result, str)
        assert len(result) > 0

    def test_calls_ai_with_correct_model(self):
        client = _make_mock_client("ok")
        rewrite_for_platform(client, "摘要", "wechat", "标题")
        call_kwargs = client.chat.completions.create.call_args.kwargs
        assert "model" in call_kwargs
        assert call_kwargs["temperature"] == 0.7
        assert len(call_kwargs["messages"]) == 2

    def test_higher_temperature_for_creativity(self):
        client = _make_mock_client("ok")
        rewrite_for_platform(client, "摘要", "douyin", "标题")
        call_kwargs = client.chat.completions.create.call_args.kwargs
        assert call_kwargs["temperature"] == 0.7

    def test_raises_for_unknown_platform(self):
        client = _make_mock_client("ok")
        with pytest.raises(ValueError, match="Unknown platform"):
            rewrite_for_platform(client, "摘要", "invalid", "标题")

    def test_strips_whitespace_from_response(self):
        client = _make_mock_client("  内容前后有空格  \n\n")
        result = rewrite_for_platform(client, "摘要", "xhs", "标题")
        assert result == "内容前后有空格"


# ═══════════════════════════════════════════════════════════════════════════
# generate_platform_content
# ═══════════════════════════════════════════════════════════════════════════


class TestGeneratePlatformContent:
    def test_saves_file_to_output_dir(self, tmp_path):
        client = _make_mock_client("小红书风格内容")

        output = tmp_path / "output"
        result = generate_platform_content(
            client, "测试摘要", "xhs", "测试标题",
            output_dir=str(output), date_str="2026-06-02",
        )

        assert result["file_path"] is not None
        assert result["platform_name"] == "小红书"

        saved = output / "daily_2026-06-02_xhs.txt"
        assert saved.exists()
        assert saved.read_text(encoding="utf-8") == "小红书风格内容"

    def test_filename_includes_date_and_platform(self, tmp_path):
        client = _make_mock_client("x")

        output = tmp_path / "out"
        result = generate_platform_content(
            client, "s", "wechat", "t",
            output_dir=str(output), date_str="2026-06-15",
        )

        file_path = result["file_path"]
        assert "2026-06-15" in file_path
        assert "wechat" in file_path

    def test_avoids_overwrite_with_counter(self, tmp_path):
        client = _make_mock_client("c")

        output = tmp_path / "out"
        # Create a file that would collide
        output.mkdir(parents=True)
        (output / "daily_2026-06-02_xhs.txt").write_text("existing")

        result = generate_platform_content(
            client, "s", "xhs", "t",
            output_dir=str(output), date_str="2026-06-02",
        )

        assert "xhs_1" in str(result["file_path"])

    def test_returns_none_filepath_when_no_output_dir(self):
        client = _make_mock_client("c")

        result = generate_platform_content(
            client, "s", "xhs", "t", output_dir=None,
        )
        assert result["file_path"] is None
        assert result["content"] == "c"


# ═══════════════════════════════════════════════════════════════════════════
# generate_all_platforms
# ═══════════════════════════════════════════════════════════════════════════


class TestGenerateAllPlatforms:
    def test_defaults_to_all_four_platforms(self, tmp_path):
        client = _make_mock_client("ok")

        output = tmp_path / "out"
        results = generate_all_platforms(
            client, "摘要", "标题", output_dir=str(output),
        )

        assert len(results) == 4
        platform_keys = {r["platform"] for r in results}
        assert platform_keys == {"wechat", "xhs", "douyin", "podcast"}

    def test_specific_platforms_only(self, tmp_path):
        client = _make_mock_client("ok")

        output = tmp_path / "out"
        results = generate_all_platforms(
            client, "摘要", "标题", output_dir=str(output),
            platforms=["xhs", "wechat"],
        )

        assert len(results) == 2
        keys = {r["platform"] for r in results}
        assert keys == {"xhs", "wechat"}

    def test_skips_unknown_platforms(self, tmp_path):
        client = _make_mock_client("ok")

        output = tmp_path / "out"
        results = generate_all_platforms(
            client, "摘要", "标题", output_dir=str(output),
            platforms=["xhs", "unknown_platform", "wechat"],
        )

        assert len(results) == 2  # unknown skipped

    def test_handles_error_gracefully_per_platform(self, tmp_path):
        # First call succeeds, second fails
        success_response = MagicMock()
        success_response.choices = [MagicMock(message=MagicMock(content="ok"))]

        client = MagicMock()
        client.chat.completions.create.side_effect = [
            success_response,
            Exception("API 调用失败"),
            success_response,
            success_response,
        ]

        output = tmp_path / "out"
        results = generate_all_platforms(
            client, "摘要", "标题", output_dir=str(output),
            platforms=["xhs", "wechat", "douyin", "podcast"],
        )

        assert len(results) == 4
        errors = [r for r in results if r.get("error")]
        assert len(errors) == 1
        assert errors[0]["platform"] == "wechat"
