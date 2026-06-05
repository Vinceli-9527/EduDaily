"""Shared fixtures for EduDaily tests."""

import json
import os
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

# Ensure project root is importable
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


# ── Sample data ──────────────────────────────────────────────────────────


@pytest.fixture
def sample_article_body():
    return (
        "教育部今日发布了《关于推进人工智能赋能教育行动的指导意见》，"
        "明确提出到2027年，在全国范围内建设100个AI教育示范区，"
        "推动人工智能技术在课堂教学、教育评价和教育管理中的深度融合。"
        "该政策将覆盖从基础教育到高等教育的全学段，重点支持中西部地区。"
        "专家表示，这是我国教育数字化转型的关键一步。"
    )


@pytest.fixture
def sample_article_title():
    return "[教育部] 2026-06-02 | 教育部发布AI赋能教育行动指导意见"


@pytest.fixture
def sample_summary_response():
    return (
        "教育部发布AI赋能教育行动指导意见，计划到2027年建设100个AI教育示范区。"
        "\n- 覆盖从基础到高等教育的全学段"
        "\n- 重点支持中西部地区"
        "\n- 推动AI在课堂教学、评价和管理中的融合"
        "\n该政策标志着我国教育数字化转型进入实质推进阶段。"
    )


# ── Mock OpenAI client ───────────────────────────────────────────────────


@pytest.fixture
def mock_openai_response():
    """Create a mock chat completion response."""

    def _make_response(content: str):
        choice = MagicMock()
        choice.message.content = content
        resp = MagicMock()
        resp.choices = [choice]
        return resp

    return _make_response


@pytest.fixture
def mock_client(mock_openai_response, sample_summary_response):
    """A fully mocked openai.OpenAI client that returns summary responses."""

    def _create_response(*args, **kwargs):
        return mock_openai_response(sample_summary_response)

    client = MagicMock()
    client.chat.completions.create.side_effect = _create_response
    return client


# ── Temp directory fixtures ──────────────────────────────────────────────


@pytest.fixture
def temp_data_dir(tmp_path, monkeypatch):
    """Create a temporary data directory with .gitkeep, patching config paths.

    Returns the Path to the temp data dir.
    """
    data_dir = tmp_path / "data" / "sample_docs"
    data_dir.mkdir(parents=True)

    # Create .gitkeep
    (data_dir / ".gitkeep").write_text("", encoding="utf-8")

    # Create output dir
    output_dir = tmp_path / "output"
    output_dir.mkdir(parents=True)

    # Create DB dir
    db_dir = tmp_path / "data"
    db_dir.mkdir(parents=True, exist_ok=True)

    # Create processed.json dir
    (tmp_path / "data").mkdir(parents=True, exist_ok=True)

    # Patch config paths
    import config

    monkeypatch.setattr(config, "BASE_DIR", tmp_path)
    monkeypatch.setattr(config, "SAMPLE_DOCS_DIR", str(data_dir))
    monkeypatch.setattr(config, "OUTPUT_DIR", str(output_dir))
    monkeypatch.setattr(config, "SQLITE_DB_PATH", str(db_dir / "test.db"))
    monkeypatch.setattr(config, "CHROMA_PERSIST_DIR", str(tmp_path / "chroma_test"))
    monkeypatch.setattr(config, "LOG_FILE", str(tmp_path / "test.log"))

    # Also patch the batch_processor's PROCESSED_LOG_PATH
    monkeypatch.setattr(
        "batch_processor.PROCESSED_LOG_PATH",
        tmp_path / "data" / "processed.json",
    )

    return data_dir


@pytest.fixture
def temp_article_files(temp_data_dir, sample_article_body, sample_article_title):
    """Create sample .txt article files in the temp data directory.

    Returns list of Path objects for the created files.
    """
    files = []
    titles = [
        "[教育部] 2026-06-02 | AI赋能教育行动指导意见",
        "[北京大学] 2026-06-02 | 新型催化材料研究取得重大突破",
        "[上海市教委] 2026-06-02 | 2026年秋季高考改革方案公布",
    ]
    bodies = [
        sample_article_body,
        "北京大学化学与分子工程学院今日宣布，"
        "其团队在新型催化材料研究方面取得重大突破，"
        "相关成果已发表于《Science》期刊。"
        "该材料可将二氧化碳转化效率提升3倍以上。",
        "上海市教育委员会今日正式公布了2026年秋季高考改革方案，"
        "新方案将在考试科目设置、招生录取方式等方面进行重大调整。"
        "物理和历史将作为必选科目之一。",
    ]

    for i, (title, body) in enumerate(zip(titles, bodies)):
        content = f"{title}\n\n{body}"
        fname = f"article_{i+1}_{i+100}.txt"
        fpath = temp_data_dir / fname
        fpath.write_text(content, encoding="utf-8")
        files.append(fpath)

    return files
