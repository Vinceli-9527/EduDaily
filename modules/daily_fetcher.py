"""Daily fetch engine — crawl news sources and extract today's articles.

Flow for each source:
1. Fetch source URL → extract text
2. LLM identifies today's articles from the page text
3. For each today-article: fetch → LLM-refine → save .txt → index into KB
4. Record in edu_articles table with source traceability
"""

import json
import logging
import re
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import urljoin, urlparse

import openai
import requests

import config
from db import repository as repo
from modules.url_ingester import fetch_url, generate_kb_document

logger = logging.getLogger(__name__)

# ── Constants ────────────────────────────────────────────────────────────

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/125.0.0.0 Safari/537.36"
    ),
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    "Accept-Language": "zh-CN,zh;q=0.9,en;q=0.8",
}
REQUEST_TIMEOUT = 20

# ── Article discovery prompt ──────────────────────────────────────────────

ARTICLE_DISCOVERY_SYSTEM = """你是一个网页内容分析助手。你的任务是从网页的文本内容中提取出当天发布的文章链接和信息。

你必须遵守以下规则：
1. 只提取文本中明确标注了今天日期的文章
2. 不要编造或猜测任何信息
3. 如果无法确认文章是否是今天发布的，不要包含它
4. 返回合法的 JSON，不要包含任何 JSON 之外的文本"""

ARTICLE_DISCOVERY_USER = """请从以下网页文本中，提取所有发布日期为 {today} 的文章信息。

网页来源：{source_name}（{source_url}）

对于每篇发布日期为 {today} 的文章，提取：
- title: 文章标题
- url: 文章链接（如有，可以是相对路径或完整URL）
- publish_date: 发布日期
- brief: 文章简介或摘要（如有，不超过100字）

如果没有找到今天发布的文章，返回空数组。

══════════ 网页文本内容 ══════════
{page_text}
══════════ 结束 ══════════

请以 JSON 数组格式返回，每个元素包含 title、url、publish_date、brief 字段。
如果没有找到今天的文章，返回：[]"""


def _discover_today_articles(
    client: openai.OpenAI,
    source_name: str,
    source_url: str,
    page_text: str,
    today: str,
) -> list[dict]:
    """Use LLM to identify articles published today from a page's text."""
    if not page_text or len(page_text) < 100:
        return []

    user_prompt = ARTICLE_DISCOVERY_USER.format(
        today=today,
        source_name=source_name,
        source_url=source_url,
        page_text=page_text[:8000],
    )

    try:
        resp = client.chat.completions.create(
            model=config.DEEPSEEK_CHAT_MODEL,
            messages=[
                {"role": "system", "content": ARTICLE_DISCOVERY_SYSTEM},
                {"role": "user", "content": user_prompt},
            ],
            temperature=0.1,
            max_tokens=2048,
        )
        raw = resp.choices[0].message.content or ""

        # Try to parse JSON from response
        from utils.helpers import safe_json_parse
        # Handle array response
        raw = raw.strip()
        m = re.search(r"\[.*\]", raw, re.DOTALL)
        if m:
            articles = json.loads(m.group(0))
            if isinstance(articles, list):
                return articles

        return []
    except Exception as e:
        logger.warning(f"Article discovery failed for {source_name}: {e}")
        return []


# ── Document save helper ──────────────────────────────────────────────────


def _sanitize_filename(text: str, max_len: int = 50) -> str:
    safe = re.sub(r"[^\w一-鿿\s-]", "", text)
    safe = re.sub(r"\s+", "_", safe.strip())
    if not safe:
        safe = "untitled"
    return safe[:max_len]


def _resolve_url(base_url: str, article_url: str) -> str:
    """Resolve a potentially relative URL against the base."""
    if not article_url:
        return ""
    if article_url.startswith(("http://", "https://")):
        return article_url
    return urljoin(base_url, article_url)


# ── Main fetch pipeline ───────────────────────────────────────────────────


def run_daily_fetch(
    client: openai.OpenAI,
    conn,
    embedding_model,
    collection,
    docs_state: list,
) -> dict:
    """Fetch today's articles from all enabled news sources.

    Returns dict with: batch_id, today, total_articles, sources_checked,
    sources_failed, articles (list of dicts), errors (list).
    """
    today = datetime.now().strftime("%Y-%m-%d")
    batch_id = f"daily_{today}_{uuid.uuid4().hex[:8]}"

    sources = repo.list_news_sources(conn)
    enabled_sources = [s for s in sources if s.get("enabled", 1)]

    if not enabled_sources:
        return {
            "batch_id": batch_id,
            "today": today,
            "total_articles": 0,
            "sources_checked": 0,
            "sources_failed": 0,
            "articles": [],
            "errors": ["没有已启用的信息源，请先添加信息源网址"],
        }

    all_articles = []
    errors = []
    sources_checked = 0
    sources_failed = 0

    for src in enabled_sources:
        source_id = src["id"]
        source_name = src["name"]
        source_url = src["url"]

        logger.info(f"Fetching source: {source_name} ({source_url})")

        # Step 1: Fetch source page
        fetched = fetch_url(source_url)
        if fetched["error"]:
            errors.append(f"{source_name}: {fetched['error']}")
            sources_failed += 1
            continue
        sources_checked += 1

        page_text = fetched["text"]

        # Step 2: LLM discovers today's articles
        discovered = _discover_today_articles(
            client, source_name, source_url, page_text, today
        )

        if not discovered:
            logger.info(f"  No articles from today found in {source_name}")
            errors.append(f"{source_name}: 未找到今日发布的文章")
            continue

        logger.info(f"  Found {len(discovered)} articles from today in {source_name}")

        # Step 3: Fetch & refine each article
        save_dir = Path(config.SAMPLE_DOCS_DIR)
        save_dir.mkdir(parents=True, exist_ok=True)

        for art in discovered:
            art_title = art.get("title", "未命名")
            art_url = _resolve_url(source_url, art.get("url", ""))
            art_date = art.get("publish_date", today)
            art_brief = art.get("brief", "")

            # Build source traceability label
            source_label = f"[{source_name}] {art_date}"

            # Try to fetch the full article
            article_text = ""
            if art_url:
                art_fetched = fetch_url(art_url)
                if not art_fetched["error"]:
                    article_text = art_fetched["text"]
                else:
                    logger.warning(f"    Failed to fetch article: {art_url} — {art_fetched['error']}")
                    # Fall back to brief if available
                    article_text = f"标题：{art_title}\n来源：{source_name}\n日期：{art_date}\n摘要：{art_brief}"

            # Use the brief from discovery if article fetch failed
            if not article_text or len(article_text) < 50:
                article_text = f"标题：{art_title}\n来源：{source_name}\n日期：{art_date}\n摘要：{art_brief}"

            # LLM refine into KB document
            try:
                kb_content = generate_kb_document(
                    client, art_url or source_url, art_title, article_text
                )
            except Exception as e:
                logger.warning(f"    LLM refine failed for '{art_title}': {e}")
                kb_content = f"{art_title}\n\n来源：{source_label}\n原文链接：{art_url}\n\n{article_text[:2000]}"

            if not kb_content or len(kb_content) < 20:
                kb_content = f"{art_title}\n\n来源：{source_label}\n\n{article_text[:2000]}"

            # Prepend source label as first line for traceability
            kb_content = f"{source_label} | {art_title}\n\n" + kb_content

            # Save to file
            safe_name = _sanitize_filename(f"{source_name}_{art_title}")
            filename = f"{safe_name}_{uuid.uuid4().hex[:6]}.txt"
            dest = save_dir / filename

            # Ensure unique
            while dest.exists():
                filename = f"{safe_name}_{uuid.uuid4().hex[:6]}.txt"
                dest = save_dir / filename

            dest.write_text(kb_content, encoding="utf-8")

            # ── Index into knowledge base ──
            doc_id = repo.insert_document(conn, filename, source_label + " | " + art_title, str(dest))

            from modules.chunker import chunk_document
            from modules.embedder import encode_text

            lines = kb_content.split("\n", 1)
            body = lines[1].strip() if len(lines) > 1 else kb_content
            chunks = chunk_document(
                doc_id, body,
                max_chars=config.CHUNK_MAX_CHARS,
                overlap_chars=config.CHUNK_OVERLAP_CHARS,
                min_chars=config.CHUNK_MIN_CHARS,
            )
            for c in chunks:
                c.chunk_id = repo.insert_chunk(conn, doc_id, c.chunk_index, c.content)
            repo.update_document_total_chunks(conn, doc_id, len(chunks))

            if chunks and collection is not None:
                chunk_ids = [f"chunk_{c.chunk_id}" for c in chunks]
                chunk_texts = [c.content for c in chunks]
                chunk_metadatas = [
                    {"document_id": str(doc_id), "chunk_index": c.chunk_index, "chunk_id": str(c.chunk_id)}
                    for c in chunks
                ]
                chunk_embeddings = [encode_text(embedding_model, ct) for ct in chunk_texts]
                collection.add(
                    ids=chunk_ids, embeddings=chunk_embeddings,
                    documents=chunk_texts, metadatas=chunk_metadatas,
                )

            # Add to in-memory state
            doc_obj = type("Doc", (), {})()
            doc_obj._db_id = doc_id
            doc_obj._chunks = chunks
            doc_obj.filename = filename
            doc_obj.title = source_label + " | " + art_title
            doc_obj.content = body
            docs_state.append(doc_obj)

            # Record in edu_articles
            article_id = repo.insert_article(
                conn,
                source_id=source_id,
                document_id=doc_id,
                title=art_title,
                original_url=art_url,
                publish_date=art_date,
                summary=art_brief,
                source_name=source_name,
                fetch_batch_id=batch_id,
            )

            all_articles.append({
                "article_id": article_id,
                "doc_id": doc_id,
                "title": art_title,
                "source_name": source_name,
                "publish_date": art_date,
                "original_url": art_url,
                "filename": filename,
                "chunks": len(chunks),
            })

            # Small delay between articles
            time.sleep(0.3)

        # Update source last_fetched timestamp
        repo.update_source_fetched_at(conn, source_id)

    return {
        "batch_id": batch_id,
        "today": today,
        "total_articles": len(all_articles),
        "sources_checked": sources_checked,
        "sources_failed": sources_failed,
        "articles": all_articles,
        "errors": errors,
    }
