"""URL Ingestion — fetch web content and refine into knowledge base documents.

Uses DeepSeek LLM to extract key information from web pages and format it
as properly structured knowledge base .txt files.
"""

import logging
import re
import time
from pathlib import Path
from urllib.parse import urlparse

import requests

import config

logger = logging.getLogger(__name__)

# ── HTML text extraction ────────────────────────────────────────────────

UNWANTED_TAGS = re.compile(
    r"<(script|style|noscript|iframe|svg|nav|footer|header|aside)[^>]*>.*?</\1>",
    re.DOTALL | re.IGNORECASE,
)
TAG_STRIP = re.compile(r"<[^>]+>")
ENTITY_FIX = re.compile(r"&[a-z]+;|&#\d+;")
WHITESPACE_COLLAPSE = re.compile(r"\n{3,}")
LINE_WS = re.compile(r"[ \t]+")


def _extract_text_from_html(html: str) -> str:
    """Strip HTML down to readable plain text."""
    html = UNWANTED_TAGS.sub(" ", html)
    html = TAG_STRIP.sub("\n", html)
    html = ENTITY_FIX.sub(" ", html)

    lines = []
    for line in html.splitlines():
        stripped = LINE_WS.sub(" ", line).strip()
        if stripped:
            lines.append(stripped)

    text = "\n".join(lines)
    text = WHITESPACE_COLLAPSE.sub("\n\n", text)
    return text.strip()


def _extract_title_from_html(html: str) -> str:
    """Try to extract the <title> from HTML."""
    m = re.search(r"<title[^>]*>(.*?)</title>", html, re.DOTALL | re.IGNORECASE)
    if m:
        return TAG_STRIP.sub("", m.group(1)).strip()
    return ""


# ── URL fetch ────────────────────────────────────────────────────────────

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/125.0.0.0 Safari/537.36"
    ),
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    "Accept-Language": "zh-CN,zh;q=0.9,en;q=0.8",
}

MAX_CONTENT_LENGTH = 2_000_000  # 2 MB max
REQUEST_TIMEOUT = 20


def fetch_url(url: str) -> dict:
    """Fetch a single URL and extract its text content.

    Returns dict with keys: url, title, text, content_type, error.
    """
    result = {
        "url": url,
        "title": "",
        "text": "",
        "content_type": "",
        "error": None,
    }

    try:
        resp = requests.get(
            url, headers=HEADERS, timeout=REQUEST_TIMEOUT,
            allow_redirects=True, stream=True,
        )

        content_type = resp.headers.get("Content-Type", "").lower()
        result["content_type"] = content_type

        if resp.status_code != 200:
            result["error"] = f"HTTP {resp.status_code}"
            return result

        # Read up to MAX_CONTENT_LENGTH
        chunks = []
        total = 0
        for chunk in resp.iter_content(chunk_size=65536, decode_unicode=False):
            if chunk:
                chunks.append(chunk)
                total += len(chunk)
                if total > MAX_CONTENT_LENGTH:
                    break
        raw = b"".join(chunks)

        # Detect encoding
        encoding = resp.encoding or "utf-8"
        try:
            html = raw.decode(encoding, errors="replace")
        except (LookupError, UnicodeDecodeError):
            html = raw.decode("utf-8", errors="replace")

        if "text/html" in content_type or "<html" in html[:1000].lower() or "<!doctype" in html[:1000].lower():
            result["title"] = _extract_title_from_html(html)
            result["text"] = _extract_text_from_html(html)
        elif "text/plain" in content_type:
            result["text"] = html.strip()
        elif "application/json" in content_type:
            result["text"] = html[:50000]
        else:
            # Non-text content (image, video, PDF, etc.)
            result["error"] = f"不支持的内容类型: {content_type}（仅支持网页、文本和 JSON）"
            return result

        if not result["text"] or len(result["text"]) < 20:
            result["error"] = "页面文本内容过少，可能为纯图片/视频页面"
            return result

        # Limit raw text sent to LLM
        if len(result["text"]) > 15000:
            result["text"] = result["text"][:15000] + "\n\n[... 内容过长，已截断 ...]"

    except requests.ConnectionError:
        result["error"] = "无法连接到该网址 (DNS/网络错误)"
    except requests.Timeout:
        result["error"] = f"请求超时 ({REQUEST_TIMEOUT}s)"
    except requests.TooManyRedirects:
        result["error"] = "重定向次数过多"
    except requests.RequestException as e:
        result["error"] = f"请求失败: {e}"

    return result


# ── LLM-based knowledge document generation ─────────────────────────────

REFINE_SYSTEM_PROMPT = """你是一个知识库文档处理助手。用户会提供从网页抓取的原始文本，你需要将其提炼为一份结构清晰、信息密集的知识库文档。

## 格式要求（非常重要）
1. **首行必须是文档标题** — 用一句精炼的话概括全文主题
2. 标题后空一行，然后开始正文
3. 用**空行**分隔不同段落/主题（系统会按空行分块）
4. 每个段落控制在 100-800 字之间
5. 保留原文中的关键数据、数字、日期、人名、地名
6. 删除广告、导航栏、评论区等无关噪音
7. 如果有表格数据，转换为列表或段落描述

## 内容组织
- 按主题分组，使用清晰的段落标题（如"一、..."、"1. ..."）
- 同一主题的信息放在同一段落中
- 不同主题之间用空行隔开
- 确保每个段落内的信息独立完整

## 输出
只输出知识库文档内容，不要输出任何解释、前缀或后缀。直接以标题开头。"""


def generate_kb_document(client, url: str, url_title: str, raw_text: str) -> str:
    """Use LLM to refine raw web text into a knowledge base document."""
    title_hint = f"（网页原标题：{url_title}）" if url_title else ""
    domain = urlparse(url).netloc

    user_prompt = f"""请将以下网页内容提炼为知识库文档。

来源网址: {url}
来源域名: {domain}
{title_hint}

══════════ 原始抓取内容 ══════════

{raw_text}

══════════ 结束 ══════════

请按系统提示的格式要求提炼上述内容。"""

    try:
        resp = client.chat.completions.create(
            model=config.DEEPSEEK_CHAT_MODEL,
            messages=[
                {"role": "system", "content": REFINE_SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt},
            ],
            temperature=0.3,
            max_tokens=4096,
            **config.deepseek_chat_options(),
        )
        content = resp.choices[0].message.content
        return content.strip() if content else ""
    except Exception as e:
        logger.error(f"LLM refinement failed for {url}: {e}")
        raise


# ── Main ingestion pipeline ──────────────────────────────────────────────


def sanitize_filename(text: str, max_len: int = 60) -> str:
    """Create a safe filename from document title or URL."""
    # Remove scheme
    text = re.sub(r"^https?://", "", text)
    # Keep only safe chars
    safe = re.sub(r"[^\w一-鿿\s-]", "", text)
    safe = re.sub(r"\s+", "_", safe.strip())
    if not safe:
        safe = "untitled"
    return safe[:max_len] + ".txt"


def ingest_urls(client, urls: list[str], save_dir: str) -> dict:
    """Full URL ingestion pipeline.

    1. Fetch each URL
    2. Extract text
    3. LLM-refine into KB document
    4. Save as .txt in save_dir

    Returns dict with keys: saved, failed, total_chars.
    """
    results = {"saved": [], "failed": []}
    Path(save_dir).mkdir(parents=True, exist_ok=True)

    for i, url in enumerate(urls):
        url = url.strip()
        if not url:
            continue
        if not url.startswith(("http://", "https://")):
            url = "https://" + url

        logger.info(f"[{i+1}/{len(urls)}] Fetching: {url}")

        # Step 1: Fetch
        fetched = fetch_url(url)
        if fetched["error"]:
            logger.warning(f"  Failed: {fetched['error']}")
            results["failed"].append({"url": url, "error": fetched["error"]})
            continue

        raw_len = len(fetched["text"])
        logger.info(f"  Extracted {raw_len} chars of text")

        # Step 2: LLM refinement
        try:
            kb_content = generate_kb_document(
                client, url, fetched["title"], fetched["text"]
            )
        except Exception as e:
            results["failed"].append({"url": url, "error": f"LLM 提炼失败: {e}"})
            continue

        if not kb_content or len(kb_content) < 30:
            results["failed"].append({"url": url, "error": "提炼后内容过短"})
            continue

        # Step 3: Determine filename from LLM output (first line is title)
        first_line = kb_content.split("\n")[0].strip()
        filename = sanitize_filename(first_line or fetched["title"] or url)

        # Ensure unique filename
        dest = Path(save_dir) / filename
        if dest.exists():
            base = Path(filename).stem
            filename = f"{base}_{int(time.time())}.txt"
            dest = Path(save_dir) / filename

        # Step 4: Save
        dest.write_text(kb_content, encoding="utf-8")
        logger.info(f"  Saved: {dest} ({len(kb_content)} chars)")

        results["saved"].append({
            "url": url,
            "title": first_line or fetched["title"],
            "filename": filename,
            "path": str(dest),
            "char_count": len(kb_content),
        })

        # Small delay between URLs to be polite
        if i < len(urls) - 1:
            time.sleep(0.5)

    return results
