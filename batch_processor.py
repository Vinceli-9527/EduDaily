#!/usr/bin/env python3
"""Batch processor — scan unprocessed articles and generate daily summary reports.

Usage:
    python batch_processor.py              # Scan and process all unprocessed files
    python batch_processor.py --force-all  # Reprocess all files, ignore processed log
"""

import argparse
import json
import logging
import sys
import time
from datetime import datetime
from pathlib import Path

import openai

import config
from db import repository as repo
from utils.helpers import setup_logging
from template_engine import (
    PLATFORMS,
    generate_platform_content,
    generate_all_platforms,
    list_platforms,
)

logger = logging.getLogger(__name__)

PROCESSED_LOG_PATH = Path(config.BASE_DIR) / "data" / "processed.json"

# ── Processed log helpers ────────────────────────────────────────────────


def load_processed_log() -> dict:
    if not PROCESSED_LOG_PATH.exists():
        return {}
    with open(PROCESSED_LOG_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


def save_processed_log(data: dict) -> None:
    PROCESSED_LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(PROCESSED_LOG_PATH, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


# ── Summary generation ───────────────────────────────────────────────────


SUMMARY_SYSTEM = (
    "你是一个教育新闻摘要助手。请对提供的文章进行简洁摘要。\n"
    "\n"
    "格式要求：\n"
    "1. 一句话概括文章核心主题\n"
    "2. 3-5 个关键要点（用 - 列表）\n"
    "3. 一句话总结影响或意义\n"
    "\n"
    "使用中文，总字数控制在 300 字以内。直接输出摘要，不要额外标题。"
)


def generate_summary(client: openai.OpenAI, text: str, title: str = "") -> str:
    """Generate a concise summary for a single article.

    Args:
        client: OpenAI-compatible client (DeepSeek)
        text: Full article body text
        title: Article title for context

    Returns:
        Summary text in Chinese
    """
    if not text or len(text) < 20:
        return "（文章内容过短，无法生成摘要）"

    user_prompt = f"文章标题：{title}\n\n文章内容：\n{text[:5000]}"

    resp = client.chat.completions.create(
        model=config.DEEPSEEK_CHAT_MODEL,
        messages=[
            {"role": "system", "content": SUMMARY_SYSTEM},
            {"role": "user", "content": user_prompt},
        ],
        temperature=0.3,
        max_tokens=800,
        timeout=config.API_TIMEOUT_SECONDS,
        **config.deepseek_chat_options(),
    )
    return resp.choices[0].message.content.strip() or ""


# ── File discovery ───────────────────────────────────────────────────────


def find_unprocessed(data_dir: str | None = None, force_all: bool = False) -> list[Path]:
    """Find .txt files that haven't been summarized yet.

    判断依据：processed.json 中没有记录的 .txt 文件（.gitkeep 会被跳过）

    Args:
        data_dir: Directory to scan (default: config.SAMPLE_DOCS_DIR)
        force_all: If True, return all .txt files regardless of processed status

    Returns:
        List of Path objects for unprocessed files
    """
    if data_dir is None:
        data_dir = config.SAMPLE_DOCS_DIR

    if force_all:
        data_path = Path(data_dir)
        return sorted(
            p for p in data_path.glob("*.txt") if p.name != ".gitkeep"
        )

    processed = load_processed_log()
    data_path = Path(data_dir)
    unprocessed = []

    for txt_file in sorted(data_path.glob("*.txt")):
        if txt_file.name == ".gitkeep":
            continue
        if txt_file.name not in processed:
            unprocessed.append(txt_file)

    return unprocessed


# ── Batch processing ─────────────────────────────────────────────────────


def process_batch(
    client: openai.OpenAI,
    conn=None,
    embedding_model=None,
    collection=None,
    data_dir: str | None = None,
    output_dir: str | None = None,
    force_all: bool = False,
    platforms: list[str] | None = None,
) -> dict:
    """Batch process all unprocessed .txt articles and generate daily report.

    Args:
        client: OpenAI-compatible client
        conn: SQLite connection (optional, for future DB recording)
        embedding_model: SentenceTransformer (reserved for interface compatibility)
        collection: ChromaDB collection (reserved for interface compatibility)
        data_dir: Directory to scan for .txt files
        output_dir: Directory to save daily report
        force_all: Reprocess even already-processed files
        platforms: List of platform keys to generate (None/[] = default report only)

    Returns:
        dict with processed / summaries / total / report_path / platform_results
    """
    if data_dir is None:
        data_dir = config.SAMPLE_DOCS_DIR
    if output_dir is None:
        output_dir = config.OUTPUT_DIR

    unprocessed = find_unprocessed(data_dir, force_all=force_all)

    if not unprocessed:
        logger.info("No unprocessed files found.")
        print("没有未处理的文件。")
        return {"processed": 0, "summaries": [], "total": 0, "report_path": None}

    total = len(unprocessed)
    logger.info(f"Found {total} unprocessed files.")
    print(f"\n找到 {total} 个未处理的文件，开始批量分析...\n")

    processed_log = {} if force_all else load_processed_log()
    summaries = []
    today = datetime.now().strftime("%Y-%m-%d")

    for i, txt_file in enumerate(unprocessed, 1):
        filename = txt_file.name

        try:
            content = txt_file.read_text(encoding="utf-8").strip()
            if not content:
                processed_log[filename] = {
                    "status": "skipped", "reason": "empty", "date": today
                }
                save_processed_log(processed_log)
                print(f"  [{i}/{total}] {filename} — 跳过（空文件）")
                continue

            lines = content.split("\n", 1)
            article_title = lines[0].strip() if lines else filename
            article_body = lines[1].strip() if len(lines) > 1 else content

            print(
                f"  [{i}/{total}] {filename} — 正在生成摘要... ",
                end="", flush=True,
            )
            t0 = time.perf_counter()

            summary = generate_summary(client, article_body, article_title)

            elapsed = int((time.perf_counter() - t0) * 1000)
            print(f"✓ ({elapsed}ms)")

            summaries.append({
                "filename": filename,
                "title": article_title,
                "summary": summary,
                "elapsed_ms": elapsed,
            })

            processed_log[filename] = {
                "status": "processed",
                "date": today,
                "title": article_title,
                "elapsed_ms": elapsed,
            }
            save_processed_log(processed_log)

        except openai.AuthenticationError:
            print("✗ API 认证失败，请检查 DEEPSEEK_API_KEY")
            logger.error("Authentication failed — aborting batch")
            processed_log[filename] = {
                "status": "error", "error": "auth_failed", "date": today
            }
            save_processed_log(processed_log)
            break
        except openai.RateLimitError:
            print("✗ API 频率限制，等待 5 秒后重试...")
            logger.warning("Rate limited, waiting 5s")
            time.sleep(5)
            try:
                summary = generate_summary(client, article_body, article_title)
                elapsed = int((time.perf_counter() - t0) * 1000)
                print(f"  ✓ 重试成功 ({elapsed}ms)")
                summaries.append({
                    "filename": filename,
                    "title": article_title,
                    "summary": summary,
                    "elapsed_ms": elapsed,
                })
                processed_log[filename] = {
                    "status": "processed", "date": today,
                    "title": article_title, "elapsed_ms": elapsed,
                }
                save_processed_log(processed_log)
            except Exception as e2:
                print(f"✗ 重试失败: {e2}")
                processed_log[filename] = {
                    "status": "error", "error": str(e2), "date": today
                }
                save_processed_log(processed_log)
        except Exception as e:
            logger.error(f"Failed to process {filename}: {e}")
            processed_log[filename] = {
                "status": "error", "error": str(e), "date": today
            }
            save_processed_log(processed_log)
            print(f"✗ 失败: {e}")

    save_processed_log(processed_log)

    # ── Generate default Markdown daily report ──
    report_path = None
    if summaries:
        report_path = generate_daily_report(today, summaries, output_dir)
        print(f"\n日报已生成于: {report_path}")

    # ── Generate platform-specific content ──
    platform_results = []
    if platforms and summaries:
        # Resolve "all" → all available platform keys
        if "all" in platforms:
            platforms = list(PLATFORMS.keys())

        valid_platforms = [p for p in platforms if p in PLATFORMS]
        if valid_platforms:
            print(f"\n生成平台版本: {', '.join(valid_platforms)}")
            for s in summaries:
                print(f"  为「{s['title'][:40]}」生成平台内容...")
                try:
                    results = generate_all_platforms(
                        client=client,
                        summary=s["summary"],
                        title=s["title"],
                        output_dir=output_dir,
                        date_str=today,
                        platforms=valid_platforms,
                    )
                    for r in results:
                        if r["file_path"]:
                            print(f"    ✓ {r['platform_name']}: {r['file_path']}")
                    platform_results.append({
                        "title": s["title"],
                        "results": results,
                    })
                except Exception as e:
                    logger.error("Platform generation failed: %s", e)
                    print(f"    ✗ 平台生成失败: {e}")

    result = {
        "processed": len(summaries),
        "summaries": summaries,
        "total": total,
        "report_path": report_path,
        "platform_results": platform_results,
    }
    skipped = total - len(summaries)
    status_line = (
        f"\n批量分析完成: {len(summaries)}/{total} 篇成功"
    )
    if skipped > 0:
        status_line += f", {skipped} 篇失败/跳过"
    print(status_line)
    return result


# ── Daily report generation ──────────────────────────────────────────────


def generate_daily_report(
    date_str: str, summaries: list[dict], output_dir: str
) -> str:
    """Compile article summaries into a daily markdown report.

    Args:
        date_str: Date string YYYY-MM-DD
        summaries: List of dicts with title, summary, filename, elapsed_ms
        output_dir: Output directory path

    Returns:
        Absolute path to the generated report file
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    report_file = output_path / f"daily_summary_{date_str}.md"

    lines = [
        f"# EduDaily 教育新闻日报 — {date_str}",
        "",
        f"> 自动生成于 {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
        f" | 共 {len(summaries)} 篇",
        "",
        "---",
        "",
        "## 目录",
        "",
    ]
    for i, s in enumerate(summaries, 1):
        lines.append(f"{i}. [{s['title']}](#article-{i})")

    lines.extend(["", "---", ""])

    for i, s in enumerate(summaries, 1):
        anchor = f'<a id="article-{i}"></a>'
        lines.append(f"## {anchor}{i}. {s['title']}")
        lines.append("")
        lines.append(
            f"*来源文件: `{s['filename']}` | 生成耗时: {s['elapsed_ms']}ms*"
        )
        lines.append("")
        lines.append(s["summary"])
        lines.extend(["", "---", ""])

    report_content = "\n".join(lines)
    report_file.write_text(report_content, encoding="utf-8")

    logger.info(
        "Daily report saved: %s (%d chars)", report_file, len(report_content)
    )
    return str(report_file)


# ── CLI ──────────────────────────────────────────────────────────────────


def main():
    parser = argparse.ArgumentParser(
        description="EduDaily Batch Processor — 批量分析未处理的新闻文章"
    )
    parser.add_argument(
        "--force-all", action="store_true",
        help="重新处理所有文件，忽略已处理记录",
    )
    parser.add_argument(
        "--data-dir", type=str, default=None,
        help="文章目录路径（默认: data/sample_docs）",
    )
    parser.add_argument(
        "--output-dir", type=str, default=None,
        help="输出目录路径（默认: output）",
    )
    parser.add_argument(
        "--platform", "-p", type=str, default=None,
        help="生成指定平台版本 (wechat / xhs / douyin / podcast / all)",
    )
    parser.add_argument(
        "--copy", "-c", action="store_true",
        help="生成后将日报内容自动复制到系统剪贴板",
    )
    parser.add_argument(
        "--publish-draft", type=str, default=None, metavar="PLATFORM",
        help="生成后将日报作为草稿发送到指定平台 (wechat / zhihu / weibo)",
    )
    args = parser.parse_args()

    setup_logging(config.LOG_FILE)

    import sqlite3
    from db.schema import init_db

    # Init DB
    db_dir = Path(config.SQLITE_DB_PATH).parent
    db_dir.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(config.SQLITE_DB_PATH)
    conn.row_factory = sqlite3.Row
    init_db(conn)

    # Init client
    api_key = config.DEEPSEEK_API_KEY
    if not api_key or api_key == "sk-your-key-here":
        print("[ERROR] DEEPSEEK_API_KEY 未配置，请在 .env 中设置")
        sys.exit(1)

    client = openai.OpenAI(
        api_key=api_key, base_url=config.DEEPSEEK_BASE_URL
    )
    print(f"[OK] DeepSeek client: {config.DEEPSEEK_BASE_URL}")

    # Resolve platform argument
    platforms = None
    if args.platform:
        if args.platform == "all":
            platforms = ["all"]
        elif "," in args.platform:
            platforms = [p.strip() for p in args.platform.split(",")]
        else:
            platforms = [args.platform]

    try:
        result = process_batch(
            client=client,
            conn=conn,
            data_dir=args.data_dir,
            output_dir=args.output_dir,
            force_all=args.force_all,
            platforms=platforms,
        )
    finally:
        conn.close()

    # ── Post-processing: copy to clipboard ──
    if args.copy and result.get("report_path"):
        from clipboard import copy_latest_report

        # If platform was specified, copy that platform version too
        if platforms:
            for p in (p for p in platforms if p != "all"):
                copy_latest_report(
                    output_dir=args.output_dir,
                    platform=p,
                )
        else:
            copy_latest_report(output_dir=args.output_dir)
    elif args.copy and not result.get("report_path"):
        print("[!] --copy: 没有生成日报，无法复制。")

    # ── Post-processing: publish draft ──
    if args.publish_draft:
        from publisher import publish_draft

        plat = args.publish_draft
        if result.get("report_path"):
            from pathlib import Path
            content = Path(result["report_path"]).read_text(encoding="utf-8")
            pub_result = publish_draft(plat, content, title=f"EduDaily 教育日报")
            if pub_result.success:
                print(f"✓ {pub_result.message}")
                if pub_result.draft_url:
                    print(f"  草稿链接: {pub_result.draft_url}")
            else:
                print(f"✗ 发布失败: {pub_result.error}")
        else:
            print(f"[!] --publish-draft: 没有生成日报，无法发布。")


if __name__ == "__main__":
    main()
