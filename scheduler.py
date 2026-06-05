#!/usr/bin/env python3
"""Scheduler — automatic daily pipeline runner for EduDaily.

Usage:
    python scheduler.py              # Start scheduler loop (blocking)
    python scheduler.py --once       # Run pipeline once and exit

Config:
    Set SCHEDULE_TIME in .env (default: 07:00)
    Set SCHEDULE_TIME env var or add to .env file

Dependencies:
    pip install schedule
"""

import argparse
import logging
import os
import sys
import time
from datetime import datetime
from pathlib import Path

import config
from utils.helpers import setup_logging

logger = logging.getLogger("scheduler")

try:
    import schedule
    HAS_SCHEDULE = True
except ImportError:
    HAS_SCHEDULE = False


def get_schedule_time() -> str:
    """Get daily run time from config. Format: HH:MM (24h)."""
    return getattr(config, "SCHEDULE_TIME", "07:00")


# ── Pipeline initialization ──────────────────────────────────────────────


def init_pipeline():
    """Initialize the full RAG pipeline (standalone, no server required).

    Returns:
        Tuple of (conn, client, embedding_model, collection, docs)
    """
    import sqlite3

    import chromadb
    import openai
    from sentence_transformers import SentenceTransformer

    from db.schema import init_db
    from modules.embedder import build_chroma_collection

    print("=" * 60)
    print("  EduDaily Scheduler — Pipeline Initialization")
    print("=" * 60)

    setup_logging(config.LOG_FILE)

    # SQLite
    db_dir = Path(config.SQLITE_DB_PATH).parent
    db_dir.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(config.SQLITE_DB_PATH)
    conn.row_factory = sqlite3.Row
    init_db(conn)
    print(f"  [OK] SQLite: {config.SQLITE_DB_PATH}")

    # DeepSeek client
    api_key = config.DEEPSEEK_API_KEY
    if not api_key or api_key == "sk-your-key-here":
        print("  [ERROR] DEEPSEEK_API_KEY 未配置，请在 .env 中设置")
        sys.exit(1)
    client = openai.OpenAI(api_key=api_key, base_url=config.DEEPSEEK_BASE_URL)
    print(f"  [OK] DeepSeek client: {config.DEEPSEEK_BASE_URL}")

    # Embedding model
    print(f"  Loading embedding model: {config.EMBEDDING_MODEL_NAME} ...")
    embedding_model = SentenceTransformer(config.EMBEDDING_MODEL_NAME)
    dim = embedding_model.get_sentence_embedding_dimension()
    print(f"  [OK] Embedding model loaded (dim={dim})")

    # ChromaDB
    chroma_client = chromadb.PersistentClient(path=config.CHROMA_PERSIST_DIR)
    try:
        collection = chroma_client.get_collection(config.CHROMA_COLLECTION_NAME)
        print(f"  [OK] ChromaDB: {collection.count()} vectors cached")
    except Exception:
        collection = build_chroma_collection(
            config.CHROMA_PERSIST_DIR, config.CHROMA_COLLECTION_NAME
        )
        print(f"  [OK] ChromaDB: new collection created")

    # Ensure dirs exist
    Path(config.SAMPLE_DOCS_DIR).mkdir(parents=True, exist_ok=True)
    Path(config.OUTPUT_DIR).mkdir(parents=True, exist_ok=True)

    # Load existing docs from DB into memory
    docs = _sync_docs_from_db(conn)

    print("=" * 60)
    print("  Pipeline ready.")
    print("=" * 60)

    return conn, client, embedding_model, collection, docs


def _sync_docs_from_db(conn) -> list:
    """Rebuild in-memory docs list from DB state (mirrors server.py logic)."""
    from modules.data_loader import load_documents

    docs = load_documents(config.SAMPLE_DOCS_DIR)
    for doc in docs:
        row = conn.execute(
            "SELECT id FROM documents WHERE filename = ? ORDER BY id DESC LIMIT 1",
            (doc.filename,),
        ).fetchone()
        if row:
            doc._db_id = row["id"]
            chunks_rows = conn.execute(
                "SELECT id, chunk_index, content, char_count "
                "FROM chunks WHERE document_id = ? ORDER BY chunk_index",
                (doc._db_id,),
            ).fetchall()
            doc._chunks = []
            for cr in chunks_rows:
                c = type("Chunk", (), {})()
                c.chunk_id = cr["id"]
                c.document_id = doc._db_id
                c.chunk_index = cr["chunk_index"]
                c.content = cr["content"] or ""
                c.char_count = cr["char_count"] or 0
                doc._chunks.append(c)
    return docs


# ── Full pipeline ────────────────────────────────────────────────────────


def run_full_pipeline(client, conn, embedding_model, collection, docs_state):
    """Execute the complete daily pipeline: fetch → batch process → report."""
    from modules.daily_fetcher import run_daily_fetch
    from batch_processor import process_batch

    now = datetime.now()
    logger.info("=" * 60)
    logger.info("Scheduled pipeline starting: %s", now.isoformat())

    # ── Step 1: Daily fetch ──
    logger.info("Step 1/2: Fetching today's articles...")
    print(f"\n{'='*60}")
    print(f"  [{now.strftime('%H:%M:%S')}] Step 1/2: 抓取新闻...")
    print(f"{'='*60}")

    try:
        fetch_result = run_daily_fetch(
            client, conn, embedding_model, collection, docs_state
        )
        logger.info(
            "Fetch done: %d articles from %d sources",
            fetch_result["total_articles"],
            fetch_result["sources_checked"],
        )
        print(f"  抓取完成: {fetch_result['total_articles']} 篇文章")
        if fetch_result["errors"]:
            for err in fetch_result["errors"]:
                print(f"    [警告] {err}")
    except Exception as e:
        logger.error("Fetch failed: %s", e)
        print(f"  抓取失败: {e}")

    # ── Step 2: Batch analyze ──
    logger.info("Step 2/2: Batch analyzing...")
    print(f"\n{'='*60}")
    print(f"  [{datetime.now().strftime('%H:%M:%S')}] Step 2/2: 批量分析...")
    print(f"{'='*60}")

    batch_result = {"processed": 0, "report_path": None}
    try:
        batch_result = process_batch(
            client, conn, embedding_model, collection
        )
        logger.info("Batch done: %d articles processed", batch_result["processed"])
    except Exception as e:
        logger.error("Batch processing failed: %s", e)
        print(f"  批量分析失败: {e}")

    logger.info(
        "Scheduled pipeline complete: %s", datetime.now().isoformat()
    )
    logger.info("=" * 60)

    if batch_result.get("report_path"):
        print(f"\n日报已生成于: {batch_result['report_path']}")


# ── Scheduler loop ───────────────────────────────────────────────────────


def start_scheduler():
    """Start the scheduling loop (blocking). Press Ctrl+C to stop."""
    if not HAS_SCHEDULE:
        print("=" * 60)
        print("  ERROR: 'schedule' library not installed!")
        print("  Run: pip install schedule")
        print("=" * 60)
        sys.exit(1)

    schedule_time = get_schedule_time()
    conn, client, embedding_model, collection, docs = init_pipeline()

    logger.info("Scheduler configured — daily at %s", schedule_time)
    print(f"\n  定时任务已配置: 每天 {schedule_time} 自动运行")
    print(f"  按 Ctrl+C 停止\n")

    schedule.every().day.at(schedule_time).do(
        run_full_pipeline, client, conn, embedding_model, collection, docs
    )

    next_run = schedule.next_run()
    if next_run:
        print(f"  下次运行时间: {next_run.strftime('%Y-%m-%d %H:%M:%S')}")

    try:
        while True:
            schedule.run_pending()
            time.sleep(60)
    except KeyboardInterrupt:
        print("\n  调度器已停止。")
        if conn:
            conn.close()


def run_once():
    """Run the full pipeline once and exit."""
    conn, client, embedding_model, collection, docs = init_pipeline()
    try:
        run_full_pipeline(client, conn, embedding_model, collection, docs)
    finally:
        if conn:
            conn.close()


# ── CLI ──────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="EduDaily Scheduler — 定时自动运行每日新闻流水线"
    )
    parser.add_argument(
        "--once", action="store_true",
        help="运行一次后退出（不启动定时循环）",
    )
    args = parser.parse_args()

    if args.once:
        run_once()
    else:
        start_scheduler()
