#!/usr/bin/env python3
"""EduDaily — RAG-powered daily education news & analysis system

Full RAG pipeline: Load → Chunk → Extract → Embed → Retrieve → Generate → Evaluate

Usage:
    python main.py              # Run full pipeline
    python main.py --query "..." # Run full pipeline + custom query
"""


import os
import re
import sys
import json
import sqlite3
import logging
import time
import argparse
import openai

from pathlib import Path
from sentence_transformers import SentenceTransformer

import config
from utils.helpers import setup_logging, timer_context
from db.schema import init_db
from db import repository as repo

# Ensure project root is on path
sys.path.insert(0, str(config.BASE_DIR))


# ── Banner ────────────────────────────────────────────────────────────

def print_banner():
    print()
    print("=" * 62)
    print("   EduDaily — Daily Education News & Analysis")
    print("   RAG Pipeline: Load → Chunk → Extract → Embed → Retrieve → Generate → Evaluate")
    print("=" * 62)
    print(f"   Chat Model:      {config.DEEPSEEK_CHAT_MODEL}")
    print(f"   Embedding Model: {config.EMBEDDING_MODEL_NAME} (local)")
    print(f"   Database:        {config.SQLITE_DB_PATH}")
    print(f"   ChromaDB:        {config.CHROMA_PERSIST_DIR}")
    print("=" * 62)
    print()


# ── Stage 0: Initialization ───────────────────────────────────────────

def stage_init():
    """Verify config, initialize DB and API client."""
    print("─" * 50)
    print("[STAGE 0] Initialization")

    # Check API key
    api_key = config.DEEPSEEK_API_KEY
    if not api_key or api_key == "sk-your-key-here":
        print("  [!] WARNING: DEEPSEEK_API_KEY not configured!")
        print("      Copy .env.example to .env and edit it with your real key.")
        print()

    # Init logging
    setup_logging(config.LOG_FILE)

    # Init SQLite
    db_dir = Path(config.SQLITE_DB_PATH).parent
    db_dir.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(config.SQLITE_DB_PATH)
    conn.row_factory = sqlite3.Row
    init_db(conn)
    print(f"  [OK] SQLite initialized: {config.SQLITE_DB_PATH}")

    # Init OpenAI client for DeepSeek
    client = openai.OpenAI(
        api_key=api_key,
        base_url=config.DEEPSEEK_BASE_URL,
    )
    print(f"  [OK] DeepSeek client ready: {config.DEEPSEEK_BASE_URL}")

    # Init local embedding model
    print(f"  Loading embedding model: {config.EMBEDDING_MODEL_NAME} ...")
    embedding_model = SentenceTransformer(config.EMBEDDING_MODEL_NAME)
    dim = embedding_model.get_sentence_embedding_dimension()
    print(f"  [OK] Embedding model loaded (dim={dim})")

    # Ensure output dir
    Path(config.OUTPUT_DIR).mkdir(parents=True, exist_ok=True)

    return conn, client, embedding_model


# ── Stage 1: Load Documents ───────────────────────────────────────────

def stage_load_documents(conn):
    """Load txt files from sample_docs directory."""
    print()
    print("[STAGE 1] Load Documents")
    from modules.data_loader import load_documents

    docs = load_documents(config.SAMPLE_DOCS_DIR)
    for doc in docs:
        doc_id = repo.insert_document(conn, doc.filename, doc.title, doc.source)
        doc._db_id = doc_id  # attach db id for later use
        print(f"  Loaded: {doc.filename} → doc_id={doc_id} | {len(doc.content)} chars")

    print(f"  [OK] {len(docs)} documents loaded")
    return docs


# ── Stage 2: Chunk Documents ──────────────────────────────────────────

def stage_chunk_documents(conn, docs):
    """Split documents into overlapping chunks."""
    print()
    print("[STAGE 2] Chunk Documents")
    from modules.chunker import chunk_document

    # Chunk now returns objects with document_id, not full document
    # We need doc._db_id from stage 1
    for doc in docs:
        doc._chunks = chunk_document(
            document_id=doc._db_id,
            text=doc.content,
            max_chars=config.CHUNK_MAX_CHARS,
            overlap_chars=config.CHUNK_OVERLAP_CHARS,
            min_chars=config.CHUNK_MIN_CHARS,
        )

    total_chunks = 0
    for doc in docs:
        for chunk in doc._chunks:
            chunk_id = repo.insert_chunk(
                conn, doc._db_id, chunk.chunk_index, chunk.content
            )
            chunk.chunk_id = chunk_id  # attach db id
            total_chunks += 1
        repo.update_document_total_chunks(conn, doc._db_id, len(doc._chunks))

    print(f"  [OK] {total_chunks} chunks created across {len(docs)} documents")
    for doc in docs:
        print(f"       {doc.filename}: {len(doc._chunks)} chunks")


# ── Stage 3: Extract Structured Fields ────────────────────────────────

def stage_extract(conn, client, docs):
    """LLM-based extraction of structured fields from chunks."""
    print()
    print("[STAGE 3] Extract Structured Fields (LLM)")

    from modules.extractor import run_extraction_pipeline

    all_chunks = []
    for doc in docs:
        all_chunks.extend(doc._chunks)

    entities = run_extraction_pipeline(
        client=client,
        chunks=all_chunks,
        conn=conn,
        repo=repo,
        model=config.DEEPSEEK_CHAT_MODEL,
        temperature=config.EXTRACTION_TEMPERATURE,
    )

    # Print sample extractions
    for i, (chunk, entity) in enumerate(zip(all_chunks, entities)):
        policy = entity.get("policy_name") or "(未识别)"
        stage = entity.get("education_stage") or "N/A"
        print(f"  chunk_{chunk.chunk_id}: {policy} | 学段={stage}")
        if i >= 5:
            break  # show first few

    valid = sum(1 for e in entities if e.get("confidence_score", 0) > 0)
    print(f"  [OK] {valid}/{len(entities)} extractions valid")
    # Store entities on docs for later use
    return entities


# ── Stage 4: Embed & Store in ChromaDB ────────────────────────────────

def stage_embed_and_store(conn, embedding_model, docs):
    """Generate embeddings and store in ChromaDB vector database."""
    print()
    print("[STAGE 4] Embed Chunks & Store in ChromaDB")

    from modules.embedder import build_chroma_collection, embed_and_store_chunks

    collection = build_chroma_collection(
        config.CHROMA_PERSIST_DIR,
        config.CHROMA_COLLECTION_NAME,
    )

    all_chunks = []
    for doc in docs:
        all_chunks.extend(doc._chunks)

    embed_and_store_chunks(
        model=embedding_model,
        collection=collection,
        chunks=all_chunks,
    )

    print(f"  [OK] {len(all_chunks)} embeddings stored in ChromaDB")
    print(f"       Persist directory: {config.CHROMA_PERSIST_DIR}")
    return collection


# ── Stage 5: Evaluation (Extraction) ──────────────────────────────────

def stage_evaluate_extraction(conn):
    """Evaluate extraction quality against ground truth."""
    print()
    print("[STAGE 5] Evaluate Extraction Quality")

    from modules.evaluator import evaluate_extraction

    entities = repo.get_all_extracted_entities(conn)
    scores = evaluate_extraction(config.GROUND_TRUTH_PATH, entities)

    print()
    print("  Extraction Evaluation Results:")
    print(f"  {'Metric':<32} | {'Score':>8}")
    print(f"  {'-'*32}-+-{'-'*8}")
    for key in sorted(scores.keys()):
        print(f"  {key:<32} | {scores[key]:>8.3f}")

    # Store in DB
    for metric_name, value in scores.items():
        repo.insert_evaluation_result(
            conn,
            evaluation_type="extraction",
            metric_name=metric_name,
            metric_value=value,
            details=None,
        )

    print(f"  [OK] {len(scores)} extraction metrics stored")
    return scores


# ── Stage 6: Query & Generate Reports ─────────────────────────────────

DEMO_QUERIES = []


def stage_query_and_generate(conn, client, embedding_model, collection):
    """Interactive / demo query loop: retrieve + generate reports."""
    print()
    print("[STAGE 6] Query & Generate Analysis Reports")

    from modules.retriever import retrieve_relevant_chunks
    from modules.generator import generate_report

    queries_to_run = list(DEMO_QUERIES)

    if not queries_to_run:
        print("  (no queries to run)")
        return

    for idx, query in enumerate(queries_to_run, 1):
        print()
        print(f"  ── Query {idx}/{len(queries_to_run)} ──")
        print(f"  Q: {query}")

        # Retrieve
        retrieved = retrieve_relevant_chunks(
            model=embedding_model,
            collection=collection,
            query=query,
            top_k=config.TOP_K_RETRIEVAL,
        )

        chunk_ids = [int(c["metadata"]["chunk_id"]) for c in retrieved]
        print(f"  Retrieved {len(retrieved)} chunks: {chunk_ids}")

        # Generate
        gen_result = generate_report(
            client=client,
            conn=conn,
            query=query,
            retrieved_chunks=retrieved,
            model=config.DEEPSEEK_CHAT_MODEL,
            temperature=config.GENERATION_TEMPERATURE,
        )
        report = gen_result["report"]
        gen_time_ms = gen_result["generation_time_ms"]

        # Save report to file
        safe_name = re.sub(r"[^\w\s-]", "", query)[:40].strip().replace(" ", "_")
        safe_name = re.sub(r"[-_]+", "_", safe_name)
        report_path = Path(config.OUTPUT_DIR) / f"report_{idx}_{safe_name}.md"
        report_path.write_text(report, encoding="utf-8")
        print(f"  Report saved: {report_path}")

        # Store in DB
        report_id = repo.insert_analysis_report(
            conn,
            query_text=query,
            retrieved_chunk_ids=chunk_ids,
            report_content=report,
            model=config.DEEPSEEK_CHAT_MODEL,
            generation_time_ms=gen_time_ms,
        )

        # Print report excerpt
        print()
        print(report[:800])
        if len(report) > 800:
            print(f"  ... ({len(report)} chars total)")

    print()
    print(f"  [OK] {len(queries_to_run)} reports generated")


# ── Stage 7: Evaluate Generation ──────────────────────────────────────

def stage_evaluate_generation(conn):
    """Evaluate generation quality."""
    print()
    print("[STAGE 7] Evaluate Generation Quality")

    from modules.evaluator import evaluate_generation

    # Get all generated reports
    cur = conn.execute("SELECT * FROM analysis_reports ORDER BY id")
    reports = cur.fetchall()

    all_scores = []
    for report in reports:
        chunk_ids = json.loads(report["retrieved_chunk_ids"] or "[]")
        entities = repo.get_extracted_entities_for_chunks(conn, chunk_ids)
        scores = evaluate_generation(report["report_content"], entities)
        all_scores.append(scores)

        # Store individual metrics
        for metric_name, value in scores.items():
            repo.insert_evaluation_result(
                conn,
                evaluation_type="generation",
                reference_id=report["id"],
                metric_name=metric_name,
                metric_value=value,
                details=None,
            )

    # Print summary
    print()
    print("  Generation Evaluation Summary:")
    if all_scores:
        # Average across all reports
        avg_scores = {}
        for key in all_scores[0]:
            avg_scores[key] = sum(s[key] for s in all_scores) / len(all_scores)
        for key, val in avg_scores.items():
            print(f"  {key:<28} | {val:>8.3f}")
    print(f"  [OK] Generation evaluation complete ({len(all_scores)} reports)")


# ── Summary ────────────────────────────────────────────────────────────


def print_summary(conn):
    """Print final summary of all artifacts and evaluation results."""
    print()
    print("=" * 62)
    print("   PIPELINE COMPLETE — EduDaily Summary")
    print("=" * 62)

    # DB stats
    doc_count = conn.execute("SELECT COUNT(*) FROM documents").fetchone()[0]
    chunk_count = conn.execute("SELECT COUNT(*) FROM chunks").fetchone()[0]
    entity_count = conn.execute("SELECT COUNT(*) FROM extracted_entities").fetchone()[0]
    report_count = conn.execute("SELECT COUNT(*) FROM analysis_reports").fetchone()[0]
    eval_count = conn.execute("SELECT COUNT(*) FROM evaluation_results").fetchone()[0]

    print(f"  Documents loaded:      {doc_count}")
    print(f"  Chunks created:        {chunk_count}")
    print(f"  Entities extracted:    {entity_count}")
    print(f"  Reports generated:     {report_count}")
    print(f"  Evaluation metrics:    {eval_count}")
    print()
    print(f"  SQLite DB:             {config.SQLITE_DB_PATH}")
    print(f"  ChromaDB:              {config.CHROMA_PERSIST_DIR}")
    print(f"  Reports:               {config.OUTPUT_DIR}")
    print(f"  Log:                   {config.LOG_FILE}")
    print("=" * 62)


# ── Main ──────────────────────────────────────────────────────────────


def main():
    parser = argparse.ArgumentParser(
        description="EduDaily — Daily Education News & Analysis"
    )
    parser.add_argument(
        "--query", "-q", type=str, default=None,
        help="Custom query to run (appended to demo queries)",
    )
    parser.add_argument(
        "--skip-extraction", action="store_true",
        help="Skip LLM extraction stage (use cached DB if exists)",
    )
    parser.add_argument(
        "--skip-generation", action="store_true",
        help="Skip report generation stage",
    )
    args = parser.parse_args()

    print_banner()

    # Stage 0: Init
    with timer_context("Pipeline Total"):
        conn, client, embedding_model = stage_init()

        # Stage 1: Load
        with timer_context("Stage 1 — Load Documents"):
            docs = stage_load_documents(conn)

        # Stage 2: Chunk
        with timer_context("Stage 2 — Chunk Documents"):
            stage_chunk_documents(conn, docs)

        # Stage 3: Extract
        if not args.skip_extraction:
            with timer_context("Stage 3 — Extract Structured Fields"):
                stage_extract(conn, client, docs)
        else:
            print("\n[STAGE 3] SKIPPED (--skip-extraction)")

        # Stage 4: Embed & Store
        with timer_context("Stage 4 — Embed & Store in ChromaDB"):
            collection = stage_embed_and_store(conn, embedding_model, docs)

        # Stage 5: Evaluate Extraction
        with timer_context("Stage 5 — Evaluate Extraction"):
            stage_evaluate_extraction(conn)

        # Stage 6: Query & Generate
        if not args.skip_generation:
            if args.query:
                DEMO_QUERIES.append(args.query)
            with timer_context("Stage 6 — Query & Generate Reports"):
                stage_query_and_generate(conn, client, embedding_model, collection)
        else:
            print("\n[STAGE 6] SKIPPED (--skip-generation)")

        # Stage 7: Evaluate Generation
        if not args.skip_generation:
            with timer_context("Stage 7 — Evaluate Generation"):
                stage_evaluate_generation(conn)

    # Summary
    print_summary(conn)
    conn.close()


if __name__ == "__main__":
    main()
