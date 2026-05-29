"""Repository helpers — CRUD operations on SQLite tables."""

import json
import sqlite3


def insert_document(conn: sqlite3.Connection, filename: str, title: str = None, source: str = None) -> int:
    cur = conn.execute(
        "INSERT INTO documents (filename, title, source) VALUES (?, ?, ?)",
        (filename, title, source),
    )
    conn.commit()
    return cur.lastrowid


def update_document_total_chunks(conn: sqlite3.Connection, doc_id: int, total: int) -> None:
    conn.execute("UPDATE documents SET total_chunks = ? WHERE id = ?", (total, doc_id))
    conn.commit()


def insert_chunk(conn: sqlite3.Connection, document_id: int, chunk_index: int, content: str) -> int:
    cur = conn.execute(
        "INSERT INTO chunks (document_id, chunk_index, content, char_count) VALUES (?, ?, ?, ?)",
        (document_id, chunk_index, content, len(content)),
    )
    conn.commit()
    return cur.lastrowid


def insert_extracted_entity(
    conn: sqlite3.Connection,
    chunk_id: int,
    document_id: int,
    entity: dict,
    extraction_raw: str,
    model: str,
) -> int:
    cur = conn.execute(
        """
        INSERT INTO extracted_entities (
            chunk_id, document_id,
            policy_name, policy_level,
            education_stage, subject_area,
            institution_name, person_name,
            event_date, reform_type,
            impact_summary, region,
            keywords,
            extraction_raw, confidence_score, extraction_model
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            chunk_id,
            document_id,
            entity.get("policy_name"),
            entity.get("policy_level"),
            entity.get("education_stage"),
            entity.get("subject_area"),
            entity.get("institution_name"),
            entity.get("person_name"),
            entity.get("event_date"),
            entity.get("reform_type"),
            entity.get("impact_summary"),
            entity.get("region"),
            json.dumps(entity.get("keywords"), ensure_ascii=False) if entity.get("keywords") else None,
            extraction_raw,
            entity.get("confidence_score"),
            model,
        ),
    )
    conn.commit()
    return cur.lastrowid


def get_extracted_entities_for_chunks(
    conn: sqlite3.Connection, chunk_ids: list[int]
) -> list[dict]:
    """Fetch extracted entities for a list of chunk IDs."""
    placeholders = ",".join("?" * len(chunk_ids))
    rows = conn.execute(
        f"SELECT * FROM extracted_entities WHERE chunk_id IN ({placeholders})",
        chunk_ids,
    ).fetchall()
    return [dict(row) for row in rows]


def get_all_extracted_entities(conn: sqlite3.Connection) -> list[dict]:
    rows = conn.execute("SELECT * FROM extracted_entities").fetchall()
    return [dict(row) for row in rows]


def insert_analysis_report(
    conn: sqlite3.Connection,
    query_text: str,
    retrieved_chunk_ids: list[int],
    report_content: str,
    model: str,
    generation_time_ms: int,
) -> int:
    cur = conn.execute(
        "INSERT INTO analysis_reports (query_text, retrieved_chunk_ids, report_content, model_used, generation_time_ms) VALUES (?, ?, ?, ?, ?)",
        (
            query_text,
            json.dumps(retrieved_chunk_ids),
            report_content,
            model,
            generation_time_ms,
        ),
    )
    conn.commit()
    return cur.lastrowid


def insert_evaluation_result(
    conn: sqlite3.Connection,
    evaluation_type: str,
    metric_name: str,
    metric_value: float,
    reference_id: int = None,
    details: str = None,
) -> int:
    cur = conn.execute(
        "INSERT INTO evaluation_results (evaluation_type, reference_id, metric_name, metric_value, details) VALUES (?, ?, ?, ?, ?)",
        (evaluation_type, reference_id, metric_name, metric_value, details),
    )
    conn.commit()
    return cur.lastrowid


# ── News Sources CRUD ─────────────────────────────────────────────────


def list_news_sources(conn: sqlite3.Connection) -> list[dict]:
    rows = conn.execute(
        "SELECT * FROM news_sources ORDER BY id"
    ).fetchall()
    return [dict(r) for r in rows]


def insert_news_source(
    conn: sqlite3.Connection, name: str, url: str, source_type: str = "web", category: str = "education"
) -> int:
    cur = conn.execute(
        "INSERT INTO news_sources (name, url, source_type, category) VALUES (?, ?, ?, ?)",
        (name, url, source_type, category),
    )
    conn.commit()
    return cur.lastrowid


def delete_news_source(conn: sqlite3.Connection, source_id: int) -> bool:
    cur = conn.execute("DELETE FROM news_sources WHERE id = ?", (source_id,))
    conn.commit()
    return cur.rowcount > 0


def update_source_fetched_at(conn: sqlite3.Connection, source_id: int) -> None:
    conn.execute(
        "UPDATE news_sources SET last_fetched_at = CURRENT_TIMESTAMP WHERE id = ?",
        (source_id,),
    )
    conn.commit()


# ── Edu Articles CRUD ──────────────────────────────────────────────────


def insert_article(
    conn: sqlite3.Connection,
    source_id: int,
    title: str,
    original_url: str = None,
    publish_date: str = None,
    summary: str = None,
    source_name: str = None,
    document_id: int = None,
    fetch_batch_id: str = None,
) -> int:
    cur = conn.execute(
        """
        INSERT INTO edu_articles (source_id, document_id, title, original_url, publish_date, summary, source_name, fetch_batch_id)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (source_id, document_id, title, original_url, publish_date, summary, source_name, fetch_batch_id),
    )
    conn.commit()
    return cur.lastrowid


def list_articles_by_date(
    conn: sqlite3.Connection, date: str = None, limit: int = 50
) -> list[dict]:
    if date:
        rows = conn.execute(
            "SELECT * FROM edu_articles WHERE publish_date = ? ORDER BY id DESC LIMIT ?",
            (date, limit),
        ).fetchall()
    else:
        rows = conn.execute(
            "SELECT * FROM edu_articles ORDER BY id DESC LIMIT ?",
            (limit,),
        ).fetchall()
    return [dict(r) for r in rows]


def list_articles_by_batch(
    conn: sqlite3.Connection, batch_id: str
) -> list[dict]:
    rows = conn.execute(
        "SELECT * FROM edu_articles WHERE fetch_batch_id = ? ORDER BY id",
        (batch_id,),
    ).fetchall()
    return [dict(r) for r in rows]
