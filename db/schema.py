"""SQLite schema — DDL for all tables."""

CREATE_DOCUMENTS_TABLE = """
CREATE TABLE IF NOT EXISTS documents (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    filename        TEXT    NOT NULL,
    title           TEXT,
    source          TEXT,
    total_chunks    INTEGER DEFAULT 0,
    created_at      TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
"""

CREATE_CHUNKS_TABLE = """
CREATE TABLE IF NOT EXISTS chunks (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    document_id     INTEGER NOT NULL REFERENCES documents(id) ON DELETE CASCADE,
    chunk_index     INTEGER NOT NULL,
    content         TEXT    NOT NULL,
    char_count      INTEGER,
    created_at      TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    UNIQUE(document_id, chunk_index)
);
"""

CREATE_EXTRACTED_ENTITIES_TABLE = """
CREATE TABLE IF NOT EXISTS extracted_entities (
    id                INTEGER PRIMARY KEY AUTOINCREMENT,
    chunk_id          INTEGER NOT NULL REFERENCES chunks(id) ON DELETE CASCADE,
    document_id       INTEGER NOT NULL REFERENCES documents(id) ON DELETE CASCADE,
    policy_name       TEXT,
    policy_level      TEXT,
    education_stage   TEXT,
    subject_area      TEXT,
    institution_name  TEXT,
    person_name       TEXT,
    event_date        TEXT,
    reform_type       TEXT,
    impact_summary    TEXT,
    region            TEXT,
    keywords          TEXT,
    extraction_raw    TEXT,
    confidence_score  REAL,
    extraction_model  TEXT,
    created_at        TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX IF NOT EXISTS idx_entities_policy ON extracted_entities(policy_name);
CREATE INDEX IF NOT EXISTS idx_entities_doc ON extracted_entities(document_id);
CREATE INDEX IF NOT EXISTS idx_entities_event_date ON extracted_entities(event_date);
"""

CREATE_ANALYSIS_REPORTS_TABLE = """
CREATE TABLE IF NOT EXISTS analysis_reports (
    id                  INTEGER PRIMARY KEY AUTOINCREMENT,
    query_text          TEXT    NOT NULL,
    retrieved_chunk_ids TEXT,
    report_content      TEXT,
    model_used          TEXT,
    generation_time_ms  INTEGER,
    created_at          TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
"""

CREATE_EVALUATION_RESULTS_TABLE = """
CREATE TABLE IF NOT EXISTS evaluation_results (
    id                INTEGER PRIMARY KEY AUTOINCREMENT,
    evaluation_type   TEXT NOT NULL,
    reference_id      INTEGER,
    metric_name       TEXT NOT NULL,
    metric_value      REAL NOT NULL,
    details           TEXT,
    created_at        TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
"""

CREATE_NEWS_SOURCES_TABLE = """
CREATE TABLE IF NOT EXISTS news_sources (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    name            TEXT    NOT NULL,
    url             TEXT    NOT NULL,
    source_type     TEXT    DEFAULT 'web',
    category        TEXT    DEFAULT 'education',
    enabled         INTEGER DEFAULT 1,
    last_fetched_at TIMESTAMP,
    created_at      TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
"""

CREATE_EDU_ARTICLES_TABLE = """
CREATE TABLE IF NOT EXISTS edu_articles (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    source_id       INTEGER NOT NULL REFERENCES news_sources(id) ON DELETE CASCADE,
    document_id     INTEGER REFERENCES documents(id) ON DELETE SET NULL,
    title           TEXT    NOT NULL,
    original_url    TEXT,
    publish_date    DATE,
    summary         TEXT,
    source_name     TEXT,
    fetch_batch_id  TEXT,
    fetched_at      TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX IF NOT EXISTS idx_articles_source ON edu_articles(source_id);
CREATE INDEX IF NOT EXISTS idx_articles_date ON edu_articles(publish_date);
CREATE INDEX IF NOT EXISTS idx_articles_batch ON edu_articles(fetch_batch_id);
"""

ALL_TABLES = [
    CREATE_DOCUMENTS_TABLE,
    CREATE_CHUNKS_TABLE,
    CREATE_EXTRACTED_ENTITIES_TABLE,
    CREATE_ANALYSIS_REPORTS_TABLE,
    CREATE_EVALUATION_RESULTS_TABLE,
    CREATE_NEWS_SOURCES_TABLE,
    CREATE_EDU_ARTICLES_TABLE,
]


def init_db(conn) -> None:
    """Run all CREATE TABLE statements on the given connection."""
    for ddl in ALL_TABLES:
        conn.executescript(ddl)
    conn.commit()
