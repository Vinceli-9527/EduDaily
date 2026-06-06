#!/usr/bin/env python3
"""EduDaily FastAPI server — wraps the RAG pipeline as a REST API.

Startup:
    python server.py

Endpoints:
    GET  /                           → Frontend UI (Vue3 SPA)
    GET  /api/health                 → Pipeline readiness check
    POST /api/query                  → Run retrieval + LLM generation
    POST /api/extract                → Run LLM extraction pipeline
    GET  /api/knowledge              → List knowledge base documents
    POST /api/knowledge/upload       → Upload .txt files into knowledge base
    DELETE /api/knowledge/{doc_id}   → Remove a document from knowledge base
"""

import json
import logging
import os
import shutil
import sqlite3
import sys
import time
import uuid
from datetime import datetime, timedelta
from contextlib import asynccontextmanager
from pathlib import Path

import chromadb
import openai
from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer

import config
from db.repository import (
    insert_analysis_report,
    insert_document,
    insert_chunk,
    update_document_total_chunks,
)
from db import repository as repo
from db.schema import init_db
from modules.chunker import chunk_document
from modules.data_loader import load_documents
from modules.embedder import (
    build_chroma_collection,
    embed_and_store_chunks,
    encode_text,
    rebuild_chroma_collection_safely,
)
from modules.extractor import run_extraction_pipeline
from modules.generator import generate_report
from modules.retriever import retrieve_relevant_chunks
from modules.url_ingester import ingest_urls
from modules.daily_fetcher import run_daily_fetch
from batch_processor import process_batch
from db.repository import list_news_sources, insert_news_source, delete_news_source
from utils.helpers import setup_logging

sys.path.insert(0, str(config.BASE_DIR))

# ── Global state ──────────────────────────────────────────────────────

state = {
    "conn": None,
    "client": None,
    "embedding_model": None,
    "collection": None,
    "docs": None,
    "ready": False,
    "extraction_done": False,
    "api_key_valid": False,
}

# ── Helpers ───────────────────────────────────────────────────────────

KEYRING_SERVICE = "EduDaily.DeepSeek"
SETTINGS_FILENAME = "settings.json"


def _settings_path() -> Path:
    return Path(config.DATA_DIR) / "data" / SETTINGS_FILENAME


def _load_settings() -> dict:
    path = _settings_path()
    if not path.exists():
        return {"api_keys": [], "active_api_key_id": None, "sample_docs_dir": None}
    try:
        return json.loads(path.read_text(encoding="utf-8-sig"))
    except Exception:
        return {"api_keys": [], "active_api_key_id": None, "sample_docs_dir": None}


def _save_settings(settings: dict) -> None:
    path = _settings_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(settings, ensure_ascii=False, indent=2), encoding="utf-8")


def _mask_api_key(api_key: str) -> str:
    if not api_key:
        return ""
    if len(api_key) <= 10:
        return api_key[:3] + "****"
    return api_key[:3] + "****" + api_key[-4:]


def _keyring_get(key_id: str) -> str:
    try:
        import keyring

        return keyring.get_password(KEYRING_SERVICE, key_id) or ""
    except Exception:
        return ""


def _keyring_set(key_id: str, api_key: str) -> None:
    import keyring

    keyring.set_password(KEYRING_SERVICE, key_id, api_key)


def _keyring_delete(key_id: str) -> None:
    try:
        import keyring

        keyring.delete_password(KEYRING_SERVICE, key_id)
    except Exception:
        pass


def _apply_api_key(api_key: str) -> None:
    config.DEEPSEEK_API_KEY = api_key
    state["client"] = openai.OpenAI(api_key=api_key, base_url=config.DEEPSEEK_BASE_URL)
    state["api_key_valid"] = True


def _copy_dir_contents(src: Path, dst: Path) -> None:
    if not src.exists():
        return
    dst.mkdir(parents=True, exist_ok=True)
    for item in src.iterdir():
        target = dst / item.name
        if item.is_dir():
            shutil.copytree(item, target, dirs_exist_ok=True)
        else:
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(item, target)


def _connect_sqlite() -> sqlite3.Connection:
    conn = sqlite3.connect(config.SQLITE_DB_PATH, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    return conn


def _sync_docs_from_db() -> list:
    """Rebuild state['docs'] from the current DB + ChromaDB state."""
    conn = state["conn"]
    docs = load_documents(config.SAMPLE_DOCS_DIR)
    for doc in docs:
        row = conn.execute(
            "SELECT id FROM documents WHERE filename = ? ORDER BY id DESC LIMIT 1",
            (doc.filename,),
        ).fetchone()
        if row:
            doc._db_id = row["id"]
            chunks_rows = conn.execute(
                "SELECT id, chunk_index, content, char_count FROM chunks WHERE document_id = ? ORDER BY chunk_index",
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


def _cleanup_indexed_document(
    doc_id: int | None = None,
    filename: str | None = None,
    disk_path: Path | None = None,
    chunk_ids: list[str] | None = None,
) -> None:
    """Best-effort rollback for partially indexed documents."""
    conn = state.get("conn")
    collection = state.get("collection")

    if conn and doc_id and chunk_ids is None:
        try:
            rows = conn.execute(
                "SELECT id FROM chunks WHERE document_id = ?", (doc_id,)
            ).fetchall()
            chunk_ids = [f"chunk_{r['id']}" for r in rows]
        except Exception:
            chunk_ids = []

    if collection and chunk_ids:
        try:
            collection.delete(ids=chunk_ids)
        except Exception:
            pass

    if conn and doc_id:
        try:
            conn.execute("DELETE FROM documents WHERE id = ?", (doc_id,))
            conn.commit()
        except Exception:
            pass

    if disk_path is None and filename:
        disk_path = Path(config.SAMPLE_DOCS_DIR) / filename
    if disk_path and disk_path.exists():
        try:
            disk_path.unlink()
        except Exception:
            pass


def _index_text_document(
    filename: str,
    content: str,
    dest_path: Path,
    *,
    write_file: bool = True,
    cleanup_disk_on_error: bool = True,
) -> dict:
    """Persist and index one text document with rollback on partial failure."""
    doc_id = None
    chunk_ids: list[str] = []
    try:
        if write_file:
            dest_path.write_text(content, encoding="utf-8")

        lines = content.split("\n", 1)
        title = lines[0].strip() if lines else filename
        body = lines[1].strip() if len(lines) > 1 else content

        doc_id = insert_document(state["conn"], filename, title, str(dest_path))
        chunks = chunk_document(
            doc_id, body,
            max_chars=config.CHUNK_MAX_CHARS,
            overlap_chars=config.CHUNK_OVERLAP_CHARS,
            min_chars=config.CHUNK_MIN_CHARS,
        )
        for c in chunks:
            c.chunk_id = insert_chunk(state["conn"], doc_id, c.chunk_index, c.content)
        update_document_total_chunks(state["conn"], doc_id, len(chunks))

        if chunks:
            chunk_ids = [f"chunk_{c.chunk_id}" for c in chunks]
            chunk_texts = [c.content for c in chunks]
            chunk_metadatas = [
                {"document_id": str(doc_id), "chunk_index": c.chunk_index, "chunk_id": str(c.chunk_id)}
                for c in chunks
            ]
            chunk_embeddings = [encode_text(state["embedding_model"], ct) for ct in chunk_texts]
            state["collection"].add(
                ids=chunk_ids,
                embeddings=chunk_embeddings,
                documents=chunk_texts,
                metadatas=chunk_metadatas,
            )

        doc_obj = type("Doc", (), {})()
        doc_obj._db_id = doc_id
        doc_obj._chunks = chunks
        doc_obj.filename = filename
        doc_obj.title = title
        doc_obj.content = body
        state["docs"].append(doc_obj)

        return {
            "filename": filename,
            "title": title,
            "doc_id": doc_id,
            "chunks": len(chunks),
            "total_chars": sum(c.char_count for c in chunks),
        }
    except Exception:
        _cleanup_indexed_document(
            doc_id,
            filename,
            dest_path if cleanup_disk_on_error else None,
            chunk_ids,
        )
        state["docs"] = _sync_docs_from_db()
        raise


def _delete_document_by_id(doc_id: int) -> dict:
    """Delete one document from SQLite, Chroma, disk, and in-memory state."""
    if not state["conn"]:
        raise HTTPException(status_code=503, detail="数据库未就绪")

    row = state["conn"].execute(
        "SELECT id, filename FROM documents WHERE id = ?", (doc_id,)
    ).fetchone()
    if not row:
        raise HTTPException(status_code=404, detail="文档不存在")

    filename = row["filename"]
    chunk_rows = state["conn"].execute(
        "SELECT id FROM chunks WHERE document_id = ?", (doc_id,)
    ).fetchall()
    chunk_ids = [f"chunk_{cr['id']}" for cr in chunk_rows]

    if chunk_ids:
        try:
            state["collection"].delete(ids=chunk_ids)
        except Exception:
            pass

    state["conn"].execute("DELETE FROM documents WHERE id = ?", (doc_id,))
    state["conn"].commit()

    disk_path = Path(config.SAMPLE_DOCS_DIR) / filename
    if disk_path.exists():
        disk_path.unlink()

    return {
        "doc_id": doc_id,
        "filename": filename,
        "removed_chunks": len(chunk_ids),
    }


def _refresh_docs_state() -> None:
    state["docs"] = _sync_docs_from_db()


def _open_runtime_database() -> None:
    if state.get("conn"):
        try:
            state["conn"].close()
        except Exception:
            pass

    db_dir = Path(config.SQLITE_DB_PATH).parent
    db_dir.mkdir(parents=True, exist_ok=True)
    conn = _connect_sqlite()
    init_db(conn)
    state["conn"] = conn


def _load_runtime_documents() -> None:
    Path(config.SAMPLE_DOCS_DIR).mkdir(parents=True, exist_ok=True)
    existing = state["conn"].execute("SELECT DISTINCT filename FROM documents").fetchall()
    existing_filenames = {r["filename"] for r in existing}

    if existing_filenames:
        state["docs"] = _sync_docs_from_db()
        return

    docs = load_documents(config.SAMPLE_DOCS_DIR)
    for doc in docs:
        doc._db_id = insert_document(state["conn"], doc.filename, doc.title, doc.source)
        doc._chunks = chunk_document(
            doc._db_id,
            doc.content,
            max_chars=config.CHUNK_MAX_CHARS,
            overlap_chars=config.CHUNK_OVERLAP_CHARS,
            min_chars=config.CHUNK_MIN_CHARS,
        )
        for chunk in doc._chunks:
            chunk.chunk_id = insert_chunk(
                state["conn"], doc._db_id, chunk.chunk_index, chunk.content
            )
        update_document_total_chunks(state["conn"], doc._db_id, len(doc._chunks))
    state["docs"] = docs


def _open_runtime_collection() -> None:
    chunk_count = state["conn"].execute("SELECT COUNT(*) FROM chunks").fetchone()[0]
    chroma_client = chromadb.PersistentClient(path=config.CHROMA_PERSIST_DIR)
    needs_rebuild = False
    try:
        state["collection"] = chroma_client.get_collection(config.CHROMA_COLLECTION_NAME)
        needs_rebuild = state["collection"].count() != chunk_count
    except Exception:
        needs_rebuild = True

    if needs_rebuild:
        all_chunks = []
        for doc in state["docs"] or []:
            all_chunks.extend(doc._chunks)
        if all_chunks:
            state["collection"] = rebuild_chroma_collection_safely(
                state["embedding_model"],
                config.CHROMA_PERSIST_DIR,
                config.CHROMA_COLLECTION_NAME,
                all_chunks,
            )
        else:
            state["collection"] = build_chroma_collection(
                config.CHROMA_PERSIST_DIR,
                config.CHROMA_COLLECTION_NAME,
            )


def _refresh_extraction_status() -> None:
    if not state.get("conn"):
        state["extraction_done"] = False
        return
    entity_count = state["conn"].execute(
        "SELECT COUNT(*) FROM extracted_entities WHERE confidence_score > 0"
    ).fetchone()[0]
    state["extraction_done"] = entity_count > 0


def _reload_runtime_storage() -> None:
    state["ready"] = False
    _open_runtime_database()
    _load_runtime_documents()
    _refresh_extraction_status()
    _open_runtime_collection()
    state["ready"] = True


def _set_dotenv_value(key: str, value: str) -> None:
    dotenv_path = Path(config.BASE_DIR) / ".env"
    lines = []
    updated = False
    if dotenv_path.exists():
        lines = dotenv_path.read_text(encoding="utf-8").splitlines()

    output = []
    for line in lines:
        if line.strip().startswith(f"{key}=") or line.strip().startswith(f"{key} ="):
            output.append(f'{key} = "{value}"')
            updated = True
        else:
            output.append(line)
    if not updated:
        output.append(f'{key} = "{value}"')

    dotenv_path.write_text("\n".join(output).strip() + "\n", encoding="utf-8")


# ── Models ────────────────────────────────────────────────────────────


class QueryRequest(BaseModel):
    query: str
    top_k: int = config.TOP_K_RETRIEVAL


class QueryResponse(BaseModel):
    query: str
    retrieved_chunks: list[dict]
    report: str
    generation_time_ms: int
    prompt_system: str = ""
    prompt_user: str = ""
    pii_redacted: int = 0
    domain: str = "general"
    persona_role: str = ""


class HealthResponse(BaseModel):
    ready: bool
    api_key_configured: bool
    extraction_done: bool
    document_count: int
    chunk_count: int
    vector_count: int


class KnowledgeItem(BaseModel):
    doc_id: int
    filename: str
    title: str
    chunk_count: int
    char_count: int
    created_at: str


class IngestURLRequest(BaseModel):
    urls: list[str]


class IngestURLResponse(BaseModel):
    saved: list[dict]
    failed: list[dict]
    save_dir: str


class NewsSourceRequest(BaseModel):
    name: str
    url: str
    source_type: str = "web"
    category: str = "education"


class NewsSourceResponse(BaseModel):
    id: int
    name: str
    url: str
    source_type: str
    category: str
    enabled: int
    last_fetched_at: str | None = None
    created_at: str | None = None


class DailyFetchResponse(BaseModel):
    batch_id: str
    today: str
    total_articles: int
    sources_checked: int
    sources_failed: int
    articles: list[dict]
    errors: list[str]


class BulkDeleteRequest(BaseModel):
    doc_ids: list[int]


class ApiKeyRequest(BaseModel):
    api_key: str


class ApiKeyCreateRequest(BaseModel):
    name: str
    api_key: str
    activate: bool = True


class PathSettingsRequest(BaseModel):
    data_dir: str | None = None
    sample_docs_dir: str | None = None
    migrate: bool = False


# ── Lifecycle ─────────────────────────────────────────────────────────


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initialize the full RAG pipeline on server startup."""
    print()
    print("=" * 60)
    print("  Starting EduDaily API Server ...")
    print("=" * 60)

    setup_logging(config.LOG_FILE)

    # ── Init SQLite ──
    db_dir = Path(config.SQLITE_DB_PATH).parent
    db_dir.mkdir(parents=True, exist_ok=True)
    conn = _connect_sqlite()
    init_db(conn)
    state["conn"] = conn
    print(f"  [OK] SQLite: {config.SQLITE_DB_PATH}")

    # ── Init DeepSeek client ──
    api_key = config.DEEPSEEK_API_KEY
    settings = _load_settings()
    active_key_id = settings.get("active_api_key_id")
    if active_key_id:
        stored_api_key = _keyring_get(active_key_id)
        if stored_api_key:
            api_key = stored_api_key
            config.DEEPSEEK_API_KEY = stored_api_key
    if settings.get("sample_docs_dir"):
        config.configure_sample_docs_dir(settings["sample_docs_dir"])
    state["api_key_valid"] = bool(api_key) and api_key not in ("", "sk-your-key-here")
    if not state["api_key_valid"]:
        print("  [!] WARNING: DEEPSEEK_API_KEY not configured!")
        print("  [!] Copy .env.example to .env and fill in your key.")
        state["client"] = None
    else:
        state["client"] = openai.OpenAI(api_key=api_key, base_url=config.DEEPSEEK_BASE_URL)
        print(f"  [OK] DeepSeek client: {config.DEEPSEEK_BASE_URL}")

    # ── Init local embedding model ──
    print(f"  Loading embedding model: {config.EMBEDDING_MODEL_NAME} ...")
    state["embedding_model"] = SentenceTransformer(
        config.EMBEDDING_MODEL_NAME, local_files_only=True
    )
    print(f"  [OK] Embedding model loaded")

    # ── Ensure sample_docs dir exists ──
    Path(config.SAMPLE_DOCS_DIR).mkdir(parents=True, exist_ok=True)

    # ── Load & chunk documents ──
    existing = conn.execute("SELECT DISTINCT filename FROM documents").fetchall()
    existing_filenames = {r["filename"] for r in existing}

    if existing_filenames:
        print(f"  Using cached documents ({len(existing_filenames)} files)")
        state["docs"] = _sync_docs_from_db()
    else:
        print("  Indexing documents for the first time ...")
        docs = load_documents(config.SAMPLE_DOCS_DIR)
        for doc in docs:
            doc._db_id = insert_document(conn, doc.filename, doc.title, doc.source)
            doc._chunks = chunk_document(
                doc._db_id, doc.content,
                max_chars=config.CHUNK_MAX_CHARS,
                overlap_chars=config.CHUNK_OVERLAP_CHARS,
                min_chars=config.CHUNK_MIN_CHARS,
            )
            for chunk in doc._chunks:
                chunk.chunk_id = insert_chunk(
                    conn, doc._db_id, chunk.chunk_index, chunk.content
                )
            update_document_total_chunks(conn, doc._db_id, len(doc._chunks))
        state["docs"] = docs

    chunk_count = conn.execute("SELECT COUNT(*) FROM chunks").fetchone()[0]
    print(f"  [OK] {len(state['docs'])} documents, {chunk_count} chunks")

    # ── Check extraction status ──
    entity_count = conn.execute(
        "SELECT COUNT(*) FROM extracted_entities WHERE confidence_score > 0"
    ).fetchone()[0]
    state["extraction_done"] = entity_count > 0
    if state["extraction_done"]:
        print(f"  [OK] Extraction cached: {entity_count} entities")

    # ── Init / rebuild ChromaDB ──
    chroma_client = chromadb.PersistentClient(path=config.CHROMA_PERSIST_DIR)
    needs_rebuild = False
    try:
        state["collection"] = chroma_client.get_collection(config.CHROMA_COLLECTION_NAME)
        vec_count = state["collection"].count()
        if vec_count != chunk_count:
            print(f"  ChromaDB mismatch ({vec_count} vs {chunk_count}), rebuilding ...")
            needs_rebuild = True
        else:
            print(f"  [OK] ChromaDB: {vec_count} vectors cached")
    except Exception:
        needs_rebuild = True

    if needs_rebuild:
        print("  Building ChromaDB collection ...")
        all_chunks = []
        for doc in state["docs"]:
            all_chunks.extend(doc._chunks)
        if all_chunks:
            state["collection"] = rebuild_chroma_collection_safely(
                state["embedding_model"],
                config.CHROMA_PERSIST_DIR,
                config.CHROMA_COLLECTION_NAME,
                all_chunks,
            )
            print(f"  [OK] ChromaDB: {state['collection'].count()} vectors indexed")
        else:
            state["collection"] = build_chroma_collection(
                config.CHROMA_PERSIST_DIR, config.CHROMA_COLLECTION_NAME
            )
            print(f"  [OK] ChromaDB initialized (empty — waiting for documents)")

    state["ready"] = True
    print("=" * 60)
    print("  EduDaily server ready! Open http://localhost:8765")
    print("=" * 60)
    print()

    yield

    if state["conn"]:
        state["conn"].close()
    print("Server stopped.")


app = FastAPI(title="EduDaily API", lifespan=lifespan)

frontend_dir = Path(__file__).parent / "frontend"
frontend_dir.mkdir(exist_ok=True)
app.mount("/static", StaticFiles(directory=str(frontend_dir)), name="static")
assets_dir = frontend_dir / "assets"
assets_dir.mkdir(exist_ok=True)
app.mount("/assets", StaticFiles(directory=str(assets_dir)), name="assets")


# ── Existing endpoints ────────────────────────────────────────────────


@app.get("/")
async def root():
    return FileResponse(
        str(frontend_dir / "index.html"),
        headers={
            "Cache-Control": "no-cache, no-store, must-revalidate",
            "Pragma": "no-cache",
            "Expires": "0",
        },
    )


@app.get("/api/health", response_model=HealthResponse)
async def health():
    vec_count = state["collection"].count() if state["collection"] else 0
    chunk_count = 0
    if state["conn"]:
        chunk_count = state["conn"].execute("SELECT COUNT(*) FROM chunks").fetchone()[0]
    return HealthResponse(
        ready=state["ready"],
        api_key_configured=state["api_key_valid"],
        extraction_done=state["extraction_done"],
        document_count=len(state["docs"]) if state["docs"] else 0,
        chunk_count=chunk_count,
        vector_count=vec_count,
    )


@app.get("/api/settings")
async def get_settings():
    settings = _load_settings()
    active_id = settings.get("active_api_key_id")
    keys = []
    for item in settings.get("api_keys", []):
        key_id = item.get("id", "")
        secret = _keyring_get(key_id)
        keys.append({
            "id": key_id,
            "name": item.get("name", ""),
            "masked": _mask_api_key(secret) if secret else item.get("masked", ""),
            "active": key_id == active_id,
            "created_at": item.get("created_at", ""),
        })

    return {
        "api_keys": keys,
        "paths": {
            "data_dir": str(config.DATA_DIR),
            "sample_docs_dir": config.SAMPLE_DOCS_DIR,
            "sqlite_db_path": config.SQLITE_DB_PATH,
            "chroma_persist_dir": config.CHROMA_PERSIST_DIR,
            "output_dir": config.OUTPUT_DIR,
            "log_file": config.LOG_FILE,
        },
        "health": (await health()).model_dump(),
    }


@app.post("/api/settings/api-keys")
async def add_api_key(req: ApiKeyCreateRequest):
    api_key = req.api_key.strip()
    name = req.name.strip()
    if not name:
        raise HTTPException(status_code=400, detail="API Key 名称不能为空")
    if not api_key:
        raise HTTPException(status_code=400, detail="API Key 不能为空")
    if not api_key.startswith("sk-"):
        raise HTTPException(status_code=400, detail="API Key 格式看起来不正确，应以 sk- 开头")

    settings = _load_settings()
    key_id = uuid.uuid4().hex
    _keyring_set(key_id, api_key)
    entry = {
        "id": key_id,
        "name": name,
        "masked": _mask_api_key(api_key),
        "created_at": datetime.now().isoformat(timespec="seconds"),
    }
    settings.setdefault("api_keys", []).append(entry)
    if req.activate or not settings.get("active_api_key_id"):
        settings["active_api_key_id"] = key_id
        _apply_api_key(api_key)
    _save_settings(settings)
    return {"status": "ok", "key": {**entry, "active": settings.get("active_api_key_id") == key_id}}


@app.post("/api/settings/api-keys/{key_id}/activate")
async def activate_api_key(key_id: str):
    settings = _load_settings()
    entry = next((x for x in settings.get("api_keys", []) if x.get("id") == key_id), None)
    if not entry:
        raise HTTPException(status_code=404, detail="API Key 不存在")
    api_key = _keyring_get(key_id)
    if not api_key:
        raise HTTPException(status_code=500, detail="系统凭据管理器中未找到该 API Key")
    settings["active_api_key_id"] = key_id
    _save_settings(settings)
    _apply_api_key(api_key)
    return {"status": "ok", "active_api_key_id": key_id, "api_key_configured": True}


@app.delete("/api/settings/api-keys/{key_id}")
async def delete_api_key(key_id: str):
    settings = _load_settings()
    before = len(settings.get("api_keys", []))
    settings["api_keys"] = [x for x in settings.get("api_keys", []) if x.get("id") != key_id]
    if len(settings["api_keys"]) == before:
        raise HTTPException(status_code=404, detail="API Key 不存在")
    _keyring_delete(key_id)
    if settings.get("active_api_key_id") == key_id:
        settings["active_api_key_id"] = None
        state["client"] = None
        state["api_key_valid"] = False
    _save_settings(settings)
    return {"status": "ok", "deleted_id": key_id}


@app.post("/api/settings/paths")
async def update_paths(req: PathSettingsRequest):
    settings = _load_settings()
    old_data_dir = Path(config.DATA_DIR)
    old_sample_dir = Path(config.SAMPLE_DOCS_DIR)
    new_data_dir = Path(req.data_dir).expanduser().resolve() if req.data_dir else old_data_dir
    new_sample_dir = (
        Path(req.sample_docs_dir).expanduser().resolve()
        if req.sample_docs_dir
        else (new_data_dir / "data" / "sample_docs" if req.data_dir else old_sample_dir)
    )

    if req.migrate:
        if new_data_dir != old_data_dir:
            _copy_dir_contents(old_data_dir, new_data_dir)
        if new_sample_dir != old_sample_dir:
            _copy_dir_contents(old_sample_dir, new_sample_dir)

    new_data_dir.mkdir(parents=True, exist_ok=True)
    new_sample_dir.mkdir(parents=True, exist_ok=True)
    config.configure_data_dir(new_data_dir)
    config.configure_sample_docs_dir(new_sample_dir)

    settings["sample_docs_dir"] = str(new_sample_dir)
    _save_settings(settings)
    _reload_runtime_storage()
    return {
        "status": "ok",
        "paths": {
            "data_dir": str(config.DATA_DIR),
            "sample_docs_dir": config.SAMPLE_DOCS_DIR,
            "sqlite_db_path": config.SQLITE_DB_PATH,
            "chroma_persist_dir": config.CHROMA_PERSIST_DIR,
        },
        "health": (await health()).model_dump(),
    }


@app.get("/api/settings/rag-diagnostics")
async def rag_diagnostics():
    checks = []

    def add(name: str, ok: bool, detail: str = ""):
        checks.append({"name": name, "ok": ok, "detail": detail})

    add("API Key", bool(state["api_key_valid"] and state["client"]), "已配置" if state["api_key_valid"] else "未配置")
    model_path = Path(config.EMBEDDING_MODEL_NAME)
    add("本地嵌入模型", model_path.exists() if model_path.is_absolute() else True, config.EMBEDDING_MODEL_NAME)

    try:
        Path(config.SQLITE_DB_PATH).parent.mkdir(parents=True, exist_ok=True)
        if state.get("conn"):
            state["conn"].execute("CREATE TEMP TABLE IF NOT EXISTS rag_diag_tmp (id INTEGER)")
            state["conn"].execute("DROP TABLE IF EXISTS rag_diag_tmp")
        add("SQLite 读写", True, config.SQLITE_DB_PATH)
    except Exception as e:
        add("SQLite 读写", False, str(e))

    try:
        count = state["collection"].count() if state.get("collection") else 0
        add("Chroma 向量库", state.get("collection") is not None, f"{count} vectors")
    except Exception as e:
        add("Chroma 向量库", False, str(e))

    try:
        h = (await health()).model_dump()
        add("完整健康检查", bool(h["ready"]), json.dumps(h, ensure_ascii=False))
    except Exception as e:
        add("完整健康检查", False, str(e))

    report_lines = [
        "EduDaily RAG Diagnostics",
        f"Time: {datetime.now().isoformat(timespec='seconds')}",
        f"Data dir: {config.DATA_DIR}",
        f"Knowledge dir: {config.SAMPLE_DOCS_DIR}",
        "",
    ]
    for check in checks:
        report_lines.append(f"[{'OK' if check['ok'] else 'FAIL'}] {check['name']}: {check['detail']}")
    return {"checks": checks, "report": "\n".join(report_lines)}


@app.post("/api/settings/rag/rebuild")
async def rebuild_rag_channel():
    if not state["ready"]:
        raise HTTPException(status_code=503, detail="管道尚未就绪")
    all_chunks = []
    for doc in state["docs"] or []:
        all_chunks.extend(doc._chunks)
    state["collection"] = rebuild_chroma_collection_safely(
        state["embedding_model"],
        config.CHROMA_PERSIST_DIR,
        config.CHROMA_COLLECTION_NAME,
        all_chunks,
    )
    return {"status": "ok", "vector_count": state["collection"].count()}


@app.post("/api/settings/rag/reload")
async def reload_rag_documents():
    _reload_runtime_storage()
    return {"status": "ok", "health": (await health()).model_dump()}


@app.post("/api/config/api-key")
async def set_api_key(req: ApiKeyRequest):
    """Backward-compatible API Key save endpoint."""
    return await add_api_key(ApiKeyCreateRequest(
        name="默认 Key",
        api_key=req.api_key,
        activate=True,
    ))

@app.post("/api/extract")
async def extract():
    if not state["ready"]:
        raise HTTPException(status_code=503, detail="管道尚未就绪，请稍后再试")
    if not state["api_key_valid"] or state["client"] is None:
        raise HTTPException(status_code=401, detail="请先配置 DEEPSEEK_API_KEY")

    all_chunks = []
    for doc in state["docs"]:
        all_chunks.extend(doc._chunks)

    try:
        entities = run_extraction_pipeline(
            client=state["client"], chunks=all_chunks,
            conn=state["conn"], repo=repo,
            model=config.DEEPSEEK_CHAT_MODEL,
            temperature=config.EXTRACTION_TEMPERATURE,
        )
    except openai.AuthenticationError:
        raise HTTPException(status_code=401, detail="DeepSeek API 认证失败")
    except openai.RateLimitError:
        raise HTTPException(status_code=429, detail="API 请求频率超限")
    except openai.APIError as e:
        raise HTTPException(status_code=502, detail=f"DeepSeek API 错误: {e}")

    state["extraction_done"] = True
    return {
        "status": "ok", "extracted": len(entities),
        "valid": sum(1 for e in entities if e.get("confidence_score", 0) > 0),
    }


@app.post("/api/query", response_model=QueryResponse)
async def query(req: QueryRequest):
    if not state["ready"]:
        raise HTTPException(status_code=503, detail="管道尚未就绪")
    if not req.query.strip():
        raise HTTPException(status_code=400, detail="查询内容不能为空")
    if not state["api_key_valid"] or state["client"] is None:
        raise HTTPException(status_code=401,
            detail="请先配置 DEEPSEEK_API_KEY：复制 .env.example 为 .env，编辑填入密钥后重启服务")

    try:
        retrieved = retrieve_relevant_chunks(
            model=state["embedding_model"], collection=state["collection"],
            query=req.query.strip(), top_k=req.top_k,
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"检索失败: {e}")

    try:
        gen_result = generate_report(
            client=state["client"], conn=state["conn"],
            query=req.query.strip(), retrieved_chunks=retrieved,
            model=config.DEEPSEEK_CHAT_MODEL,
            temperature=config.GENERATION_TEMPERATURE,
        )
    except openai.AuthenticationError:
        raise HTTPException(status_code=401, detail="DeepSeek API 认证失败 (401)")
    except openai.RateLimitError:
        raise HTTPException(status_code=429, detail="API 请求频率超限")
    except openai.APITimeoutError:
        raise HTTPException(status_code=504, detail="API 响应超时")
    except openai.APIError as e:
        raise HTTPException(status_code=502, detail=f"DeepSeek API 错误: {e}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"报告生成失败: {e}")

    chunk_ids = [int(c["metadata"]["chunk_id"]) for c in retrieved]
    try:
        insert_analysis_report(state["conn"], query_text=req.query.strip(),
            retrieved_chunk_ids=chunk_ids, report_content=gen_result["report"],
            model=config.DEEPSEEK_CHAT_MODEL,
            generation_time_ms=gen_result["generation_time_ms"])
    except Exception:
        pass

    chunks_out = []
    for c in retrieved:
        chunks_out.append({
            "chunk_id": c["metadata"].get("chunk_id", ""),
            "document_id": c["metadata"].get("document_id", ""),
            "distance": round(c.get("distance", 0), 4),
            "text": c["document"][:300] + ("..." if len(c["document"]) > 300 else ""),
        })

    return QueryResponse(
        query=req.query.strip(), retrieved_chunks=chunks_out,
        report=gen_result["report"],
        generation_time_ms=gen_result["generation_time_ms"],
        prompt_system=gen_result["prompt_system"],
        prompt_user=gen_result["prompt_user"],
        pii_redacted=gen_result.get("pii_redacted", 0),
        domain=gen_result.get("domain", "general"),
        persona_role=gen_result.get("persona_role", ""),
    )


# ── Knowledge Base Management ─────────────────────────────────────────


@app.get("/api/knowledge")
async def list_knowledge():
    """List all documents currently in the knowledge base."""
    if not state["conn"]:
        raise HTTPException(status_code=503, detail="数据库未就绪")

    rows = state["conn"].execute("""
        SELECT
            d.id, d.filename, d.title, d.created_at,
            COUNT(c.id) AS chunk_count,
            COALESCE(SUM(c.char_count), 0) AS total_chars
        FROM documents d
        LEFT JOIN chunks c ON c.document_id = d.id
        GROUP BY d.id
        ORDER BY d.id
    """).fetchall()

    items = []
    for r in rows:
        items.append(KnowledgeItem(
            doc_id=r["id"],
            filename=r["filename"],
            title=r["title"] or "(无标题)",
            chunk_count=r["chunk_count"],
            char_count=r["total_chars"],
            created_at=str(r["created_at"] or ""),
        ))
    return {"documents": [i.model_dump() for i in items]}


@app.post("/api/knowledge/upload")
async def upload_knowledge(files: list[UploadFile] = File(...)):
    """Upload .txt files into the knowledge base — auto chunk, embed, index."""
    if not state["ready"]:
        raise HTTPException(status_code=503, detail="管道尚未就绪")

    uploaded = []
    skipped = []

    for file in files:
        if not file.filename or not file.filename.lower().endswith(".txt"):
            skipped.append({"filename": file.filename, "reason": "仅支持 .txt 文件"})
            continue

        content = (await file.read()).decode("utf-8", errors="replace").strip()
        if not content:
            skipped.append({"filename": file.filename, "reason": "文件内容为空"})
            continue

        # Check duplicate filename
        existing = state["conn"].execute(
            "SELECT id FROM documents WHERE filename = ?", (file.filename,)
        ).fetchone()
        if existing:
            skipped.append({"filename": file.filename, "reason": "文件名已存在，请重命名后上传"})
            continue

        dest_path = Path(config.SAMPLE_DOCS_DIR) / file.filename
        try:
            uploaded.append(_index_text_document(file.filename, content, dest_path))
        except Exception as e:
            skipped.append({"filename": file.filename, "reason": f"入库失败: {e}"})

    return {
        "uploaded": uploaded,
        "skipped": skipped,
        "total_documents": len(state["docs"]),
        "total_vectors": state["collection"].count(),
    }


@app.delete("/api/knowledge/{doc_id:int}")
async def delete_knowledge(doc_id: int):
    """Remove a document and all its data from the knowledge base."""
    deleted = _delete_document_by_id(doc_id)
    _refresh_docs_state()

    return {
        "status": "ok",
        "deleted_doc_id": deleted["doc_id"],
        "deleted_filename": deleted["filename"],
        "removed_chunks": deleted["removed_chunks"],
        "total_documents": len(state["docs"]),
        "total_vectors": state["collection"].count(),
    }


@app.delete("/api/knowledge-bulk")
async def bulk_delete_knowledge(req: BulkDeleteRequest):
    """Delete multiple knowledge-base documents in one request."""
    unique_ids = []
    seen = set()
    for doc_id in req.doc_ids:
        if doc_id not in seen:
            unique_ids.append(doc_id)
            seen.add(doc_id)

    if not unique_ids:
        raise HTTPException(status_code=400, detail="请至少选择一个知识库文档")

    deleted = []
    failed = []
    for doc_id in unique_ids:
        try:
            deleted.append(_delete_document_by_id(doc_id))
        except HTTPException as e:
            failed.append({"doc_id": doc_id, "reason": e.detail})
        except Exception as e:
            failed.append({"doc_id": doc_id, "reason": str(e)})

    _refresh_docs_state()
    return {
        "status": "ok",
        "deleted": deleted,
        "failed": failed,
        "deleted_count": len(deleted),
        "failed_count": len(failed),
        "total_documents": len(state["docs"]),
        "total_vectors": state["collection"].count(),
    }


@app.delete("/api/knowledge-clear-yesterday")
async def delete_yesterday_knowledge():
    """Delete documents created yesterday according to local server date."""
    if not state["conn"]:
        raise HTTPException(status_code=503, detail="数据库未就绪")

    yesterday = (datetime.now() - timedelta(days=1)).strftime("%Y-%m-%d")
    rows = state["conn"].execute(
        "SELECT id FROM documents WHERE date(created_at) = date(?)",
        (yesterday,),
    ).fetchall()
    doc_ids = [r["id"] for r in rows]

    deleted = []
    failed = []
    for doc_id in doc_ids:
        try:
            deleted.append(_delete_document_by_id(doc_id))
        except Exception as e:
            failed.append({"doc_id": doc_id, "reason": str(e)})

    _refresh_docs_state()
    return {
        "status": "ok",
        "target_date": yesterday,
        "deleted": deleted,
        "failed": failed,
        "deleted_count": len(deleted),
        "failed_count": len(failed),
        "total_documents": len(state["docs"]),
        "total_vectors": state["collection"].count(),
    }


# ── URL Ingestion ───────────────────────────────────────────────────────


@app.post("/api/knowledge/ingest-url", response_model=IngestURLResponse)
async def ingest_url(req: IngestURLRequest):
    """Fetch URLs, extract text, use LLM to refine into KB documents, and save to disk.

    Documents are saved to data/sample_docs/ but NOT auto-indexed.
    Call /api/knowledge/import-saved to index them into the knowledge base.
    """
    if not state["ready"]:
        raise HTTPException(status_code=503, detail="管道尚未就绪")
    if not state["api_key_valid"] or state["client"] is None:
        raise HTTPException(status_code=401, detail="请先配置 DEEPSEEK_API_KEY")
    if not req.urls:
        raise HTTPException(status_code=400, detail="请至少提供一个网址")

    clean_urls = [u.strip() for u in req.urls if u.strip()]
    if not clean_urls:
        raise HTTPException(status_code=400, detail="请至少提供一个有效网址")

    try:
        result = ingest_urls(state["client"], clean_urls, config.SAMPLE_DOCS_DIR)
    except openai.AuthenticationError:
        raise HTTPException(status_code=401, detail="DeepSeek API 认证失败")
    except openai.RateLimitError:
        raise HTTPException(status_code=429, detail="API 请求频率超限，请稍后再试")
    except openai.APIError as e:
        raise HTTPException(status_code=502, detail=f"DeepSeek API 错误: {e}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"网址处理失败: {e}")

    return IngestURLResponse(
        saved=result["saved"],
        failed=result["failed"],
        save_dir=config.SAMPLE_DOCS_DIR,
    )


@app.post("/api/knowledge/import-saved")
async def import_saved_files():
    """Scan data/sample_docs/ for .txt files not yet in DB, and index them into ChromaDB."""
    if not state["ready"]:
        raise HTTPException(status_code=503, detail="管道尚未就绪")

    existing = set()
    if state["conn"]:
        rows = state["conn"].execute("SELECT filename FROM documents").fetchall()
        existing = {r["filename"] for r in rows}

    import_dir = Path(config.SAMPLE_DOCS_DIR)
    imported = []
    skipped = []

    for txt_file in sorted(import_dir.glob("*.txt")):
        if txt_file.name in existing:
            skipped.append(txt_file.name)
            continue

        content = txt_file.read_text(encoding="utf-8").strip()
        if not content:
            continue

        try:
            imported.append(
                _index_text_document(
                    txt_file.name,
                    content,
                    txt_file,
                    write_file=False,
                    cleanup_disk_on_error=False,
                )
            )
        except Exception as e:
            skipped.append({"filename": txt_file.name, "reason": f"入库失败: {e}"})

    return {
        "imported": imported,
        "skipped": skipped,
        "total_documents": len(state["docs"]),
        "total_vectors": state["collection"].count(),
    }


# ── News Sources Management ─────────────────────────────────────────────


@app.get("/api/sources")
async def list_sources():
    """List all configured news sources."""
    if not state["conn"]:
        raise HTTPException(status_code=503, detail="数据库未就绪")
    sources = list_news_sources(state["conn"])
    return {"sources": sources, "total": len(sources)}


@app.post("/api/sources")
async def add_source(req: NewsSourceRequest):
    """Add a new news source URL."""
    if not state["conn"]:
        raise HTTPException(status_code=503, detail="数据库未就绪")
    if not req.name.strip() or not req.url.strip():
        raise HTTPException(status_code=400, detail="名称和网址不能为空")
    if not req.url.startswith(("http://", "https://")):
        raise HTTPException(status_code=400, detail="网址必须以 http:// 或 https:// 开头")

    source_id = insert_news_source(
        state["conn"], req.name.strip(), req.url.strip(),
        req.source_type, req.category,
    )
    return {"status": "ok", "id": source_id, "name": req.name.strip(), "url": req.url.strip()}


@app.delete("/api/sources/{source_id}")
async def remove_source(source_id: int):
    """Delete a news source and its associated articles."""
    if not state["conn"]:
        raise HTTPException(status_code=503, detail="数据库未就绪")
    ok = delete_news_source(state["conn"], source_id)
    if not ok:
        raise HTTPException(status_code=404, detail="信息源不存在")
    return {"status": "ok", "deleted_id": source_id}


# ── Daily Fetch ──────────────────────────────────────────────────────────


@app.post("/api/daily-fetch", response_model=DailyFetchResponse)
async def daily_fetch():
    """Fetch today's articles from all enabled news sources and index into KB."""
    if not state["ready"]:
        raise HTTPException(status_code=503, detail="管道尚未就绪")
    if not state["api_key_valid"] or state["client"] is None:
        raise HTTPException(status_code=401, detail="请先配置 DEEPSEEK_API_KEY")

    try:
        result = run_daily_fetch(
            client=state["client"],
            conn=state["conn"],
            embedding_model=state["embedding_model"],
            collection=state["collection"],
            docs_state=state["docs"],
        )
    except openai.AuthenticationError:
        raise HTTPException(status_code=401, detail="DeepSeek API 认证失败")
    except openai.RateLimitError:
        raise HTTPException(status_code=429, detail="API 请求频率超限，请稍后再试")
    except openai.APIError as e:
        raise HTTPException(status_code=502, detail=f"DeepSeek API 错误: {e}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"日报获取失败: {e}")

    return DailyFetchResponse(
        batch_id=result["batch_id"],
        today=result["today"],
        total_articles=result["total_articles"],
        sources_checked=result["sources_checked"],
        sources_failed=result["sources_failed"],
        articles=result["articles"],
        errors=result["errors"],
    )


# ── Batch Analysis ────────────────────────────────────────────────────


class BatchAnalyzeResponse(BaseModel):
    processed: int
    total: int
    report_path: str | None = None
    summaries: list[dict] = []
    platform_results: list[dict] = []


@app.post("/api/batch-analyze", response_model=BatchAnalyzeResponse)
async def batch_analyze(force_all: bool = False, platform: str | None = None):
    """Batch process all unanalyzed .txt articles and generate a daily report.

    Scans data/sample_docs/ for .txt files not yet recorded in
    data/processed.json, generates LLM summaries for each, and compiles
    them into output/daily_summary_DATE.md.

    Args:
        force_all: Reprocess even already-processed files
        platform: Generate platform-specific versions (wechat/xhs/douyin/podcast/all)
    """
    if not state["ready"]:
        raise HTTPException(status_code=503, detail="管道尚未就绪")
    if not state["api_key_valid"] or state["client"] is None:
        raise HTTPException(status_code=401, detail="请先配置 DEEPSEEK_API_KEY")

    # Resolve platform argument
    platforms = None
    if platform:
        if platform == "all":
            platforms = ["all"]
        elif "," in platform:
            platforms = [p.strip() for p in platform.split(",")]
        else:
            platforms = [platform]

    try:
        result = process_batch(
            client=state["client"],
            conn=state["conn"],
            embedding_model=state["embedding_model"],
            collection=state["collection"],
            force_all=force_all,
            platforms=platforms,
        )
    except openai.AuthenticationError:
        raise HTTPException(status_code=401, detail="DeepSeek API 认证失败")
    except openai.RateLimitError:
        raise HTTPException(status_code=429, detail="API 请求频率超限，请稍后再试")
    except openai.APIError as e:
        raise HTTPException(status_code=502, detail=f"DeepSeek API 错误: {e}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"批量分析失败: {e}")

    return BatchAnalyzeResponse(
        processed=result["processed"],
        total=result["total"],
        report_path=result["report_path"],
        summaries=[
            {
                "filename": s["filename"],
                "title": s["title"],
                "summary": s["summary"][:200] + "..."
                if len(s["summary"]) > 200
                else s["summary"],
            }
            for s in result["summaries"]
        ],
        platform_results=[
            {
                "title": pr["title"],
                "platforms": [
                    {"platform": r["platform"], "file_path": r["file_path"]}
                    for r in pr.get("results", [])
                    if r.get("file_path")
                ],
            }
            for pr in result.get("platform_results", [])
        ],
    )


# ── Main ──────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("server:app", host="0.0.0.0", port=8765, reload=False)
