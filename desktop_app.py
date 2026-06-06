#!/usr/bin/env python3
"""Windowed desktop entrypoint for EduDaily.

This uses pywebview to host the existing Vue HTML and calls backend logic
through a Python bridge instead of starting a local HTTP server.
"""

from __future__ import annotations

import asyncio
import json
import traceback
import os
import sys
import webbrowser
import threading
from pathlib import Path
from urllib.parse import parse_qs, urlparse

import openai
import webview
from fastapi import HTTPException
from pydantic import BaseModel

import config

APP_NAME = "EduDaily"
APP_CONFIG_DIR = Path(
    os.environ.get("EDUDAILY_DESKTOP_CONFIG_DIR")
    or Path(os.environ.get("APPDATA", Path.home())) / APP_NAME
)
APP_CONFIG_PATH = APP_CONFIG_DIR / "desktop.json"
APP_LOG_PATH = APP_CONFIG_DIR / "desktop.log"
KEYRING_SERVICE = "EduDaily.DeepSeek"
KEYRING_USERNAME = "default"

window = None


def resource_path(*parts: str) -> Path:
    base = Path(getattr(sys, "_MEIPASS", Path(__file__).resolve().parent))
    return base.joinpath(*parts)


def load_app_config() -> dict:
    if not APP_CONFIG_PATH.exists():
        return {}
    try:
        return json.loads(APP_CONFIG_PATH.read_text(encoding="utf-8-sig"))
    except Exception:
        return {}


def save_app_config(data: dict) -> None:
    APP_CONFIG_DIR.mkdir(parents=True, exist_ok=True)
    APP_CONFIG_PATH.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def write_desktop_log(message: str) -> None:
    try:
        APP_CONFIG_DIR.mkdir(parents=True, exist_ok=True)
        with APP_LOG_PATH.open("a", encoding="utf-8") as f:
            f.write(message.rstrip() + "\n")
    except Exception:
        pass


def default_data_dir() -> Path:
    return APP_CONFIG_DIR / "data"


def choose_initial_data_dir() -> Path:
    app_cfg = load_app_config()
    existing = app_cfg.get("data_dir")
    if existing:
        return Path(existing)

    selected = None
    try:
        import tkinter as tk
        from tkinter import filedialog

        root = tk.Tk()
        root.withdraw()
        selected = filedialog.askdirectory(
            title="请选择 EduDaily 数据保存目录",
            initialdir=str(default_data_dir().parent),
        )
        root.destroy()
    except Exception:
        selected = None

    data_dir = Path(selected) if selected else default_data_dir()
    app_cfg["data_dir"] = str(data_dir)
    save_app_config(app_cfg)
    return data_dir


def load_secret_api_key() -> str:
    try:
        import keyring

        return keyring.get_password(KEYRING_SERVICE, KEYRING_USERNAME) or ""
    except Exception:
        return ""


def save_secret_api_key(api_key: str) -> None:
    import keyring

    keyring.set_password(KEYRING_SERVICE, KEYRING_USERNAME, api_key)


def configure_runtime() -> None:
    write_desktop_log("Resolving data directory")
    data_dir = choose_initial_data_dir()
    data_dir.mkdir(parents=True, exist_ok=True)
    config.configure_data_dir(data_dir)
    write_desktop_log(f"Configured data dir: {config.DATA_DIR}")

    write_desktop_log("Loading API key from system credential manager")
    api_key = load_secret_api_key()
    if api_key:
        config.DEEPSEEK_API_KEY = api_key
        write_desktop_log("API key loaded from system credential manager")
    else:
        write_desktop_log("No API key found in system credential manager")


def serialize(value):
    if isinstance(value, BaseModel):
        return value.model_dump()
    if isinstance(value, dict):
        return {k: serialize(v) for k, v in value.items()}
    if isinstance(value, list):
        return [serialize(v) for v in value]
    if hasattr(value, "model_dump"):
        return value.model_dump()
    return value


def run_async(coro):
    return asyncio.run(coro)


class DesktopApi:
    def __init__(self) -> None:
        write_desktop_log("Starting EduDaily desktop runtime")
        self._api_lock = threading.RLock()
        configure_runtime()
        write_desktop_log(f"Data dir: {config.DATA_DIR}")
        import server

        self.server = server
        self._lifespan = server.lifespan(server.app)
        run_async(self._lifespan.__aenter__())
        write_desktop_log("Backend runtime ready")

    def shutdown(self) -> None:
        try:
            run_async(self._lifespan.__aexit__(None, None, None))
        except Exception:
            pass

    def _ok(self, data):
        return {"ok": True, "data": serialize(data)}

    def _error(self, exc: Exception):
        if isinstance(exc, HTTPException):
            return {"ok": False, "status": exc.status_code, "detail": exc.detail}
        return {"ok": False, "status": 500, "detail": str(exc)}

    def _query_value(self, parsed, key: str, default=None):
        values = parse_qs(parsed.query).get(key)
        return values[0] if values else default

    def _bool_query(self, parsed, key: str, default=False) -> bool:
        value = self._query_value(parsed, key)
        if value is None:
            return default
        return str(value).lower() in {"1", "true", "yes", "on"}

    def call(self, url: str, method: str = "GET", body=None):
        try:
            with self._api_lock:
                return self._ok(run_async(self._dispatch(url, method.upper(), body or {})))
        except Exception as exc:
            return self._error(exc)

    def open_external(self, url: str):
        try:
            webbrowser.open(url, new=2)
            return {"ok": True}
        except Exception as exc:
            return self._error(exc)

    async def _dispatch(self, url: str, method: str, body):
        s = self.server
        parsed = urlparse(url)
        path = parsed.path

        if method == "GET" and path == "/api/health":
            return await s.health()
        if method == "GET" and path == "/api/settings":
            return await s.get_settings()
        if method == "POST" and path == "/api/settings/api-keys":
            return await s.add_api_key(s.ApiKeyCreateRequest(**body))
        if method == "POST" and path.startswith("/api/settings/api-keys/") and path.endswith("/activate"):
            return await s.activate_api_key(path.split("/")[-2])
        if method == "DELETE" and path.startswith("/api/settings/api-keys/"):
            return await s.delete_api_key(path.rsplit("/", 1)[1])
        if method == "POST" and path == "/api/settings/paths":
            return await s.update_paths(s.PathSettingsRequest(**body))
        if method == "GET" and path == "/api/settings/rag-diagnostics":
            return await s.rag_diagnostics()
        if method == "POST" and path == "/api/settings/rag/rebuild":
            return await s.rebuild_rag_channel()
        if method == "POST" and path == "/api/settings/rag/reload":
            return await s.reload_rag_documents()
        if method == "POST" and path == "/api/config/api-key":
            return self._set_api_key(body)
        if method == "POST" and path == "/api/extract":
            return await s.extract()
        if method == "POST" and path == "/api/query":
            return await s.query(s.QueryRequest(**body))
        if method == "GET" and path == "/api/knowledge":
            return await s.list_knowledge()
        if method == "DELETE" and path.startswith("/api/knowledge/"):
            return await s.delete_knowledge(int(path.rsplit("/", 1)[1]))
        if method == "DELETE" and path == "/api/knowledge-bulk":
            return await s.bulk_delete_knowledge(s.BulkDeleteRequest(**body))
        if method == "DELETE" and path == "/api/knowledge-clear-yesterday":
            return await s.delete_yesterday_knowledge()
        if method == "POST" and path == "/api/knowledge/ingest-url":
            return await s.ingest_url(s.IngestURLRequest(**body))
        if method == "POST" and path == "/api/knowledge/import-saved":
            return await s.import_saved_files()
        if method == "GET" and path == "/api/sources":
            return await s.list_sources()
        if method == "POST" and path == "/api/sources":
            return await s.add_source(s.NewsSourceRequest(**body))
        if method == "DELETE" and path.startswith("/api/sources/"):
            return await s.remove_source(int(path.rsplit("/", 1)[1]))
        if method == "POST" and path == "/api/daily-fetch":
            return await s.daily_fetch()
        if method == "POST" and path == "/api/batch-analyze":
            return await s.batch_analyze(
                force_all=self._bool_query(parsed, "force_all"),
                platform=self._query_value(parsed, "platform"),
            )

        raise HTTPException(status_code=404, detail=f"Unsupported desktop API route: {method} {path}")

    def _set_api_key(self, body):
        return run_async(self.server.add_api_key(self.server.ApiKeyCreateRequest(
            name=(body or {}).get("name") or "默认 Key",
            api_key=(body or {}).get("api_key", ""),
            activate=True,
        )))

    def upload_knowledge_files(self):
        with self._api_lock:
            return self._upload_knowledge_files_locked()

    def _upload_knowledge_files_locked(self):
        try:
            paths = window.create_file_dialog(
                webview.OPEN_DIALOG,
                allow_multiple=True,
                file_types=("Text files (*.txt)",),
            )
            if not paths:
                return {"uploaded": [], "skipped": [], "total_documents": len(self.server.state["docs"])}

            uploaded = []
            skipped = []
            save_dir = Path(config.SAMPLE_DOCS_DIR)
            save_dir.mkdir(parents=True, exist_ok=True)
            for item in paths:
                path = Path(item)
                if path.suffix.lower() != ".txt":
                    skipped.append({"filename": path.name, "reason": "仅支持 .txt 文件"})
                    continue
                content = path.read_text(encoding="utf-8", errors="replace").strip()
                if not content:
                    skipped.append({"filename": path.name, "reason": "文件内容为空"})
                    continue
                try:
                    uploaded.append(
                        self.server._index_text_document(
                            path.name,
                            content,
                            save_dir / path.name,
                        )
                    )
                except Exception as exc:
                    skipped.append({"filename": path.name, "reason": f"入库失败: {exc}"})

            return {
                "uploaded": serialize(uploaded),
                "skipped": skipped,
                "total_documents": len(self.server.state["docs"]),
                "total_vectors": self.server.state["collection"].count(),
            }
        except Exception as exc:
            return self._error(exc)


def main() -> None:
    global window
    try:
        api = DesktopApi()
        html_path = resource_path("frontend", "index.html")
        write_desktop_log(f"Frontend path: {html_path}")
        window = webview.create_window(
            "EduDaily",
            html_path.as_uri(),
            js_api=api,
            width=1280,
            height=860,
            min_size=(1024, 700),
        )
        window.events.closed += api.shutdown
        webview.start(debug=False)
    except Exception as exc:
        write_desktop_log(traceback.format_exc())
        try:
            import tkinter as tk
            from tkinter import messagebox

            root = tk.Tk()
            root.withdraw()
            messagebox.showerror(
                "EduDaily 启动失败",
                f"{exc}\n\n日志位置：{APP_LOG_PATH}",
            )
            root.destroy()
        except Exception:
            pass
        raise


if __name__ == "__main__":
    main()
