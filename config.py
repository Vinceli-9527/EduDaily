"""Central configuration for EduDaily.

Secrets are loaded from .env file. Copy .env.example to .env and fill in your key.
"""

import os
from pathlib import Path

# ---- Load .env file (simple parser, no external deps) ----
def _load_dotenv(dotenv_path: str) -> None:
    """Parse a .env file and set KEY=VALUE in os.environ (only if not already set)."""
    if not os.path.isfile(dotenv_path):
        return
    with open(dotenv_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            if "=" not in line:
                continue
            key, _, value = line.partition("=")
            key = key.strip()
            value = value.strip().strip('"').strip("'")
            if key and key not in os.environ:
                os.environ[key] = value

# Load .env from project root (environment wins over .env)
_load_dotenv(str(Path(__file__).resolve().parent / ".env"))

# ---- Paths ----
BASE_DIR = Path(__file__).resolve().parent

# ---- DeepSeek API ----
DEEPSEEK_API_KEY = os.environ.get("DEEPSEEK_API_KEY", "")
DEEPSEEK_BASE_URL = "https://api.deepseek.com"
DEEPSEEK_CHAT_MODEL = os.environ.get("DEEPSEEK_CHAT_MODEL", "deepseek-v4-pro")
DEEPSEEK_REASONING_EFFORT = os.environ.get("DEEPSEEK_REASONING_EFFORT", "high")
DEEPSEEK_THINKING_ENABLED = os.environ.get(
    "DEEPSEEK_THINKING_ENABLED", "true"
).lower() not in {"0", "false", "no", "off"}
# Local embedding model (Chinese-optimized, runs offline, no API needed)
LOCAL_EMBEDDING_MODEL_DIR = BASE_DIR / "models" / "bge-small-zh-v1.5"
EMBEDDING_MODEL_NAME = os.environ.get(
    "EMBEDDING_MODEL_NAME",
    str(LOCAL_EMBEDDING_MODEL_DIR)
    if LOCAL_EMBEDDING_MODEL_DIR.exists()
    else "BAAI/bge-small-zh-v1.5",
)


def deepseek_chat_options() -> dict:
    """Return DeepSeek V4 Pro-specific options for chat completion calls."""
    options = {}
    if DEEPSEEK_REASONING_EFFORT:
        options["reasoning_effort"] = DEEPSEEK_REASONING_EFFORT
    if DEEPSEEK_THINKING_ENABLED:
        options["extra_body"] = {"thinking": {"type": "enabled"}}
    return options

CHROMA_COLLECTION_NAME = "edu_documents"
DATA_DIR = Path(os.environ.get("EDUDAILY_DATA_DIR", str(BASE_DIR)))
CHROMA_PERSIST_DIR = str(DATA_DIR / "chroma_store")
SQLITE_DB_PATH = str(DATA_DIR / "data" / "structured.db")
SAMPLE_DOCS_DIR = str(DATA_DIR / "data" / "sample_docs")
GROUND_TRUTH_PATH = str(DATA_DIR / "data" / "ground_truth.json")
OUTPUT_DIR = str(DATA_DIR / "output")
LOG_FILE = str(DATA_DIR / "pipeline.log")


def configure_data_dir(data_dir: str | os.PathLike) -> None:
    """Point all user-writable runtime paths at a selected data directory."""
    global DATA_DIR, CHROMA_PERSIST_DIR, SQLITE_DB_PATH, SAMPLE_DOCS_DIR
    global GROUND_TRUTH_PATH, OUTPUT_DIR, LOG_FILE

    DATA_DIR = Path(data_dir).expanduser().resolve()
    CHROMA_PERSIST_DIR = str(DATA_DIR / "chroma_store")
    SQLITE_DB_PATH = str(DATA_DIR / "data" / "structured.db")
    SAMPLE_DOCS_DIR = str(DATA_DIR / "data" / "sample_docs")
    GROUND_TRUTH_PATH = str(DATA_DIR / "data" / "ground_truth.json")
    OUTPUT_DIR = str(DATA_DIR / "output")
    LOG_FILE = str(DATA_DIR / "pipeline.log")

# ---- Chunking ----
CHUNK_MAX_CHARS = 1000
CHUNK_OVERLAP_CHARS = 200
CHUNK_MIN_CHARS = 50

# ---- Retrieval ----
TOP_K_RETRIEVAL = 5

# ---- Scheduling ----
SCHEDULE_TIME = os.environ.get("SCHEDULE_TIME", "07:00")

# ---- API Settings ----
EXTRACTION_TEMPERATURE = 0.1
GENERATION_TEMPERATURE = 0.3
MAX_RETRIES = 3
API_TIMEOUT_SECONDS = 60
