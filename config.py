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

# ---- DeepSeek API ----
DEEPSEEK_API_KEY = os.environ.get("DEEPSEEK_API_KEY", "")
DEEPSEEK_BASE_URL = "https://api.deepseek.com"
DEEPSEEK_CHAT_MODEL = "deepseek-chat"
# Local embedding model (Chinese-optimized, runs offline, no API needed)
EMBEDDING_MODEL_NAME = "BAAI/bge-small-zh-v1.5"

# ---- Paths ----
BASE_DIR = Path(__file__).resolve().parent
CHROMA_PERSIST_DIR = str(BASE_DIR / "chroma_store")
CHROMA_COLLECTION_NAME = "edu_documents"
SQLITE_DB_PATH = str(BASE_DIR / "data" / "structured.db")
SAMPLE_DOCS_DIR = str(BASE_DIR / "data" / "sample_docs")
GROUND_TRUTH_PATH = str(BASE_DIR / "data" / "ground_truth.json")
OUTPUT_DIR = str(BASE_DIR / "output")
LOG_FILE = str(BASE_DIR / "pipeline.log")

# ---- Chunking ----
CHUNK_MAX_CHARS = 1000
CHUNK_OVERLAP_CHARS = 200
CHUNK_MIN_CHARS = 50

# ---- Retrieval ----
TOP_K_RETRIEVAL = 5

# ---- API Settings ----
EXTRACTION_TEMPERATURE = 0.1
GENERATION_TEMPERATURE = 0.3
MAX_RETRIES = 3
API_TIMEOUT_SECONDS = 60
