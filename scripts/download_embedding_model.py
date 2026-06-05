#!/usr/bin/env python3
"""Download the local embedding model needed by the offline desktop build."""

from pathlib import Path

from sentence_transformers import SentenceTransformer


MODEL_ID = "BAAI/bge-small-zh-v1.5"
DEST = Path(__file__).resolve().parents[1] / "models" / "bge-small-zh-v1.5"


def main() -> None:
    DEST.parent.mkdir(parents=True, exist_ok=True)
    model = SentenceTransformer(MODEL_ID)
    model.save(str(DEST))
    print(f"Saved {MODEL_ID} to {DEST}")


if __name__ == "__main__":
    main()
