from __future__ import annotations

import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]

# Default locations (work without any env vars)
INPUT_DIR = Path(os.getenv("RAG_INPUT_DIR", PROJECT_ROOT / "input")).resolve()
CACHE_DIR = Path(os.getenv("RAG_CACHE_DIR", PROJECT_ROOT / "cache")).resolve()

MARKDOWN_DIR = CACHE_DIR / "markdown"
CHUNKS_DIR = CACHE_DIR / "chunk"
EMBED_DIR = CACHE_DIR / "embed"
META_DIR = CACHE_DIR / "meta"

def ensure_dirs() -> None:
    INPUT_DIR.mkdir(parents=True, exist_ok=True)
    for d in (CACHE_DIR, MARKDOWN_DIR, CHUNKS_DIR, EMBED_DIR, META_DIR):
        d.mkdir(parents=True, exist_ok=True)

