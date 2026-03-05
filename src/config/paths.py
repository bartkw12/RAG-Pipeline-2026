from __future__ import annotations

import os
from pathlib import Path

# ── Project root ────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parents[2]

# ── User-facing input directory ─────────────────────────────────
INPUT_DIR     = Path(os.getenv("RAG_INPUT_DIR", PROJECT_ROOT / "input")).resolve()
PROCESSED_DIR = INPUT_DIR / "processed"          # drop-folder files move here after ingestion

# ── Derived / cache directories ─────────────────────────────────
CACHE_DIR    = Path(os.getenv("RAG_CACHE_DIR", PROJECT_ROOT / "cache")).resolve()
MARKDOWN_DIR = CACHE_DIR / "markdown"
CHUNK_DIR    = CACHE_DIR / "chunk"
EMBED_DIR    = CACHE_DIR / "embed"
META_DIR     = CACHE_DIR / "meta"

# ── Ingestion registry (tracks what has already been ingested) ──
REGISTRY_FILE = META_DIR / "ingestion_registry.json"

# ── Default manifest location ──────────────────────────────────
MANIFEST_DEFAULT = PROJECT_ROOT / "manifest.json"

# ── Supported file extensions for ingestion ─────────────────────
SUPPORTED_EXTENSIONS: frozenset[str] = frozenset({".pdf", ".docx"})


def ensure_dirs() -> None:
    """Create all required directories if they don't already exist."""
    for d in (
        INPUT_DIR,
        PROCESSED_DIR,
        CACHE_DIR,
        MARKDOWN_DIR,
        CHUNK_DIR,
        EMBED_DIR,
        META_DIR,
    ):
        d.mkdir(parents=True, exist_ok=True)

