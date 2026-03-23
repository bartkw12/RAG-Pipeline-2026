"""Chunking pipeline — orchestrates chunking for one or all documents.

Public API
----------
``chunk_document_file(doc_id, config) -> ChunkedDocument``
    Read a cached Markdown file, chunk it, persist the result, and return it.

``chunk_all(config) -> list[ChunkedDocument]``
    Chunk every Markdown file in the cache directory.
"""

from __future__ import annotations

import logging
from pathlib import Path

from src.chunking.config import ChunkConfig
from src.chunking.models import ChunkedDocument
from src.chunking.strategy import chunk_document
from src.chunking.writer import write_chunks
from src.config.paths import MARKDOWN_DIR

log = logging.getLogger(__name__)


# ── Public API ──────────────────────────────────────────────────


def chunk_document_file(
    doc_id: str,
    config: ChunkConfig | None = None,
) -> ChunkedDocument:
    """Read ``cache/markdown/{doc_id}.md``, chunk it, and write the result.

    Parameters
    ----------
    doc_id:
        SHA-256 hex digest that identifies the source file.
    config:
        Chunking configuration.  Uses defaults when ``None``.

    Returns
    -------
    ChunkedDocument
        The fully chunked document (also persisted to disk).

    Raises
    ------
    FileNotFoundError
        If the Markdown file does not exist in the cache.
    """
    md_path = MARKDOWN_DIR / f"{doc_id}.md"
    if not md_path.exists():
        raise FileNotFoundError(
            f"Markdown file not found: {md_path}"
        )

    markdown = md_path.read_text(encoding="utf-8")
    chunked_doc = chunk_document(markdown, doc_id, config)

    out_path = write_chunks(chunked_doc)
    log.info(
        "Chunked %s -> %d chunks, written to %s",
        doc_id[:12], len(chunked_doc.chunks), out_path.name,
    )

    return chunked_doc


def chunk_all(
    config: ChunkConfig | None = None,
) -> list[ChunkedDocument]:
    """Chunk every Markdown file in ``cache/markdown/``.

    Skips files that fail and logs warnings for them.

    Returns
    -------
    list[ChunkedDocument]
        Successfully chunked documents.
    """
    results: list[ChunkedDocument] = []
    md_files = sorted(MARKDOWN_DIR.glob("*.md"))

    if not md_files:
        log.warning("No Markdown files found in %s", MARKDOWN_DIR)
        return results

    for md_path in md_files:
        doc_id = md_path.stem
        try:
            chunked_doc = chunk_document_file(doc_id, config)
            results.append(chunked_doc)
        except Exception:
            log.exception("Failed to chunk %s", doc_id[:12])

    log.info("Chunked %d / %d documents", len(results), len(md_files))
    return results
