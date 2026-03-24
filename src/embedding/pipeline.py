"""Embedding pipeline orchestration — load, enrich, embed, store.

Reads chunked documents from ``cache/chunk/``, enriches tier-3 chunks
with metadata prefixes, calls Azure OpenAI Ada-002 for embeddings,
and upserts the results into a persistent ChromaDB collection.

Usage::

    from src.embedding.pipeline import embed_all, embed_document

    # Embed one document
    result = embed_document("f4b48797a6a2...")

    # Embed all chunked documents
    results = embed_all()
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from ..config.paths import CHUNK_DIR
from .client import embed_texts
from .enrich import enrich_chunks
from .store import EmbeddingStore

logger = logging.getLogger(__name__)

# Tier value for atomic (leaf) chunks.
_TIER_ATOMIC = 3


# ── Result dataclass ────────────────────────────────────────────


@dataclass
class EmbeddingResult:
    """Summary of an embedding run for a single document."""

    doc_id: str
    doc_title: str = ""
    total_chunks: int = 0      # tier-3 chunks in the document
    embedded: int = 0          # newly embedded this run
    skipped: int = 0           # already in store
    failed: int = 0
    errors: list[str] = field(default_factory=list)


# ── Public API ──────────────────────────────────────────────────


def embed_document(
    doc_id: str,
    *,
    force: bool = False,
    store: EmbeddingStore | None = None,
) -> EmbeddingResult:
    """Embed a single document's tier-3 chunks.

    Parameters
    ----------
    doc_id:
        SHA-256 identifier of the document (matches a chunk JSON file
        in ``cache/chunk/``).
    force:
        If ``True``, delete existing vectors for this document and
        re-embed everything.  If ``False``, skip chunks already in
        the store.
    store:
        Optional pre-initialised ``EmbeddingStore``.  One will be
        created (with defaults) if not provided.

    Returns
    -------
    EmbeddingResult
        Counts of embedded / skipped / failed chunks.
    """
    result = EmbeddingResult(doc_id=doc_id)

    # ── Load chunk JSON ─────────────────────────────────────────
    chunk_path = CHUNK_DIR / f"{doc_id}.json"
    if not chunk_path.exists():
        raise FileNotFoundError(
            f"Chunk file not found: {chunk_path}\n"
            f"Run the chunking pipeline first."
        )

    data = json.loads(chunk_path.read_text(encoding="utf-8"))
    doc_metadata: dict[str, Any] = data.get("doc_metadata", {})
    all_chunks: list[dict[str, Any]] = data.get("chunks", [])
    result.doc_title = doc_metadata.get("doc_title", "")

    # ── Filter to tier-3 only ───────────────────────────────────
    atomic_chunks = [c for c in all_chunks if c.get("tier") == _TIER_ATOMIC]
    result.total_chunks = len(atomic_chunks)

    if not atomic_chunks:
        logger.info("No tier-3 chunks in %s — nothing to embed.", doc_id[:12])
        return result

    # ── Initialise store ────────────────────────────────────────
    if store is None:
        store = EmbeddingStore()

    # ── Force mode: delete stale vectors first ──────────────────
    if force:
        deleted = store.delete_by_doc(doc_id)
        if deleted:
            logger.info(
                "Force mode: removed %d existing vectors for %s.",
                deleted, doc_id[:12],
            )

    # ── Determine which chunks need embedding ───────────────────
    chunk_ids = [c["chunk_id"] for c in atomic_chunks]

    if force:
        to_embed = atomic_chunks
    else:
        existing_ids = store.get_existing_ids(chunk_ids)
        to_embed = [c for c in atomic_chunks if c["chunk_id"] not in existing_ids]
        result.skipped = len(atomic_chunks) - len(to_embed)

    if not to_embed:
        logger.info(
            "%s: all %d chunks already embedded — skipping.",
            doc_id[:12], result.total_chunks,
        )
        return result

    # ── Enrich texts ────────────────────────────────────────────
    enriched_texts = enrich_chunks(to_embed, doc_metadata)

    # ── Embed via Azure OpenAI ──────────────────────────────────
    try:
        vectors = embed_texts(enriched_texts)
    except (RuntimeError, ValueError) as exc:
        result.failed = len(to_embed)
        result.errors.append(str(exc))
        logger.error("Embedding failed for %s: %s", doc_id[:12], exc)
        return result

    # ── Build ChromaDB metadata ─────────────────────────────────
    ids: list[str] = []
    metadatas: list[dict[str, Any]] = []

    for chunk in to_embed:
        meta = chunk.get("metadata", {})
        ids.append(chunk["chunk_id"])
        metadatas.append(_build_chroma_metadata(chunk, meta, doc_metadata))

    # ── Upsert ──────────────────────────────────────────────────
    store.upsert(
        ids=ids,
        embeddings=vectors,  # type: ignore[arg-type]
        documents=enriched_texts,
        metadatas=metadatas,  # type: ignore[arg-type]
    )
    result.embedded = len(ids)

    logger.info(
        "%s (%s): embedded %d, skipped %d, failed %d  [total tier-3: %d]",
        doc_id[:12], result.doc_title[:40],
        result.embedded, result.skipped, result.failed, result.total_chunks,
    )
    return result


def embed_all(
    *,
    force: bool = False,
) -> list[EmbeddingResult]:
    """Embed all chunked documents in ``cache/chunk/``.

    Parameters
    ----------
    force:
        Passed through to :func:`embed_document`.

    Returns
    -------
    list[EmbeddingResult]
        One result per document processed.
    """
    chunk_files = sorted(CHUNK_DIR.glob("*.json"))
    if not chunk_files:
        logger.warning("No chunk files found in %s.", CHUNK_DIR)
        return []

    # Share a single store instance across all documents.
    store = EmbeddingStore()
    results: list[EmbeddingResult] = []

    for path in chunk_files:
        doc_id = path.stem
        try:
            result = embed_document(doc_id, force=force, store=store)
            results.append(result)
        except Exception:
            logger.exception("Failed to embed %s", doc_id[:12])
            results.append(EmbeddingResult(
                doc_id=doc_id,
                failed=1,
                errors=[f"Unexpected error for {doc_id[:12]}"],
            ))

    return results


# ── Internals ───────────────────────────────────────────────────


def _build_chroma_metadata(
    chunk: dict[str, Any],
    meta: dict[str, Any],
    doc_metadata: dict[str, Any],
) -> dict[str, str | int | float | bool]:
    """Build a flat metadata dict for ChromaDB.

    ChromaDB values must be str, int, float, or bool — no lists.
    List fields are joined with ``";"`` delimiters.
    Only non-empty values are included.
    """
    m: dict[str, str | int | float | bool] = {}

    # Document-level
    _set_str(m, "doc_id", chunk.get("doc_id", ""))
    _set_str(m, "doc_type", doc_metadata.get("doc_type", ""))
    _set_str(m, "module_name", doc_metadata.get("module_name", ""))

    # Chunk-level
    _set_str(m, "chunk_type", chunk.get("chunk_type", ""))
    _set_str(m, "section_number", meta.get("section_number", ""))
    _set_str(m, "heading", meta.get("heading", ""))

    # Booleans
    if meta.get("has_table"):
        m["has_table"] = True
    if meta.get("has_figure"):
        m["has_figure"] = True

    # Token count
    token_count = chunk.get("token_count", 0)
    if token_count:
        m["token_count"] = token_count

    # Test-case fields
    _set_str(m, "test_case_id", meta.get("test_case_id", ""))
    _set_str(m, "test_result", meta.get("test_result", ""))

    # List fields → semicolon-joined strings
    _set_joined(m, "requirement_ids", meta.get("requirement_ids", []))
    _set_joined(m, "traceability_ids", meta.get("traceability_ids", []))

    return m


def _set_str(
    d: dict[str, str | int | float | bool],
    key: str,
    value: str,
) -> None:
    """Set a string value only if non-empty."""
    if value:
        d[key] = value


def _set_joined(
    d: dict[str, str | int | float | bool],
    key: str,
    values: list[str],
) -> None:
    """Join a list into a semicolon-delimited string, if non-empty."""
    if values:
        d[key] = ";".join(values)
