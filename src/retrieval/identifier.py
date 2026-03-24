"""Direct identifier lookup — bypass hybrid search for exact-match queries.

When the query analyzer classifies a query as ``EXACT_LOOKUP``, this
module queries ChromaDB metadata directly to retrieve the matching
chunks.  This is fast, deterministic, and avoids unnecessary embedding
or BM25 scoring.

Identifier types and their lookup strategies:

* **test_case_id** — exact equality filter on ``test_case_id`` metadata.
* **requirement_ids** — semicolon-joined string; needs client-side
  substring matching since ChromaDB ``$contains`` doesn't support it.
* **traceability_ids** — same semicolon-joined format, same approach.
* **component_ids** — scanned within chunk documents (text search).
* **cross_references** — scanned within traceability_ids metadata.

Usage::

    from src.retrieval.identifier import lookup_by_identifiers

    analysis = analyze_query("FVTR_MECH_01")
    results = lookup_by_identifiers(analysis, store)
    # → [ScoredChunk(chunk_id=..., score=1.0, ...)]
"""

from __future__ import annotations

import logging
from typing import Any

from ..embedding.store import EmbeddingStore
from .models import QueryAnalysis, ScoredChunk

logger = logging.getLogger(__name__)


def lookup_by_identifiers(
    analysis: QueryAnalysis,
    store: EmbeddingStore,
) -> list[ScoredChunk] | None:
    """Retrieve chunks matching the identifiers in *analysis*.

    Only called when ``analysis.strategy == EXACT_LOOKUP``.

    Parameters
    ----------
    analysis:
        A ``QueryAnalysis`` with at least one identifier populated.
    store:
        The ``EmbeddingStore`` to query.

    Returns
    -------
    list[ScoredChunk] | None
        Matched chunks with ``score=1.0``, or ``None`` if nothing was
        found (caller should fall through to hybrid search).
    """
    if not analysis.has_identifiers:
        return None

    results: dict[str, ScoredChunk] = {}  # keyed by chunk_id to deduplicate

    # ── Test-case IDs — exact equality on `test_case_id` ────────
    for tc_id in analysis.test_case_ids:
        _merge_hits(results, _lookup_by_field(store, "test_case_id", tc_id))

    # ── Requirement IDs — semicolon-joined, need client-side scan
    for req_id in analysis.requirement_ids:
        _merge_hits(results, _lookup_semicolon_field(store, "requirement_ids", req_id))

    # ── Cross-references — check traceability_ids ───────────────
    for xref in analysis.cross_references:
        _merge_hits(results, _lookup_semicolon_field(store, "traceability_ids", xref))

    # ── Component IDs — rare in queries, check requirement_ids
    #    and traceability_ids fields as well as document text ─────
    for comp_id in analysis.component_ids:
        _merge_hits(results, _lookup_text_scan(store, comp_id))

    if not results:
        logger.debug("Identifier lookup found no matches.")
        return None

    chunks = sorted(results.values(), key=lambda c: c.chunk_id)
    logger.info("Identifier lookup returned %d chunk(s).", len(chunks))
    return chunks


# ── Internal helpers ────────────────────────────────────────────


def _lookup_by_field(
    store: EmbeddingStore,
    field: str,
    value: str,
) -> list[ScoredChunk]:
    """Exact-equality metadata lookup on a single field."""
    try:
        raw = store._collection.get(
            where={field: value},
            include=["documents", "metadatas"],
        )
    except Exception as e:
        logger.warning("ChromaDB get(where={%s: %r}) failed: %s", field, value, e)
        return []

    return _raw_to_scored(raw)


def _lookup_semicolon_field(
    store: EmbeddingStore,
    field: str,
    target: str,
) -> list[ScoredChunk]:
    """Find chunks where *field* contains *target* as a semicolon-delimited value.

    ChromaDB doesn't support substring matching on string metadata,
    so we fetch all chunks that have a non-empty value for *field*
    and filter client-side.

    To keep this efficient, we first try an exact-equality match
    (covers the common single-value case), then fall back to a
    scan only if needed.
    """
    # Fast path: exact equality (single-value fields).
    exact = _lookup_by_field(store, field, target)
    if exact:
        return exact

    # Slow path: fetch all chunks with this field populated and
    # filter for semicolon-delimited substring match.
    try:
        raw = store._collection.get(
            where={field: {"$ne": ""}},
            include=["documents", "metadatas"],
        )
    except Exception as e:
        logger.warning("ChromaDB scan for %s failed: %s", field, e)
        return []

    hits: list[ScoredChunk] = []
    target_lower = target.lower()
    for i, meta in enumerate(raw["metadatas"]):
        field_val = str(meta.get(field, "")).lower()
        # Check each semicolon-separated value.
        values = [v.strip() for v in field_val.split(";")]
        if target_lower in values:
            hits.append(_meta_to_scored(
                chunk_id=raw["ids"][i],
                document=raw["documents"][i] if raw["documents"] else "",
                metadata=meta,
            ))

    return hits


def _lookup_text_scan(
    store: EmbeddingStore,
    target: str,
) -> list[ScoredChunk]:
    """Scan stored document texts for *target* substring.

    Used as a last resort for component IDs that may not appear
    in structured metadata fields.  Limited to first 500 results
    to avoid scanning the entire collection.
    """
    try:
        raw = store._collection.get(
            limit=500,
            include=["documents", "metadatas"],
        )
    except Exception as e:
        logger.warning("ChromaDB text scan failed: %s", e)
        return []

    target_lower = target.lower()
    hits: list[ScoredChunk] = []
    for i, doc_text in enumerate(raw["documents"]):
        if target_lower in doc_text.lower():
            hits.append(_meta_to_scored(
                chunk_id=raw["ids"][i],
                document=doc_text,
                metadata=raw["metadatas"][i],
            ))

    return hits


def _raw_to_scored(raw: dict[str, Any]) -> list[ScoredChunk]:
    """Convert a ChromaDB ``get()`` result dict to ``ScoredChunk`` list."""
    results: list[ScoredChunk] = []
    ids = raw.get("ids", [])
    docs = raw.get("documents") or [""] * len(ids)
    metas = raw.get("metadatas") or [{}] * len(ids)

    for i, chunk_id in enumerate(ids):
        results.append(_meta_to_scored(
            chunk_id=chunk_id,
            document=docs[i] if i < len(docs) else "",
            metadata=metas[i] if i < len(metas) else {},
        ))
    return results


def _meta_to_scored(
    chunk_id: str,
    document: str,
    metadata: dict[str, Any],
) -> ScoredChunk:
    """Build a ``ScoredChunk`` from ChromaDB metadata."""
    return ScoredChunk(
        chunk_id=chunk_id,
        doc_id=str(metadata.get("doc_id", "")),
        text=document,
        chunk_type=str(metadata.get("chunk_type", "")),
        tier=3,
        score=1.0,          # perfect match by definition
        vector_score=0.0,
        bm25_score=0.0,
        rerank_score=0.0,
        metadata=dict(metadata),
    )


def _merge_hits(
    target: dict[str, ScoredChunk],
    hits: list[ScoredChunk],
) -> None:
    """Merge *hits* into *target*, deduplicating by chunk_id."""
    for chunk in hits:
        if chunk.chunk_id not in target:
            target[chunk.chunk_id] = chunk
