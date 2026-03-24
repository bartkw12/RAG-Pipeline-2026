"""Hybrid search — fuse vector similarity and BM25 via Reciprocal Rank Fusion.

Combines two complementary retrieval signals:

* **Vector search** — semantic similarity via Ada-002 embeddings and
  ChromaDB cosine distance.  Captures meaning even when exact keywords
  do not match.
* **BM25 search** — keyword relevance via the domain-aware tokeniser.
  Excels at exact identifier and terminology matching.

Fusion uses **Reciprocal Rank Fusion (RRF)**: for each chunk appearing
in either result set its fused score is
``sum(1 / (k + rank_in_list))`` across both lists, where *k* is a
damping constant (default 60).

Scope-aware filtering applies ChromaDB ``where`` metadata filters for
``SCOPED_SEMANTIC`` queries and constrains BM25 results to matching
``doc_ids``.  ``UNCONSTRAINED`` queries search the full collection.

Usage::

    from src.retrieval.hybrid import hybrid_search

    scored = hybrid_search(analysis, config, store)
    for chunk in scored[:5]:
        print(chunk.chunk_id, chunk.score, chunk.vector_score, chunk.bm25_score)
"""

from __future__ import annotations

import logging
import time
from typing import Any

from ..embedding.client import embed_texts
from ..embedding.store import EmbeddingStore
from .abbreviations import expand_query
from .bm25 import bm25_search
from .models import QueryAnalysis, QueryStrategy, RetrievalConfig, ScoredChunk

logger = logging.getLogger(__name__)


# ── Public API ──────────────────────────────────────────────────


def hybrid_search(
    analysis: QueryAnalysis,
    config: RetrievalConfig,
    store: EmbeddingStore,
) -> list[ScoredChunk]:
    """Run scope-aware hybrid (vector + BM25) search and fuse results.

    Parameters
    ----------
    analysis:
        A ``QueryAnalysis`` produced by the query analyzer.  Must have
        ``strategy`` of ``SCOPED_SEMANTIC`` or ``UNCONSTRAINED``.
    config:
        Retrieval configuration (candidate counts, RRF k, etc.).
    store:
        The ``EmbeddingStore`` to query for vector similarity.

    Returns
    -------
    list[ScoredChunk]
        Up to ``config.top_k_broad`` chunks sorted by fused RRF score
        (descending).  Each chunk carries its individual ``vector_score``
        and ``bm25_score`` for downstream diagnostics.
    """
    timings: dict[str, float] = {}

    # ── 1. Abbreviation expansion ───────────────────────────────
    t0 = time.perf_counter()
    query_text = analysis.semantic_remainder or analysis.original_query
    if config.expand_abbreviations:
        # Use the first scope doc_id for scoped expansion, if available.
        scope_doc = analysis.scope_doc_ids[0] if analysis.scope_doc_ids else None
        expanded = expand_query(query_text, scope_doc_id=scope_doc)
    else:
        expanded = query_text

    # Store the expanded form back on the analysis for traceability.
    analysis.expanded_query = expanded
    timings["abbreviation_expand"] = time.perf_counter() - t0
    logger.info("Expanded query: %r", expanded)

    # ── 2. Vector search ────────────────────────────────────────
    t0 = time.perf_counter()
    vec_chunks = _vector_search(
        expanded, analysis, config.top_k_broad, store,
    )
    timings["vector_search"] = time.perf_counter() - t0
    logger.info("Vector search returned %d candidates.", len(vec_chunks))

    # ── 3. BM25 search ─────────────────────────────────────────
    t0 = time.perf_counter()
    bm25_chunks = _bm25_search(
        expanded, analysis, config.top_k_broad,
    )
    timings["bm25_search"] = time.perf_counter() - t0
    logger.info("BM25 search returned %d candidates.", len(bm25_chunks))

    # ── 4. Reciprocal Rank Fusion ───────────────────────────────
    t0 = time.perf_counter()
    fused = _reciprocal_rank_fusion(
        vec_chunks, bm25_chunks, k=config.rrf_k,
    )
    timings["rrf_fusion"] = time.perf_counter() - t0

    # ── 5. Backfill text/metadata for BM25-only chunks ─────────
    t0 = time.perf_counter()
    _backfill_metadata(fused, store)
    timings["backfill"] = time.perf_counter() - t0

    # Trim to top_k_broad.
    fused = fused[: config.top_k_broad]
    logger.info(
        "Hybrid search: %d fused candidates (vec=%d, bm25=%d). "
        "Timings: %s",
        len(fused), len(vec_chunks), len(bm25_chunks),
        {k: f"{v:.3f}s" for k, v in timings.items()},
    )

    return fused


# ── Internal helpers ────────────────────────────────────────────


def _vector_search(
    query: str,
    analysis: QueryAnalysis,
    top_k: int,
    store: EmbeddingStore,
) -> list[ScoredChunk]:
    """Embed *query* and retrieve nearest neighbours from ChromaDB.

    Applies scope-aware metadata filters for ``SCOPED_SEMANTIC``
    queries; no filter for ``UNCONSTRAINED``.
    """
    # Embed the query text.
    vectors = embed_texts([query])
    if not vectors:
        logger.warning("Query embedding returned empty — skipping vector search.")
        return []
    query_vec = vectors[0]

    # Build the scope filter.
    where: dict[str, Any] | None = None
    if analysis.strategy == QueryStrategy.SCOPED_SEMANTIC and analysis.scope_filters:
        where = _build_where_filter(analysis.scope_filters)

    # Query ChromaDB.
    result = store.query(query_vec, n=top_k, where=where)

    # Convert to ScoredChunk.
    chunks: list[ScoredChunk] = []
    for i, chunk_id in enumerate(result.ids):
        # ChromaDB cosine distance ∈ [0, 2]; convert to similarity ∈ [0, 1].
        distance = result.distances[i] if i < len(result.distances) else 0.0
        similarity = max(0.0, 1.0 - distance)

        meta = result.metadatas[i] if i < len(result.metadatas) else {}
        chunks.append(ScoredChunk(
            chunk_id=chunk_id,
            doc_id=str(meta.get("doc_id", "")),
            text=result.documents[i] if i < len(result.documents) else "",
            chunk_type=str(meta.get("chunk_type", "")),
            tier=3,
            score=0.0,             # will be set by RRF
            vector_score=similarity,
            bm25_score=0.0,
            rerank_score=0.0,
            metadata=dict(meta),
        ))

    return chunks


def _bm25_search(
    query: str,
    analysis: QueryAnalysis,
    top_k: int,
) -> list[ScoredChunk]:
    """Run BM25 keyword search, optionally scoped to specific documents."""
    doc_ids: list[str] | None = None
    if analysis.strategy == QueryStrategy.SCOPED_SEMANTIC and analysis.scope_doc_ids:
        doc_ids = analysis.scope_doc_ids

    hits = bm25_search(query, n=top_k, doc_ids=doc_ids)

    if not hits:
        return []

    # Normalise BM25 scores to [0, 1] relative to the top hit.
    max_score = hits[0].score if hits[0].score > 0 else 1.0

    chunks: list[ScoredChunk] = []
    for hit in hits:
        chunks.append(ScoredChunk(
            chunk_id=hit.chunk_id,
            doc_id=hit.doc_id,
            text="",              # text will be filled from the vector result
            chunk_type="",
            tier=3,
            score=0.0,            # will be set by RRF
            vector_score=0.0,
            bm25_score=hit.score / max_score,
            rerank_score=0.0,
            metadata={},
        ))

    return chunks


def _reciprocal_rank_fusion(
    vec_chunks: list[ScoredChunk],
    bm25_chunks: list[ScoredChunk],
    k: int = 60,
) -> list[ScoredChunk]:
    """Fuse two ranked lists using Reciprocal Rank Fusion (RRF).

    For each chunk, its fused score is::

        score = sum(1 / (k + rank_i))

    where ``rank_i`` is the 1-based rank in each list the chunk
    appears in.  Chunks appearing in both lists get contributions
    from both ranks.

    The chunk metadata is taken preferentially from the vector result
    (which includes full text and metadata from ChromaDB); BM25 results
    supplement chunks not found by vector search.
    """
    # Build lookup: chunk_id → best ScoredChunk (prefer vector version).
    chunk_map: dict[str, ScoredChunk] = {}
    for chunk in vec_chunks:
        chunk_map[chunk.chunk_id] = chunk
    for chunk in bm25_chunks:
        if chunk.chunk_id not in chunk_map:
            chunk_map[chunk.chunk_id] = chunk

    # Accumulate RRF scores.
    rrf_scores: dict[str, float] = {}

    for rank, chunk in enumerate(vec_chunks, start=1):
        rrf_scores[chunk.chunk_id] = rrf_scores.get(chunk.chunk_id, 0.0) + 1.0 / (k + rank)

    for rank, chunk in enumerate(bm25_chunks, start=1):
        rrf_scores[chunk.chunk_id] = rrf_scores.get(chunk.chunk_id, 0.0) + 1.0 / (k + rank)

    # Merge individual scores onto the canonical chunk objects.
    for chunk_id, rrf_score in rrf_scores.items():
        entry = chunk_map[chunk_id]
        entry.score = rrf_score

        # Carry over BM25 normalised score if the canonical is from vector.
        for bm in bm25_chunks:
            if bm.chunk_id == chunk_id:
                entry.bm25_score = bm.bm25_score
                break

        # Carry over vector score if the canonical is from BM25.
        for vc in vec_chunks:
            if vc.chunk_id == chunk_id:
                entry.vector_score = vc.vector_score
                # Also fill in text/metadata from the richer vector result.
                if not entry.text and vc.text:
                    entry.text = vc.text
                    entry.metadata = vc.metadata
                    entry.doc_id = vc.doc_id
                    entry.chunk_type = vc.chunk_type
                break

    # Sort by fused score descending.
    ranked = sorted(chunk_map.values(), key=lambda c: c.score, reverse=True)
    return ranked


def _build_where_filter(scope_filters: dict[str, str]) -> dict[str, Any] | None:
    """Convert scope_filters to a ChromaDB ``where`` clause.

    Single-key filters are passed as-is.  Multiple keys are combined
    with ``$and``.  Returns ``None`` if no filters.
    """
    if not scope_filters:
        return None

    conditions = [{k: v} for k, v in scope_filters.items()]

    if len(conditions) == 1:
        return conditions[0]

    return {"$and": conditions}


def _backfill_metadata(
    chunks: list[ScoredChunk],
    store: EmbeddingStore,
) -> None:
    """Fetch text and metadata from ChromaDB for chunks that lack them.

    BM25-only results carry no text or metadata because the BM25 index
    only stores chunk IDs and scores.  This function batch-fetches the
    missing data from ChromaDB so downstream stages (re-ranking,
    context assembly) always have full chunk content.
    """
    missing_ids = [c.chunk_id for c in chunks if not c.text]
    if not missing_ids:
        return

    try:
        raw = store._collection.get(
            ids=missing_ids,
            include=["documents", "metadatas"],
        )
    except Exception as e:
        logger.warning("Backfill fetch failed: %s", e)
        return

    # Build a quick lookup from the response.
    fetched: dict[str, tuple[str, dict[str, Any]]] = {}
    ids = raw["ids"] or []
    docs = raw["documents"] or []
    metas = raw["metadatas"] or []
    for i, cid in enumerate(ids):
        doc = docs[i] if i < len(docs) else ""
        meta = dict(metas[i]) if i < len(metas) else {}
        fetched[cid] = (doc, meta)

    # Patch the chunks in-place.
    for chunk in chunks:
        if chunk.text or chunk.chunk_id not in fetched:
            continue
        text, meta = fetched[chunk.chunk_id]
        chunk.text = text
        chunk.metadata = meta
        chunk.doc_id = str(meta.get("doc_id", chunk.doc_id))
        chunk.chunk_type = str(meta.get("chunk_type", chunk.chunk_type))
