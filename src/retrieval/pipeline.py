"""Retrieval pipeline orchestrator — single entry point for query → context.

Wires together all retrieval components:

1. **Query analysis** — classify the query and extract scope/identifiers.
2. **Search** — exact identifier lookup *or* hybrid (vector + BM25).
3. **Re-ranking** — narrow candidates via cross-encoder or LLM.
4. **Context assembly** — expand chunks into a hierarchy-aware window.

Usage::

    from src.retrieval.pipeline import retrieve

    result = retrieve("DIM-V thermal test results")
    print(result.context.to_prompt_text())
    print(result.timings)
"""

from __future__ import annotations

import logging
import time

from ..embedding.store import EmbeddingStore
from .analyzer import analyze_query
from .context import assemble_context
from .hybrid import hybrid_search
from .identifier import lookup_by_identifiers
from .models import (
    ContextWindow,
    QueryStrategy,
    RetrievalConfig,
    RetrievalResult,
)
from .rerank import get_reranker

logger = logging.getLogger(__name__)

# Singleton store — opened once per process.
_store: EmbeddingStore | None = None


def _get_store() -> EmbeddingStore:
    """Return the shared ``EmbeddingStore`` singleton."""
    global _store
    if _store is None:
        _store = EmbeddingStore()
    return _store


def retrieve(
    query: str,
    config: RetrievalConfig | None = None,
    store: EmbeddingStore | None = None,
) -> RetrievalResult:
    """Run the full retrieval pipeline for *query*.

    Parameters
    ----------
    query:
        Natural-language or identifier query string.
    config:
        Retrieval configuration.  Uses defaults when ``None``.
    store:
        Optional ``EmbeddingStore`` instance.  A shared singleton
        is used when not supplied.

    Returns
    -------
    RetrievalResult
        Context window, scored chunks, query analysis, and timings.
    """
    cfg = config or RetrievalConfig()
    emb_store = store or _get_store()
    timings: dict[str, float] = {}

    # ── 1. Query analysis ───────────────────────────────────────
    t0 = time.perf_counter()
    analysis = analyze_query(query)
    timings["analysis"] = time.perf_counter() - t0

    logger.info(
        "Query analyzed: strategy=%s  ids=%s  scope=%s",
        analysis.strategy.value,
        analysis.has_identifiers,
        analysis.scope_filters or "none",
    )

    # ── 2. Search ───────────────────────────────────────────────
    t0 = time.perf_counter()

    if analysis.strategy == QueryStrategy.EXACT_LOOKUP:
        # Direct metadata lookup — fast path.
        exact_hits = lookup_by_identifiers(analysis, emb_store)
        if exact_hits:
            timings["search"] = time.perf_counter() - t0
            logger.info("Exact lookup returned %d chunk(s).", len(exact_hits))

            # Assemble context directly (no re-ranking needed).
            t0 = time.perf_counter()
            context = assemble_context(exact_hits, cfg)
            timings["assembly"] = time.perf_counter() - t0

            return RetrievalResult(
                context=context,
                scored_chunks=exact_hits,
                query_analysis=analysis,
                timings=timings,
                strategy=analysis.strategy.value,
            )

        # No exact matches — fall through to hybrid search.
        logger.info("Exact lookup found nothing; falling through to hybrid search.")

    # Hybrid search (SCOPED_SEMANTIC, UNCONSTRAINED, or EXACT_LOOKUP fallback).
    candidates = hybrid_search(analysis, cfg, emb_store)
    timings["search"] = time.perf_counter() - t0

    if not candidates:
        logger.warning("Hybrid search returned no candidates.")
        return RetrievalResult(
            context=ContextWindow(),
            scored_chunks=[],
            query_analysis=analysis,
            timings=timings,
            strategy=analysis.strategy.value,
        )

    # ── 3. Re-ranking ──────────────────────────────────────────
    t0 = time.perf_counter()
    reranker = get_reranker(cfg.reranker_type)
    rerank_query = analysis.expanded_query or analysis.original_query
    reranked = reranker.rerank(rerank_query, candidates, top_k=cfg.top_k_final)
    timings["rerank"] = time.perf_counter() - t0

    # ── 4. Context assembly ─────────────────────────────────────
    t0 = time.perf_counter()
    context = assemble_context(reranked, cfg)
    timings["assembly"] = time.perf_counter() - t0

    logger.info(
        "Pipeline complete: strategy=%s  candidates=%d  reranked=%d  "
        "sections=%d  tokens=%d  timings=%s",
        analysis.strategy.value,
        len(candidates),
        len(reranked),
        len(context.sections),
        context.total_tokens,
        {k: f"{v:.3f}s" for k, v in timings.items()},
    )

    return RetrievalResult(
        context=context,
        scored_chunks=reranked,
        query_analysis=analysis,
        timings=timings,
        strategy=analysis.strategy.value,
    )
