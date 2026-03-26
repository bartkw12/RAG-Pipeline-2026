"""Stratified evaluation framework for the retrieval pipeline.

Runs every query in a golden test set through the pipeline, computes
IR metrics per-query and aggregated per query-type, then produces a
structured report.

Metrics
-------
* **Precision@k** — fraction of top-k results that are relevant.
* **Recall@k** — fraction of relevant chunks found in top-k.
* **MRR** — reciprocal rank of the first relevant result.
* **nDCG@k** — normalised discounted cumulative gain.
* **Hit Rate** — fraction of queries with ≥ 1 relevant chunk in top-k.

Usage::

    from src.retrieval.evaluate import evaluate

    report = evaluate("tests/golden_retrieval.json")
    report.print_table()          # human-readable
    report.to_dict()              # machine-readable
"""

from __future__ import annotations

import json
import logging
import math
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .models import RetrievalConfig
from .pipeline import retrieve

logger = logging.getLogger(__name__)


# ── Per-query result ────────────────────────────────────────────


@dataclass
class QueryResult:
    """Metrics for a single golden query."""

    query: str = ""
    query_type: str = ""
    expected_chunk_ids: list[str] = field(default_factory=list)
    retrieved_chunk_ids: list[str] = field(default_factory=list)
    expected_doc_ids: list[str] = field(default_factory=list)
    retrieved_doc_ids: list[str] = field(default_factory=list)
    strategy_used: str = ""
    elapsed_s: float = 0.0

    # Metrics
    precision: float = 0.0
    recall: float = 0.0
    mrr: float = 0.0
    ndcg: float = 0.0
    hit: bool = False


# ── Aggregate row ───────────────────────────────────────────────


@dataclass
class TypeAggregate:
    """Averaged metrics for one query-type stratum (or overall)."""

    query_type: str = ""
    count: int = 0
    precision: float = 0.0
    recall: float = 0.0
    mrr: float = 0.0
    ndcg: float = 0.0
    hit_rate: float = 0.0


# ── Full report ─────────────────────────────────────────────────


@dataclass
class EvalReport:
    """Complete evaluation output."""

    query_results: list[QueryResult] = field(default_factory=list)
    type_aggregates: list[TypeAggregate] = field(default_factory=list)
    overall: TypeAggregate = field(default_factory=TypeAggregate)
    config: dict[str, Any] = field(default_factory=dict)
    total_elapsed_s: float = 0.0

    # ── Rendering ───────────────────────────────────────────────

    def print_table(self, *, per_query: bool = False) -> None:
        """Print a human-readable stratified table to stdout."""

        if per_query:
            hdr = (
                f"{'Query':<52} {'Type':<24} {'P@k':>5} {'R@k':>5} "
                f"{'MRR':>5} {'nDCG':>5} {'Hit':>4} {'Time':>6}"
            )
            print(hdr)
            print("─" * len(hdr))
            for qr in self.query_results:
                q_disp = (qr.query[:49] + "...") if len(qr.query) > 52 else qr.query
                print(
                    f"{q_disp:<52} {qr.query_type:<24} "
                    f"{qr.precision:5.3f} {qr.recall:5.3f} "
                    f"{qr.mrr:5.3f} {qr.ndcg:5.3f} "
                    f"{'  ✓' if qr.hit else '  ✗':>4} "
                    f"{qr.elapsed_s:5.2f}s"
                )
            print()

        # Stratified table
        hdr = (
            f"{'Query Type':<28} {'#':>3} {'P@k':>6} {'R@k':>6} "
            f"{'MRR':>6} {'nDCG':>6} {'Hit%':>6}"
        )
        print(hdr)
        print("─" * len(hdr))
        for ta in self.type_aggregates:
            print(
                f"{ta.query_type:<28} {ta.count:3d} "
                f"{ta.precision:6.3f} {ta.recall:6.3f} "
                f"{ta.mrr:6.3f} {ta.ndcg:6.3f} {ta.hit_rate:6.1%}"
            )
        print("─" * len(hdr))
        ov = self.overall
        print(
            f"{'OVERALL':<28} {ov.count:3d} "
            f"{ov.precision:6.3f} {ov.recall:6.3f} "
            f"{ov.mrr:6.3f} {ov.ndcg:6.3f} {ov.hit_rate:6.1%}"
        )
        print(f"\nTotal time: {self.total_elapsed_s:.1f}s")

    def to_dict(self) -> dict[str, Any]:
        """Serialise the report to a plain dict (JSON-friendly)."""
        return {
            "config": self.config,
            "total_elapsed_s": round(self.total_elapsed_s, 3),
            "overall": _agg_dict(self.overall),
            "by_type": [_agg_dict(ta) for ta in self.type_aggregates],
            "queries": [_qr_dict(qr) for qr in self.query_results],
        }


# ── Metric helpers ──────────────────────────────────────────────


def _precision_at_k(retrieved: list[str], relevant: set[str]) -> float:
    """Fraction of retrieved items that are relevant."""
    if not retrieved:
        return 0.0
    return sum(1 for r in retrieved if r in relevant) / len(retrieved)


def _recall_at_k(retrieved: list[str], relevant: set[str]) -> float:
    """Fraction of relevant items found in retrieved."""
    if not relevant:
        return 1.0  # vacuously true
    return sum(1 for r in retrieved if r in relevant) / len(relevant)


def _mrr(retrieved: list[str], relevant: set[str]) -> float:
    """Reciprocal rank of the first relevant result."""
    for i, r in enumerate(retrieved, 1):
        if r in relevant:
            return 1.0 / i
    return 0.0


def _dcg(gains: list[float]) -> float:
    """Discounted cumulative gain."""
    return sum(g / math.log2(i + 2) for i, g in enumerate(gains))


def _ndcg_at_k(retrieved: list[str], relevant: set[str]) -> float:
    """Normalised DCG for binary relevance."""
    if not relevant:
        return 1.0
    gains = [1.0 if r in relevant else 0.0 for r in retrieved]
    actual = _dcg(gains)
    # Ideal: all relevant items ranked first
    ideal_len = min(len(relevant), len(retrieved))
    ideal = _dcg([1.0] * ideal_len)
    if ideal == 0.0:
        return 0.0
    return actual / ideal


# ── Core evaluation ─────────────────────────────────────────────


def evaluate(
    golden_path: str | Path,
    config: RetrievalConfig | None = None,
) -> EvalReport:
    """Run evaluation against a golden test set.

    Parameters
    ----------
    golden_path:
        Path to the golden JSON file (``tests/golden_retrieval.json``).
    config:
        Pipeline configuration override.  Uses defaults when ``None``.

    Returns
    -------
    EvalReport
    """
    cfg = config or RetrievalConfig()
    golden = _load_golden(golden_path)
    queries = golden["queries"]

    report = EvalReport(
        config={
            "reranker_type": cfg.reranker_type,
            "top_k_broad": cfg.top_k_broad,
            "top_k_final": cfg.top_k_final,
            "max_context_tokens": cfg.max_context_tokens,
            "expand_abbreviations": cfg.expand_abbreviations,
        },
    )

    total_t0 = time.perf_counter()

    for i, entry in enumerate(queries, 1):
        query = entry["query"]
        query_type = entry["query_type"]
        expected_chunks = set(entry.get("expected_chunk_ids", []))
        expected_docs = set(entry.get("expected_doc_ids", []))

        logger.info("[%d/%d] %s  (%s)", i, len(queries), query, query_type)

        t0 = time.perf_counter()
        result = retrieve(query, cfg)
        elapsed = time.perf_counter() - t0

        retrieved_ids = [sc.chunk_id for sc in result.scored_chunks]
        retrieved_docs = list(dict.fromkeys(
            sc.doc_id for sc in result.scored_chunks
        ))

        qr = QueryResult(
            query=query,
            query_type=query_type,
            expected_chunk_ids=list(expected_chunks),
            retrieved_chunk_ids=retrieved_ids,
            expected_doc_ids=list(expected_docs),
            retrieved_doc_ids=retrieved_docs,
            strategy_used=result.strategy,
            elapsed_s=elapsed,
            precision=_precision_at_k(retrieved_ids, expected_chunks),
            recall=_recall_at_k(retrieved_ids, expected_chunks),
            mrr=_mrr(retrieved_ids, expected_chunks),
            ndcg=_ndcg_at_k(retrieved_ids, expected_chunks),
            hit=any(r in expected_chunks for r in retrieved_ids),
        )
        report.query_results.append(qr)

    report.total_elapsed_s = time.perf_counter() - total_t0

    # ── Aggregate per query type ────────────────────────────────
    type_order: list[str] = []
    type_buckets: dict[str, list[QueryResult]] = {}
    for qr in report.query_results:
        if qr.query_type not in type_buckets:
            type_order.append(qr.query_type)
            type_buckets[qr.query_type] = []
        type_buckets[qr.query_type].append(qr)

    for qt in type_order:
        bucket = type_buckets[qt]
        report.type_aggregates.append(_aggregate(qt, bucket))

    report.overall = _aggregate("overall", report.query_results)
    return report


# ── Internal helpers ────────────────────────────────────────────


def _load_golden(path: str | Path) -> dict[str, Any]:
    """Load and validate the golden test set JSON."""
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Golden test set not found: {p}")
    data = json.loads(p.read_text(encoding="utf-8"))
    if "queries" not in data or not isinstance(data["queries"], list):
        raise ValueError(f"Golden file must contain a 'queries' list: {p}")
    return data


def _aggregate(label: str, results: list[QueryResult]) -> TypeAggregate:
    """Compute mean metrics for a list of query results."""
    n = len(results)
    if n == 0:
        return TypeAggregate(query_type=label)
    return TypeAggregate(
        query_type=label,
        count=n,
        precision=sum(qr.precision for qr in results) / n,
        recall=sum(qr.recall for qr in results) / n,
        mrr=sum(qr.mrr for qr in results) / n,
        ndcg=sum(qr.ndcg for qr in results) / n,
        hit_rate=sum(1 for qr in results if qr.hit) / n,
    )


def _agg_dict(ta: TypeAggregate) -> dict[str, Any]:
    """Serialise a TypeAggregate to a plain dict."""
    return {
        "query_type": ta.query_type,
        "count": ta.count,
        "precision": round(ta.precision, 4),
        "recall": round(ta.recall, 4),
        "mrr": round(ta.mrr, 4),
        "ndcg": round(ta.ndcg, 4),
        "hit_rate": round(ta.hit_rate, 4),
    }


def _qr_dict(qr: QueryResult) -> dict[str, Any]:
    """Serialise a QueryResult to a plain dict."""
    return {
        "query": qr.query,
        "query_type": qr.query_type,
        "strategy_used": qr.strategy_used,
        "expected_chunk_ids": qr.expected_chunk_ids,
        "retrieved_chunk_ids": qr.retrieved_chunk_ids,
        "precision": round(qr.precision, 4),
        "recall": round(qr.recall, 4),
        "mrr": round(qr.mrr, 4),
        "ndcg": round(qr.ndcg, 4),
        "hit": qr.hit,
        "elapsed_s": round(qr.elapsed_s, 3),
    }
