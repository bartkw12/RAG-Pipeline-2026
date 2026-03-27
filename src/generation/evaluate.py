"""Stratified evaluation framework for the generation pipeline.

Runs every query in a golden test set through the full
retrieval → generation pipeline, computes generation-quality
metrics per-query and aggregated per query-type, then produces a
structured report.

Metrics
-------
* **Fact Recall** — fraction of expected facts found in the answer.
* **Abstention Accuracy** — did the pipeline abstain when expected?
* **Partial Accuracy** — did the pipeline flag partial when expected?
* **Citation Coverage** — fraction of claims with ≥ 1 valid citation.
* **Verification Pass** — all structural groundedness checks pass?
* **Confidence Alignment** — system confidence ≥ minimum expected?
* **Doc-ID Precision** — fraction of cited docs that were expected.
* **Doc-ID Recall** — fraction of expected docs that were cited.

Usage::

    from src.generation.evaluate import evaluate_generation

    report = evaluate_generation("tests/golden_generation.json")
    report.print_table()          # human-readable
    report.to_dict()              # machine-readable
"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from openai import AzureOpenAI

from ..config.env_config import load_azure_generation_config
from ..retrieval.models import RetrievalConfig
from .models import ConfidenceLevel, GenerationConfig, GenerationResult
from .pipeline import generate

logger = logging.getLogger(__name__)

# ── Confidence ordinal for comparison ───────────────────────────

_CONF_RANK: dict[str, int] = {"LOW": 0, "MEDIUM": 1, "HIGH": 2}


# ── Per-query result ────────────────────────────────────────────


@dataclass
class GenQueryResult:
    """Metrics for a single golden generation query."""

    # Identity
    query: str = ""
    query_type: str = ""
    notes: str = ""

    # Expected
    expected_facts: list[str] = field(default_factory=list)
    expected_doc_ids: list[str] = field(default_factory=list)
    expected_abstained: bool = False
    expected_partial: bool = False
    min_confidence: str = "LOW"

    # Actual (from GenerationResult)
    answer: str = ""
    actual_abstained: bool = False
    actual_partial: bool = False
    actual_confidence: str = "LOW"
    model_confidence: str = "LOW"
    strategy_used: str = ""
    cited_doc_ids: list[str] = field(default_factory=list)
    num_claims: int = 0
    elapsed_s: float = 0.0
    total_elapsed_s: float = 0.0
    error: str = ""

    # ── Computed metrics ────────────────────────────────────────
    fact_recall: float = 0.0
    """Fraction of expected facts found (case-insensitive substring)."""

    abstention_correct: bool = True
    """``actual_abstained == expected_abstained``."""

    partial_correct: bool = True
    """``actual_partial == expected_partial``."""

    citation_coverage: float = 0.0
    """From ``verification.citation_coverage_ratio``."""

    verification_pass: bool = True
    """``all_citations_resolved and abstention_consistent``."""

    confidence_aligned: bool = True
    """``actual_confidence >= min_confidence`` (ordinal)."""

    doc_precision: float = 0.0
    """Fraction of cited doc IDs that are in expected_doc_ids."""

    doc_recall: float = 0.0
    """Fraction of expected doc IDs that were cited."""

    matched_facts: list[str] = field(default_factory=list)
    """Which expected facts were found."""

    missed_facts: list[str] = field(default_factory=list)
    """Which expected facts were NOT found."""


# ── Aggregate row ───────────────────────────────────────────────


@dataclass
class GenTypeAggregate:
    """Averaged metrics for one query-type stratum (or overall)."""

    query_type: str = ""
    count: int = 0
    fact_recall: float = 0.0
    abstention_accuracy: float = 0.0
    partial_accuracy: float = 0.0
    citation_coverage: float = 0.0
    verification_rate: float = 0.0
    confidence_alignment: float = 0.0
    doc_precision: float = 0.0
    doc_recall: float = 0.0
    error_rate: float = 0.0


# ── Full report ─────────────────────────────────────────────────


@dataclass
class GenEvalReport:
    """Complete generation evaluation output."""

    query_results: list[GenQueryResult] = field(default_factory=list)
    type_aggregates: list[GenTypeAggregate] = field(default_factory=list)
    overall: GenTypeAggregate = field(default_factory=GenTypeAggregate)
    config: dict[str, Any] = field(default_factory=dict)
    total_elapsed_s: float = 0.0

    # ── Rendering ───────────────────────────────────────────────

    def print_table(self, *, per_query: bool = False) -> None:
        """Print a human-readable stratified table to stdout."""

        if per_query:
            hdr = (
                f"{'Query':<44} {'Type':<20} {'FRec':>5} {'Abst':>5} "
                f"{'Part':>5} {'CCov':>5} {'Vfy':>4} {'Conf':>5} "
                f"{'DPrc':>5} {'DRec':>5} {'Time':>6}"
            )
            print(hdr)
            print("─" * len(hdr))
            for qr in self.query_results:
                q_disp = (
                    (qr.query[:41] + "...")
                    if len(qr.query) > 44
                    else qr.query
                )
                err_flag = " ⚠" if qr.error else ""
                print(
                    f"{q_disp:<44} {qr.query_type:<20} "
                    f"{qr.fact_recall:5.2f} "
                    f"{'  ✓' if qr.abstention_correct else '  ✗':>5} "
                    f"{'  ✓' if qr.partial_correct else '  ✗':>5} "
                    f"{qr.citation_coverage:5.2f} "
                    f"{'  ✓' if qr.verification_pass else '  ✗':>4} "
                    f"{'  ✓' if qr.confidence_aligned else '  ✗':>5} "
                    f"{qr.doc_precision:5.2f} "
                    f"{qr.doc_recall:5.2f} "
                    f"{qr.total_elapsed_s:5.1f}s{err_flag}"
                )
            print()

        # Stratified table
        hdr = (
            f"{'Query Type':<24} {'#':>3} {'FRec':>6} {'Abst':>6} "
            f"{'Part':>6} {'CCov':>6} {'Vfy%':>6} {'Conf':>6} "
            f"{'DPrc':>6} {'DRec':>6} {'Err%':>6}"
        )
        print(hdr)
        print("─" * len(hdr))
        for ta in self.type_aggregates:
            self._print_agg_row(ta)
        print("─" * len(hdr))
        self._print_agg_row(self.overall)
        print(f"\nTotal time: {self.total_elapsed_s:.1f}s")

    @staticmethod
    def _print_agg_row(ta: GenTypeAggregate) -> None:
        """Print a single aggregate row."""
        print(
            f"{ta.query_type:<24} {ta.count:3d} "
            f"{ta.fact_recall:6.3f} {ta.abstention_accuracy:6.1%} "
            f"{ta.partial_accuracy:6.1%} {ta.citation_coverage:6.3f} "
            f"{ta.verification_rate:6.1%} {ta.confidence_alignment:6.1%} "
            f"{ta.doc_precision:6.3f} {ta.doc_recall:6.3f} "
            f"{ta.error_rate:6.1%}"
        )

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


def _fact_recall(
    answer: str,
    expected_facts: list[str],
) -> tuple[float, list[str], list[str]]:
    """Compute case-insensitive substring fact recall.

    Returns
    -------
    tuple[float, list[str], list[str]]
        ``(recall, matched, missed)``
    """
    if not expected_facts:
        return 1.0, [], []  # vacuously true

    answer_lower = answer.lower()
    matched: list[str] = []
    missed: list[str] = []
    for fact in expected_facts:
        if fact.lower() in answer_lower:
            matched.append(fact)
        else:
            missed.append(fact)

    recall = len(matched) / len(expected_facts)
    return recall, matched, missed


def _semantic_fact_recall(
    answer: str,
    expected_facts: list[str],
    judge_client: AzureOpenAI,
    judge_model: str,
) -> tuple[float, list[str], list[str]]:
    """Compute fact recall using an LLM-as-judge for semantic matching.

    For each expected fact, asks a small model whether the answer
    contains the semantic equivalent of that fact.  More accurate
    than substring matching for paraphrased answers.

    Parameters
    ----------
    answer:
        The generation pipeline's answer text.
    expected_facts:
        List of expected fact strings from the golden set.
    judge_client:
        An ``AzureOpenAI`` client instance for the judge model.
    judge_model:
        Deployment name for the judge (e.g. ``"gpt-5-nano"``).

    Returns
    -------
    tuple[float, list[str], list[str]]
        ``(recall, matched, missed)``
    """
    if not expected_facts:
        return 1.0, [], []

    # First try substring — avoids an API call for obvious matches
    answer_lower = answer.lower()
    matched: list[str] = []
    to_judge: list[str] = []
    for fact in expected_facts:
        if fact.lower() in answer_lower:
            matched.append(fact)
        else:
            to_judge.append(fact)

    # Use LLM judge for remaining facts
    for fact in to_judge:
        try:
            response = judge_client.chat.completions.create(
                model=judge_model,
                messages=[
                    {
                        "role": "system",
                        "content": (
                            "You are a fact-matching judge. Given an ANSWER "
                            "and a FACT, determine if the ANSWER contains "
                            "the semantic equivalent of the FACT. The fact "
                            "may be paraphrased, use different units, or "
                            "different wording. Respond with only YES or NO."
                        ),
                    },
                    {
                        "role": "user",
                        "content": f"ANSWER:\n{answer}\n\nFACT:\n{fact}",
                    },
                ],
                max_completion_tokens=8,
                temperature=0,
            )
            verdict = (response.choices[0].message.content or "").strip().upper()
            if verdict.startswith("YES"):
                matched.append(fact)
            else:
                logger.debug("Judge: fact '%s' NOT found in answer.", fact)
        except Exception as exc:
            logger.warning("Judge call failed for fact '%s': %s", fact, exc)
            # Fall through — fact counts as missed

    missed = [f for f in expected_facts if f not in matched]
    recall = len(matched) / len(expected_facts)
    return recall, matched, missed


def _get_judge_client() -> tuple[AzureOpenAI, str] | None:
    """Create an AzureOpenAI client for the LLM-as-judge (GPT-5-nano).

    Returns ``None`` if credentials cannot be loaded.
    """
    try:
        creds = load_azure_generation_config(
            section="OpenAI2",
            model_key="GPT-5-nano",
        )
        client = AzureOpenAI(
            azure_endpoint=creds["endpoint"],
            api_key=creds["api_key"],
            api_version=creds["api_version"],
        )
        return client, creds["model"]
    except Exception as exc:
        logger.warning("Could not load judge credentials: %s", exc)
        return None


def _doc_precision(
    cited: list[str],
    expected: list[str],
) -> float:
    """Fraction of cited doc IDs that are in expected."""
    if not cited:
        return 1.0  # vacuously true when nothing cited
    expected_set = set(expected)
    return sum(1 for d in cited if d in expected_set) / len(cited)


def _doc_recall(
    cited: list[str],
    expected: list[str],
) -> float:
    """Fraction of expected doc IDs that appear in cited."""
    if not expected:
        return 1.0  # vacuously true
    cited_set = set(cited)
    return sum(1 for d in expected if d in cited_set) / len(expected)


def _confidence_ge(actual: str, minimum: str) -> bool:
    """Check if actual confidence is ≥ minimum (ordinal)."""
    return _CONF_RANK.get(actual, 0) >= _CONF_RANK.get(minimum, 0)


# ── Core evaluation ─────────────────────────────────────────────


def evaluate_generation(
    golden_path: str | Path,
    retrieval_config: RetrievalConfig | None = None,
    generation_config: GenerationConfig | None = None,
    *,
    query_type_filter: str | None = None,
    semantic_facts: bool = False,
) -> GenEvalReport:
    """Run generation evaluation against a golden test set.

    Parameters
    ----------
    golden_path:
        Path to the golden JSON file
        (``tests/golden_generation.json``).
    retrieval_config:
        Retrieval configuration override.  Uses defaults when ``None``.
    generation_config:
        Generation configuration override.  Uses defaults when ``None``.
    query_type_filter:
        When set, only evaluate queries of this type.
    semantic_facts:
        When ``True``, use an LLM-as-judge (GPT-5-nano) for semantic
        fact matching instead of substring matching.

    Returns
    -------
    GenEvalReport
    """
    ret_cfg = retrieval_config or RetrievalConfig()
    gen_cfg = generation_config or GenerationConfig()
    golden = _load_golden(golden_path)

    # ── Optional LLM judge for semantic fact matching ───────────
    judge_info: tuple[AzureOpenAI, str] | None = None
    if semantic_facts:
        judge_info = _get_judge_client()
        if judge_info is None:
            logger.warning(
                "Semantic facts requested but judge unavailable — "
                "falling back to substring matching."
            )

    # Filter out comment entries and optionally by query type
    queries = [
        q for q in golden["queries"]
        if "query" in q
        and (query_type_filter is None or q.get("query_type") == query_type_filter)
    ]

    report = GenEvalReport(
        config={
            "model": gen_cfg.model,
            "temperature": gen_cfg.temperature,
            "max_output_tokens": gen_cfg.max_output_tokens,
            "reasoning_effort": gen_cfg.reasoning_effort,
            "reranker_type": ret_cfg.reranker_type,
            "top_k_broad": ret_cfg.top_k_broad,
            "top_k_final": ret_cfg.top_k_final,
            "max_context_tokens": ret_cfg.max_context_tokens,
            "golden_path": str(golden_path),
            "query_type_filter": query_type_filter,
            "semantic_facts": semantic_facts,
        },
    )

    total_t0 = time.perf_counter()

    for i, entry in enumerate(queries, 1):
        query = entry["query"]
        query_type = entry.get("query_type", "")
        expected_facts = entry.get("expected_facts", [])
        expected_doc_ids = entry.get("expected_doc_ids", [])
        expected_abstained = entry.get("expected_abstained", False)
        expected_partial = entry.get("expected_partial", False)
        min_confidence = entry.get("min_confidence", "LOW")
        notes = entry.get("notes", "")

        logger.info(
            "[%d/%d] %s  (%s)", i, len(queries), query, query_type,
        )

        # ── Run the pipeline ────────────────────────────────────
        result = generate(query, ret_cfg, gen_cfg)

        # ── Extract actuals ─────────────────────────────────────
        cited_doc_ids = list(dict.fromkeys(
            c.doc_id for c in result.citations if c.doc_id
        ))

        actual_conf = result.system_confidence.value
        model_conf = result.model_confidence.value

        # ── Compute per-query metrics ───────────────────────────
        if judge_info is not None:
            fr, matched, missed = _semantic_fact_recall(
                result.answer, expected_facts,
                judge_info[0], judge_info[1],
            )
        else:
            fr, matched, missed = _fact_recall(result.answer, expected_facts)
        dp = _doc_precision(cited_doc_ids, expected_doc_ids)
        dr = _doc_recall(cited_doc_ids, expected_doc_ids)
        abst_ok = result.abstained == expected_abstained
        part_ok = result.partial == expected_partial
        vfy_pass = (
            result.verification.all_citations_resolved
            and result.verification.abstention_consistent
            and not result.verification.contains_unmapped_citations
        )
        conf_ok = _confidence_ge(actual_conf, min_confidence)

        qr = GenQueryResult(
            query=query,
            query_type=query_type,
            notes=notes,
            # Expected
            expected_facts=expected_facts,
            expected_doc_ids=expected_doc_ids,
            expected_abstained=expected_abstained,
            expected_partial=expected_partial,
            min_confidence=min_confidence,
            # Actual
            answer=result.answer,
            actual_abstained=result.abstained,
            actual_partial=result.partial,
            actual_confidence=actual_conf,
            model_confidence=model_conf,
            strategy_used=result.strategy,
            cited_doc_ids=cited_doc_ids,
            num_claims=len(result.claims),
            elapsed_s=result.elapsed_s,
            total_elapsed_s=result.total_elapsed_s,
            error=result.error,
            # Metrics
            fact_recall=fr,
            abstention_correct=abst_ok,
            partial_correct=part_ok,
            citation_coverage=result.verification.citation_coverage_ratio,
            verification_pass=vfy_pass,
            confidence_aligned=conf_ok,
            doc_precision=dp,
            doc_recall=dr,
            matched_facts=matched,
            missed_facts=missed,
        )
        report.query_results.append(qr)

    report.total_elapsed_s = time.perf_counter() - total_t0

    # ── Aggregate per query type ────────────────────────────────
    type_order: list[str] = []
    type_buckets: dict[str, list[GenQueryResult]] = {}
    for qr in report.query_results:
        if qr.query_type not in type_buckets:
            type_order.append(qr.query_type)
            type_buckets[qr.query_type] = []
        type_buckets[qr.query_type].append(qr)

    for qt in type_order:
        report.type_aggregates.append(_aggregate(qt, type_buckets[qt]))

    report.overall = _aggregate("OVERALL", report.query_results)
    return report


# ── Internal helpers ────────────────────────────────────────────


def _load_golden(path: str | Path) -> dict[str, Any]:
    """Load and validate the golden test set JSON."""
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Golden test set not found: {p}")
    data = json.loads(p.read_text(encoding="utf-8"))
    if "queries" not in data or not isinstance(data["queries"], list):
        raise ValueError(
            f"Golden file must contain a 'queries' list: {p}"
        )
    return data


def _aggregate(
    label: str,
    results: list[GenQueryResult],
) -> GenTypeAggregate:
    """Compute mean metrics for a list of query results."""
    n = len(results)
    if n == 0:
        return GenTypeAggregate(query_type=label)
    return GenTypeAggregate(
        query_type=label,
        count=n,
        fact_recall=sum(qr.fact_recall for qr in results) / n,
        abstention_accuracy=sum(
            1 for qr in results if qr.abstention_correct
        ) / n,
        partial_accuracy=sum(
            1 for qr in results if qr.partial_correct
        ) / n,
        citation_coverage=sum(
            qr.citation_coverage for qr in results
        ) / n,
        verification_rate=sum(
            1 for qr in results if qr.verification_pass
        ) / n,
        confidence_alignment=sum(
            1 for qr in results if qr.confidence_aligned
        ) / n,
        doc_precision=sum(qr.doc_precision for qr in results) / n,
        doc_recall=sum(qr.doc_recall for qr in results) / n,
        error_rate=sum(1 for qr in results if qr.error) / n,
    )


def _agg_dict(ta: GenTypeAggregate) -> dict[str, Any]:
    """Serialise a GenTypeAggregate to a plain dict."""
    return {
        "query_type": ta.query_type,
        "count": ta.count,
        "fact_recall": round(ta.fact_recall, 4),
        "abstention_accuracy": round(ta.abstention_accuracy, 4),
        "partial_accuracy": round(ta.partial_accuracy, 4),
        "citation_coverage": round(ta.citation_coverage, 4),
        "verification_rate": round(ta.verification_rate, 4),
        "confidence_alignment": round(ta.confidence_alignment, 4),
        "doc_precision": round(ta.doc_precision, 4),
        "doc_recall": round(ta.doc_recall, 4),
        "error_rate": round(ta.error_rate, 4),
    }


def _qr_dict(qr: GenQueryResult) -> dict[str, Any]:
    """Serialise a GenQueryResult to a plain dict."""
    return {
        "query": qr.query,
        "query_type": qr.query_type,
        "answer": qr.answer,
        "strategy_used": qr.strategy_used,
        # Expected
        "expected_facts": qr.expected_facts,
        "expected_doc_ids": qr.expected_doc_ids,
        "expected_abstained": qr.expected_abstained,
        "expected_partial": qr.expected_partial,
        "min_confidence": qr.min_confidence,
        # Actual
        "actual_abstained": qr.actual_abstained,
        "actual_partial": qr.actual_partial,
        "actual_confidence": qr.actual_confidence,
        "model_confidence": qr.model_confidence,
        "cited_doc_ids": qr.cited_doc_ids,
        "num_claims": qr.num_claims,
        # Metrics
        "fact_recall": round(qr.fact_recall, 4),
        "abstention_correct": qr.abstention_correct,
        "partial_correct": qr.partial_correct,
        "citation_coverage": round(qr.citation_coverage, 4),
        "verification_pass": qr.verification_pass,
        "confidence_aligned": qr.confidence_aligned,
        "doc_precision": round(qr.doc_precision, 4),
        "doc_recall": round(qr.doc_recall, 4),
        "matched_facts": qr.matched_facts,
        "missed_facts": qr.missed_facts,
        "error": qr.error,
        "elapsed_s": round(qr.elapsed_s, 3),
        "total_elapsed_s": round(qr.total_elapsed_s, 3),
    }
