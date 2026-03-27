"""Hybrid confidence scoring — combines retrieval and generation signals.

Computes a system-level confidence score from retrieval quality,
citation coverage, and structural verification results.  The final
confidence is the conservative minimum of the model's self-assessment
and this system-computed score.
"""

from __future__ import annotations

import logging

from .models import ConfidenceLevel, VerificationResult

logger = logging.getLogger(__name__)


def compute_system_confidence(
    model_confidence: ConfidenceLevel,
    scored_chunks: list,
    verification: VerificationResult,
) -> tuple[ConfidenceLevel, dict[str, float]]:
    """Compute hybrid confidence from retrieval + verification signals.

    Parameters
    ----------
    model_confidence:
        The LLM's self-assessed confidence from the structured output.
    scored_chunks:
        ``RetrievalResult.scored_chunks`` — used to extract the best
        vector similarity score.  Each item must have a
        ``vector_score`` attribute.
    verification:
        The ``VerificationResult`` from ``verify_generation()``.

    Returns
    -------
    tuple[ConfidenceLevel, dict[str, float]]
        ``(system_confidence, components)`` where *components* is a
        dict with ``retrieval_support``, ``citation_coverage``, and
        ``verification_pass`` scores in [0, 1].
    """
    # ── Component 1: Retrieval support ──────────────────────────
    # Max vector similarity score from the retrieved chunks.
    if scored_chunks:
        best_vector = max(
            (getattr(sc, "vector_score", 0.0) for sc in scored_chunks),
            default=0.0,
        )
    else:
        best_vector = 0.0

    if best_vector >= 0.85:
        retrieval_support = 1.0
    elif best_vector >= 0.75:
        retrieval_support = 0.85
    elif best_vector >= 0.65:
        retrieval_support = 0.65
    else:
        retrieval_support = 0.4

    # ── Component 2: Citation coverage ──────────────────────────
    citation_coverage = verification.citation_coverage_ratio

    # ── Component 3: Verification pass ──────────────────────────
    verification_pass = 1.0 if (
        verification.all_citations_resolved
        and verification.abstention_consistent
        and not verification.contains_unmapped_citations
    ) else 0.0

    # ── Aggregate → ConfidenceLevel ─────────────────────────────
    components = {
        "retrieval_support": round(retrieval_support, 2),
        "citation_coverage": round(citation_coverage, 4),
        "verification_pass": verification_pass,
    }

    if all(v >= 0.75 for v in components.values()):
        system = ConfidenceLevel.HIGH
    elif all(v >= 0.5 for v in components.values()):
        system = ConfidenceLevel.MEDIUM
    else:
        system = ConfidenceLevel.LOW

    # Conservative: final = min(model, system).
    ranking = {ConfidenceLevel.LOW: 0, ConfidenceLevel.MEDIUM: 1, ConfidenceLevel.HIGH: 2}
    if ranking[system] <= ranking[model_confidence]:
        final = system
    else:
        final = model_confidence

    logger.info(
        "Confidence: system=%s, model=%s → final=%s  "
        "(retrieval=%.2f, coverage=%.2f, verify=%.0f)",
        system.value, model_confidence.value, final.value,
        retrieval_support, citation_coverage, verification_pass,
    )

    return final, components
