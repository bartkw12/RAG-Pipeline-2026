"""Deterministic structural-groundedness verifier for generation output.

Runs lightweight, rule-based checks on the LLM's structured response
to ensure citations are valid, claims are backed by evidence, and
abstention flags are internally consistent.  No LLM calls — purely
deterministic.
"""

from __future__ import annotations

import logging
from typing import Any

from .models import Citation, VerificationResult

logger = logging.getLogger(__name__)


def verify_generation(
    raw_response: dict[str, Any],
    source_manifest: list[dict[str, Any]],
    citations: list[Citation],
) -> VerificationResult:
    """Run structural groundedness checks on a generation result.

    Parameters
    ----------
    raw_response:
        Parsed JSON from the LLM's structured output (must contain
        ``claims``, ``abstained``, etc.).
    source_manifest:
        The manifest list from ``build_source_manifest()`` — defines
        which source IDs are valid.
    citations:
        Resolved ``Citation`` objects from ``resolve_citations()``.

    Returns
    -------
    VerificationResult
        Flags and metrics describing the structural soundness of the
        generation.
    """
    claims = raw_response.get("claims", [])
    abstained = raw_response.get("abstained", False)

    valid_source_ids: set[int] = {
        entry["source_id"] for entry in source_manifest
    }

    # ── Check 1: All citations resolved ─────────────────────────
    # Every source_id referenced in claims must exist in the manifest.
    unmapped: list[int] = []
    for claim in claims:
        for sid in claim.get("source_ids", []):
            if sid not in valid_source_ids:
                unmapped.append(sid)
    # Deduplicate while preserving order.
    seen: set[int] = set()
    unmapped_unique: list[int] = []
    for sid in unmapped:
        if sid not in seen:
            seen.add(sid)
            unmapped_unique.append(sid)

    all_citations_resolved = len(unmapped_unique) == 0
    contains_unmapped = len(unmapped_unique) > 0

    # ── Check 2: All claims cited ───────────────────────────────
    # Every claim must have at least one source_id.
    uncited_indices: list[int] = [
        i for i, claim in enumerate(claims)
        if not claim.get("source_ids")
    ]
    all_claims_cited = len(uncited_indices) == 0

    # ── Check 3: Abstention consistency ─────────────────────────
    # If abstained, claims should be empty.
    # If not abstained, claims should be non-empty.
    if abstained:
        abstention_consistent = len(claims) == 0
    else:
        abstention_consistent = len(claims) > 0

    # ── Check 4: Citation coverage ratio ────────────────────────
    if claims:
        cited_count = sum(
            1 for claim in claims if claim.get("source_ids")
        )
        coverage = cited_count / len(claims)
    else:
        coverage = 1.0  # vacuously true when no claims

    # ── Assemble result ─────────────────────────────────────────
    result = VerificationResult(
        all_citations_resolved=all_citations_resolved,
        all_claims_cited=all_claims_cited,
        contains_unmapped_citations=contains_unmapped,
        abstention_consistent=abstention_consistent,
        citation_coverage_ratio=coverage,
        unmapped_source_ids=unmapped_unique,
        uncited_claim_indices=uncited_indices,
    )

    # ── Log summary ─────────────────────────────────────────────
    if contains_unmapped:
        logger.warning(
            "Unmapped source IDs in generation: %s", unmapped_unique,
        )
    if uncited_indices:
        logger.warning(
            "Uncited claims at indices: %s", uncited_indices,
        )
    if not abstention_consistent:
        logger.warning(
            "Abstention inconsistency: abstained=%s but %d claims present.",
            abstained, len(claims),
        )

    logger.info(
        "Verification: citations_resolved=%s, claims_cited=%s, "
        "coverage=%.0f%%, abstention_consistent=%s",
        all_citations_resolved, all_claims_cited,
        coverage * 100, abstention_consistent,
    )

    return result
