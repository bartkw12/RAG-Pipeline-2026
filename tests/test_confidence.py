"""Unit tests for the hybrid confidence scoring module.

Tests ``compute_system_confidence()`` with controlled
``ScoredChunk`` and ``VerificationResult`` inputs — no API calls.
"""

from __future__ import annotations

import pytest

from src.generation.confidence import compute_system_confidence
from src.generation.models import ConfidenceLevel, VerificationResult

from conftest import make_scored_chunk, make_verification_result


# ════════════════════════════════════════════════════════════════
# Helper — run confidence with shorthand args
# ════════════════════════════════════════════════════════════════


def _compute(
    model: ConfidenceLevel = ConfidenceLevel.HIGH,
    vector_scores: list[float] | None = None,
    coverage: float = 1.0,
    all_resolved: bool = True,
    abstention_consistent: bool = True,
    contains_unmapped: bool = False,
) -> tuple[ConfidenceLevel, dict[str, float]]:
    """Shorthand wrapper around ``compute_system_confidence``."""
    chunks = [make_scored_chunk(vs) for vs in (vector_scores or [0.90])]
    verification = make_verification_result(
        citation_coverage_ratio=coverage,
        all_citations_resolved=all_resolved,
        abstention_consistent=abstention_consistent,
        contains_unmapped_citations=contains_unmapped,
    )
    return compute_system_confidence(model, chunks, verification)


# ════════════════════════════════════════════════════════════════
# 1. All signals high → HIGH
# ════════════════════════════════════════════════════════════════


class TestAllSignalsHigh:
    """Best-case: high retrieval, full coverage, verification passes."""

    def test_system_confidence_high(self):
        final, components = _compute(
            model=ConfidenceLevel.HIGH,
            vector_scores=[0.90],
            coverage=1.0,
        )
        assert final == ConfidenceLevel.HIGH

    def test_retrieval_support_1_0(self):
        _, components = _compute(vector_scores=[0.90])
        assert components["retrieval_support"] == 1.0

    def test_citation_coverage_1_0(self):
        _, components = _compute(coverage=1.0)
        assert components["citation_coverage"] == 1.0

    def test_verification_pass_1_0(self):
        _, components = _compute()
        assert components["verification_pass"] == 1.0


# ════════════════════════════════════════════════════════════════
# 2. Low retrieval score → LOW
# ════════════════════════════════════════════════════════════════


class TestLowRetrieval:
    """vector_score < 0.65 → retrieval_support = 0.4 → forces LOW."""

    def test_low_retrieval_support(self):
        _, components = _compute(vector_scores=[0.50])
        assert components["retrieval_support"] == 0.4

    def test_system_low(self):
        final, _ = _compute(
            model=ConfidenceLevel.HIGH,
            vector_scores=[0.50],
        )
        assert final == ConfidenceLevel.LOW

    def test_very_low_vector_score(self):
        final, components = _compute(vector_scores=[0.10])
        assert components["retrieval_support"] == 0.4
        assert final == ConfidenceLevel.LOW


# ════════════════════════════════════════════════════════════════
# 3. Medium retrieval score → retrieval_support = 0.65 or 0.85
# ════════════════════════════════════════════════════════════════


class TestMediumRetrieval:
    """vector_score in [0.65, 0.85) → retrieval_support = 0.65 or 0.85."""

    def test_retrieval_support_0_85(self):
        _, components = _compute(vector_scores=[0.75])
        assert components["retrieval_support"] == 0.85

    def test_boundary_at_0_65(self):
        _, components = _compute(vector_scores=[0.65])
        assert components["retrieval_support"] == 0.65

    def test_boundary_at_0_70(self):
        _, components = _compute(vector_scores=[0.70])
        assert components["retrieval_support"] == 0.65

    def test_boundary_below_0_85(self):
        _, components = _compute(vector_scores=[0.84])
        assert components["retrieval_support"] == 0.85

    def test_system_high_with_0_85_support(self):
        """0.85 ≥ 0.75 threshold → system can be HIGH."""
        final, _ = _compute(
            model=ConfidenceLevel.HIGH,
            vector_scores=[0.75],
        )
        assert final == ConfidenceLevel.HIGH


# ════════════════════════════════════════════════════════════════
# 4. Verification failure → verification_pass = 0.0
# ════════════════════════════════════════════════════════════════


class TestVerificationFailure:
    """Any verification flag failure → verification_pass = 0.0 → LOW."""

    def test_unmapped_citations(self):
        final, components = _compute(
            contains_unmapped=True,
            all_resolved=False,
        )
        assert components["verification_pass"] == 0.0
        assert final == ConfidenceLevel.LOW

    def test_abstention_inconsistent(self):
        final, components = _compute(abstention_consistent=False)
        assert components["verification_pass"] == 0.0
        assert final == ConfidenceLevel.LOW

    def test_all_resolved_false_alone(self):
        _, components = _compute(all_resolved=False)
        assert components["verification_pass"] == 0.0


# ════════════════════════════════════════════════════════════════
# 5. Partial citation coverage
# ════════════════════════════════════════════════════════════════


class TestPartialCoverage:
    """Coverage < 1.0 influences the system confidence level."""

    def test_half_coverage_caps_at_medium(self):
        """coverage=0.5 is at the MEDIUM boundary (all ≥ 0.5)."""
        final, components = _compute(
            model=ConfidenceLevel.HIGH,
            vector_scores=[0.90],     # retrieval_support = 1.0
            coverage=0.5,             # citation_coverage = 0.5
        )
        assert components["citation_coverage"] == 0.5
        # 0.5 < 0.75 threshold → cannot be HIGH
        assert final != ConfidenceLevel.HIGH

    def test_zero_coverage_forces_low(self):
        final, _ = _compute(coverage=0.0)
        assert final == ConfidenceLevel.LOW


# ════════════════════════════════════════════════════════════════
# 6. Conservative min(model, system) logic
# ════════════════════════════════════════════════════════════════


class TestConservativeMin:
    """Final confidence = min(model, system)."""

    def test_model_low_overrides_system_high(self):
        """system=HIGH but model=LOW → final=LOW."""
        final, _ = _compute(
            model=ConfidenceLevel.LOW,
            vector_scores=[0.90],
            coverage=1.0,
        )
        assert final == ConfidenceLevel.LOW

    def test_system_low_overrides_model_high(self):
        """model=HIGH but system=LOW (bad retrieval) → final=LOW."""
        final, _ = _compute(
            model=ConfidenceLevel.HIGH,
            vector_scores=[0.50],  # forces system LOW
        )
        assert final == ConfidenceLevel.LOW

    def test_model_medium_system_high(self):
        """model=MEDIUM, system=HIGH → final=MEDIUM."""
        final, _ = _compute(
            model=ConfidenceLevel.MEDIUM,
            vector_scores=[0.90],
            coverage=1.0,
        )
        assert final == ConfidenceLevel.MEDIUM

    def test_both_medium(self):
        final, _ = _compute(
            model=ConfidenceLevel.MEDIUM,
            vector_scores=[0.75],  # retrieval_support = 0.85
            coverage=0.8,
        )
        # retrieval=0.85, coverage=0.8, verify=1.0 → all ≥ 0.75 → HIGH
        # min(MEDIUM, HIGH) = MEDIUM
        assert final == ConfidenceLevel.MEDIUM


# ════════════════════════════════════════════════════════════════
# 7. Edge cases
# ════════════════════════════════════════════════════════════════


class TestEdgeCases:
    """Boundary and degenerate inputs."""

    def test_empty_scored_chunks(self):
        """No chunks → best_vector = 0.0 → retrieval_support = 0.4."""
        verification = make_verification_result()
        final, components = compute_system_confidence(
            ConfidenceLevel.HIGH, [], verification,
        )
        assert components["retrieval_support"] == 0.4
        assert final == ConfidenceLevel.LOW

    def test_boundary_at_0_85(self):
        """vector_score == 0.85 → retrieval_support = 1.0 (≥ threshold)."""
        _, components = _compute(vector_scores=[0.85])
        assert components["retrieval_support"] == 1.0

    def test_max_of_multiple_chunks(self):
        """Best score from multiple chunks is used."""
        _, components = _compute(vector_scores=[0.50, 0.60, 0.90])
        assert components["retrieval_support"] == 1.0

    def test_components_dict_keys(self):
        """Result dict always has exactly 3 component keys."""
        _, components = _compute()
        assert set(components.keys()) == {
            "retrieval_support",
            "citation_coverage",
            "verification_pass",
        }
