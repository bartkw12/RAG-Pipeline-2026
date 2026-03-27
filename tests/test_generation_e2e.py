"""End-to-end generation pipeline tests — live Azure OpenAI calls.

Every test in this module is marked ``@pytest.mark.e2e`` so the suite
can be skipped in CI or offline environments::

    # Run only E2E tests
    python -m pytest tests/test_generation_e2e.py -m e2e

    # Skip E2E tests
    python -m pytest tests/ -m "not e2e"

These tests validate that the generation pipeline produces grounded,
structurally-sound results against the real ingested DIM-V / PAM corpus.
"""

from __future__ import annotations

import pytest

from src.generation.models import (
    ConfidenceLevel,
    GenerationConfig,
    GenerationResult,
    VerificationResult,
)
from src.generation.pipeline import generate, generate_from_context
from src.retrieval.models import (
    ContextWindow,
    QueryAnalysis,
    QueryStrategy,
    RetrievalConfig,
    RetrievalResult,
)


# ── Shared config ───────────────────────────────────────────────

_GEN_CFG = GenerationConfig(model="gpt-5-mini")


# ── Helpers ─────────────────────────────────────────────────────


def _assert_is_valid_result(result: GenerationResult) -> None:
    """Basic structural assertions that apply to every result."""
    assert isinstance(result, GenerationResult)
    assert isinstance(result.answer, str)
    assert len(result.answer) > 0
    assert result.error == ""
    assert isinstance(result.verification, VerificationResult)
    assert isinstance(result.model_confidence, ConfidenceLevel)
    assert isinstance(result.system_confidence, ConfidenceLevel)


# ════════════════════════════════════════════════════════════════
#  Test cases
# ════════════════════════════════════════════════════════════════


@pytest.mark.e2e
class TestExactLookupGroundedAnswer:
    """Query a known test case ID and verify the answer is grounded."""

    QUERY = "What are the results for FVTR_HVT_01?"

    def test_does_not_abstain(self):
        result = generate(self.QUERY, generation_config=_GEN_CFG)
        _assert_is_valid_result(result)
        assert result.abstained is False, "Expected a substantive answer"

    def test_has_claims(self):
        result = generate(self.QUERY, generation_config=_GEN_CFG)
        assert len(result.claims) > 0, "Expected at least one claim"

    def test_all_citations_resolved(self):
        result = generate(self.QUERY, generation_config=_GEN_CFG)
        v = result.verification
        assert v.all_citations_resolved, (
            f"Unmapped source IDs: {v.unmapped_source_ids}"
        )

    def test_citation_coverage_full(self):
        result = generate(self.QUERY, generation_config=_GEN_CFG)
        assert result.verification.citation_coverage_ratio == 1.0

    def test_answer_mentions_expected_terms(self):
        result = generate(self.QUERY, generation_config=_GEN_CFG)
        lower = result.answer.lower()
        assert any(
            term in lower
            for term in ["fvtr_hvt_01", "hvt_01", "high voltage", "isolation"]
        ), f"Answer does not mention expected terms: {result.answer[:200]}"


@pytest.mark.e2e
class TestEmptyContextAbstains:
    """When context is empty the pipeline should abstain without an LLM call."""

    def test_abstains(self):
        empty_retrieval = RetrievalResult(
            query_analysis=QueryAnalysis(
                original_query="Unrelated question",
                strategy=QueryStrategy.UNCONSTRAINED,
            ),
            scored_chunks=[],
            context=ContextWindow(sections=[], total_tokens=0),
            strategy="unconstrained",
        )
        result = generate_from_context(
            "Unrelated question about nothing in the corpus",
            empty_retrieval,
            _GEN_CFG,
        )
        assert result.abstained is True

    def test_no_claims(self):
        empty_retrieval = RetrievalResult(
            query_analysis=QueryAnalysis(
                original_query="Unrelated question",
                strategy=QueryStrategy.UNCONSTRAINED,
            ),
            scored_chunks=[],
            context=ContextWindow(sections=[], total_tokens=0),
            strategy="unconstrained",
        )
        result = generate_from_context(
            "Unrelated question about nothing in the corpus",
            empty_retrieval,
            _GEN_CFG,
        )
        assert result.claims == []

    def test_answer_indicates_abstention(self):
        empty_retrieval = RetrievalResult(
            query_analysis=QueryAnalysis(
                original_query="Unrelated question",
                strategy=QueryStrategy.UNCONSTRAINED,
            ),
            scored_chunks=[],
            context=ContextWindow(sections=[], total_tokens=0),
            strategy="unconstrained",
        )
        result = generate_from_context(
            "Unrelated question about nothing in the corpus",
            empty_retrieval,
            _GEN_CFG,
        )
        assert "don't have enough information" in result.answer.lower()


@pytest.mark.e2e
class TestVerificationPassesOnRealResponse:
    """A well-scoped query should produce a fully-verified response."""

    QUERY = "What are the DIM-V high voltage test results?"

    def test_all_citations_resolved(self):
        result = generate(self.QUERY, generation_config=_GEN_CFG)
        _assert_is_valid_result(result)
        assert result.verification.all_citations_resolved

    def test_abstention_consistent(self):
        result = generate(self.QUERY, generation_config=_GEN_CFG)
        assert result.verification.abstention_consistent

    def test_citation_coverage_high(self):
        result = generate(self.QUERY, generation_config=_GEN_CFG)
        assert result.verification.citation_coverage_ratio >= 0.8, (
            f"Coverage too low: {result.verification.citation_coverage_ratio}"
        )


@pytest.mark.e2e
class TestConfidenceNotHighOnWeakQuery:
    """A vague or off-topic query should not yield HIGH system confidence."""

    QUERY = "What is the color of the DIM-V module?"

    def test_system_confidence_not_high(self):
        result = generate(self.QUERY, generation_config=_GEN_CFG)
        # The documents don't describe the module's colour, so
        # the pipeline should either abstain or assign ≤ MEDIUM.
        assert (
            result.system_confidence != ConfidenceLevel.HIGH
            or result.abstained
        ), (
            f"Expected LOW/MEDIUM confidence or abstention, "
            f"got {result.system_confidence.value} with "
            f"abstained={result.abstained}"
        )


@pytest.mark.e2e
class TestResultSerialisation:
    """Verify to_dict() and to_text() work on a real pipeline result."""

    QUERY = "What test cases are in the DIM-V FVTR?"

    def test_to_dict_has_expected_keys(self):
        result = generate(self.QUERY, generation_config=_GEN_CFG)
        _assert_is_valid_result(result)
        d = result.to_dict()
        assert isinstance(d, dict)
        for key in (
            "answer",
            "claims",
            "citations",
            "abstained",
            "model_confidence",
            "system_confidence",
            "verification",
            "query",
        ):
            assert key in d, f"Missing key in to_dict(): {key}"

    def test_to_text_contains_markers(self):
        result = generate(self.QUERY, generation_config=_GEN_CFG)
        text = result.to_text()
        assert isinstance(text, str)
        assert "Answer:" in text
        assert "Confidence:" in text

    def test_to_dict_does_not_raise(self):
        result = generate(self.QUERY, generation_config=_GEN_CFG)
        # Should not throw regardless of content
        d = result.to_dict()
        assert d is not None

    def test_to_text_does_not_raise(self):
        result = generate(self.QUERY, generation_config=_GEN_CFG)
        text = result.to_text()
        assert text is not None
