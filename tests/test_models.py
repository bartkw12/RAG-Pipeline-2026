"""Unit tests for GenerationResult serialisation (to_dict / to_text).

Validates that ``to_dict()`` produces all expected keys with correct
types and that ``to_text()`` renders the required sections.
"""

from __future__ import annotations

import pytest

from src.generation.models import (
    Citation,
    Claim,
    ConfidenceLevel,
    GenerationResult,
    VerificationResult,
)

from conftest import (
    DOC_ID_DIMV_FVTR,
    DOC_ID_PAM_FVTR,
    _CHUNK_A,
    _CHUNK_B,
)


# ════════════════════════════════════════════════════════════════
# Fixtures — pre-built GenerationResult instances
# ════════════════════════════════════════════════════════════════


@pytest.fixture()
def full_result() -> GenerationResult:
    """A fully-populated GenerationResult with 2 citations/claims."""
    return GenerationResult(
        answer="The DIM-V mechanical test passed [Source 1]. PAM tests also passed [Source 2].",
        claims=[
            Claim(statement="DIM-V mechanical test passed.", source_ids=[1]),
            Claim(statement="PAM tests passed.", source_ids=[2]),
        ],
        citations=[
            Citation(
                source_id=1,
                label="FVTR DIM-V, §5 — Mechanical tests",
                chunk_id=_CHUNK_A,
                doc_id=DOC_ID_DIMV_FVTR,
                doc_type="FVTR",
                section_heading="Mechanical tests",
                section_number="5",
                quoted_text="Test passed.",
            ),
            Citation(
                source_id=2,
                label="FVTR PAM, §4 — Functional tests",
                chunk_id=_CHUNK_B,
                doc_id=DOC_ID_PAM_FVTR,
                doc_type="FVTR",
                section_heading="Functional tests",
                section_number="4",
            ),
        ],
        abstained=False,
        partial=False,
        unanswered_aspects=[],
        contradictions_noted=False,
        model_confidence=ConfidenceLevel.HIGH,
        model_confidence_reasoning="Direct evidence from FVTR documents.",
        system_confidence=ConfidenceLevel.HIGH,
        confidence_components={
            "retrieval_support": 1.0,
            "citation_coverage": 1.0,
            "verification_pass": 1.0,
        },
        verification=VerificationResult(
            all_citations_resolved=True,
            all_claims_cited=True,
            contains_unmapped_citations=False,
            abstention_consistent=True,
            citation_coverage_ratio=1.0,
        ),
        query="What are the DIM-V mechanical test results?",
        strategy="exact_lookup",
        model="gpt-5-mini",
        usage={"prompt_tokens": 500, "completion_tokens": 200, "total_tokens": 700},
        elapsed_s=1.234,
        total_elapsed_s=1.567,
    )


@pytest.fixture()
def abstained_result() -> GenerationResult:
    """An abstained GenerationResult with no citations or claims."""
    return GenerationResult(
        answer="I don't have enough information to answer.",
        claims=[],
        citations=[],
        abstained=True,
        model_confidence=ConfidenceLevel.LOW,
        system_confidence=ConfidenceLevel.LOW,
        verification=VerificationResult(abstention_consistent=True),
        query="What is the color of the module?",
        strategy="unconstrained",
        model="gpt-5-mini",
        elapsed_s=0.8,
        total_elapsed_s=1.0,
    )


@pytest.fixture()
def partial_result() -> GenerationResult:
    """A partial-answer GenerationResult with unanswered aspects."""
    return GenerationResult(
        answer="The DIM-V module has 32 digital inputs [Source 1].",
        claims=[
            Claim(statement="DIM-V has 32 digital inputs.", source_ids=[1]),
        ],
        citations=[
            Citation(
                source_id=1,
                label="HwIRS DIM-V, §3",
                chunk_id=_CHUNK_A,
                doc_id=DOC_ID_DIMV_FVTR,
                doc_type="HwIRS",
            ),
        ],
        abstained=False,
        partial=True,
        unanswered_aspects=["voltage levels", "timing characteristics"],
        model_confidence=ConfidenceLevel.MEDIUM,
        system_confidence=ConfidenceLevel.MEDIUM,
        verification=VerificationResult(
            all_citations_resolved=True,
            all_claims_cited=True,
            abstention_consistent=True,
            citation_coverage_ratio=1.0,
        ),
        query="What are the DIM-V input specs?",
        strategy="scoped_semantic",
        model="gpt-5-mini",
        elapsed_s=1.0,
        total_elapsed_s=1.5,
    )


# ════════════════════════════════════════════════════════════════
# 1. to_dict() — key presence and types
# ════════════════════════════════════════════════════════════════


class TestToDict:
    """Verify ``to_dict()`` produces correct structure."""

    EXPECTED_TOP_KEYS = {
        "answer",
        "claims",
        "citations",
        "abstained",
        "partial",
        "unanswered_aspects",
        "contradictions_noted",
        "model_confidence",
        "model_confidence_reasoning",
        "system_confidence",
        "confidence_components",
        "verification",
        "query",
        "strategy",
        "model",
        "usage",
        "elapsed_s",
        "total_elapsed_s",
        "error",
    }

    EXPECTED_VERIFICATION_KEYS = {
        "all_citations_resolved",
        "all_claims_cited",
        "contains_unmapped_citations",
        "abstention_consistent",
        "citation_coverage_ratio",
        "unmapped_source_ids",
        "uncited_claim_indices",
    }

    def test_all_top_level_keys_present(self, full_result):
        d = full_result.to_dict()
        assert set(d.keys()) == self.EXPECTED_TOP_KEYS

    def test_verification_keys_present(self, full_result):
        d = full_result.to_dict()
        assert set(d["verification"].keys()) == self.EXPECTED_VERIFICATION_KEYS

    def test_confidence_serialised_as_string(self, full_result):
        d = full_result.to_dict()
        assert d["model_confidence"] == "HIGH"
        assert d["system_confidence"] == "HIGH"

    def test_claims_structure(self, full_result):
        d = full_result.to_dict()
        assert len(d["claims"]) == 2
        for claim in d["claims"]:
            assert "statement" in claim
            assert "source_ids" in claim
            assert isinstance(claim["source_ids"], list)

    def test_citations_structure(self, full_result):
        d = full_result.to_dict()
        assert len(d["citations"]) == 2
        for citation in d["citations"]:
            assert "source_id" in citation
            assert "label" in citation
            assert "chunk_id" in citation
            assert "doc_id" in citation

    def test_elapsed_rounded(self, full_result):
        d = full_result.to_dict()
        # to_dict rounds to 3 decimal places
        assert d["elapsed_s"] == 1.234
        assert d["total_elapsed_s"] == 1.567

    def test_coverage_ratio_rounded(self, full_result):
        d = full_result.to_dict()
        # to_dict rounds coverage to 4 decimal places
        assert isinstance(d["verification"]["citation_coverage_ratio"], float)

    def test_abstained_result_dict(self, abstained_result):
        d = abstained_result.to_dict()
        assert d["abstained"] is True
        assert d["claims"] == []
        assert d["citations"] == []
        assert d["model_confidence"] == "LOW"

    def test_usage_dict(self, full_result):
        d = full_result.to_dict()
        assert d["usage"]["total_tokens"] == 700

    def test_error_empty_on_success(self, full_result):
        d = full_result.to_dict()
        assert d["error"] == ""


# ════════════════════════════════════════════════════════════════
# 2. to_text() — human-readable rendering
# ════════════════════════════════════════════════════════════════


class TestToText:
    """Verify ``to_text()`` renders key sections."""

    def test_contains_answer_header(self, full_result):
        text = full_result.to_text()
        assert "Answer:" in text

    def test_contains_answer_text(self, full_result):
        text = full_result.to_text()
        assert "DIM-V mechanical test passed" in text

    def test_contains_confidence(self, full_result):
        text = full_result.to_text()
        assert "Confidence:" in text
        assert "HIGH (system)" in text
        assert "HIGH (model)" in text

    def test_contains_sources_section(self, full_result):
        text = full_result.to_text()
        assert "Sources:" in text
        assert "[1]" in text
        assert "[2]" in text

    def test_contains_verification_checkmark(self, full_result):
        text = full_result.to_text()
        assert "✓ Verification:" in text
        assert "100% claims cited" in text

    def test_contains_timings(self, full_result):
        text = full_result.to_text()
        assert "Retrieval:" in text
        assert "Generation:" in text
        assert "Total:" in text

    def test_quoted_text_shown(self, full_result):
        text = full_result.to_text()
        assert '"Test passed."' in text

    def test_abstention_no_sources(self, abstained_result):
        text = abstained_result.to_text()
        assert "Sources:" not in text

    def test_abstention_shows_answer(self, abstained_result):
        text = abstained_result.to_text()
        assert "don't have enough information" in text

    def test_partial_shows_unanswered(self, partial_result):
        text = partial_result.to_text()
        assert "Not covered by available evidence:" in text
        assert "voltage levels" in text
        assert "timing characteristics" in text

    def test_contradictions_warning(self):
        result = GenerationResult(
            answer="Conflicting data.",
            contradictions_noted=True,
            elapsed_s=0.5,
            total_elapsed_s=1.0,
        )
        text = result.to_text()
        assert "Contradictions" in text

    def test_verification_issues_shown(self):
        result = GenerationResult(
            answer="Answer text.",
            verification=VerificationResult(
                all_citations_resolved=False,
                unmapped_source_ids=[99],
            ),
            elapsed_s=0.5,
            total_elapsed_s=1.0,
        )
        text = result.to_text()
        assert "⚠ Verification issues:" in text
        assert "unmapped sources" in text

    def test_empty_answer_shows_placeholder(self):
        result = GenerationResult(elapsed_s=0.0, total_elapsed_s=0.0)
        text = result.to_text()
        assert "(no answer)" in text
