"""Shared pytest fixtures for generation and RAG pipeline tests.

Provides synthetic data structures that mirror real pipeline output
without requiring API calls or disk access.  Every fixture is
deterministic and self-contained.
"""

from __future__ import annotations

import pytest

from src.generation.models import (
    Citation,
    Claim,
    ConfidenceLevel,
    GenerationConfig,
    GenerationResult,
    VerificationResult,
)
from src.retrieval.models import (
    ContextSection,
    ContextWindow,
    QueryAnalysis,
    QueryStrategy,
    RetrievalConfig,
    RetrievalResult,
    ScoredChunk,
)

# ── Real document IDs from the ingested corpus ──────────────────

DOC_ID_DIMV_FVTR = (
    "8250cded0140bf39e80a93bc21040baa473c3d5e672edd37ddd83be9a482c2a9"
)
DOC_ID_PAM_FVTR = (
    "f4b48797a6a272a5a74f96ceb5a2133338a7606c541478a97f3d1528b0558143"
)
DOC_ID_DIM_HWIRS = (
    "eceebed4d7a903ef3500ea30b96aaf3571a000156497b07c98aef6c410ae942f"
)

# ── Synthetic chunk IDs (deterministic, not real hashes) ────────

_CHUNK_A = "aaaa" * 16  # 64-char hex
_CHUNK_B = "bbbb" * 16
_CHUNK_C = "cccc" * 16


# ── pytest markers ──────────────────────────────────────────────


def pytest_configure(config: pytest.Config) -> None:
    """Register custom markers so pytest does not emit warnings."""
    config.addinivalue_line(
        "markers",
        "e2e: marks tests as end-to-end (require live Azure credentials)",
    )


# ════════════════════════════════════════════════════════════════
# Source manifest fixtures
# ════════════════════════════════════════════════════════════════


@pytest.fixture()
def sample_source_manifest() -> list[dict]:
    """Three-entry source manifest matching the three ingested docs."""
    return [
        {
            "source_id": 1,
            "label": "FVTR DIM-V, §5 — Verification of mechanical tests",
            "doc_id": DOC_ID_DIMV_FVTR,
            "doc_type": "FVTR",
            "section_heading": "Verification of mechanical tests",
            "section_number": "5",
            "chunk_ids": [_CHUNK_A],
            "revision": "07",
            "revision_date": "2021-03-02",
        },
        {
            "source_id": 2,
            "label": "FVTR PAM, §4 — Functional test overview",
            "doc_id": DOC_ID_PAM_FVTR,
            "doc_type": "FVTR",
            "section_heading": "Functional test overview",
            "section_number": "4",
            "chunk_ids": [_CHUNK_B],
            "revision": "07",
            "revision_date": "2021-03-08",
        },
        {
            "source_id": 3,
            "label": "HwIRS DIM-V, §3 — Hardware requirements",
            "doc_id": DOC_ID_DIM_HWIRS,
            "doc_type": "HwIRS",
            "section_heading": "Hardware requirements",
            "section_number": "3",
            "chunk_ids": [_CHUNK_C],
            "revision": "005",
            "revision_date": "2021-03-05",
        },
    ]


# ════════════════════════════════════════════════════════════════
# Raw LLM response fixtures (structured output dicts)
# ════════════════════════════════════════════════════════════════


@pytest.fixture()
def sample_raw_response_clean() -> dict:
    """Well-formed LLM response — 2 cited claims, no issues."""
    return {
        "answer": (
            "The DIM-V mechanical tests (FVTR_MECH_01) verified "
            "labelling and assembly [Source 1]. The PAM functional "
            "tests confirmed correct operation [Source 2]."
        ),
        "claims": [
            {
                "statement": (
                    "DIM-V mechanical tests verified labelling "
                    "and assembly."
                ),
                "source_ids": [1],
            },
            {
                "statement": (
                    "PAM functional tests confirmed correct operation."
                ),
                "source_ids": [2],
            },
        ],
        "abstained": False,
        "partial": False,
        "unanswered_aspects": [],
        "contradictions_noted": False,
        "confidence": "HIGH",
        "confidence_reasoning": (
            "Direct evidence from both FVTR documents."
        ),
    }


@pytest.fixture()
def sample_raw_response_abstained() -> dict:
    """Abstained response — no useful evidence available."""
    return {
        "answer": (
            "I don't have enough information in the provided "
            "documents to answer this question."
        ),
        "claims": [],
        "abstained": True,
        "partial": False,
        "unanswered_aspects": [],
        "contradictions_noted": False,
        "confidence": "LOW",
        "confidence_reasoning": "No relevant evidence found.",
    }


@pytest.fixture()
def sample_raw_response_uncited() -> dict:
    """Response with one uncited claim (empty source_ids)."""
    return {
        "answer": (
            "The DIM-V module supports 32 digital inputs [Source 1]. "
            "It also uses 3.3V logic levels."
        ),
        "claims": [
            {
                "statement": "The DIM-V module supports 32 digital inputs.",
                "source_ids": [1],
            },
            {
                "statement": "It uses 3.3V logic levels.",
                "source_ids": [],  # ← uncited
            },
        ],
        "abstained": False,
        "partial": False,
        "unanswered_aspects": [],
        "contradictions_noted": False,
        "confidence": "MEDIUM",
        "confidence_reasoning": (
            "One claim lacks a source citation."
        ),
    }


@pytest.fixture()
def sample_raw_response_unmapped() -> dict:
    """Response citing a source_id that doesn't exist in the manifest."""
    return {
        "answer": (
            "The thermal test passed at 85°C [Source 1]. "
            "Additional data is in [Source 99]."
        ),
        "claims": [
            {
                "statement": "The thermal test passed at 85°C.",
                "source_ids": [1],
            },
            {
                "statement": "Additional data is available.",
                "source_ids": [99],  # ← not in manifest
            },
        ],
        "abstained": False,
        "partial": False,
        "unanswered_aspects": [],
        "contradictions_noted": False,
        "confidence": "MEDIUM",
        "confidence_reasoning": "One source reference could not be resolved.",
    }


# ════════════════════════════════════════════════════════════════
# Retrieval result fixtures
# ════════════════════════════════════════════════════════════════


def _make_scored_chunks() -> list[ScoredChunk]:
    """Three scored chunks with decreasing vector similarity."""
    return [
        ScoredChunk(
            chunk_id=_CHUNK_A,
            doc_id=DOC_ID_DIMV_FVTR,
            text=(
                "FVTR_MECH_01 — Labelling and assembly verification. "
                "Test passed for DIM-V module."
            ),
            chunk_type="test_case",
            tier=3,
            score=0.92,
            vector_score=0.90,
            bm25_score=0.85,
            rerank_score=0.92,
            metadata={
                "doc_type": "FVTR",
                "section_number": "5",
                "test_case_id": "FVTR_MECH_01",
            },
        ),
        ScoredChunk(
            chunk_id=_CHUNK_B,
            doc_id=DOC_ID_PAM_FVTR,
            text=(
                "FVTR_MECH_01 — Plugging and unplugging test. "
                "Verified for PAM module."
            ),
            chunk_type="test_case",
            tier=3,
            score=0.78,
            vector_score=0.75,
            bm25_score=0.70,
            rerank_score=0.78,
            metadata={
                "doc_type": "FVTR",
                "section_number": "4",
                "test_case_id": "FVTR_MECH_01",
            },
        ),
        ScoredChunk(
            chunk_id=_CHUNK_C,
            doc_id=DOC_ID_DIM_HWIRS,
            text=(
                "HW-IRS_DIM_VI_100 — The DIM-V module shall support "
                "32 digital input channels."
            ),
            chunk_type="requirement",
            tier=3,
            score=0.62,
            vector_score=0.60,
            bm25_score=0.55,
            rerank_score=0.62,
            metadata={
                "doc_type": "HwIRS",
                "section_number": "3",
                "requirement_id": "HW-IRS_DIM_VI_100",
            },
        ),
    ]


def _make_context_window(
    chunks: list[ScoredChunk],
) -> ContextWindow:
    """Build a ContextWindow from scored chunks (two sections)."""
    section_1 = ContextSection(
        section_heading="5 Verification of mechanical tests",
        section_number="5",
        preamble="This section covers mechanical verification.",
        child_chunks=[chunks[0]],
        doc_id=DOC_ID_DIMV_FVTR,
        doc_type="FVTR",
        content_type_hint="test_case",
        token_count=120,
    )
    section_2 = ContextSection(
        section_heading="4 Functional test overview",
        section_number="4",
        preamble="Overview of PAM functional tests.",
        child_chunks=[chunks[1]],
        doc_id=DOC_ID_PAM_FVTR,
        doc_type="FVTR",
        content_type_hint="test_case",
        token_count=110,
    )
    return ContextWindow(
        sections=[section_1, section_2],
        total_tokens=230,
        chunk_ids=[chunks[0].chunk_id, chunks[1].chunk_id],
        doc_ids=[DOC_ID_DIMV_FVTR, DOC_ID_PAM_FVTR],
    )


@pytest.fixture()
def sample_scored_chunks() -> list[ScoredChunk]:
    """Three ScoredChunks with vector_scores 0.90, 0.75, 0.60."""
    return _make_scored_chunks()


@pytest.fixture()
def sample_retrieval_result() -> RetrievalResult:
    """Non-empty RetrievalResult with 2 context sections, 3 scored chunks."""
    chunks = _make_scored_chunks()
    return RetrievalResult(
        context=_make_context_window(chunks),
        scored_chunks=chunks,
        query_analysis=QueryAnalysis(
            original_query="FVTR_MECH_01",
            strategy=QueryStrategy.EXACT_LOOKUP,
            test_case_ids=["FVTR_MECH_01"],
        ),
        timings={"analysis": 0.01, "search": 0.15, "assembly": 0.02},
        strategy="exact_lookup",
    )


@pytest.fixture()
def empty_retrieval_result() -> RetrievalResult:
    """Empty RetrievalResult — no context, no chunks."""
    return RetrievalResult(
        context=ContextWindow(),
        scored_chunks=[],
        query_analysis=QueryAnalysis(
            original_query="nonexistent query xyz",
            strategy=QueryStrategy.UNCONSTRAINED,
        ),
        timings={"analysis": 0.01, "search": 0.10},
        strategy="unconstrained",
    )


# ════════════════════════════════════════════════════════════════
# Citation fixtures
# ════════════════════════════════════════════════════════════════


@pytest.fixture()
def sample_citations() -> list[Citation]:
    """Two resolved Citation objects for source IDs 1 and 2."""
    return [
        Citation(
            source_id=1,
            label="FVTR DIM-V, §5 — Verification of mechanical tests",
            chunk_id=_CHUNK_A,
            doc_id=DOC_ID_DIMV_FVTR,
            doc_type="FVTR",
            section_heading="Verification of mechanical tests",
            section_number="5",
        ),
        Citation(
            source_id=2,
            label="FVTR PAM, §4 — Functional test overview",
            chunk_id=_CHUNK_B,
            doc_id=DOC_ID_PAM_FVTR,
            doc_type="FVTR",
            section_heading="Functional test overview",
            section_number="4",
        ),
    ]


# ════════════════════════════════════════════════════════════════
# Convenience helpers for tests that need custom variants
# ════════════════════════════════════════════════════════════════


def make_verification_result(**overrides) -> VerificationResult:
    """Build a ``VerificationResult`` with defaults, overriding any fields."""
    return VerificationResult(**overrides)


def make_scored_chunk(
    vector_score: float = 0.90,
    **overrides,
) -> ScoredChunk:
    """Build a minimal ``ScoredChunk`` — only `vector_score` is meaningful
    for confidence tests."""
    defaults = {
        "chunk_id": _CHUNK_A,
        "doc_id": DOC_ID_DIMV_FVTR,
        "text": "Test chunk text.",
        "vector_score": vector_score,
    }
    defaults.update(overrides)
    return ScoredChunk(**defaults)
