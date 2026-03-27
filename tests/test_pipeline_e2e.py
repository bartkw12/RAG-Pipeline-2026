"""Full RAG pipeline smoke tests — end-to-end through retrieval + generation.

Every test is marked ``@pytest.mark.e2e`` so the suite can be excluded
in offline or CI environments::

    # Run only E2E tests
    python -m pytest tests/test_pipeline_e2e.py -m e2e

    # Skip E2E tests
    python -m pytest tests/ -m "not e2e"

These are lightweight "does the whole thing work?" checks — they call
``generate()`` once for a known query and assert on structural
properties of the result (non-empty answer, timing fields populated,
usage dict present, no error).
"""

from __future__ import annotations

import pytest

from src.generation.models import GenerationConfig, GenerationResult
from src.generation.pipeline import generate


# ── Shared config & query ───────────────────────────────────────

_GEN_CFG = GenerationConfig(model="gpt-5-mini")
_QUERY = "FVTR_HVT_01"


# ── Single shared result (avoid redundant API calls) ────────────

@pytest.fixture(scope="module")
def pipeline_result() -> GenerationResult:
    """Run the full pipeline once and share the result across tests."""
    return generate(_QUERY, generation_config=_GEN_CFG)


# ════════════════════════════════════════════════════════════════
#  Smoke tests
# ════════════════════════════════════════════════════════════════


@pytest.mark.e2e
class TestFullPipelineReturnsResult:
    """The pipeline returns a well-formed GenerationResult."""

    def test_returns_generation_result(self, pipeline_result):
        assert isinstance(pipeline_result, GenerationResult)

    def test_answer_non_empty(self, pipeline_result):
        assert isinstance(pipeline_result.answer, str)
        assert len(pipeline_result.answer) > 0

    def test_query_echoed(self, pipeline_result):
        assert pipeline_result.query == _QUERY


@pytest.mark.e2e
class TestFullPipelineErrorFieldEmpty:
    """A successful run should have an empty error field."""

    def test_error_empty(self, pipeline_result):
        assert pipeline_result.error == "", (
            f"Unexpected error: {pipeline_result.error}"
        )


@pytest.mark.e2e
class TestFullPipelineTimingsPopulated:
    """Timing telemetry should be present and sensible."""

    def test_generation_elapsed_positive(self, pipeline_result):
        assert pipeline_result.elapsed_s > 0

    def test_total_elapsed_positive(self, pipeline_result):
        assert pipeline_result.total_elapsed_s > 0

    def test_total_greater_than_generation(self, pipeline_result):
        assert pipeline_result.total_elapsed_s >= pipeline_result.elapsed_s


@pytest.mark.e2e
class TestFullPipelineUsagePopulated:
    """Token usage dict should contain expected keys with positive values."""

    def test_total_tokens_positive(self, pipeline_result):
        assert pipeline_result.usage.get("total_tokens", 0) > 0

    def test_prompt_tokens_present(self, pipeline_result):
        assert "prompt_tokens" in pipeline_result.usage

    def test_completion_tokens_present(self, pipeline_result):
        assert "completion_tokens" in pipeline_result.usage
