"""Unit tests for the structural-groundedness verifier.

Tests ``verify_generation()`` with parametrized LLM response
variants — no API calls, fully deterministic.
"""

from __future__ import annotations

import pytest

from src.generation.verification import verify_generation


# ════════════════════════════════════════════════════════════════
# 1. Clean response — all checks pass
# ════════════════════════════════════════════════════════════════


class TestCleanResponse:
    """Well-formed response with all claims cited and valid source IDs."""

    def test_all_citations_resolved(
        self, sample_raw_response_clean, sample_source_manifest, sample_citations,
    ):
        result = verify_generation(
            sample_raw_response_clean, sample_source_manifest, sample_citations,
        )
        assert result.all_citations_resolved is True

    def test_all_claims_cited(
        self, sample_raw_response_clean, sample_source_manifest, sample_citations,
    ):
        result = verify_generation(
            sample_raw_response_clean, sample_source_manifest, sample_citations,
        )
        assert result.all_claims_cited is True

    def test_no_unmapped_citations(
        self, sample_raw_response_clean, sample_source_manifest, sample_citations,
    ):
        result = verify_generation(
            sample_raw_response_clean, sample_source_manifest, sample_citations,
        )
        assert result.contains_unmapped_citations is False
        assert result.unmapped_source_ids == []

    def test_abstention_consistent(
        self, sample_raw_response_clean, sample_source_manifest, sample_citations,
    ):
        result = verify_generation(
            sample_raw_response_clean, sample_source_manifest, sample_citations,
        )
        assert result.abstention_consistent is True

    def test_full_coverage(
        self, sample_raw_response_clean, sample_source_manifest, sample_citations,
    ):
        result = verify_generation(
            sample_raw_response_clean, sample_source_manifest, sample_citations,
        )
        assert result.citation_coverage_ratio == 1.0

    def test_no_uncited_claims(
        self, sample_raw_response_clean, sample_source_manifest, sample_citations,
    ):
        result = verify_generation(
            sample_raw_response_clean, sample_source_manifest, sample_citations,
        )
        assert result.uncited_claim_indices == []


# ════════════════════════════════════════════════════════════════
# 2. Uncited claim — one claim has empty source_ids
# ════════════════════════════════════════════════════════════════


class TestUncitedClaim:
    """Response where one claim has ``source_ids: []``."""

    def test_all_claims_cited_false(
        self, sample_raw_response_uncited, sample_source_manifest, sample_citations,
    ):
        result = verify_generation(
            sample_raw_response_uncited, sample_source_manifest, sample_citations,
        )
        assert result.all_claims_cited is False

    def test_uncited_claim_index(
        self, sample_raw_response_uncited, sample_source_manifest, sample_citations,
    ):
        result = verify_generation(
            sample_raw_response_uncited, sample_source_manifest, sample_citations,
        )
        assert 1 in result.uncited_claim_indices

    def test_coverage_below_one(
        self, sample_raw_response_uncited, sample_source_manifest, sample_citations,
    ):
        result = verify_generation(
            sample_raw_response_uncited, sample_source_manifest, sample_citations,
        )
        assert result.citation_coverage_ratio == 0.5

    def test_citations_still_resolved(
        self, sample_raw_response_uncited, sample_source_manifest, sample_citations,
    ):
        """The cited claim (source 1) is valid — only the empty one is the issue."""
        result = verify_generation(
            sample_raw_response_uncited, sample_source_manifest, sample_citations,
        )
        assert result.all_citations_resolved is True
        assert result.contains_unmapped_citations is False


# ════════════════════════════════════════════════════════════════
# 3. Unmapped source ID — claim cites source_id not in manifest
# ════════════════════════════════════════════════════════════════


class TestUnmappedSource:
    """Response citing ``[Source 99]`` which doesn't exist in manifest."""

    def test_contains_unmapped(
        self, sample_raw_response_unmapped, sample_source_manifest, sample_citations,
    ):
        result = verify_generation(
            sample_raw_response_unmapped, sample_source_manifest, sample_citations,
        )
        assert result.contains_unmapped_citations is True

    def test_all_citations_resolved_false(
        self, sample_raw_response_unmapped, sample_source_manifest, sample_citations,
    ):
        result = verify_generation(
            sample_raw_response_unmapped, sample_source_manifest, sample_citations,
        )
        assert result.all_citations_resolved is False

    def test_unmapped_ids_list(
        self, sample_raw_response_unmapped, sample_source_manifest, sample_citations,
    ):
        result = verify_generation(
            sample_raw_response_unmapped, sample_source_manifest, sample_citations,
        )
        assert result.unmapped_source_ids == [99]

    def test_claims_still_cited(
        self, sample_raw_response_unmapped, sample_source_manifest, sample_citations,
    ):
        """Both claims have non-empty source_ids, even though one is invalid."""
        result = verify_generation(
            sample_raw_response_unmapped, sample_source_manifest, sample_citations,
        )
        assert result.all_claims_cited is True


# ════════════════════════════════════════════════════════════════
# 4. Clean abstention — abstained=True, empty claims
# ════════════════════════════════════════════════════════════════


class TestCleanAbstention:
    """Properly abstained response — no claims, no citations."""

    def test_abstention_consistent(
        self, sample_raw_response_abstained, sample_source_manifest,
    ):
        result = verify_generation(
            sample_raw_response_abstained, sample_source_manifest, [],
        )
        assert result.abstention_consistent is True

    def test_all_claims_cited_vacuously(
        self, sample_raw_response_abstained, sample_source_manifest,
    ):
        """No claims → vacuously true."""
        result = verify_generation(
            sample_raw_response_abstained, sample_source_manifest, [],
        )
        assert result.all_claims_cited is True

    def test_coverage_vacuously_one(
        self, sample_raw_response_abstained, sample_source_manifest,
    ):
        """No claims → coverage is 1.0 (vacuously true)."""
        result = verify_generation(
            sample_raw_response_abstained, sample_source_manifest, [],
        )
        assert result.citation_coverage_ratio == 1.0

    def test_no_unmapped(
        self, sample_raw_response_abstained, sample_source_manifest,
    ):
        result = verify_generation(
            sample_raw_response_abstained, sample_source_manifest, [],
        )
        assert result.contains_unmapped_citations is False


# ════════════════════════════════════════════════════════════════
# 5. Inconsistent abstention — abstained=True but claims present
# ════════════════════════════════════════════════════════════════


class TestInconsistentAbstentionWithClaims:
    """abstained=True but claims list is non-empty."""

    @pytest.fixture()
    def response(self):
        return {
            "abstained": True,
            "claims": [
                {"statement": "Some claim.", "source_ids": [1]},
            ],
        }

    def test_abstention_inconsistent(
        self, response, sample_source_manifest, sample_citations,
    ):
        result = verify_generation(response, sample_source_manifest, sample_citations)
        assert result.abstention_consistent is False


# ════════════════════════════════════════════════════════════════
# 6. Inconsistent non-abstention — abstained=False, no claims
# ════════════════════════════════════════════════════════════════


class TestInconsistentNonAbstention:
    """abstained=False but claims list is empty."""

    @pytest.fixture()
    def response(self):
        return {
            "abstained": False,
            "claims": [],
        }

    def test_abstention_inconsistent(
        self, response, sample_source_manifest, sample_citations,
    ):
        result = verify_generation(response, sample_source_manifest, sample_citations)
        assert result.abstention_consistent is False

    def test_coverage_vacuously_one(
        self, response, sample_source_manifest, sample_citations,
    ):
        """Empty claims → coverage 1.0 (vacuous)."""
        result = verify_generation(response, sample_source_manifest, sample_citations)
        assert result.citation_coverage_ratio == 1.0


# ════════════════════════════════════════════════════════════════
# 7. Edge cases
# ════════════════════════════════════════════════════════════════


class TestEdgeCases:
    """Boundary conditions and degenerate inputs."""

    def test_empty_manifest(self, sample_raw_response_clean):
        """All source_ids become unmapped when manifest is empty."""
        result = verify_generation(sample_raw_response_clean, [], [])
        assert result.contains_unmapped_citations is True
        assert result.all_citations_resolved is False
        assert set(result.unmapped_source_ids) == {1, 2}

    def test_empty_response(self, sample_source_manifest):
        """Completely empty response dict."""
        result = verify_generation({}, sample_source_manifest, [])
        # No claims, not abstained → abstention_consistent should be False
        # (abstained defaults to False, claims defaults to [])
        assert result.abstention_consistent is False
        assert result.all_claims_cited is True  # vacuously
        assert result.citation_coverage_ratio == 1.0  # vacuously

    def test_claim_with_multiple_valid_sources(self, sample_source_manifest):
        """A single claim citing multiple valid sources."""
        response = {
            "abstained": False,
            "claims": [
                {"statement": "Multi-source claim.", "source_ids": [1, 2, 3]},
            ],
        }
        result = verify_generation(response, sample_source_manifest, [])
        assert result.all_citations_resolved is True
        assert result.all_claims_cited is True
        assert result.citation_coverage_ratio == 1.0

    def test_duplicate_unmapped_ids_deduplicated(self, sample_source_manifest):
        """Same invalid source_id cited twice → appears once in unmapped."""
        response = {
            "abstained": False,
            "claims": [
                {"statement": "Claim A.", "source_ids": [99]},
                {"statement": "Claim B.", "source_ids": [99]},
            ],
        }
        result = verify_generation(response, sample_source_manifest, [])
        assert result.unmapped_source_ids == [99]
