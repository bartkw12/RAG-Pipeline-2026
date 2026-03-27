"""Unit tests for prompt construction and citation resolution.

Tests ``get_response_schema()``, ``build_source_manifest()``, and
``resolve_citations()`` — no API or disk access required.
"""

from __future__ import annotations

import pytest

from src.generation.prompt import (
    build_source_manifest,
    get_response_schema,
    resolve_citations,
)
from src.retrieval.models import ContextSection, ContextWindow, ScoredChunk

from conftest import (
    DOC_ID_DIMV_FVTR,
    DOC_ID_PAM_FVTR,
    _CHUNK_A,
    _CHUNK_B,
)


# ════════════════════════════════════════════════════════════════
# 1. get_response_schema() — structural validation
# ════════════════════════════════════════════════════════════════


class TestResponseSchema:
    """Validate the JSON Schema returned by ``get_response_schema``."""

    REQUIRED_FIELDS = [
        "answer",
        "claims",
        "abstained",
        "partial",
        "unanswered_aspects",
        "contradictions_noted",
        "confidence",
        "confidence_reasoning",
    ]

    def test_has_all_required_fields(self):
        schema = get_response_schema()
        assert schema["required"] == self.REQUIRED_FIELDS

    def test_no_additional_properties(self):
        schema = get_response_schema()
        assert schema.get("additionalProperties") is False

    def test_top_level_type_is_object(self):
        schema = get_response_schema()
        assert schema["type"] == "object"

    def test_claims_items_schema(self):
        schema = get_response_schema()
        claims_items = schema["properties"]["claims"]["items"]
        assert claims_items["type"] == "object"
        assert "statement" in claims_items["properties"]
        assert "source_ids" in claims_items["properties"]
        assert claims_items.get("additionalProperties") is False
        assert set(claims_items["required"]) == {"statement", "source_ids"}

    def test_confidence_enum_values(self):
        schema = get_response_schema()
        conf_prop = schema["properties"]["confidence"]
        assert set(conf_prop["enum"]) == {"HIGH", "MEDIUM", "LOW"}

    def test_source_ids_item_type_is_integer(self):
        schema = get_response_schema()
        source_ids = (
            schema["properties"]["claims"]["items"]
            ["properties"]["source_ids"]
        )
        assert source_ids["items"]["type"] == "integer"


# ════════════════════════════════════════════════════════════════
# 2. build_source_manifest() — numbering and labels
# ════════════════════════════════════════════════════════════════


class TestBuildSourceManifest:
    """Verify manifest construction from a ``ContextWindow``."""

    @pytest.fixture()
    def two_section_context(self, sample_scored_chunks):
        """ContextWindow with two sections for manifest building."""
        section_1 = ContextSection(
            section_heading="Verification of mechanical tests",
            section_number="5",
            child_chunks=[sample_scored_chunks[0]],
            doc_id=DOC_ID_DIMV_FVTR,
            doc_type="FVTR",
            token_count=120,
        )
        section_2 = ContextSection(
            section_heading="Functional test overview",
            section_number="4",
            child_chunks=[sample_scored_chunks[1]],
            doc_id=DOC_ID_PAM_FVTR,
            doc_type="FVTR",
            token_count=110,
        )
        return ContextWindow(
            sections=[section_1, section_2],
            total_tokens=230,
            chunk_ids=[_CHUNK_A, _CHUNK_B],
            doc_ids=[DOC_ID_DIMV_FVTR, DOC_ID_PAM_FVTR],
        )

    def test_manifest_length(self, two_section_context):
        _, manifest = build_source_manifest(two_section_context)
        assert len(manifest) == 2

    def test_source_ids_sequential(self, two_section_context):
        _, manifest = build_source_manifest(two_section_context)
        ids = [entry["source_id"] for entry in manifest]
        assert ids == [1, 2]

    def test_manifest_contains_doc_ids(self, two_section_context):
        _, manifest = build_source_manifest(two_section_context)
        doc_ids = {entry["doc_id"] for entry in manifest}
        assert DOC_ID_DIMV_FVTR in doc_ids
        assert DOC_ID_PAM_FVTR in doc_ids

    def test_manifest_contains_chunk_ids(self, two_section_context):
        _, manifest = build_source_manifest(two_section_context)
        assert _CHUNK_A in manifest[0]["chunk_ids"]
        assert _CHUNK_B in manifest[1]["chunk_ids"]

    def test_prompt_text_contains_source_labels(self, two_section_context):
        prompt_text, _ = build_source_manifest(two_section_context)
        assert "Source 1:" in prompt_text
        assert "Source 2:" in prompt_text

    def test_prompt_text_contains_heading(self, two_section_context):
        prompt_text, _ = build_source_manifest(two_section_context)
        assert "mechanical tests" in prompt_text.lower()
        assert "functional test" in prompt_text.lower()

    def test_empty_context_produces_empty_manifest(self):
        prompt_text, manifest = build_source_manifest(ContextWindow())
        assert manifest == []
        assert "SOURCE MANIFEST" in prompt_text


# ════════════════════════════════════════════════════════════════
# 3. resolve_citations() — mapping source IDs to Citation objects
# ════════════════════════════════════════════════════════════════


class TestResolveCitations:
    """Verify citation resolution from raw response + manifest."""

    def test_resolves_valid_sources(
        self, sample_raw_response_clean, sample_source_manifest,
    ):
        citations = resolve_citations(
            sample_raw_response_clean, sample_source_manifest,
        )
        assert len(citations) == 2
        assert citations[0].source_id == 1
        assert citations[1].source_id == 2

    def test_citation_has_correct_chunk_id(
        self, sample_raw_response_clean, sample_source_manifest,
    ):
        citations = resolve_citations(
            sample_raw_response_clean, sample_source_manifest,
        )
        assert citations[0].chunk_id == _CHUNK_A
        assert citations[1].chunk_id == _CHUNK_B

    def test_citation_has_correct_doc_id(
        self, sample_raw_response_clean, sample_source_manifest,
    ):
        citations = resolve_citations(
            sample_raw_response_clean, sample_source_manifest,
        )
        assert citations[0].doc_id == DOC_ID_DIMV_FVTR
        assert citations[1].doc_id == DOC_ID_PAM_FVTR

    def test_citation_has_label(
        self, sample_raw_response_clean, sample_source_manifest,
    ):
        citations = resolve_citations(
            sample_raw_response_clean, sample_source_manifest,
        )
        assert citations[0].label != ""
        assert citations[1].label != ""

    def test_citation_doc_type(
        self, sample_raw_response_clean, sample_source_manifest,
    ):
        citations = resolve_citations(
            sample_raw_response_clean, sample_source_manifest,
        )
        assert citations[0].doc_type == "FVTR"
        assert citations[1].doc_type == "FVTR"

    def test_unmapped_source_produces_placeholder(
        self, sample_raw_response_unmapped, sample_source_manifest,
    ):
        citations = resolve_citations(
            sample_raw_response_unmapped, sample_source_manifest,
        )
        # Source 1 resolves, source 99 gets placeholder
        unmapped = [c for c in citations if c.source_id == 99]
        assert len(unmapped) == 1
        assert "(unmapped source 99)" in unmapped[0].label
        assert unmapped[0].chunk_id == ""
        assert unmapped[0].doc_id == ""

    def test_abstained_response_no_citations(
        self, sample_raw_response_abstained, sample_source_manifest,
    ):
        citations = resolve_citations(
            sample_raw_response_abstained, sample_source_manifest,
        )
        assert citations == []

    def test_citations_ordered_by_source_id(
        self, sample_source_manifest,
    ):
        """Sources referenced out of order are returned sorted."""
        response = {
            "claims": [
                {"statement": "A", "source_ids": [3]},
                {"statement": "B", "source_ids": [1]},
            ],
        }
        citations = resolve_citations(response, sample_source_manifest)
        ids = [c.source_id for c in citations]
        assert ids == [1, 3]

    def test_deduplicates_source_ids(
        self, sample_source_manifest,
    ):
        """Same source cited in multiple claims → single Citation."""
        response = {
            "claims": [
                {"statement": "A", "source_ids": [1]},
                {"statement": "B", "source_ids": [1, 2]},
            ],
        }
        citations = resolve_citations(response, sample_source_manifest)
        assert len(citations) == 2  # sources 1 and 2, not 3
