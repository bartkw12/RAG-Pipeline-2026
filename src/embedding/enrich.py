"""Metadata-enriched text builder for embedding input.

Prepends structured context (document type, module, chunk type, heading,
and type-specific identifiers) to the raw chunk text before embedding.
This gives the embedding model richer semantic signal and improves
retrieval accuracy when queries mention document types, module names,
section context, or specific IDs.

The enriched format is::

    [FVTR | PAM | Test Case | 5 VERIFICATION OF MECHANICAL CHARACTERISTICS]
    FVTR_OPT_01: Labelling and assembly — Result: Passed
    ---
    <raw chunk text>
"""

from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger(__name__)

# ── Human-readable labels for chunk types ───────────────────────

_TYPE_LABELS: dict[str, str] = {
    "document_meta": "Document Meta",
    "front_matter": "Front Matter",
    "section": "Section",
    "test_case": "Test Case",
    "requirement": "Requirement",
    "table": "Table",
    "definition_table": "Definition Table",
    "abbreviation_table": "Abbreviation Table",
    "prose": "Prose",
    "figure": "Figure",
    "list": "List",
}

_MAX_SAFE_TOKENS = 8_000  # warn if enriched text likely exceeds this


# ── Public API ──────────────────────────────────────────────────


def enrich_chunk_text(chunk: dict[str, Any], doc_metadata: dict[str, Any]) -> str:
    """Build an enriched text string for a single chunk.

    Parameters
    ----------
    chunk:
        A chunk dict as loaded from ``cache/chunk/{doc_id}.json``.
    doc_metadata:
        The ``doc_metadata`` dict from the same JSON file.

    Returns
    -------
    str
        Enriched text: structured prefix + separator + raw text.
    """
    meta = chunk.get("metadata", {})
    chunk_type = chunk.get("chunk_type", "")
    text = chunk.get("text", "")

    # ── Build bracket prefix ────────────────────────────────────
    parts: list[str] = []

    doc_type = doc_metadata.get("doc_type", "")
    if doc_type:
        parts.append(doc_type)

    module_name = doc_metadata.get("module_name", "")
    if module_name:
        parts.append(module_name)

    type_label = _TYPE_LABELS.get(chunk_type, chunk_type)
    if type_label:
        parts.append(type_label)

    heading = meta.get("heading") or ""
    if not heading:
        section_path = meta.get("section_path", [])
        if section_path:
            heading = section_path[-1]
    if heading:
        parts.append(heading)

    bracket_line = f"[{' | '.join(parts)}]" if parts else ""

    # ── Build type-specific summary line ────────────────────────
    summary_line = _build_summary_line(chunk_type, meta)

    # ── Assemble ────────────────────────────────────────────────
    header_parts = [p for p in (bracket_line, summary_line) if p]
    if header_parts:
        enriched = "\n".join(header_parts) + "\n---\n" + text
    else:
        enriched = text

    # Warn on unexpectedly large enriched text (defensive).
    token_estimate = len(enriched) // 4  # rough char-based estimate
    if token_estimate > _MAX_SAFE_TOKENS:
        logger.warning(
            "Enriched text for chunk %s is ~%d tokens (estimate) — "
            "may approach Ada-002's 8191-token limit.",
            chunk.get("chunk_id", "?")[:12], token_estimate,
        )

    return enriched


def enrich_chunks(
    chunks: list[dict[str, Any]],
    doc_metadata: dict[str, Any],
) -> list[str]:
    """Enrich a list of chunks from the same document.

    Parameters
    ----------
    chunks:
        Chunk dicts from the ``"chunks"`` array of a chunk JSON file.
    doc_metadata:
        The ``doc_metadata`` dict from the same file.

    Returns
    -------
    list[str]
        Enriched text strings, one per input chunk, in the same order.
    """
    return [enrich_chunk_text(c, doc_metadata) for c in chunks]


# ── Internals ───────────────────────────────────────────────────


def _build_summary_line(chunk_type: str, meta: dict[str, Any]) -> str:
    """Produce a one-line type-specific summary, or empty string."""

    if chunk_type == "test_case":
        return _test_case_summary(meta)

    if chunk_type == "requirement":
        return _requirement_summary(meta)

    return ""


def _test_case_summary(meta: dict[str, Any]) -> str:
    """E.g. ``FVTR_OPT_01: Labelling and assembly — Result: Passed``."""
    parts: list[str] = []

    tc_id = meta.get("test_case_id", "")
    tc_name = meta.get("test_name", "")
    if tc_id and tc_name:
        parts.append(f"{tc_id}: {tc_name}")
    elif tc_id:
        parts.append(tc_id)

    result = meta.get("test_result", "")
    if result:
        parts.append(f"Result: {result}")

    return " — ".join(parts)


def _requirement_summary(meta: dict[str, Any]) -> str:
    """E.g. ``HW-IRS_DIM_VI_275 — Safety: NA, Verification: Test``."""
    parts: list[str] = []

    req_ids = meta.get("requirement_ids", [])
    if req_ids:
        parts.append(", ".join(req_ids))

    attrs: list[str] = []
    safety = meta.get("safety", "")
    if safety:
        attrs.append(f"Safety: {safety}")
    verification = meta.get("verification_method", "")
    if verification:
        attrs.append(f"Verification: {verification}")
    if attrs:
        parts.append(", ".join(attrs))

    return " — ".join(parts)
