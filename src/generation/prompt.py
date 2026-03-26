"""Prompt construction and structured-output schema for grounded generation.

Assembles the full LLM prompt from retrieval context, builds a numbered
source manifest for citation tracking, defines the JSON output schema
for Azure OpenAI structured outputs, and resolves citations from the
model's response back to real chunk identifiers.
"""

from __future__ import annotations

import logging
from typing import Any

from ..chunking.writer import load_chunks
from ..retrieval.models import ContextWindow

logger = logging.getLogger(__name__)


# ── Structured output schema ───────────────────────────────────


def get_response_schema() -> dict[str, Any]:
    """Return the JSON Schema for the generation structured output.

    Designed for Azure OpenAI's ``response_format`` parameter with
    ``"type": "json_schema"`` and ``"strict": True``.  Every field
    is required and ``additionalProperties`` is ``false`` at each
    object level, as mandated by strict mode.

    Schema fields
    -------------
    answer : str
        Full answer text with ``[Source N]`` markers.
    claims : array of {statement, source_ids}
        Per-claim breakdown for verification.
    abstained : bool
        Hard abstention — no useful answer possible.
    partial : bool
        Evidence covers part of the query but not all.
    unanswered_aspects : array of str
        Gaps in the evidence (populated when ``partial`` is true).
    contradictions_noted : bool
        Whether conflicting information was detected.
    confidence : str
        ``"HIGH"``, ``"MEDIUM"``, or ``"LOW"``.
    confidence_reasoning : str
        Brief explanation of the confidence assessment.
    """
    return {
        "type": "object",
        "properties": {
            "answer": {
                "type": "string",
                "description": (
                    "Full answer text. Use [Source N] markers to cite "
                    "evidence from the source manifest. If abstaining, "
                    "explain what information is missing."
                ),
            },
            "claims": {
                "type": "array",
                "description": (
                    "Each factual claim in the answer, with the source "
                    "IDs that support it. Every assertion of fact must "
                    "appear here."
                ),
                "items": {
                    "type": "object",
                    "properties": {
                        "statement": {
                            "type": "string",
                            "description": (
                                "A single factual claim from the answer."
                            ),
                        },
                        "source_ids": {
                            "type": "array",
                            "description": (
                                "Source numbers ([Source N]) supporting "
                                "this claim. Must not be empty."
                            ),
                            "items": {"type": "integer"},
                        },
                    },
                    "required": ["statement", "source_ids"],
                    "additionalProperties": False,
                },
            },
            "abstained": {
                "type": "boolean",
                "description": (
                    "True when the context provides no useful evidence "
                    "to answer the question. Claims must be empty when "
                    "abstaining."
                ),
            },
            "partial": {
                "type": "boolean",
                "description": (
                    "True when the evidence covers part of the query "
                    "but not all. List the gaps in unanswered_aspects."
                ),
            },
            "unanswered_aspects": {
                "type": "array",
                "description": (
                    "Aspects of the query the evidence does not cover. "
                    "Populate when partial is true; empty otherwise."
                ),
                "items": {"type": "string"},
            },
            "contradictions_noted": {
                "type": "boolean",
                "description": (
                    "True if retrieved sources contain conflicting "
                    "information. State the conflict in the answer."
                ),
            },
            "confidence": {
                "type": "string",
                "enum": ["HIGH", "MEDIUM", "LOW"],
                "description": (
                    "HIGH: direct, unambiguous evidence. "
                    "MEDIUM: indirect or partial evidence. "
                    "LOW: weak or tangential evidence only."
                ),
            },
            "confidence_reasoning": {
                "type": "string",
                "description": (
                    "Brief explanation of the confidence assessment."
                ),
            },
        },
        "required": [
            "answer",
            "claims",
            "abstained",
            "partial",
            "unanswered_aspects",
            "contradictions_noted",
            "confidence",
            "confidence_reasoning",
        ],
        "additionalProperties": False,
    }


# ── Source manifest ─────────────────────────────────────────────


def build_source_manifest(
    context: ContextWindow,
) -> tuple[str, list[dict[str, Any]]]:
    """Build a numbered source manifest from the assembled context.

    Each ``ContextSection`` in the window becomes one numbered source
    that the LLM can reference with ``[Source N]`` markers.  The
    manifest includes document metadata (type, revision, date) so
    the model can reason about recency and authority.

    Parameters
    ----------
    context:
        The ``ContextWindow`` produced by the retrieval pipeline.

    Returns
    -------
    tuple[str, list[dict]]
        ``(prompt_text, manifest)`` where *prompt_text* is the
        formatted block to include in the system / user message and
        *manifest* is the structured list used later by
        ``resolve_citations()`` to map source IDs back to chunks.
    """
    manifest: list[dict[str, Any]] = []
    lines: list[str] = ["SOURCE MANIFEST", ""]

    # Cache doc metadata per doc_id to avoid repeated disk reads.
    doc_meta_cache: dict[str, dict[str, str]] = {}

    for idx, section in enumerate(context.sections, 1):
        chunk_ids = [sc.chunk_id for sc in section.child_chunks]

        # Resolve document-level metadata (revision, date, module).
        meta = _get_doc_meta(section.doc_id, doc_meta_cache)

        # Human-readable label: "FVTR DIM-V, §5.2 — Heading text"
        label_parts: list[str] = []
        if section.doc_type:
            label_parts.append(section.doc_type)
        if meta.get("module_name"):
            label_parts.append(meta["module_name"])
        prefix = " ".join(label_parts)

        if section.section_number:
            ref = f"§{section.section_number}"
        else:
            ref = ""

        heading = section.section_heading or "(untitled section)"

        if prefix and ref:
            label = f"{prefix}, {ref} — {heading}"
        elif prefix:
            label = f"{prefix} — {heading}"
        elif ref:
            label = f"{ref} — {heading}"
        else:
            label = heading

        # Structured manifest entry.
        entry: dict[str, Any] = {
            "source_id": idx,
            "label": label,
            "doc_id": section.doc_id,
            "doc_type": section.doc_type,
            "section_heading": section.section_heading,
            "section_number": section.section_number,
            "chunk_ids": chunk_ids,
            "revision": meta.get("revision", ""),
            "revision_date": meta.get("revision_date", ""),
        }
        manifest.append(entry)

        # Prompt text lines.
        line = f"Source {idx}: {label}"
        if entry["revision"]:
            line += f"  (Rev {entry['revision']}"
            if entry["revision_date"]:
                line += f", {entry['revision_date']}"
            line += ")"
        lines.append(line)
        lines.append(f"  Chunk IDs: {', '.join(cid[:12] for cid in chunk_ids)}")
        lines.append("")

    prompt_text = "\n".join(lines).rstrip()
    return prompt_text, manifest


def _get_doc_meta(
    doc_id: str,
    cache: dict[str, dict[str, str]],
) -> dict[str, str]:
    """Load document metadata for *doc_id*, with caching.

    Returns a flat dict with keys ``module_name``, ``revision``,
    ``revision_date``.  Returns empty strings on failure.
    """
    if doc_id in cache:
        return cache[doc_id]

    empty: dict[str, str] = {
        "module_name": "",
        "revision": "",
        "revision_date": "",
    }

    if not doc_id:
        cache[doc_id] = empty
        return empty

    try:
        doc = load_chunks(doc_id)
        meta = {
            "module_name": doc.doc_metadata.module_name or "",
            "revision": doc.doc_metadata.revision or "",
            "revision_date": doc.doc_metadata.revision_date or "",
        }
    except FileNotFoundError:
        logger.warning("Chunk file for doc_id=%s… not found.", doc_id[:12])
        meta = empty

    cache[doc_id] = meta
    return meta
