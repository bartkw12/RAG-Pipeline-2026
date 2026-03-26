"""Prompt construction and structured-output schema for grounded generation.

Assembles the full LLM prompt from retrieval context, builds a numbered
source manifest for citation tracking, defines the JSON output schema
for Azure OpenAI structured outputs, and resolves citations from the
model's response back to real chunk identifiers.
"""

from __future__ import annotations

import logging
from typing import Any

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
