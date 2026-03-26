"""Generation data models — configuration, citations, claims, confidence, and results.

Defines the core data structures for the grounded generation pipeline:

* **Configuration** — all tunable generation parameters.
* **Citation** — a resolved source reference with evidence excerpt.
* **Claim** — a single factual statement with source mappings.
* **Confidence** — tiered confidence level (HIGH / MEDIUM / LOW).
* **Verification** — deterministic structural-groundedness check results.
* **Generation result** — final output combining answer, citations, and diagnostics.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any


# ── Generation configuration ────────────────────────────────────


@dataclass
class GenerationConfig:
    """All tunable parameters for the generation pipeline.

    Defaults target GPT-5 mini on the ``OpenAI2`` Azure endpoint
    with conservative settings for engineering QA (low temperature,
    structured JSON output).
    """

    # ── Model selection ─────────────────────────────────────────
    model: str = "gpt-5-mini"
    """Azure OpenAI deployment name for the generation model."""

    config_section: str = "OpenAI2"
    """Top-level key in the project config JSON from which to load
    Azure credentials (endpoint, API key, API version)."""

    # ── Generation parameters ───────────────────────────────────
    temperature: float = 0.1
    """Sampling temperature.  Low values (0.0–0.1) favour
    consistency and repeatability for engineering QA."""

    max_output_tokens: int = 2048
    """Maximum tokens the model may generate in a single response."""

    reasoning_effort: str | None = None
    """Optional reasoning-effort control for GPT-5 family models
    (e.g. ``"low"``, ``"medium"``, ``"high"``).  Silently ignored
    when the deployment does not support it."""


# ── Citation ────────────────────────────────────────────────────


@dataclass
class Citation:
    """A resolved source reference linking an LLM claim to retrieved evidence.

    Built by ``resolve_citations()`` after generation, by mapping the
    ``[Source N]`` markers in the LLM's answer back to the numbered
    source manifest that was included in the prompt.
    """

    source_id: int
    """The ``[Source N]`` number used in the answer text."""

    label: str
    """Human-readable reference, e.g.
    ``"FVTR DIM-V, §5.2 — Verification of thermal performance"``."""

    chunk_id: str
    """Primary chunk identifier for machine traceability."""

    doc_id: str
    """SHA-256 document hash of the source file."""

    doc_type: str = ""
    """Document type: ``"FVTR"``, ``"HwIRS"``, or ``""``."""

    section_heading: str = ""
    """Tier-2 section heading text from the context window."""

    section_number: str = ""
    """Numeric section identifier, e.g. ``"5.2"``."""

    quoted_text: str = ""
    """Short evidence excerpt provided by the LLM to support
    the cited claim.  May be empty if the model did not
    include a direct quote."""


# ── Claim ───────────────────────────────────────────────────────


@dataclass
class Claim:
    """A single factual statement extracted from the LLM's answer.

    Part of the structured output schema — each claim carries
    explicit ``source_ids`` so the verifier can confirm that every
    assertion is backed by retrieved evidence.
    """

    statement: str
    """The factual claim text, e.g.
    ``"The FVTR DIM-V thermal test passed at 85°C."``."""

    source_ids: list[int] = field(default_factory=list)
    """``[Source N]`` numbers that support this claim.  An empty
    list means the claim is uncited (flagged by the verifier)."""
