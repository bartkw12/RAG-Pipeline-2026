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


# ── Confidence level ────────────────────────────────────────────


class ConfidenceLevel(str, Enum):
    """Tiered confidence for a generation result.

    Used for both the model's self-assessment and the
    system-computed confidence derived from retrieval signals
    and structural verification.
    """

    HIGH = "HIGH"
    """Direct, unambiguous evidence in the retrieved context."""

    MEDIUM = "MEDIUM"
    """Indirect or partial evidence — caveats should be stated."""

    LOW = "LOW"
    """Weak or tangential evidence only, or hard abstention."""


# ── Verification result ─────────────────────────────────────────


@dataclass
class VerificationResult:
    """Deterministic structural-groundedness check on a generation.

    Produced by ``verify_generation()`` — validates that the LLM's
    structured output is internally consistent and that every cited
    source maps to real retrieved context.  No LLM calls; purely
    rule-based.
    """

    all_citations_resolved: bool = True
    """Every ``source_id`` in ``claims[].source_ids`` maps to an
    entry in the source manifest."""

    all_claims_cited: bool = True
    """Every claim in ``claims[]`` has at least one ``source_id``."""

    contains_unmapped_citations: bool = False
    """The model cited a ``source_id`` that does not exist in the
    source manifest (potential hallucinated reference)."""

    abstention_consistent: bool = True
    """If ``abstained=True`` the claims list is empty; if
    ``abstained=False`` the claims list is non-empty."""

    citation_coverage_ratio: float = 1.0
    """Fraction of claims that have at least one valid citation.
    ``1.0`` when all claims are cited; ``0.0`` when none are."""

    unmapped_source_ids: list[int] = field(default_factory=list)
    """Specific ``[Source N]`` numbers that were cited but do not
    exist in the source manifest."""

    uncited_claim_indices: list[int] = field(default_factory=list)
    """Zero-based indices into ``claims[]`` for entries whose
    ``source_ids`` list is empty."""


# ── Generation result ───────────────────────────────────────────


@dataclass
class GenerationResult:
    """Complete output of a single generation call.

    Bundles the grounded answer with per-claim citations, structural
    verification, hybrid confidence, retrieval diagnostics, and
    usage telemetry.  Supports dual rendering via ``to_text()``
    (human-readable) and ``to_dict()`` (JSON for agents).
    """

    # ── Answer ──────────────────────────────────────────────────
    answer: str = ""
    """The LLM's response text, potentially containing
    ``[Source N]`` markers."""

    claims: list[Claim] = field(default_factory=list)
    """Per-claim breakdown with source mappings from the
    structured output."""

    citations: list[Citation] = field(default_factory=list)
    """Resolved source references (one per unique source used)."""

    # ── Abstention / partial answer ─────────────────────────────
    abstained: bool = False
    """Hard abstention — the LLM determined no useful answer
    is possible from the retrieved context."""

    partial: bool = False
    """Partial answer — the evidence supports part of the query
    but not all of it."""

    unanswered_aspects: list[str] = field(default_factory=list)
    """Aspects of the query the evidence does not cover.
    Populated when ``partial=True``."""

    contradictions_noted: bool = False
    """The LLM flagged conflicting information between sources."""

    # ── Confidence ──────────────────────────────────────────────
    model_confidence: ConfidenceLevel = ConfidenceLevel.LOW
    """The LLM's self-assessed confidence level."""

    model_confidence_reasoning: str = ""
    """The LLM's explanation of its confidence assessment."""

    system_confidence: ConfidenceLevel = ConfidenceLevel.LOW
    """Computed confidence from retrieval signals and verification.
    Final confidence is ``min(model, system)``."""

    confidence_components: dict[str, float] = field(default_factory=dict)
    """Breakdown: ``retrieval_support``, ``citation_coverage``,
    ``verification_pass``."""

    # ── Verification ────────────────────────────────────────────
    verification: VerificationResult = field(default_factory=VerificationResult)
    """Structural groundedness check output."""

    # ── Provenance ──────────────────────────────────────────────
    query: str = ""
    """Original user query."""

    strategy: str = ""
    """Retrieval strategy used (``exact_lookup``,
    ``scoped_semantic``, ``unconstrained``)."""

    retrieval_result: Any = None
    """Full ``RetrievalResult`` from the retrieval pipeline.
    Typed as ``Any`` to avoid circular imports; always a
    ``RetrievalResult`` at runtime."""

    # ── Telemetry ───────────────────────────────────────────────
    model: str = ""
    """Azure OpenAI deployment name used for generation."""

    usage: dict[str, int] = field(default_factory=dict)
    """Token usage: ``prompt_tokens``, ``completion_tokens``,
    ``total_tokens``."""

    elapsed_s: float = 0.0
    """Generation latency in seconds (LLM call only)."""

    total_elapsed_s: float = 0.0
    """End-to-end latency (retrieval + generation)."""

    error: str = ""
    """Non-empty if the generation encountered a recoverable
    error."""
