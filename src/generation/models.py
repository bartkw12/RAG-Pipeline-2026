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
