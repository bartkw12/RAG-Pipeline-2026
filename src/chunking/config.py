"""Chunking configuration — all tuneable knobs for the chunking pipeline.

Sensible defaults target the Ada-002 embedding model (``cl100k_base``
encoding, 8 191-token context window) with a sweet-spot chunk size of
200–500 tokens for retrieval quality.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ChunkConfig:
    """All knobs for the chunking pipeline.

    Sensible defaults are provided for every field.  Override via CLI
    flags or by constructing directly.

    Token budgets
    -------------
    * **target_tokens** — Ideal chunk size for embedding quality.
    * **min_tokens** — Chunks smaller than this are merged with an
      adjacent sibling under the same heading.
    * **max_tokens** — Splittable chunk types (prose, lists) are split
      at sentence / paragraph boundaries when they exceed this.
    * **split_threshold** — Even atomic chunk types (test cases,
      requirements, tables) are force-split when they exceed this.
      Below this limit, atomic chunks are always kept whole.
    * **overlap_tokens** — When a prose/list chunk is split, this many
      tokens of trailing context are duplicated at the start of the
      next sub-chunk for continuity.
    """

    # ── Token budgets ───────────────────────────────────────────
    target_tokens: int = 400
    """Ideal chunk size (tokens).  The chunker tries to produce
    chunks close to this size when merging small blocks."""

    min_tokens: int = 50
    """Minimum chunk size.  Blocks below this are merged with
    an adjacent sibling under the same heading."""

    max_tokens: int = 800
    """Maximum size for *splittable* chunk types (prose, lists).
    Blocks exceeding this are split at natural boundaries."""

    split_threshold: int = 1000
    """Hard ceiling for *atomic* chunk types (test cases,
    requirements, standalone tables).  Blocks above this are
    force-split even though they are normally kept whole."""

    overlap_tokens: int = 50
    """Number of trailing-context tokens duplicated at the start
    of the next sub-chunk when splitting prose or lists."""

    # ── Tokeniser ───────────────────────────────────────────────
    encoding_name: str = "cl100k_base"
    """``tiktoken`` encoding name.  ``cl100k_base`` covers the
    GPT-4 / Ada-002 family.  Change if you switch embedding models."""
