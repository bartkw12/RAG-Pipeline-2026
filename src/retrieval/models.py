"""Retrieval data models — query analysis, scoring, configuration, and context.

Defines the core data structures for the multi-strategy retrieval pipeline:

* **Query analysis** — parsed query with routing strategy and scope filters.
* **Scored chunks** — retrieval candidates with per-stage scores.
* **Configuration** — all tunable retrieval parameters.
* **Context window** — assembled, hierarchy-aware context for generation.
* **Retrieval result** — final output combining context, scores, and diagnostics.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any


# ── Query routing strategy ──────────────────────────────────────


class QueryStrategy(str, Enum):
    """How the retrieval pipeline should handle a query."""

    EXACT_LOOKUP = "exact_lookup"
    """Query contains only identifiers — bypass vector/BM25,
    use direct metadata lookup."""

    SCOPED_SEMANTIC = "scoped_semantic"
    """Query contains identifiers or scope signals *plus*
    natural language — run hybrid search with metadata filters."""

    UNCONSTRAINED = "unconstrained"
    """Pure natural language — run hybrid search without filters."""


# ── Query analysis ──────────────────────────────────────────────


@dataclass
class QueryAnalysis:
    """Parsed structure of an incoming retrieval query.

    Produced by the query analyzer; consumed by every downstream stage.
    """

    original_query: str = ""
    """The raw query string as submitted by the user."""

    expanded_query: str = ""
    """Query after abbreviation augmentation (original terms preserved,
    expansions appended).  Set by the hybrid search stage."""

    semantic_remainder: str = ""
    """The natural-language portion of the query after stripping
    detected identifiers and scope tokens.  Used for embedding."""

    strategy: QueryStrategy = QueryStrategy.UNCONSTRAINED
    """Routing decision based on query content."""

    # ── Detected identifiers ────────────────────────────────────

    test_case_ids: list[str] = field(default_factory=list)
    """Test case identifiers found in the query,
    e.g. ``["FVTR_OPT_01"]``."""

    requirement_ids: list[str] = field(default_factory=list)
    """Requirement identifiers found in the query,
    e.g. ``["HW-IRS_DIM_VI_275"]``."""

    component_ids: list[str] = field(default_factory=list)
    """Component / item numbers found in the query,
    e.g. ``["7HA-02944-AAAA"]``."""

    cross_references: list[str] = field(default_factory=list)
    """Cross-reference tags found in the query,
    e.g. ``["HWADD:TOP:0012"]``."""

    # ── Scope filters ───────────────────────────────────────────

    scope_filters: dict[str, str] = field(default_factory=dict)
    """ChromaDB ``where`` filter dict derived from detected scope,
    e.g. ``{"doc_type": "FVTR"}`` or ``{"module_name": "DIM-V"}``.
    Empty for ``UNCONSTRAINED`` queries."""

    scope_doc_ids: list[str] = field(default_factory=list)
    """Specific ``doc_id`` values implied by the query scope.
    Used to constrain BM25 post-filtering."""

    @property
    def has_identifiers(self) -> bool:
        """True if any structured identifiers were detected."""
        return bool(
            self.test_case_ids
            or self.requirement_ids
            or self.component_ids
            or self.cross_references
        )


# ── Scored chunk ────────────────────────────────────────────────


@dataclass
class ScoredChunk:
    """A retrieval candidate with per-stage relevance scores.

    Carries the chunk text and metadata through the entire
    retrieve → rerank → assemble pipeline.
    """

    chunk_id: str
    doc_id: str
    text: str
    chunk_type: str = ""
    tier: int = 3

    # ── Scores (set by different pipeline stages) ───────────────
    score: float = 0.0
    """Final fused / composite score used for ranking."""

    vector_score: float = 0.0
    """Cosine similarity score from the vector store (lower distance
    = higher similarity).  Normalised to [0, 1]."""

    bm25_score: float = 0.0
    """BM25 keyword relevance score.  Normalised to [0, 1]."""

    rerank_score: float = 0.0
    """Score assigned by the re-ranker (cross-encoder or LLM).
    0.0 if re-ranking was skipped."""

    # ── Chunk metadata (flat dict, mirrors ChromaDB metadata) ───
    metadata: dict[str, Any] = field(default_factory=dict)


# ── Retrieval configuration ─────────────────────────────────────


@dataclass
class RetrievalConfig:
    """All tunable parameters for the retrieval pipeline.

    Defaults are reasonable starting points; benchmarking (Step 12)
    determines optimal values for the target document corpus.
    """

    # ── Candidate counts ────────────────────────────────────────
    top_k_broad: int = 20
    """Number of candidates from the hybrid search stage."""

    top_k_final: int = 8
    """Number of chunks after re-ranking, sent to context assembly."""

    # ── Score fusion ────────────────────────────────────────────
    rrf_k: int = 60
    """Reciprocal Rank Fusion constant.  Higher values dampen
    rank-position differences between vector and BM25 lists."""

    # ── Re-ranker ───────────────────────────────────────────────
    reranker_type: str = "cross-encoder"
    """Which re-ranker to use: ``"cross-encoder"``, ``"llm"``,
    or ``"none"`` (passthrough)."""

    # ── Context assembly ────────────────────────────────────────
    max_context_tokens: int = 4000
    """Token budget for the assembled context window."""

    # ── Abbreviation handling ───────────────────────────────────
    expand_abbreviations: bool = True
    """Whether to augment queries with abbreviation expansions."""


# ── Context assembly models ─────────────────────────────────────


@dataclass
class ContextSection:
    """One section within the assembled context window.

    Groups related chunks under their tier-2 section heading, with
    optional preamble text for table/figure context.
    """

    section_heading: str = ""
    """Tier-2 section heading text (e.g. ``"5 VERIFICATION OF …"``)."""

    section_number: str = ""
    """Numeric section identifier (e.g. ``"5"``)."""

    preamble: str = ""
    """Introductory prose before the first child chunk — included
    when a table/figure/list chunk needs surrounding context."""

    child_chunks: list[ScoredChunk] = field(default_factory=list)
    """Tier-3 chunks belonging to this section, ordered by their
    position in the original document."""

    doc_id: str = ""
    """Source document identifier."""

    doc_type: str = ""
    """Document type (``"FVTR"``, ``"HwIRS"``, ``""``).  Propagated
    from the document-level metadata."""

    content_type_hint: str = ""
    """Dominant content type in this section: ``"table"``,
    ``"figure"``, ``"test_case"``, ``"requirement"``,
    ``"prose"``, or ``"mixed"``.  Informs generation formatting."""

    token_count: int = 0
    """Total tokens in this section (heading + preamble + children)."""


@dataclass
class ContextWindow:
    """Assembled context ready for injection into a generation prompt.

    Contains ordered sections with hierarchy, provenance metadata,
    and a rendering method for prompt construction.
    """

    sections: list[ContextSection] = field(default_factory=list)
    """Sections in display order (highest-scoring first,
    then by section number for tie-breaking)."""

    total_tokens: int = 0
    """Aggregate token count across all sections."""

    chunk_ids: list[str] = field(default_factory=list)
    """All contributing chunk IDs (for citation tracking)."""

    doc_ids: list[str] = field(default_factory=list)
    """Unique document IDs represented in the context."""

    def to_prompt_text(self) -> str:
        """Render the context window as formatted text for LLM prompt.

        Each section is demarcated with its heading and source document.
        Chunks within a section are separated by blank lines.
        """
        if not self.sections:
            return ""

        parts: list[str] = []
        for section in self.sections:
            # Section header with provenance
            header = f"### {section.section_heading}"
            if section.doc_type:
                header += f"  [{section.doc_type}]"
            parts.append(header)

            # Preamble (introductory prose for tables / figures)
            if section.preamble:
                parts.append(section.preamble)

            # Child chunk texts
            for chunk in section.child_chunks:
                parts.append(chunk.text)

            parts.append("")  # blank line between sections

        return "\n\n".join(parts).rstrip()


# ── Retrieval result ────────────────────────────────────────────


@dataclass
class RetrievalResult:
    """Complete output of a single retrieval call.

    Bundles the assembled context with diagnostics useful for
    debugging, benchmarking, and generation-layer introspection.
    """

    context: ContextWindow = field(default_factory=ContextWindow)
    """The assembled context window for generation."""

    scored_chunks: list[ScoredChunk] = field(default_factory=list)
    """All scored chunks *after* re-ranking (before context assembly
    filtering).  Useful for inspection and evaluation."""

    query_analysis: QueryAnalysis = field(default_factory=QueryAnalysis)
    """Parsed query structure produced by the analyzer."""

    timings: dict[str, float] = field(default_factory=dict)
    """Wall-clock durations (seconds) for each pipeline phase:
    ``{"analysis": …, "search": …, "rerank": …, "assembly": …}``."""

    strategy: str = ""
    """Human-readable label of the strategy that was used,
    e.g. ``"exact_lookup"``, ``"scoped_semantic"``,
    ``"unconstrained"``."""

    error: str = ""
    """Non-empty if the retrieval encountered a recoverable error."""
