"""Chunking data models — types, tiers, and structured chunk representations.

Defines the core data structures for the three-tier chunking pipeline:

* **Tier 1 — Document**:  One per ingested file; holds document-level metadata.
* **Tier 2 — Section**:   One per major (``##``) heading; context container.
* **Tier 3 — Atomic**:    Leaf chunks (test cases, requirements, tables, prose, …).

Every chunk carries rich metadata and explicit parent/child links so the
retrieval layer can navigate the hierarchy at query time.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any


# ── Enumerations ────────────────────────────────────────────────


class ChunkType(str, Enum):
    """Semantic type of a chunk's content."""

    DOCUMENT_META = "document_meta"
    FRONT_MATTER = "front_matter"
    SECTION = "section"
    TEST_CASE = "test_case"
    REQUIREMENT = "requirement"
    TABLE = "table"
    DEFINITION_TABLE = "definition_table"
    ABBREVIATION_TABLE = "abbreviation_table"
    PROSE = "prose"
    FIGURE = "figure"
    LIST = "list"


class ChunkTier(int, Enum):
    """Hierarchy level within the three-tier chunk model."""

    DOCUMENT = 1
    SECTION = 2
    ATOMIC = 3


# ── Chunk-level metadata ───────────────────────────────────────


@dataclass
class ChunkMetadata:
    """Rich metadata attached to every chunk.

    Fields are populated selectively based on ``ChunkType``:

    * **Universal fields** are set for all chunks.
    * **Test-case fields** are set only when ``chunk_type == TEST_CASE``.
    * **Requirement fields** are set only when ``chunk_type == REQUIREMENT``.
    """

    # ── Universal fields (all chunk types) ──────────────────────
    section_path: list[str] = field(default_factory=list)
    """Ordered list of heading texts from root to the chunk's
    nearest enclosing section, e.g.
    ``["1 INTRODUCTION", "1.5 REFERENCES", "1.5.2 Additional references"]``."""

    section_number: str | None = None
    """Extracted numeric section identifier (e.g. ``"1.5.2"``)."""

    heading: str | None = None
    """Text of the nearest enclosing heading."""

    has_table: bool = False
    """True if the chunk text contains a Markdown pipe-table."""

    has_figure: bool = False
    """True if the chunk text contains a figure reference or VLM description."""

    cross_references: list[str] = field(default_factory=list)
    """Bracket-style document references found in the text,
    e.g. ``["HWADD:TOP:0012", "CD_PAM"]``."""

    component_ids: list[str] = field(default_factory=list)
    """Thales item numbers found in the text,
    e.g. ``["7HA-02944-AAAA"]``."""

    # ── Test-case fields (ChunkType.TEST_CASE only) ─────────────
    test_case_id: str | None = None
    """Identifier like ``"FVTR_OPT_01"`` or ``"FVTR_FUNC_13"``."""

    test_name: str | None = None
    """Short name of the test (e.g. ``"Labelling and assembly"``)."""

    test_result: str | None = None
    """Outcome: ``"Passed"``, ``"Failed"``, etc."""

    test_item: str | None = None
    """Unit(s) under test (e.g. ``"MAV8"`` or ``"MAV6, MAV7"``)."""

    date: str | None = None
    """Date the test was performed."""

    tester: str | None = None
    """Person who carried out the test."""

    verifier: str | None = None
    """Person who verified the test."""

    failure_criteria: str | None = None
    """Stated failure criteria for the test."""

    # ── Shared traceability (TEST_CASE and REQUIREMENT) ─────────
    traceability_ids: list[str] = field(default_factory=list)
    """IDs from ``**Traceability:**`` lines, e.g.
    ``["FVTSR_PAM_0002", "HW-IRS_PAM_219"]``."""

    reference_ids: list[str] = field(default_factory=list)
    """IDs from ``**Reference:**`` lines (test cases only)."""

    # ── Requirement fields (ChunkType.REQUIREMENT only) ─────────
    requirement_ids: list[str] = field(default_factory=list)
    """All ``HW-IRS_xxx_nnn`` identifiers in the block."""

    category: str | None = None
    """DOORS category: ``"Requirement"``, ``"Background"``,
    ``"Definition"``, ``"Picture"``, etc."""

    allocation: str | None = None
    """DOORS allocation: ``"HW"``, ``"NA"``, etc."""

    priority: str | None = None
    """DOORS priority: ``"Mandatory"``, ``"NA"``, etc."""

    safety: str | None = None
    """DOORS safety flag: ``"Yes"``, ``"No"``, ``"NA"``."""

    verification_method: str | None = None
    """DOORS verification method: ``"Test"``, ``"Analysis"``,
    ``"Inspection"``, ``"Demonstration"``, ``"NA"``."""

    is_background: bool = False
    """True when the requirement block's category is ``"Background"``
    (contextual info, not a testable requirement)."""


# ── Document-level metadata ─────────────────────────────────────


@dataclass
class DocumentMeta:
    """Metadata extracted from the document's front-matter and headings.

    Populated once per document and inherited by every chunk via the
    Tier 1 document node.
    """

    doc_title: str = ""
    """Primary document title (first ``##`` heading)."""

    doc_type: str = ""
    """Classified document type: ``"FVTR"``, ``"HwIRS"``, or ``""``
    for unrecognised formats."""

    module_name: str = ""
    """Short module identifier, e.g. ``"PAM"``, ``"DIM-V"``."""

    module_full_name: str = ""
    """Full module name, e.g. ``"PWM and Analogue I/O Module"``."""

    system: str = ""
    """System identifier, e.g. ``"TOP"``."""

    source_file: str = ""
    """Original filename of the ingested document."""

    revision: str = ""
    """Latest revision number from the change-log table."""

    revision_date: str = ""
    """Date of the latest revision."""

    authors: list[str] = field(default_factory=list)
    """Names from the ``Written by`` row of the approval table."""

    approvers: list[str] = field(default_factory=list)
    """Names from the ``Approved by`` row of the approval table."""

    page_count: int = 0
    """Number of pages in the source document."""


# ── Chunk ───────────────────────────────────────────────────────


@dataclass
class Chunk:
    """A single chunk in the three-tier hierarchy.

    Attributes
    ----------
    chunk_id:
        Deterministic identifier: ``sha256(doc_id + section_path + index)``.
    doc_id:
        SHA-256 hex digest of the source file (from the ingestion registry).
    chunk_type:
        Semantic type (test case, requirement, table, prose, …).
    tier:
        Hierarchy level (1 = document, 2 = section, 3 = atomic).
    text:
        The chunk's Markdown content.
    token_count:
        Number of tokens in *text* (computed via ``tiktoken``).
    parent_id:
        ``chunk_id`` of the parent chunk (Tier 2 for atomics,
        Tier 1 for sections, ``None`` for the document node).
    children_ids:
        ``chunk_id`` values of direct children.  Populated for
        Tier 1 and Tier 2 chunks; empty for Tier 3 leaves.
    metadata:
        Rich metadata specific to this chunk.
    """

    chunk_id: str
    doc_id: str
    chunk_type: ChunkType
    tier: ChunkTier
    text: str
    token_count: int = 0
    parent_id: str | None = None
    children_ids: list[str] = field(default_factory=list)
    metadata: ChunkMetadata = field(default_factory=ChunkMetadata)\


# ── Chunked document (top-level container) ──────────────────────


@dataclass
class ChunkedDocument:
    """Complete chunking result for a single ingested document.

    Serialised to ``cache/chunk/{doc_id}.json`` by the writer module.
    """

    doc_id: str
    doc_metadata: DocumentMeta = field(default_factory=DocumentMeta)
    chunks: list[Chunk] = field(default_factory=list)

    # ── Convenience accessors ───────────────────────────────────

    @property
    def document_chunk(self) -> Chunk | None:
        """Return the single Tier 1 (document) chunk, if present."""
        for c in self.chunks:
            if c.tier == ChunkTier.DOCUMENT:
                return c
        return None

    @property
    def section_chunks(self) -> list[Chunk]:
        """Return all Tier 2 (section) chunks."""
        return [c for c in self.chunks if c.tier == ChunkTier.SECTION]

    @property
    def atomic_chunks(self) -> list[Chunk]:
        """Return all Tier 3 (atomic / leaf) chunks."""
        return [c for c in self.chunks if c.tier == ChunkTier.ATOMIC]

    def get_chunk(self, chunk_id: str) -> Chunk | None:
        """Look up a chunk by its ID."""
        for c in self.chunks:
            if c.chunk_id == chunk_id:
                return c
        return None

    def get_children(self, chunk_id: str) -> list[Chunk]:
        """Return the direct children of the given chunk."""
        parent = self.get_chunk(chunk_id)
        if parent is None:
            return []
        return [c for c in self.chunks if c.chunk_id in parent.children_ids]
