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



