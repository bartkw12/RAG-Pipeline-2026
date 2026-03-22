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
