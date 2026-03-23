"""Content-block classifier — maps each ``ContentBlock`` to a ``ChunkType``.

Takes a ``ContentBlock`` (from the structural tree parser) and the
``SectionNode`` it belongs to, and returns the appropriate ``ChunkType``
enum value.  This is used by the chunk-assembly layer to decide which
metadata extractors to run and how to size each chunk.
"""

from __future__ import annotations

import re

from src.chunking.models import ChunkType
from src.chunking.tree import ContentBlock, SectionNode


# ── Regex patterns for sub-classification ───────────────────────

# Matches FVTR test-case blocks: **Test case:** FVTR_xxx
_RE_TEST_CASE = re.compile(r"\*\*Test\s+case:\*\*", re.IGNORECASE)

# Matches HwIRS requirement IDs: **HW-IRS_DIM_VI_275** or HW-IRS_PAM_219
_RE_REQUIREMENT_ID = re.compile(r"HW-IRS_\w+", re.IGNORECASE)

# Matches the [Category: ... | Allocation: ... ] metadata tag found in
# HwIRS requirement blocks (including those without an explicit HW-IRS_ ID).
_RE_CATEGORY_TAG = re.compile(
    r"\[Category:\s*\w+\s*\|", re.IGNORECASE
)

# Matches section headings that indicate a definition table.
_RE_DEFINITIONS = re.compile(
    r"\bdefinitions?\b", re.IGNORECASE
)

# Matches section headings that indicate an abbreviation table.
_RE_ABBREVIATIONS = re.compile(
    r"\babbreviations?\b", re.IGNORECASE
)


# ── Public API ──────────────────────────────────────────────────


def classify_block(
    block: ContentBlock,
    section_context: SectionNode,
) -> ChunkType:
    """Determine the ``ChunkType`` for a given content block.

    Parameters
    ----------
    block:
        A ``ContentBlock`` produced by ``build_section_tree()``.
    section_context:
        The ``SectionNode`` that directly contains *block*.  Used to
        resolve context-dependent types (e.g. definition vs. plain
        table).

    Returns
    -------
    ChunkType
        The semantic type to assign to the chunk built from *block*.
    """
    bt = block.block_type

    if bt == "atomic_delimited":
        return _classify_atomic(block.text)

    if bt == "table":
        return _classify_table(section_context)

    if bt == "list":
        return ChunkType.LIST

    if bt == "figure":
        return ChunkType.FIGURE

    # Default: prose
    return ChunkType.PROSE


# ── Private helpers ─────────────────────────────────────────────


def _classify_atomic(text: str) -> ChunkType:
    """Sub-classify an ``atomic_delimited`` block."""
    if _RE_TEST_CASE.search(text):
        return ChunkType.TEST_CASE

    if _RE_REQUIREMENT_ID.search(text) or _RE_CATEGORY_TAG.search(text):
        return ChunkType.REQUIREMENT

    return ChunkType.PROSE


def _classify_table(section: SectionNode) -> ChunkType:
    """Sub-classify a table block based on its section heading."""
    heading = section.heading

    if _RE_DEFINITIONS.search(heading):
        return ChunkType.DEFINITION_TABLE

    if _RE_ABBREVIATIONS.search(heading):
        return ChunkType.ABBREVIATION_TABLE

    return ChunkType.TABLE
