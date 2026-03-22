"""Markdown structural tree parser — builds a heading hierarchy with content blocks.

Walks a clean Markdown document line-by-line and produces a tree of
``SectionNode`` objects.  Each section contains child sections (sub-headings)
and ``ContentBlock`` objects representing the leaf content between headings.

The primary chunking boundary is the ``---`` horizontal-rule delimiter that
the parser inserts around FVTR test cases and HwIRS requirements.  Everything
between two consecutive ``---`` lines is kept together as a single
``atomic_delimited`` content block — even if it contains embedded tables,
lists, or figure references.

Content that is NOT inside ``---`` delimiters is segmented into separate
block types (``table``, ``list``, ``figure``, ``prose``) so the chunker
can apply type-specific size rules.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field


# ── Regex patterns ──────────────────────────────────────────────

# Matches a Markdown heading line:  # Title  /  ## Section  / etc.
_RE_HEADING = re.compile(r"^(#{1,6})\s+(.+)$")

# Matches the leading section number in a heading, e.g.
# "1.5.3 Standards" → group(1)="1.5.3", group(2)="Standards"
# "5 VERIFICATION" → group(1)="5", group(2)="VERIFICATION"
# Handles both "1." and "1" numbering styles.
_RE_SECTION_NUMBER = re.compile(
    r"^(\d+(?:\.\d+)*\.?)\s+(.*)"
)

# Matches a horizontal rule used as an atomic-block delimiter.
# Must be exactly "---" (possibly with surrounding whitespace) on its
# own line.  We intentionally do NOT match "----" or "***" variants
# to avoid false positives.
_RE_SEPARATOR = re.compile(r"^\s*-{3}\s*$")

# Matches a Markdown pipe-table row:  | cell | cell | ...
_RE_TABLE_ROW = re.compile(r"^\|.*\|\s*$")

# Matches the start of an unordered or ordered list item.
_RE_LIST_ITEM = re.compile(r"^(\s*[-*+]\s|\s*\d+[.)]\s)")

# Matches a figure reference or VLM description tag.
_RE_FIGURE = re.compile(
    r"\[VLM\s*-|"           # [VLM - Figure N]
    r"\[Figure\s*[—–-]|"    # [Figure — see source document]
    r"^Figure\s+\d+:"       # Figure 1: Test Set-up
    , re.IGNORECASE
)


