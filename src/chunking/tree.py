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


# ── Data structures ─────────────────────────────────────────────


@dataclass
class ContentBlock:
    """A contiguous block of non-heading content within a section.

    Attributes
    ----------
    block_type:
        One of ``"atomic_delimited"``, ``"table"``, ``"list"``,
        ``"figure"``, or ``"prose"``.
    text:
        Raw Markdown text of this block (newlines preserved).
    start_line:
        1-based line number where the block starts in the source Markdown.
    end_line:
        1-based line number where the block ends (inclusive).
    """

    block_type: str
    text: str
    start_line: int
    end_line: int


@dataclass
class SectionNode:
    """A node in the Markdown heading hierarchy.

    The root node (level 0) is a synthetic container that holds the
    entire document.  Real sections correspond to ``#``–``######``
    headings (levels 1–6).

    Attributes
    ----------
    heading:
        Heading text without the ``#`` prefix (e.g. ``"5 VERIFICATION
        OF MECHANICAL CHARACTERISTICS"``).  Empty string for the root.
    level:
        Heading depth: 0 for root, 1 for ``#``, 2 for ``##``, etc.
    section_number:
        Extracted numeric prefix (e.g. ``"1.5.3"``), or ``None``.
    start_line:
        1-based line number of the heading (0 for root).
    end_line:
        1-based line number of the last line belonging to this section
        (set during tree construction).
    children:
        Child ``SectionNode`` objects (sub-headings).
    content_blocks:
        Leaf content between the heading and the first child heading
        (or the end of the section).
    """

    heading: str = ""
    level: int = 0
    section_number: str | None = None
    start_line: int = 0
    end_line: int = 0
    children: list[SectionNode] = field(default_factory=list)
    content_blocks: list[ContentBlock] = field(default_factory=list)


# ── Content segmentation ────────────────────────────────────────


def _segment_content_lines(
    lines: list[str],
    base_line: int,
) -> list[ContentBlock]:
    """Segment a run of non-heading lines into typed ``ContentBlock`` objects.

    Parameters
    ----------
    lines:
        The raw lines (no headings) to segment.
    base_line:
        1-based line number of the first element in *lines* within
        the overall Markdown document.

    Returns
    -------
    list[ContentBlock]
        Ordered content blocks.  Empty input yields an empty list.
    """
    if not lines:
        return []

    blocks: list[ContentBlock] = []

    # ── Pass 1: Extract atomic-delimited blocks ─────────────────
    # The ``---`` lines act as SEPARATORS between atomic blocks,
    # NOT as open/close pairs.  Every run of content between two
    # consecutive ``---`` lines is one ``atomic_delimited`` block.
    # Content before the first ``---`` and after the last ``---``
    # are non-delimited gaps.

    gaps: list[tuple[int, int]] = []       # (start_idx, end_idx) of non-delimited runs
    delimited: list[tuple[int, int]] = []  # (start_idx, end_idx) of delimited blocks

    # Collect indices of all separator lines.
    sep_indices = [
        idx for idx, line in enumerate(lines)
        if _RE_SEPARATOR.match(line)
    ]

    if not sep_indices:
        # No delimiters at all — everything is a gap.
        gaps.append((0, len(lines) - 1))
    else:
        # Content before the first separator → gap.
        if sep_indices[0] > 0:
            gaps.append((0, sep_indices[0] - 1))

        # Content between consecutive separators → atomic_delimited.
        for i in range(len(sep_indices) - 1):
            content_start = sep_indices[i] + 1
            content_end = sep_indices[i + 1] - 1
            if content_end >= content_start:
                delimited.append((content_start, content_end))

        # Content after the last separator → gap.
        if sep_indices[-1] < len(lines) - 1:
            gaps.append((sep_indices[-1] + 1, len(lines) - 1))

    # Build atomic_delimited ContentBlocks.
    for start_idx, end_idx in delimited:
        text = "\n".join(lines[start_idx : end_idx + 1]).strip()
        if text:
            blocks.append(ContentBlock(
                block_type="atomic_delimited",
                text=text,
                start_line=base_line + start_idx,
                end_line=base_line + end_idx,
            ))

    # ── Pass 2: Segment the gaps (non-delimited content) ────────
    for gap_start_idx, gap_end_idx in gaps:
        gap_lines = lines[gap_start_idx : gap_end_idx + 1]
        gap_base = base_line + gap_start_idx
        gap_blocks = _segment_non_delimited(gap_lines, gap_base)
        blocks.extend(gap_blocks)

    # Sort all blocks by start_line so they appear in document order.
    blocks.sort(key=lambda b: b.start_line)

    return blocks


def _segment_non_delimited(
    lines: list[str],
    base_line: int,
) -> list[ContentBlock]:
    """Segment non-delimited lines into table / list / figure / prose blocks.

    Called for content that is NOT between ``---`` delimiters.
    """
    blocks: list[ContentBlock] = []
    if not lines:
        return blocks

    current_type: str | None = None
    current_lines: list[str] = []
    current_start = 0

    def _flush() -> None:
        """Emit the accumulated lines as a ContentBlock."""
        nonlocal current_type, current_lines, current_start
        if current_type is None or not current_lines:
            current_lines = []
            current_type = None
            return
        text = "\n".join(current_lines).strip()
        if text:
            blocks.append(ContentBlock(
                block_type=current_type,
                text=text,
                start_line=base_line + current_start,
                end_line=base_line + current_start + len(current_lines) - 1,
            ))
        current_lines = []
        current_type = None

    for idx, line in enumerate(lines):
        line_type = _classify_line(line)

        # Blank lines: flush current accumulator except inside tables
        # (table separator rows look blank-ish but match _RE_TABLE_ROW).
        if not line.strip():
            if current_type == "table":
                # Blank line ends a table.
                _flush()
            elif current_type is not None:
                # For prose / list: blank line is part of the block
                # (paragraphs are separated by blanks).
                current_lines.append(line)
            continue

        # If the line type matches what we're accumulating, extend.
        if line_type == current_type:
            current_lines.append(line)
            continue

        # Type changed — flush previous, start new.
        # Special case: within a list, non-list continuation lines
        # (indented or prose) stay with the list.
        if current_type == "list" and line_type == "prose" and _is_list_continuation(line):
            current_lines.append(line)
            continue

        _flush()
        current_type = line_type
        current_start = idx
        current_lines = [line]

    _flush()
    return blocks


def _classify_line(line: str) -> str:
    """Return the block type for a single non-blank line."""
    if _RE_TABLE_ROW.match(line):
        return "table"
    if _RE_LIST_ITEM.match(line):
        return "list"
    if _RE_FIGURE.search(line):
        return "figure"
    return "prose"


def _is_list_continuation(line: str) -> bool:
    """Return True if *line* looks like an indented continuation of a list item."""
    # Lines indented by 2+ spaces after a list are continuations.
    return line.startswith("  ") or line.startswith("\t")


# ── Section number extraction ───────────────────────────────────


def _extract_section_number(heading: str) -> tuple[str | None, str]:
    """Extract a leading section number from a heading.

    Returns
    -------
    tuple[str | None, str]
        ``(section_number, remaining_heading_text)``.
        ``section_number`` is ``None`` if no number is found.
    """
    m = _RE_SECTION_NUMBER.match(heading.strip())
    if m:
        num = m.group(1).rstrip(".")
        rest = m.group(2).strip()
        return num, rest
    return None, heading.strip()


# ── Tree builder ────────────────────────────────────────────────


def build_section_tree(markdown: str) -> SectionNode:
    """Parse a Markdown document into a hierarchical section tree.

    Parameters
    ----------
    markdown:
        The full Markdown text of a parsed document (from
        ``cache/markdown/{doc_id}.md``).

    Returns
    -------
    SectionNode
        A synthetic root node (level 0) whose children are the
        top-level sections.  Content before the first heading is
        attached as ``content_blocks`` of the root.
    """
    all_lines = markdown.split("\n")
    total_lines = len(all_lines)

    # Synthetic root node that wraps the entire document.
    root = SectionNode(
        heading="",
        level=0,
        section_number=None,
        start_line=0,
        end_line=total_lines,
    )

    # Stack of (SectionNode, accumulated_content_lines, content_base_line).
    # We track content lines between headings so we can segment them
    # into ContentBlocks once a new heading (or EOF) is reached.
    stack: list[tuple[SectionNode, list[str], int]] = [(root, [], 1)]

    for line_idx, line in enumerate(all_lines):
        line_no = line_idx + 1  # 1-based

        heading_match = _RE_HEADING.match(line)
        if heading_match is None:
            # Non-heading line — accumulate for the current section.
            _top = stack[-1]
            _top[1].append(line)
            continue

        # ── Heading line detected ───────────────────────────────
        hashes = heading_match.group(1)
        heading_text = heading_match.group(2).strip()
        level = len(hashes)

        section_number, _ = _extract_section_number(heading_text)

        new_node = SectionNode(
            heading=heading_text,
            level=level,
            section_number=section_number,
            start_line=line_no,
            end_line=total_lines,  # will be narrowed later
        )

        # Flush accumulated content for the current top-of-stack.
        _flush_content(stack)

        # Pop stack until we find a parent with a strictly lower level.
        while len(stack) > 1 and stack[-1][0].level >= level:
            _finalise_node(stack, line_no - 1)

        # Attach new_node as a child of the current top-of-stack.
        parent = stack[-1][0]
        parent.children.append(new_node)

        # Push new_node with an empty content accumulator.
        # Content lines start on the line AFTER the heading.
        stack.append((new_node, [], line_no + 1))

    # ── EOF: flush everything remaining on the stack ────────────
    _flush_content(stack)
    while len(stack) > 1:
        _finalise_node(stack, total_lines)

    # Finalise the root.
    root.end_line = total_lines

    return root

