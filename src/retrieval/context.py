"""Table/figure-aware context assembly — expand re-ranked chunks into
a structured, hierarchy-aware context window for generation.

Transforms a flat list of re-ranked ``ScoredChunk`` objects into a
``ContextWindow`` of grouped ``ContextSection`` s, applying:

* **Parent expansion** — resolve each chunk's tier-2 section heading.
* **Content-type-aware preamble** — for table, figure, and list chunks
  include the immediately preceding prose sibling that introduces them.
* **Sibling grouping** — chunks sharing a tier-2 parent are merged
  under one section header (no repeated headings).
* **Token budget** — cumulative tracking via ``count_tokens()``, stops
  adding sections when reaching ``config.max_context_tokens``.
* **Ordering** — sections sorted by best chunk score, then by section
  number for tie-breaking; children ordered by document position.

Usage::

    from src.retrieval.context import assemble_context

    window = assemble_context(reranked_chunks, config)
    prompt_text = window.to_prompt_text()
"""

from __future__ import annotations

import logging
from collections import OrderedDict

from ..chunking.models import Chunk, ChunkedDocument, ChunkTier, ChunkType
from ..chunking.tokens import count_tokens
from ..chunking.writer import load_chunks
from .models import ContextSection, ContextWindow, RetrievalConfig, ScoredChunk

logger = logging.getLogger(__name__)

# Chunk types that benefit from a preceding prose preamble.
_NEEDS_PREAMBLE = {
    ChunkType.TABLE,
    ChunkType.DEFINITION_TABLE,
    ChunkType.FIGURE,
    ChunkType.LIST,
}


def assemble_context(
    chunks: list[ScoredChunk],
    config: RetrievalConfig,
) -> ContextWindow:
    """Build a hierarchy-aware ``ContextWindow`` from re-ranked chunks.

    Parameters
    ----------
    chunks:
        Re-ranked ``ScoredChunk`` list (typically 5–8 after re-ranking).
    config:
        Retrieval configuration — mainly ``max_context_tokens``.

    Returns
    -------
    ContextWindow
        Assembled context with sections, token counts, and provenance.
    """
    if not chunks:
        return ContextWindow()

    # ── 1. Load chunk hierarchies for each referenced document ──
    doc_cache: dict[str, ChunkedDocument] = {}
    for sc in chunks:
        if sc.doc_id and sc.doc_id not in doc_cache:
            try:
                doc_cache[sc.doc_id] = load_chunks(sc.doc_id)
            except FileNotFoundError:
                logger.warning("Chunk file for doc_id=%s… not found.", sc.doc_id[:12])

    # ── 2. Build chunk_id → Chunk lookup across all docs ────────
    chunk_lookup: dict[str, Chunk] = {}
    for doc in doc_cache.values():
        for c in doc.chunks:
            chunk_lookup[c.chunk_id] = c

    # ── 3. Group scored chunks by their tier-2 parent ───────────
    #    key = parent_id (or chunk_id itself if no parent found)
    #    value = list of (ScoredChunk, sibling_index) in insertion order
    SectionGroup = list[tuple[ScoredChunk, int]]
    groups: OrderedDict[str, SectionGroup] = OrderedDict()
    group_best_score: dict[str, float] = {}

    for sc in chunks:
        raw = chunk_lookup.get(sc.chunk_id)
        parent_id = raw.parent_id if raw and raw.parent_id else sc.chunk_id

        # Determine the child's position within its parent's children
        # (used later for document-order sorting within a section).
        sibling_idx = _sibling_index(raw, chunk_lookup) if raw else 0

        groups.setdefault(parent_id, []).append((sc, sibling_idx))
        if parent_id not in group_best_score or sc.score > group_best_score[parent_id]:
            group_best_score[parent_id] = sc.score

    # ── 4. Sort groups: best score desc, then section_number asc ─
    def _sort_key(parent_id: str) -> tuple[float, str]:
        parent = chunk_lookup.get(parent_id)
        sec_num = parent.metadata.section_number if parent and parent.metadata.section_number else "zzz"
        return (-group_best_score.get(parent_id, 0.0), sec_num)

    sorted_parent_ids = sorted(groups.keys(), key=_sort_key)

    # ── 5. Assemble sections within token budget ────────────────
    sections: list[ContextSection] = []
    running_tokens = 0
    all_chunk_ids: list[str] = []
    seen_doc_ids: set[str] = set()

    for parent_id in sorted_parent_ids:
        group = groups[parent_id]
        parent_chunk = chunk_lookup.get(parent_id)

        # Resolve section heading.
        heading = ""
        section_number = ""
        doc_id = ""
        doc_type = ""
        if parent_chunk and parent_chunk.tier == ChunkTier.SECTION:
            heading = parent_chunk.metadata.heading or parent_chunk.text.split("\n", 1)[0]
            section_number = parent_chunk.metadata.section_number or ""
            doc_id = parent_chunk.doc_id
        elif group:
            doc_id = group[0][0].doc_id

        # Infer doc_type from the document metadata.
        if doc_id and doc_id in doc_cache:
            doc_type = doc_cache[doc_id].doc_metadata.doc_type

        # Sort children by their sibling index (document order).
        group.sort(key=lambda t: t[1])

        # Build the preamble and collect child texts.
        preamble = ""
        child_scored: list[ScoredChunk] = []
        preamble_ids: set[str] = set()

        for sc, sib_idx in group:
            raw_chunk = chunk_lookup.get(sc.chunk_id)
            chunk_type = ChunkType(sc.chunk_type) if sc.chunk_type else None

            # Detect whether this chunk needs a preamble.
            needs_preamble = (
                chunk_type in _NEEDS_PREAMBLE
                or (raw_chunk and raw_chunk.metadata.has_table)
                or (raw_chunk and raw_chunk.metadata.has_figure)
            )

            if needs_preamble and raw_chunk and parent_chunk:
                pre = _find_preceding_prose(raw_chunk, parent_chunk, chunk_lookup)
                if pre and pre.chunk_id not in preamble_ids:
                    preamble_ids.add(pre.chunk_id)
                    if preamble:
                        preamble += "\n\n" + pre.text
                    else:
                        preamble = pre.text

            child_scored.append(sc)

        if not child_scored:
            continue

        # Estimate token cost of this section.
        section_text_parts = [heading, preamble] + [c.text for c in child_scored]
        section_tokens = count_tokens("\n\n".join(p for p in section_text_parts if p))

        # Check budget.  Always allow at least one section through
        # so we never return completely empty context.
        if running_tokens + section_tokens > config.max_context_tokens:
            if sections:
                logger.info(
                    "Token budget reached (%d + %d > %d). Stopping at %d sections.",
                    running_tokens, section_tokens, config.max_context_tokens, len(sections),
                )
                break
            # First section exceeds budget — still include it but stop here.
            logger.info(
                "First section (%d tokens) exceeds budget (%d). Including it anyway.",
                section_tokens, config.max_context_tokens,
            )

        # Determine content type hint.
        content_hint = _infer_content_hint([sc.chunk_type for sc in child_scored])

        section = ContextSection(
            section_heading=heading,
            section_number=section_number,
            preamble=preamble,
            child_chunks=child_scored,
            doc_id=doc_id,
            doc_type=doc_type,
            content_type_hint=content_hint,
            token_count=section_tokens,
        )
        sections.append(section)
        running_tokens += section_tokens

        for sc in child_scored:
            all_chunk_ids.append(sc.chunk_id)
        for pid in preamble_ids:
            all_chunk_ids.append(pid)
        if doc_id:
            seen_doc_ids.add(doc_id)

    window = ContextWindow(
        sections=sections,
        total_tokens=running_tokens,
        chunk_ids=all_chunk_ids,
        doc_ids=sorted(seen_doc_ids),
    )

    logger.info(
        "Context assembled: %d sections, %d tokens, %d chunks from %d doc(s).",
        len(sections), running_tokens, len(all_chunk_ids), len(seen_doc_ids),
    )
    return window


# ── Internal helpers ────────────────────────────────────────────


def _sibling_index(chunk: Chunk, lookup: dict[str, Chunk]) -> int:
    """Return the 0-based position of *chunk* within its parent's children."""
    if not chunk.parent_id:
        return 0
    parent = lookup.get(chunk.parent_id)
    if not parent:
        return 0
    try:
        return parent.children_ids.index(chunk.chunk_id)
    except ValueError:
        return 0


def _find_preceding_prose(
    chunk: Chunk,
    parent: Chunk,
    lookup: dict[str, Chunk],
) -> Chunk | None:
    """Find the prose sibling immediately before *chunk* in document order.

    Walks backwards through the parent's ``children_ids`` from
    *chunk*'s position.  Returns the first ``PROSE`` sibling found,
    or ``None`` if there is none.
    """
    try:
        idx = parent.children_ids.index(chunk.chunk_id)
    except ValueError:
        return None

    # Walk backwards from the sibling just before this chunk.
    for i in range(idx - 1, -1, -1):
        sib = lookup.get(parent.children_ids[i])
        if sib and sib.chunk_type == ChunkType.PROSE:
            return sib
        # Stop at non-prose (avoid jumping over a table to find
        # an unrelated earlier prose block).
        break

    return None


def _infer_content_hint(chunk_types: list[str]) -> str:
    """Infer the dominant content type label for a section group."""
    if not chunk_types:
        return "prose"

    type_set = set(chunk_types)

    # Single type → use it directly.
    if len(type_set) == 1:
        ct = type_set.pop()
        if ct in ("table", "definition_table"):
            return "table"
        return ct

    # Mixed — check for dominant patterns.
    if "test_case" in type_set:
        return "test_case"
    if "requirement" in type_set:
        return "requirement"
    if "table" in type_set or "definition_table" in type_set:
        return "table"
    if "figure" in type_set:
        return "figure"
    return "mixed"
