"""Chunk-assembly strategy — the main orchestrator for the chunking pipeline.

Walks the structural tree built by ``tree.py``, classifies every content
block, applies token-budget rules (merge / keep / split), attaches rich
metadata, and produces a fully linked ``ChunkedDocument``.

Public API
----------
``chunk_document(markdown, doc_id, config) -> ChunkedDocument``
    Entry point called by the pipeline layer.

``extract_document_metadata(markdown, tree) -> DocumentMeta``
    Parses front-matter headings and tables for document-level metadata.
"""

from __future__ import annotations

import hashlib
import re
from dataclasses import replace

from src.chunking.classify import classify_block
from src.chunking.config import ChunkConfig
from src.chunking.extractors import (
    detect_embedded_figures,
    detect_embedded_tables,
    extract_component_ids,
    extract_cross_references,
    extract_requirement_fields,
    extract_test_case_fields,
)
from src.chunking.models import (
    Chunk,
    ChunkedDocument,
    ChunkMetadata,
    ChunkTier,
    ChunkType,
    DocumentMeta,
)
from src.chunking.tokens import count_tokens
from src.chunking.tree import SectionNode, build_section_tree


# ── Regex helpers for document metadata extraction ──────────────

_RE_TABLE_ROW = re.compile(r"^\|.*\|", re.MULTILINE)
_RE_PIPE_CELLS = re.compile(r"\|")


# ── Public API ──────────────────────────────────────────────────


def chunk_document(
    markdown: str,
    doc_id: str,
    config: ChunkConfig | None = None,
) -> ChunkedDocument:
    """Chunk a parsed Markdown document into a three-tier hierarchy.

    Parameters
    ----------
    markdown:
        Full Markdown text (from ``cache/markdown/{doc_id}.md``).
    doc_id:
        SHA-256 hex digest that identifies the source file.
    config:
        Chunking configuration.  Uses defaults when ``None``.

    Returns
    -------
    ChunkedDocument
        Complete chunking result with Tier 1 / 2 / 3 chunks.
    """
    if config is None:
        config = ChunkConfig()

    # 1. Build the structural tree.
    tree = build_section_tree(markdown)

    # 2. Extract document-level metadata.
    doc_meta = extract_document_metadata(markdown, tree)

    # 3. Create Tier 1 — document chunk.
    doc_chunk_id = _make_id(doc_id, ["__document__"], 0, 0)
    doc_text = _build_document_summary(doc_meta)
    doc_chunk = Chunk(
        chunk_id=doc_chunk_id,
        doc_id=doc_id,
        chunk_type=ChunkType.DOCUMENT_META,
        tier=ChunkTier.DOCUMENT,
        text=doc_text,
        token_count=count_tokens(doc_text, config.encoding_name),
        parent_id=None,
        children_ids=[],
        metadata=ChunkMetadata(),
    )

    all_chunks: list[Chunk] = [doc_chunk]

    # Identify front-matter sections (everything before the first
    # section that has a numeric section_number, e.g. "1 Introduction").
    front_matter_sections: list[SectionNode] = []
    body_sections: list[SectionNode] = []
    found_numbered = False
    for section in tree.children:
        if not found_numbered and section.section_number is None:
            front_matter_sections.append(section)
        else:
            found_numbered = True
            body_sections.append(section)

    # Also include root-level content blocks (before the first heading)
    # as front-matter.
    root_blocks = tree.content_blocks

    # 4a. Create front-matter Tier 2 + Tier 3 chunks.
    if front_matter_sections or root_blocks:
        fm_chunks = _build_front_matter_chunks(
            root_blocks, front_matter_sections, doc_id, doc_chunk_id, config,
        )
        all_chunks.extend(fm_chunks)
        # Link the section chunk (first item) to the document.
        if fm_chunks:
            doc_chunk.children_ids.append(fm_chunks[0].chunk_id)

    # 4b & 5. Walk numbered ## sections → Tier 2 + depth-first → Tier 3.
    for sec_idx, section in enumerate(body_sections):
        if section.level != 2:
            # Some documents have level-1 headings at top; treat the
            # same as level-2 for Tier 2 purposes.
            pass

        section_path = [section.heading]
        sec_chunk_id = _make_id(doc_id, ["__section__"] + section_path, sec_idx, 0)
        sec_text = _build_section_summary(section)
        sec_chunk = Chunk(
            chunk_id=sec_chunk_id,
            doc_id=doc_id,
            chunk_type=ChunkType.SECTION,
            tier=ChunkTier.SECTION,
            text=sec_text,
            token_count=count_tokens(sec_text, config.encoding_name),
            parent_id=doc_chunk_id,
            children_ids=[],
            metadata=ChunkMetadata(
                section_path=section_path,
                section_number=section.section_number,
                heading=section.heading,
            ),
        )

        # Collect all leaf content blocks from this section subtree.
        leaf_blocks = _collect_leaf_blocks(section)

        # Apply merge / keep / split rules → produce Tier 3 chunks.
        atomic_chunks = _process_leaf_blocks(
            leaf_blocks, doc_id, section_path, sec_idx, sec_chunk_id, config,
        )

        # Link section → atomics.
        sec_chunk.children_ids = [c.chunk_id for c in atomic_chunks]
        all_chunks.append(sec_chunk)
        all_chunks.extend(atomic_chunks)

        # Link document → section.
        doc_chunk.children_ids.append(sec_chunk_id)

    return ChunkedDocument(
        doc_id=doc_id,
        doc_metadata=doc_meta,
        chunks=all_chunks,
    )


# ── Document metadata extraction ────────────────────────────────


def extract_document_metadata(
    markdown: str,
    tree: SectionNode,
) -> DocumentMeta:
    """Parse front-matter headings and tables for document-level info.

    Extracts title, module name, doc type, system, revision info,
    and author / approver names.
    """
    meta = DocumentMeta()

    headings = [c.heading for c in tree.children]

    # Title & doc type — scan the first few ## headings.
    _extract_title_and_type(headings, meta)

    # Module name — second heading often names the module.
    _extract_module_info(headings, meta)

    # System — look for "TOP" or similar.
    _extract_system(headings, meta)

    # Revision info — parse the LOG OF CHANGES table.
    _extract_revision(tree, markdown, meta)

    # Authors / approvers — parse the APPROVAL table.
    _extract_approval(markdown, meta)

    return meta


# ── Title / doc-type extraction helpers ─────────────────────────

_FVTR_KEYWORDS = re.compile(
    r"verification\s+test\s+report|FVTR", re.IGNORECASE
)
_HWIRS_KEYWORDS = re.compile(
    r"requirements?\s+specification|HwIRS|HWIRS", re.IGNORECASE
)


def _extract_title_and_type(headings: list[str], meta: DocumentMeta) -> None:
    """Set ``doc_title`` and ``doc_type`` from the first headings."""
    for h in headings[:5]:
        if _FVTR_KEYWORDS.search(h):
            meta.doc_title = h
            meta.doc_type = "FVTR"
            return
        if _HWIRS_KEYWORDS.search(h):
            meta.doc_title = h
            meta.doc_type = "HwIRS"
            return
    # Fallback: use the first heading as title.
    if headings:
        meta.doc_title = headings[0]


_RE_MODULE_PARENS = re.compile(r"\(([A-Z][A-Z0-9-]+)\)")
_RE_MODULE_DASH = re.compile(
    r"(?:module|board)\s*[-–—]\s*(?:vital\s*)?\(?([A-Z][A-Z0-9-]+)",
    re.IGNORECASE,
)

# Abbreviations that refer to the system or doc type, not the module.
_NON_MODULE_ABBREVS = frozenset({"TOP", "FVTR", "HWIRS", "TAS"})


def _extract_module_info(headings: list[str], meta: DocumentMeta) -> None:
    """Set ``module_name`` and ``module_full_name``."""
    # First pass: look for a dedicated module heading.
    for h in headings[:5]:
        if re.search(r"follow-up|introduction|evolutions", h, re.IGNORECASE):
            continue
        m = _RE_MODULE_PARENS.search(h)
        if m and m.group(1).upper() not in _NON_MODULE_ABBREVS:
            meta.module_name = m.group(1)
            full = h[: m.start()].strip().rstrip("-–—").strip()
            if full:
                meta.module_full_name = full
            return
        m2 = _RE_MODULE_DASH.search(h)
        if m2 and m2.group(1).upper() not in _NON_MODULE_ABBREVS:
            meta.module_name = m2.group(1)
            return

    # Second pass: extract module from doc-type headings that embed
    # module info, e.g. "HwIRS - Digital Inputs Module - Vital (DIM Vital)".
    for h in headings[:5]:
        if _FVTR_KEYWORDS.search(h) or _HWIRS_KEYWORDS.search(h):
            for m in _RE_MODULE_PARENS.finditer(h):
                abbrev = m.group(1)
                if abbrev.upper() not in _NON_MODULE_ABBREVS:
                    meta.module_name = abbrev
                    return
            m2 = _RE_MODULE_DASH.search(h)
            if m2 and m2.group(1).upper() not in _NON_MODULE_ABBREVS:
                meta.module_name = m2.group(1)
                return

    # Fallback: look for known short names.
    combined = " ".join(headings[:5])
    for candidate in ("DIM-V", "DIM", "PAM", "DOM", "AOM", "COM"):
        if candidate in combined:
            meta.module_name = candidate
            break


def _extract_system(headings: list[str], meta: DocumentMeta) -> None:
    """Set ``system`` — usually 'TOP'."""
    combined = " ".join(headings[:5])
    if "TOP" in combined or "TAS-TOP" in combined:
        meta.system = "TOP"


def _extract_revision(
    tree: SectionNode, markdown: str, meta: DocumentMeta,
) -> None:
    """Extract latest revision from the LOG OF CHANGES table.

    Falls back to the APPROVAL table's Date column when no dedicated
    revision log is present (e.g. DIM-V FVTR).
    """
    # Find the "Follow-up of the evolutions" section.
    evolutions = None
    for child in tree.children:
        if re.search(r"follow-up|evolutions", child.heading, re.IGNORECASE):
            evolutions = child
            break
    if evolutions is None:
        return

    # Look for the LOG OF CHANGES table (not the APPROVAL table).
    for block in evolutions.content_blocks:
        if block.block_type != "table":
            continue
        upper = block.text.upper()
        # The revision table contains "LOG OF CHANGES" or a
        # "Revision" column header.  APPROVAL tables don't.
        if "LOG OF CHANGES" not in upper and "REVISION" not in upper:
            continue
        rows = _parse_table_rows(block.text)
        if rows:
            last = rows[-1]
            if len(last) >= 2:
                meta.revision = last[0].strip()
                meta.revision_date = last[1].strip()
        break

    # Fallback: if no revision was found, try extracting the latest
    # date from the APPROVAL table's "Date" column.
    if not meta.revision_date:
        _extract_revision_from_approval(evolutions, meta)


_RE_DATE = re.compile(r"\d{4}-\d{2}-\d{2}")


def _extract_revision_from_approval(
    section: SectionNode, meta: DocumentMeta,
) -> None:
    """Extract the most recent date from the APPROVAL table as a fallback."""
    for block in section.content_blocks:
        if block.block_type != "table":
            continue
        if "APPROVAL" not in block.text.upper():
            continue
        # Find all YYYY-MM-DD dates in the table rows.
        dates: list[str] = []
        rows = _parse_table_rows(block.text)
        for row in rows:
            for cell in row:
                m = _RE_DATE.search(cell)
                if m:
                    dates.append(m.group())
        if dates:
            dates.sort()
            meta.revision_date = dates[-1]
        break


def _extract_approval(markdown: str, meta: DocumentMeta) -> None:
    """Extract author / approver names from the APPROVAL table."""
    # The approval table is typically within the first ~30 lines of
    # actual table content. We search for rows containing "Written by"
    # or "Approved by".
    lines = markdown.split("\n")
    in_approval = False
    for line in lines[:80]:
        if not _RE_TABLE_ROW.match(line):
            if in_approval:
                break
            continue
        low = line.lower()
        if "approval" in low or "written by" in low or "approved by" in low:
            in_approval = True
        if not in_approval:
            continue
        cells = _split_table_cells(line)
        if len(cells) < 2:
            continue
        label = cells[0].strip().lower()
        name = cells[1].strip()
        if "written by" in label and name:
            meta.authors = [name]
        elif "approved by" in label and name:
            meta.approvers = [name]


# ── Front-matter chunk builder ──────────────────────────────────


def _build_front_matter_chunks(
    root_blocks: list,
    front_matter_sections: list[SectionNode],
    doc_id: str,
    doc_chunk_id: str,
    config: ChunkConfig,
) -> list[Chunk]:
    """Create a synthetic 'Front Matter' Tier 2 section plus Tier 3 children.

    The front matter includes root-level content blocks (before the first
    heading) and any un-numbered sections that precede the first numbered
    body section (title pages, revision logs, approval tables, etc.).

    Returns
    -------
    list[Chunk]
        The Tier 2 section chunk followed by zero or more Tier 3 chunks.
        The caller links the first element (Tier 2) to the document chunk.
    """
    section_path = ["Front Matter"]
    sec_chunk_id = _make_id(doc_id, ["__section__"] + section_path, 0, 0)

    # Collect (text, node_or_None) pairs from root blocks, then from each
    # front-matter section's subtree.
    raw_items: list[tuple[str, SectionNode | None]] = []

    # Synthetic root node used as the "parent" for root-level blocks.
    _root_node = SectionNode(heading="Front Matter", level=0)

    for block in root_blocks:
        raw_items.append((block.text, _root_node))

    for section in front_matter_sections:
        # Include the heading itself as context prefix for the section's
        # content blocks, then walk the subtree.
        for block in section.content_blocks:
            raw_items.append((block.text, section))
        for child in section.children:
            for block in child.content_blocks:
                raw_items.append((block.text, child))

    if not raw_items:
        return []

    # Build summary text for the Tier 2 chunk.
    summary_parts = ["## Front Matter"]
    for section in front_matter_sections:
        summary_parts.append(f"- {section.heading}")
    sec_text = "\n".join(summary_parts)

    sec_chunk = Chunk(
        chunk_id=sec_chunk_id,
        doc_id=doc_id,
        chunk_type=ChunkType.SECTION,
        tier=ChunkTier.SECTION,
        text=sec_text,
        token_count=count_tokens(sec_text, config.encoding_name),
        parent_id=doc_chunk_id,
        children_ids=[],
        metadata=ChunkMetadata(
            section_path=section_path,
            section_number=None,
            heading="Front Matter",
        ),
    )

    # Produce Tier 3 FRONT_MATTER chunks — one per content block.
    atomic_chunks: list[Chunk] = []
    for emit_idx, (text, node) in enumerate(raw_items):
        text = text.strip()
        if not text:
            continue
        tokens = count_tokens(text, config.encoding_name)
        if tokens > config.split_threshold:
            # Oversized: split (tables row-wise, others paragraph-wise).
            splits = _split_atomic(text, ChunkType.FRONT_MATTER, config)
            for sub_idx, sub_text in enumerate(splits):
                cid = _make_id(doc_id, section_path, emit_idx, sub_idx)
                atomic_chunks.append(Chunk(
                    chunk_id=cid,
                    doc_id=doc_id,
                    chunk_type=ChunkType.FRONT_MATTER,
                    tier=ChunkTier.ATOMIC,
                    text=sub_text,
                    token_count=count_tokens(sub_text, config.encoding_name),
                    parent_id=sec_chunk_id,
                    children_ids=[],
                    metadata=ChunkMetadata(
                        section_path=section_path,
                        heading=node.heading if node else "Front Matter",
                        has_table=detect_embedded_tables(sub_text),
                        has_figure=detect_embedded_figures(sub_text),
                    ),
                ))
        else:
            cid = _make_id(doc_id, section_path, emit_idx, 0)
            atomic_chunks.append(Chunk(
                chunk_id=cid,
                doc_id=doc_id,
                chunk_type=ChunkType.FRONT_MATTER,
                tier=ChunkTier.ATOMIC,
                text=text,
                token_count=tokens,
                parent_id=sec_chunk_id,
                children_ids=[],
                metadata=ChunkMetadata(
                    section_path=section_path,
                    heading=node.heading if node else "Front Matter",
                    has_table=detect_embedded_tables(text),
                    has_figure=detect_embedded_figures(text),
                ),
            ))

    sec_chunk.children_ids = [c.chunk_id for c in atomic_chunks]
    return [sec_chunk] + atomic_chunks


# ── Table parsing helpers ───────────────────────────────────────


def _parse_table_rows(text: str) -> list[list[str]]:
    """Parse a Markdown pipe-table into a list of rows (lists of cell values).

    Skips the header-separator row (``|---|---|``).
    """
    rows: list[list[str]] = []
    for line in text.split("\n"):
        line = line.strip()
        if not line.startswith("|"):
            continue
        # Skip separator row.
        if re.match(r"^\|[\s\-:|]+\|$", line):
            continue
        cells = _split_table_cells(line)
        rows.append(cells)
    # Skip the header row (first row) if there are data rows.
    if len(rows) > 1:
        return rows[1:]
    return rows


def _split_table_cells(line: str) -> list[str]:
    """Split a pipe-table row into cell values."""
    # Strip leading/trailing pipes, then split on |.
    stripped = line.strip().strip("|")
    return [c.strip() for c in stripped.split("|")]


# ── Section summary builder ─────────────────────────────────────


def _build_document_summary(meta: DocumentMeta) -> str:
    """Create synthetic text for the Tier 1 document chunk."""
    parts = []
    if meta.doc_title:
        parts.append(meta.doc_title)
    if meta.module_full_name:
        parts.append(f"Module: {meta.module_full_name} ({meta.module_name})")
    elif meta.module_name:
        parts.append(f"Module: {meta.module_name}")
    if meta.doc_type:
        parts.append(f"Document type: {meta.doc_type}")
    if meta.system:
        parts.append(f"System: {meta.system}")
    if meta.revision:
        parts.append(f"Revision: {meta.revision} ({meta.revision_date})")
    if meta.authors:
        parts.append(f"Authors: {', '.join(meta.authors)}")
    return "\n".join(parts)


def _build_section_summary(section: SectionNode) -> str:
    """Create summary text for a Tier 2 section chunk.

    Includes the section heading plus the first sentence or paragraph
    of each child sub-section.
    """
    parts = [f"## {section.heading}"]

    # Include first prose block of this section (if any).
    for block in section.content_blocks:
        if block.block_type in ("prose", "atomic_delimited"):
            first_para = block.text.split("\n\n")[0].strip()
            if first_para:
                parts.append(first_para)
            break

    # Include headings + first sentences from child sections.
    for child in section.children:
        parts.append(f"### {child.heading}")
        for block in child.content_blocks:
            if block.block_type in ("prose", "atomic_delimited"):
                first_sent = _first_sentence(block.text)
                if first_sent:
                    parts.append(first_sent)
                break

    return "\n\n".join(parts)


def _first_sentence(text: str) -> str:
    """Return the first sentence of *text* (up to the first '. ')."""
    text = text.strip()
    # Try period followed by space or end of text.
    m = re.search(r"\.\s", text)
    if m and m.start() < 300:
        return text[: m.start() + 1]
    # Fallback: first line.
    first_line = text.split("\n")[0].strip()
    return first_line[:300]


# ── Leaf block collection ───────────────────────────────────────


def _collect_leaf_blocks(
    section: SectionNode,
) -> list[tuple[SectionNode, int]]:
    """Depth-first collect of (node, block_index) for all content blocks."""
    result: list[tuple[SectionNode, int]] = []
    for idx, _block in enumerate(section.content_blocks):
        result.append((section, idx))
    for child in section.children:
        result.extend(_collect_leaf_blocks(child))
    return result


# ── Section path builder ────────────────────────────────────────


def _build_section_path(node: SectionNode, root_heading: str) -> list[str]:
    """Build the section path from the top-level section down to *node*.

    For simplicity, when the node IS the top-level section the path is
    just ``[root_heading]``.  For sub-sections the path includes
    intermediate headings.
    """
    # This is called per-block. The top-level section heading is always
    # the first element; we add the node's heading if it differs.
    if node.heading == root_heading:
        return [root_heading]
    return [root_heading, node.heading]


# ── Chunk ID generation ────────────────────────────────────────


def _make_id(
    doc_id: str,
    section_path: list[str],
    block_index: int,
    sub_index: int,
) -> str:
    """Deterministic SHA-256 chunk ID."""
    raw = f"{doc_id}|{'|'.join(section_path)}|{block_index}|{sub_index}"
    return hashlib.sha256(raw.encode()).hexdigest()


# ── Tier 3 processing: merge / keep / split ─────────────────────


_SPLITTABLE_TYPES = frozenset({
    ChunkType.PROSE,
    ChunkType.LIST,
})

_ATOMIC_TYPES = frozenset({
    ChunkType.TEST_CASE,
    ChunkType.REQUIREMENT,
    ChunkType.TABLE,
    ChunkType.DEFINITION_TABLE,
    ChunkType.ABBREVIATION_TABLE,
    ChunkType.FRONT_MATTER,
    ChunkType.FIGURE,
})


def _process_leaf_blocks(
    leaf_blocks: list[tuple[SectionNode, int]],
    doc_id: str,
    top_section_path: list[str],
    sec_idx: int,
    parent_chunk_id: str,
    config: ChunkConfig,
) -> list[Chunk]:
    """Apply size rules and emit Tier 3 chunks for one ## section."""
    # First pass: build a list of (text, chunk_type, node, block_idx) items.
    items: list[tuple[str, ChunkType, SectionNode, int]] = []
    for node, block_idx in leaf_blocks:
        block = node.content_blocks[block_idx]
        ctype = classify_block(block, node)
        items.append((block.text, ctype, node, block_idx))

    # Second pass: merge small blocks with adjacent siblings.
    merged = _merge_small_blocks(items, config)

    # Third pass: split oversized blocks; emit final chunks.
    chunks: list[Chunk] = []
    for emit_idx, (text, ctype, node, _bidx) in enumerate(merged):
        section_path = _build_section_path(node, top_section_path[0])
        # Include sec_idx in the ID path to disambiguate sections with
        # identical headings (e.g. two "## Notes:" sections).
        id_path = [f"__s{sec_idx}__"] + section_path
        tokens = count_tokens(text, config.encoding_name)

        if ctype in _SPLITTABLE_TYPES and tokens > config.max_tokens:
            # Split prose / list at sentence / paragraph boundaries.
            splits = _split_text(text, config)
            for sub_idx, sub_text in enumerate(splits):
                chunk = _make_chunk(
                    sub_text, ctype, doc_id, section_path,
                    emit_idx, sub_idx, parent_chunk_id, node, config,
                    id_path=id_path,
                )
                chunks.append(chunk)
        elif ctype in _ATOMIC_TYPES and tokens > config.split_threshold:
            # Force-split oversized atomic block.
            splits = _split_atomic(text, ctype, config)
            for sub_idx, sub_text in enumerate(splits):
                chunk = _make_chunk(
                    sub_text, ctype, doc_id, section_path,
                    emit_idx, sub_idx, parent_chunk_id, node, config,
                    id_path=id_path,
                )
                chunks.append(chunk)
        else:
            # Within budget — emit as one chunk.
            chunk = _make_chunk(
                text, ctype, doc_id, section_path,
                emit_idx, 0, parent_chunk_id, node, config,
                id_path=id_path,
            )
            chunks.append(chunk)

    return chunks


def _merge_small_blocks(
    items: list[tuple[str, ChunkType, SectionNode, int]],
    config: ChunkConfig,
) -> list[tuple[str, ChunkType, SectionNode, int]]:
    """Merge blocks under ``min_tokens`` with an adjacent sibling.

    Only merges blocks that share the same parent ``SectionNode``
    and are both splittable types (prose, list).
    """
    if not items:
        return []

    result: list[tuple[str, ChunkType, SectionNode, int]] = []
    i = 0
    while i < len(items):
        text, ctype, node, bidx = items[i]
        tokens = count_tokens(text, config.encoding_name)

        if tokens < config.min_tokens and ctype in _SPLITTABLE_TYPES:
            # Try to merge with the next block if it shares the same
            # parent node and is also a splittable type.
            if (
                i + 1 < len(items)
                and items[i + 1][2] is node
                and items[i + 1][1] in _SPLITTABLE_TYPES
            ):
                next_text, next_ctype, next_node, next_bidx = items[i + 1]
                merged_text = text.strip() + "\n\n" + next_text.strip()
                result.append((merged_text, ctype, node, bidx))
                i += 2
                continue
            # Try to merge with the previous result block.
            if (
                result
                and result[-1][2] is node
                and result[-1][1] in _SPLITTABLE_TYPES
            ):
                prev_text, prev_ctype, prev_node, prev_bidx = result[-1]
                merged_text = prev_text.strip() + "\n\n" + text.strip()
                result[-1] = (merged_text, prev_ctype, prev_node, prev_bidx)
                i += 1
                continue

        result.append((text, ctype, node, bidx))
        i += 1

    return result


# ── Text splitting helpers ──────────────────────────────────────


def _split_text(
    text: str,
    config: ChunkConfig,
) -> list[str]:
    """Split prose or list text at natural boundaries with overlap.

    Tries paragraph breaks (``\\n\\n``) first, then sentence endings
    (``. `` followed by an uppercase letter).
    """
    paragraphs = re.split(r"\n\n+", text.strip())
    if len(paragraphs) == 1:
        # No paragraph breaks — split at sentence boundaries.
        paragraphs = re.split(r"(?<=\.)\s+(?=[A-Z])", text.strip())

    return _pack_segments(paragraphs, config)


def _split_atomic(
    text: str,
    ctype: ChunkType,
    config: ChunkConfig,
) -> list[str]:
    """Force-split an oversized atomic block.

    For tables: split row-wise, duplicating the header row.
    For others: split at paragraph or line boundaries.
    """
    if ctype in (ChunkType.TABLE, ChunkType.DEFINITION_TABLE,
                 ChunkType.ABBREVIATION_TABLE):
        return _split_table(text, config)

    # Test cases / requirements — split at paragraph boundaries.
    paragraphs = re.split(r"\n\n+", text.strip())
    if len(paragraphs) <= 1:
        # Last resort: split at line boundaries.
        paragraphs = text.strip().split("\n")
    return _pack_segments(paragraphs, config)


def _split_table(text: str, config: ChunkConfig) -> list[str]:
    """Split a large table row-wise, duplicating the header."""
    lines = text.strip().split("\n")
    # Identify header: first row + separator row.
    header_lines: list[str] = []
    data_start = 0
    for idx, line in enumerate(lines):
        if re.match(r"^\|[\s\-:|]+\|$", line.strip()):
            header_lines = lines[: idx + 1]
            data_start = idx + 1
            break

    if not header_lines:
        # No separator found — treat as prose split.
        paragraphs = re.split(r"\n\n+", text.strip())
        return _pack_segments(paragraphs, config)

    header_text = "\n".join(header_lines)
    data_lines = lines[data_start:]

    # Pack data rows into chunks, prepending the header to each.
    splits: list[str] = []
    current: list[str] = []
    current_tokens = count_tokens(header_text, config.encoding_name)

    for row in data_lines:
        row_tokens = count_tokens(row, config.encoding_name)
        if current and current_tokens + row_tokens > config.max_tokens:
            splits.append(header_text + "\n" + "\n".join(current))
            current = [row]
            current_tokens = count_tokens(header_text, config.encoding_name) + row_tokens
        else:
            current.append(row)
            current_tokens += row_tokens

    if current:
        splits.append(header_text + "\n" + "\n".join(current))

    return splits if splits else [text]


def _pack_segments(
    segments: list[str],
    config: ChunkConfig,
) -> list[str]:
    """Pack text segments into chunks respecting ``max_tokens``.

    Applies ``overlap_tokens`` of trailing context to each subsequent
    chunk for continuity.
    """
    if not segments:
        return []

    splits: list[str] = []
    current_parts: list[str] = []
    current_tokens = 0

    for seg in segments:
        seg = seg.strip()
        if not seg:
            continue
        seg_tokens = count_tokens(seg, config.encoding_name)

        # If a single segment exceeds max_tokens, break it into
        # individual lines so it can be packed across chunks.
        if seg_tokens > config.max_tokens and "\n" in seg:
            sub_lines = seg.split("\n")
            for sub in sub_lines:
                sub = sub.strip()
                if not sub:
                    continue
                sub_tokens = count_tokens(sub, config.encoding_name)
                if current_parts and current_tokens + sub_tokens > config.max_tokens:
                    splits.append("\n\n".join(current_parts))
                    overlap = _get_overlap(current_parts, config)
                    if overlap:
                        current_parts = [overlap, sub]
                        current_tokens = (
                            count_tokens(overlap, config.encoding_name) + sub_tokens
                        )
                    else:
                        current_parts = [sub]
                        current_tokens = sub_tokens
                else:
                    current_parts.append(sub)
                    current_tokens += sub_tokens
            continue

        if current_parts and current_tokens + seg_tokens > config.max_tokens:
            # Emit current chunk.
            splits.append("\n\n".join(current_parts))
            # Start new chunk with overlap from previous.
            overlap = _get_overlap(current_parts, config)
            if overlap:
                current_parts = [overlap, seg]
                current_tokens = (
                    count_tokens(overlap, config.encoding_name) + seg_tokens
                )
            else:
                current_parts = [seg]
                current_tokens = seg_tokens
        else:
            current_parts.append(seg)
            current_tokens += seg_tokens

    if current_parts:
        splits.append("\n\n".join(current_parts))

    return splits if splits else ["\n\n".join(segments)]


def _get_overlap(parts: list[str], config: ChunkConfig) -> str:
    """Return up to ``overlap_tokens`` of trailing text from *parts*."""
    if not parts or config.overlap_tokens <= 0:
        return ""
    # Take from the last part, up to overlap_tokens.
    last = parts[-1]
    tokens = count_tokens(last, config.encoding_name)
    if tokens <= config.overlap_tokens:
        return last
    # Approximate: take the last N characters proportionally.
    ratio = config.overlap_tokens / max(tokens, 1)
    char_count = max(1, int(len(last) * ratio))
    return last[-char_count:]


# ── Chunk construction helper ───────────────────────────────────


def _make_chunk(
    text: str,
    ctype: ChunkType,
    doc_id: str,
    section_path: list[str],
    block_index: int,
    sub_index: int,
    parent_id: str,
    node: SectionNode,
    config: ChunkConfig,
    id_path: list[str] | None = None,
) -> Chunk:
    """Construct a Tier 3 chunk with metadata."""
    chunk_id = _make_id(doc_id, id_path or section_path, block_index, sub_index)
    token_count = count_tokens(text, config.encoding_name)

    meta = _build_metadata(text, ctype, node, section_path)

    return Chunk(
        chunk_id=chunk_id,
        doc_id=doc_id,
        chunk_type=ctype,
        tier=ChunkTier.ATOMIC,
        text=text,
        token_count=token_count,
        parent_id=parent_id,
        children_ids=[],
        metadata=meta,
    )


def _build_metadata(
    text: str,
    ctype: ChunkType,
    node: SectionNode,
    section_path: list[str],
) -> ChunkMetadata:
    """Populate ``ChunkMetadata`` for a Tier 3 chunk."""
    meta = ChunkMetadata(
        section_path=list(section_path),
        section_number=node.section_number,
        heading=node.heading,
        has_table=detect_embedded_tables(text),
        has_figure=detect_embedded_figures(text),
        cross_references=extract_cross_references(text),
        component_ids=extract_component_ids(text),
    )

    # Type-specific field extraction.
    if ctype == ChunkType.TEST_CASE:
        fields = extract_test_case_fields(text)
        meta.test_case_id = fields.get("test_case_id")
        meta.test_name = fields.get("test_name")
        meta.test_result = fields.get("test_result")
        meta.test_item = fields.get("test_item")
        meta.date = fields.get("date")
        meta.tester = fields.get("tester")
        meta.verifier = fields.get("verifier")
        meta.failure_criteria = fields.get("failure_criteria")
        meta.traceability_ids = fields.get("traceability_ids", [])
        meta.reference_ids = fields.get("reference_ids", [])

    elif ctype == ChunkType.REQUIREMENT:
        fields = extract_requirement_fields(text)
        meta.requirement_ids = fields.get("requirement_ids", [])
        meta.category = fields.get("category")
        meta.allocation = fields.get("allocation")
        meta.priority = fields.get("priority")
        meta.safety = fields.get("safety")
        meta.verification_method = fields.get("verification_method")
        meta.is_background = fields.get("is_background", False)
        meta.traceability_ids = fields.get("traceability_ids", [])

    return meta
