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

    # 4 & 5. Walk ## sections → Tier 2 + depth-first → Tier 3.
    for sec_idx, section in enumerate(tree.children):
        if section.level != 2:
            # Some documents have level-1 headings at top; treat the
            # same as level-2 for Tier 2 purposes.
            pass

        section_path = [section.heading]
        sec_chunk_id = _make_id(doc_id, section_path, sec_idx, 0)
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
            leaf_blocks, doc_id, section_path, sec_chunk_id, config,
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
    """Extract latest revision from the LOG OF CHANGES table."""
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

