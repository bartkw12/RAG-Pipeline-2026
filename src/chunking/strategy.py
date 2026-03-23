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
