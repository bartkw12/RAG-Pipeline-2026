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

