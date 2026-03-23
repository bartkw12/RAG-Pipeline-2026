"""Chunk persistence — serialize a ChunkedDocument to disk as JSON.

Public API
----------
``write_chunks(chunked_doc) -> Path``
    Serialize and write to ``cache/chunk/{doc_id}.json``.

``load_chunks(doc_id) -> ChunkedDocument``
    Read a previously written JSON file back into a ``ChunkedDocument``.
"""

from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path

from src.chunking.models import (
    Chunk,
    ChunkedDocument,
    ChunkMetadata,
    ChunkTier,
    ChunkType,
    DocumentMeta,
)
from src.config.paths import CHUNK_DIR


# ── Public API ──────────────────────────────────────────────────


def write_chunks(chunked_doc: ChunkedDocument) -> Path:
    """Serialize *chunked_doc* to ``cache/chunk/{doc_id}.json``.

    Creates the output directory if it doesn't exist.

    Returns
    -------
    Path
        Absolute path of the written JSON file.
    """
    CHUNK_DIR.mkdir(parents=True, exist_ok=True)
    out_path = CHUNK_DIR / f"{chunked_doc.doc_id}.json"

    payload = _serialize_document(chunked_doc)
    out_path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    return out_path


def load_chunks(doc_id: str) -> ChunkedDocument:
    """Read ``cache/chunk/{doc_id}.json`` back into a ``ChunkedDocument``.

    Raises
    ------
    FileNotFoundError
        If the chunk file does not exist.
    """
    path = CHUNK_DIR / f"{doc_id}.json"
    raw = json.loads(path.read_text(encoding="utf-8"))
    return _deserialize_document(raw)


# ── Serialization helpers ───────────────────────────────────────


def _serialize_document(doc: ChunkedDocument) -> dict:
    """Convert a ``ChunkedDocument`` to a JSON-safe dictionary."""
    return {
        "doc_id": doc.doc_id,
        "doc_metadata": _serialize_doc_meta(doc.doc_metadata),
        "chunks": [_serialize_chunk(c) for c in doc.chunks],
    }


def _serialize_doc_meta(meta: DocumentMeta) -> dict:
    """Convert ``DocumentMeta`` to a plain dict."""
    return asdict(meta)


def _serialize_chunk(chunk: Chunk) -> dict:
    """Convert a ``Chunk`` to a plain dict with enum values as strings."""
    return {
        "chunk_id": chunk.chunk_id,
        "doc_id": chunk.doc_id,
        "chunk_type": chunk.chunk_type.value,
        "tier": chunk.tier.value,
        "text": chunk.text,
        "token_count": chunk.token_count,
        "parent_id": chunk.parent_id,
        "children_ids": chunk.children_ids,
        "metadata": _serialize_metadata(chunk.metadata),
    }


def _serialize_metadata(meta: ChunkMetadata) -> dict:
    """Convert ``ChunkMetadata`` — omit None/empty/False fields to keep
    the JSON compact."""
    raw = asdict(meta)
    return {k: v for k, v in raw.items() if v not in (None, "", False, [], 0)}


# ── Deserialization helpers ─────────────────────────────────────


def _deserialize_document(raw: dict) -> ChunkedDocument:
    """Reconstruct a ``ChunkedDocument`` from a parsed JSON dict."""
    return ChunkedDocument(
        doc_id=raw["doc_id"],
        doc_metadata=_deserialize_doc_meta(raw["doc_metadata"]),
        chunks=[_deserialize_chunk(c) for c in raw["chunks"]],
    )


def _deserialize_doc_meta(raw: dict) -> DocumentMeta:
    """Reconstruct ``DocumentMeta`` from a plain dict."""
    return DocumentMeta(**raw)


def _deserialize_chunk(raw: dict) -> Chunk:
    """Reconstruct a ``Chunk`` from a plain dict."""
    return Chunk(
        chunk_id=raw["chunk_id"],
        doc_id=raw["doc_id"],
        chunk_type=ChunkType(raw["chunk_type"]),
        tier=ChunkTier(raw["tier"]),
        text=raw["text"],
        token_count=raw.get("token_count", 0),
        parent_id=raw.get("parent_id"),
        children_ids=raw.get("children_ids", []),
        metadata=_deserialize_metadata(raw.get("metadata", {})),
    )


def _deserialize_metadata(raw: dict) -> ChunkMetadata:
    """Reconstruct ``ChunkMetadata``, filling defaults for missing keys."""
    return ChunkMetadata(
        section_path=raw.get("section_path", []),
        section_number=raw.get("section_number"),
        heading=raw.get("heading"),
        has_table=raw.get("has_table", False),
        has_figure=raw.get("has_figure", False),
        cross_references=raw.get("cross_references", []),
        component_ids=raw.get("component_ids", []),
        test_case_id=raw.get("test_case_id"),
        test_name=raw.get("test_name"),
        test_result=raw.get("test_result"),
        test_item=raw.get("test_item"),
        date=raw.get("date"),
        tester=raw.get("tester"),
        verifier=raw.get("verifier"),
        failure_criteria=raw.get("failure_criteria"),
        traceability_ids=raw.get("traceability_ids", []),
        reference_ids=raw.get("reference_ids", []),
        requirement_ids=raw.get("requirement_ids", []),
        category=raw.get("category"),
        allocation=raw.get("allocation"),
        priority=raw.get("priority"),
        safety=raw.get("safety"),
        verification_method=raw.get("verification_method"),
        is_background=raw.get("is_background", False),
    )
