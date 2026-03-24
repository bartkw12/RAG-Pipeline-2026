"""Embedding pipeline — vectorise chunks and store in ChromaDB."""

from __future__ import annotations

from .pipeline import embed_all, embed_document, EmbeddingResult

__all__ = [
    "embed_all",
    "embed_document",
    "EmbeddingResult",
]
