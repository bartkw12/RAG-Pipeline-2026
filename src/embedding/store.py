"""ChromaDB vector store adapter.

Wraps a persistent ChromaDB collection with helpers for upserting
embeddings, querying by vector similarity, and managing documents.
All vectors live in a single collection (``uc37_chunks`` by default)
with rich filterable metadata.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, List

import chromadb
from chromadb.api.types import Embedding, Metadata

from ..config.paths import EMBED_DIR

logger = logging.getLogger(__name__)

# ── Default collection name ─────────────────────────────────────
_DEFAULT_COLLECTION = "uc37_chunks"


# ── Query result ────────────────────────────────────────────────


@dataclass
class QueryResult:
    """Container for a single similarity-search result set."""

    ids: list[str] = field(default_factory=list)
    distances: list[float] = field(default_factory=list)
    documents: list[str] = field(default_factory=list)
    metadatas: list[dict[str, Any]] = field(default_factory=list)


# ── Store class ─────────────────────────────────────────────────


class EmbeddingStore:
    """Persistent ChromaDB collection for chunk embeddings.

    Parameters
    ----------
    persist_dir:
        Directory for ChromaDB's on-disk storage.
        Defaults to ``cache/embed/``.
    collection_name:
        Name of the ChromaDB collection.
    """

    def __init__(
        self,
        persist_dir: Path | str | None = None,
        collection_name: str = _DEFAULT_COLLECTION,
    ) -> None:
        self._persist_dir = Path(persist_dir or EMBED_DIR)
        self._persist_dir.mkdir(parents=True, exist_ok=True)

        self._client = chromadb.PersistentClient(
            path=str(self._persist_dir),
        )
        self._collection = self._client.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"},
        )
        logger.info(
            "ChromaDB collection '%s' opened  (%d vectors, persist=%s)",
            collection_name, self._collection.count(), self._persist_dir,
        )

    # ── Write operations ────────────────────────────────────────

    def upsert(
        self,
        ids: list[str],
        embeddings: List[Embedding],
        documents: list[str],
        metadatas: List[Metadata],
    ) -> None:
        """Insert or update vectors in the collection.

        Parameters
        ----------
        ids:
            Unique identifiers (``chunk_id`` values).
        embeddings:
            Float vectors (1536-dim for Ada-002).
        documents:
            The enriched text that was embedded (stored for inspection).
        metadatas:
            Per-vector metadata dicts (all values must be str/int/float/bool).
        """
        self._collection.upsert(
            ids=ids,
            embeddings=embeddings,
            documents=documents,
            metadatas=metadatas,
        )
        logger.debug("Upserted %d vectors.", len(ids))

    def delete_by_doc(self, doc_id: str) -> int:
        """Remove all vectors belonging to a document.

        Returns the number of vectors deleted.
        """
        # Fetch IDs matching this doc_id, then delete them.
        existing = self._collection.get(
            where={"doc_id": doc_id},
            include=[],
        )
        ids = existing["ids"]
        if ids:
            self._collection.delete(ids=ids)
            logger.info("Deleted %d vectors for doc_id=%s…", len(ids), doc_id[:12])
        return len(ids)

    # ── Read operations ─────────────────────────────────────────

    def count(self) -> int:
        """Return the total number of vectors in the collection."""
        return self._collection.count()

    def get_existing_ids(self, ids: list[str]) -> set[str]:
        """Return the subset of *ids* that already exist in the store."""
        if not ids:
            return set()
        result = self._collection.get(ids=ids, include=[])
        return set(result["ids"])

    def query(
        self,
        embedding: list[float],
        n: int = 10,
        where: dict[str, Any] | None = None,
    ) -> QueryResult:
        """Find the *n* most similar vectors.

        Parameters
        ----------
        embedding:
            Query vector (same dimensionality as stored vectors).
        n:
            Number of results to return.
        where:
            Optional ChromaDB metadata filter, e.g.
            ``{"doc_type": "HwIRS"}`` or
            ``{"chunk_type": "test_case"}``.

        Returns
        -------
        QueryResult
            Matching IDs, distances, documents, and metadata.
        """
        kwargs: dict[str, Any] = {
            "query_embeddings": [embedding],
            "n_results": min(n, self.count() or 1),
            "include": ["distances", "documents", "metadatas"],
        }
        if where:
            kwargs["where"] = where

        raw = self._collection.query(**kwargs)

        return QueryResult(
            ids=raw["ids"][0] if raw["ids"] else [],
            distances=raw["distances"][0] if raw["distances"] else [],
            documents=raw["documents"][0] if raw["documents"] else [],
            metadatas=list(raw["metadatas"][0]) if raw["metadatas"] else [],  # type: ignore[arg-type]
        )
