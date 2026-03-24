"""In-memory BM25 keyword index with domain-aware tokenisation.

Builds a BM25Okapi index over all tier-3 (atomic) chunks at first use
and caches it for the lifetime of the process.  The custom tokeniser
preserves compound engineering identifiers as whole tokens while also
emitting their sub-components, giving BM25 the best of both worlds:

* **Exact compound matches** score highest (the full token appears in
  both query and document).
* **Partial / component matches** still contribute (sub-parts like
  ``dim``, ``275`` are also indexed).

Engineering patterns recognised:

* Requirement IDs:  ``HW-IRS_DIM_VI_275``
* Test-case IDs:    ``FVTR_OPT_01``
* Component IDs:    ``7HA-02944-AAAA``
* Cross-references: ``HWADD:TOP:0012``
* Module names:     ``DIM-V``, ``DIM-NV``

Usage::

    from src.retrieval.bm25 import bm25_search

    hits = bm25_search("HW-IRS_DIM_VI_275 thermal protection", n=20)
    for hit in hits:
        print(hit.chunk_id, hit.score)
"""

from __future__ import annotations

import logging
import re
import threading
from dataclasses import dataclass
from pathlib import Path

from rank_bm25 import BM25Okapi

from ..chunking.models import ChunkTier
from ..chunking.writer import load_chunks
from ..config.paths import CHUNK_DIR

logger = logging.getLogger(__name__)


# ── Data structures ─────────────────────────────────────────────


@dataclass
class BM25Hit:
    """A single BM25 search result."""

    chunk_id: str
    doc_id: str
    score: float


# ── Engineering-domain tokeniser ────────────────────────────────

# Patterns that should be preserved as compound tokens.
# Order matters: longer / more specific patterns first.
_COMPOUND_PATTERNS: list[re.Pattern[str]] = [
    # Requirement IDs: HW-IRS_DIM_VI_275
    re.compile(r"HW-IRS_\w+", re.IGNORECASE),
    # Test-case / traceability IDs: FVTR_OPT_01, FVTSR_PAM_0009
    re.compile(r"[A-Z]{3,}_[A-Z]+_\d+", re.IGNORECASE),
    # Component / item numbers: 7HA-02944-AAAA or 7HA 02944 AAAA
    re.compile(r"7HA[\s-]\d{5}[\s-][A-Z]{4}", re.IGNORECASE),
    # Cross-reference tags (without brackets): HWADD:TOP:0012
    re.compile(r"[A-Z]{2,}(?::[A-Z0-9_]+){1,}", re.IGNORECASE),
    # Hyphenated module names: DIM-V, DIM-NV, HW-IRS (standalone)
    re.compile(r"[A-Z]{2,}-[A-Z]{1,3}\b", re.IGNORECASE),
]

# Characters used to split compound tokens into sub-parts.
_SPLIT_CHARS = re.compile(r"[-_:]+")

# Characters treated as token boundaries for the standard pass.
_WORD_SPLIT = re.compile(r"[^a-zA-Z0-9_-]+")

# Tokens too short / common to be useful.
_MIN_TOKEN_LEN = 1


def tokenize(text: str) -> list[str]:
    """Tokenise *text* with engineering-domain awareness.

    Returns a flat list of lowercased tokens.  Compound engineering
    identifiers are emitted as whole tokens *and* as sub-components.

    Example::

        >>> tokenize("HW-IRS_DIM_VI_275 thermal protection")
        ['hw-irs_dim_vi_275', 'hw', 'irs', 'dim', 'vi', '275',
         'thermal', 'protection']
    """
    tokens: list[str] = []
    remaining = text

    # First pass: collect all compound-match spans.
    raw_spans: list[tuple[int, int, str]] = []
    for pattern in _COMPOUND_PATTERNS:
        for m in pattern.finditer(remaining):
            raw_spans.append((m.start(), m.end(), m.group().lower()))

    # Merge overlapping / contained spans — keep the longest match
    # for each region so "HW-IRS_DIM_VI_275" absorbs "DIM_VI_275".
    raw_spans.sort(key=lambda s: (s[0], -s[1]))
    merged: list[tuple[int, int, str]] = []
    for start, end, matched in raw_spans:
        if merged and start < merged[-1][1]:
            # Overlaps with previous — keep whichever is longer.
            prev_start, prev_end, prev_text = merged[-1]
            if end > prev_end:
                merged[-1] = (prev_start, end, matched)
            continue
        merged.append((start, end, matched))

    # Emit compound tokens and their sub-components (deduplicated).
    for _start, _end, compound in merged:
        tokens.append(compound)
        parts = _SPLIT_CHARS.split(compound)
        for part in parts:
            if len(part) >= _MIN_TOKEN_LEN:
                tokens.append(part)

    # Mask matched spans from the standard pass.
    if merged:
        chars = list(remaining)
        for start, end, _ in merged:
            for i in range(start, min(end, len(chars))):
                chars[i] = " "
        remaining = "".join(chars)

    # Second pass: standard word splitting on the remaining text.
    words = _WORD_SPLIT.split(remaining.lower())
    for word in words:
        if len(word) >= _MIN_TOKEN_LEN:
            tokens.append(word)

    return tokens


# ── BM25 index singleton ───────────────────────────────────────

_lock = threading.Lock()
_index: _BM25Index | None = None


class _BM25Index:
    """Internal BM25 index over tier-3 chunks."""

    __slots__ = ("_bm25", "_chunk_ids", "_doc_ids")

    def __init__(
        self,
        corpus: list[list[str]],
        chunk_ids: list[str],
        doc_ids: list[str],
    ) -> None:
        self._bm25 = BM25Okapi(corpus)
        self._chunk_ids = chunk_ids
        self._doc_ids = doc_ids

    def search(
        self,
        query: str,
        n: int = 20,
        doc_ids: list[str] | None = None,
    ) -> list[BM25Hit]:
        """Score all documents and return the top-*n* hits.

        Parameters
        ----------
        query:
            Raw query string (tokenised internally).
        n:
            Maximum results to return.
        doc_ids:
            Optional whitelist — only return chunks from these documents.
        """
        query_tokens = tokenize(query)
        if not query_tokens:
            return []

        scores = self._bm25.get_scores(query_tokens)

        # Build (index, score) pairs, optionally filtering by doc_id.
        candidates: list[tuple[int, float]] = []
        for idx, score in enumerate(scores):
            if score <= 0:
                continue
            if doc_ids and self._doc_ids[idx] not in doc_ids:
                continue
            candidates.append((idx, float(score)))

        # Sort descending by score, take top-n.
        candidates.sort(key=lambda x: x[1], reverse=True)
        candidates = candidates[:n]

        return [
            BM25Hit(
                chunk_id=self._chunk_ids[idx],
                doc_id=self._doc_ids[idx],
                score=score,
            )
            for idx, score in candidates
        ]

    @property
    def size(self) -> int:
        """Number of chunks in the index."""
        return len(self._chunk_ids)


def _build_index(chunk_dir: Path | None = None) -> _BM25Index:
    """Build the BM25 index from all cached chunk JSON files.

    Loads every tier-3 chunk, tokenises its text, and constructs
    a ``BM25Okapi`` index.
    """
    chunk_dir = Path(chunk_dir or CHUNK_DIR)
    json_files = sorted(chunk_dir.glob("*.json"))

    if not json_files:
        logger.warning("No chunk files found in %s — BM25 index is empty.", chunk_dir)
        return _BM25Index(corpus=[[""]], chunk_ids=[""], doc_ids=[""])

    corpus: list[list[str]] = []
    chunk_ids: list[str] = []
    doc_ids: list[str] = []

    for json_path in json_files:
        doc_id = json_path.stem
        try:
            doc = load_chunks(doc_id)
        except Exception:
            logger.warning("Failed to load chunks for %s — skipping.", doc_id[:12])
            continue

        for chunk in doc.chunks:
            if chunk.tier != ChunkTier.ATOMIC:
                continue
            tokens = tokenize(chunk.text)
            if not tokens:
                continue
            corpus.append(tokens)
            chunk_ids.append(chunk.chunk_id)
            doc_ids.append(chunk.doc_id)

    if not corpus:
        logger.warning("No tier-3 chunks found — BM25 index is empty.")
        return _BM25Index(corpus=[[""]], chunk_ids=[""], doc_ids=[""])

    logger.info("BM25 index built: %d tier-3 chunks from %d documents.",
                len(corpus), len(json_files))
    return _BM25Index(corpus=corpus, chunk_ids=chunk_ids, doc_ids=doc_ids)


def _get_index() -> _BM25Index:
    """Return the singleton BM25 index, building it on first access."""
    global _index
    if _index is None:
        with _lock:
            if _index is None:
                _index = _build_index()
    return _index


def invalidate_index() -> None:
    """Discard the cached index so it is rebuilt on next search.

    Call this after re-chunking or re-ingesting documents.
    """
    global _index
    with _lock:
        _index = None
    logger.debug("BM25 index invalidated.")


# ── Public API ──────────────────────────────────────────────────


def bm25_search(
    query: str,
    n: int = 20,
    doc_ids: list[str] | None = None,
) -> list[BM25Hit]:
    """Search the BM25 index and return the top-*n* hits.

    Parameters
    ----------
    query:
        Natural-language or identifier query string.
    n:
        Maximum number of results (default: 20).
    doc_ids:
        Optional whitelist of document IDs to restrict results.

    Returns
    -------
    list[BM25Hit]
        Hits sorted by descending BM25 score.
    """
    index = _get_index()
    return index.search(query, n=n, doc_ids=doc_ids)
