"""Scoped abbreviation lookup and query augmentation.

Extracts abbreviation → expansion mappings from
``ChunkType.ABBREVIATION_TABLE`` chunks in the chunk cache and
provides query-time expansion that **preserves** the original acronym
and **appends** the expansion.

Mappings are stored per-document so that ambiguous acronyms can be
resolved using scope signals (document ID, module, doc type).  A
merged global dictionary is also maintained for unscoped queries.

Usage::

    from src.retrieval.abbreviations import expand_query, get_abbreviations

    # Unscoped — includes all known expansions
    q = expand_query("DIM-V thermal")
    # → "DIM-V DIGITAL INPUT MODULE - VITAL thermal"

    # Scoped to a specific document
    q = expand_query("DIM-V thermal", scope_doc_id="8250cded…")
"""

from __future__ import annotations

import logging
import re
import threading
from dataclasses import dataclass, field
from pathlib import Path

from ..chunking.models import ChunkTier, ChunkType
from ..chunking.writer import load_chunks
from ..config.paths import CHUNK_DIR

logger = logging.getLogger(__name__)


# ── Internal store ──────────────────────────────────────────────


@dataclass
class _AbbreviationStore:
    """Parsed abbreviation data from all cached documents."""

    # Per-document: {doc_id: {acronym_upper: expansion}}
    by_doc: dict[str, dict[str, str]] = field(default_factory=dict)

    # Merged global: {acronym_upper: [expansion, …]}
    # Multiple expansions when the same acronym appears in different
    # documents with different meanings.
    global_map: dict[str, list[str]] = field(default_factory=dict)

    # Reverse lookup: {doc_id: module_name}  (for scope resolution)
    doc_modules: dict[str, str] = field(default_factory=dict)


# ── Markdown table parser ───────────────────────────────────────

# Matches a pipe-table data row (not the separator row like |---|---|)
_RE_TABLE_ROW = re.compile(
    r"^\|\s*(?P<abbr>[^|]+?)\s*\|\s*(?P<meaning>[^|]+?)\s*\|",
    re.MULTILINE,
)
_RE_SEPARATOR = re.compile(r"^[\s|:-]+$")


def _parse_abbreviation_table(text: str) -> dict[str, str]:
    """Parse a markdown pipe-table into ``{ACRONYM: expansion}`` pairs.

    Handles multi-line cell wrapping by joining fragmented rows and
    normalising whitespace.  The header row (``Abbreviation | Meaning``)
    and separator rows (``|---|---|``) are skipped.
    """
    mappings: dict[str, str] = {}

    for m in _RE_TABLE_ROW.finditer(text):
        abbr = m.group("abbr").strip()
        meaning = m.group("meaning").strip()

        # Skip header row and separator rows.
        if not abbr or not meaning:
            continue
        if abbr.lower() == "abbreviation" or meaning.lower() == "meaning":
            continue
        if _RE_SEPARATOR.match(abbr) or _RE_SEPARATOR.match(meaning):
            continue

        # Normalise whitespace (handles line-wrap artifacts).
        abbr = " ".join(abbr.split())
        meaning = " ".join(meaning.split())

        # Store with uppercase key for case-insensitive lookup.
        mappings[abbr.upper()] = meaning

    return mappings


# ── Index construction ──────────────────────────────────────────

_lock = threading.Lock()
_store: _AbbreviationStore | None = None


def _build_store(chunk_dir: Path | None = None) -> _AbbreviationStore:
    """Scan all chunk JSONs and extract abbreviation tables."""
    chunk_dir = Path(chunk_dir or CHUNK_DIR)
    json_files = sorted(chunk_dir.glob("*.json"))

    store = _AbbreviationStore()

    for json_path in json_files:
        doc_id = json_path.stem
        try:
            doc = load_chunks(doc_id)
        except Exception:
            logger.warning("Failed to load chunks for %s — skipping.", doc_id[:12])
            continue

        store.doc_modules[doc_id] = doc.doc_metadata.module_name

        abbr_chunks = [
            c for c in doc.chunks
            if c.chunk_type == ChunkType.ABBREVIATION_TABLE
        ]
        if not abbr_chunks:
            continue

        doc_map: dict[str, str] = {}
        for chunk in abbr_chunks:
            doc_map.update(_parse_abbreviation_table(chunk.text))

        if doc_map:
            store.by_doc[doc_id] = doc_map
            for acronym, expansion in doc_map.items():
                existing = store.global_map.setdefault(acronym, [])
                if expansion not in existing:
                    existing.append(expansion)

    total = sum(len(m) for m in store.by_doc.values())
    logger.info(
        "Abbreviation store built: %d unique acronyms from %d documents.",
        len(store.global_map), len(store.by_doc),
    )
    return store


def _get_store() -> _AbbreviationStore:
    """Return the singleton abbreviation store, building on first access."""
    global _store
    if _store is None:
        with _lock:
            if _store is None:
                _store = _build_store()
    return _store


def invalidate_store() -> None:
    """Discard the cached store so it is rebuilt on next access."""
    global _store
    with _lock:
        _store = None
    logger.debug("Abbreviation store invalidated.")


# ── Acronym detection in queries ────────────────────────────────

# Tokens that look like acronyms: uppercase words (2+ chars),
# optionally hyphenated (DIM-V, DIM-NV).
_RE_ACRONYM_CANDIDATE = re.compile(
    r"\b([A-Z][A-Z0-9]{1,}(?:-[A-Z]{1,3})?)\b"
)


def _find_acronyms_in_query(query: str) -> list[str]:
    """Extract potential acronyms from *query*.

    Returns uppercase strings that look like abbreviations.
    """
    return list(dict.fromkeys(_RE_ACRONYM_CANDIDATE.findall(query)))


# ── Public API ──────────────────────────────────────────────────


def get_abbreviations(
    scope_doc_id: str | None = None,
) -> dict[str, str]:
    """Return the abbreviation → expansion mapping.

    Parameters
    ----------
    scope_doc_id:
        If provided, return only that document's abbreviations.
        Otherwise return the merged global map (first expansion per
        acronym if there are multiple).

    Returns
    -------
    dict[str, str]
        ``{ACRONYM: expansion}`` with uppercase keys.
    """
    store = _get_store()

    if scope_doc_id and scope_doc_id in store.by_doc:
        return dict(store.by_doc[scope_doc_id])

    # Global map — take first expansion for each acronym.
    return {k: v[0] for k, v in store.global_map.items()}


def expand_query(
    query: str,
    scope_doc_id: str | None = None,
) -> str:
    """Augment *query* by appending abbreviation expansions.

    The original acronym is **preserved** in the query; its expansion
    is appended so both the short form and long form contribute to
    vector and BM25 scoring.

    When *scope_doc_id* is provided, only that document's abbreviation
    table is consulted.  If an acronym is ambiguous (different meanings
    across documents) and no scope is given, all known expansions are
    appended.

    Parameters
    ----------
    query:
        The raw or partially processed query string.
    scope_doc_id:
        Optional document ID to scope abbreviation lookup.

    Returns
    -------
    str
        The augmented query.  Unchanged if no acronyms were found or
        no expansions exist.
    """
    store = _get_store()
    candidates = _find_acronyms_in_query(query)

    if not candidates:
        return query

    expansions: list[str] = []

    for acronym in candidates:
        key = acronym.upper()

        if scope_doc_id and scope_doc_id in store.by_doc:
            # Scoped: prefer the document's own table.
            doc_map = store.by_doc[scope_doc_id]
            if key in doc_map:
                expansions.append(doc_map[key])
                continue

        # Unscoped or not found in scoped doc: use global map.
        if key in store.global_map:
            meanings = store.global_map[key]
            if len(meanings) == 1:
                expansions.append(meanings[0])
            else:
                # Ambiguous — include all distinct expansions.
                expansions.extend(meanings)

    if not expansions:
        return query

    # Append expansions after the original query.
    suffix = " ".join(dict.fromkeys(expansions))
    return f"{query} {suffix}"
