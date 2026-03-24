"""Query analyzer — parse, classify, and extract scope from retrieval queries.

Examines an incoming query string and produces a ``QueryAnalysis`` that
tells the rest of the pipeline **how** to search:

* **EXACT_LOOKUP** — the query contains only structured identifiers
  (test-case IDs, requirement IDs, component numbers) with no
  meaningful natural-language content.  Direct metadata lookup.
* **SCOPED_SEMANTIC** — the query contains identifiers *or* scope
  signals (doc type, module name, system) combined with natural
  language.  Hybrid search with ChromaDB ``where`` filters.
* **UNCONSTRAINED** — pure natural language with no scope signals.
  Broad hybrid search.

Usage::

    from src.retrieval.analyzer import analyze_query

    analysis = analyze_query("DIM-V thermal test results")
    # analysis.strategy == QueryStrategy.SCOPED_SEMANTIC
    # analysis.scope_filters == {"module_name": "DIM-V"}
    # analysis.semantic_remainder == "thermal test results"
"""

from __future__ import annotations

import logging
import re

from .models import QueryAnalysis, QueryStrategy

logger = logging.getLogger(__name__)


# ── Identifier patterns ─────────────────────────────────────────
# Reuse the same conventions as src/chunking/extractors.py.

# Test-case IDs: FVTR_OPT_01, FVTSR_PAM_0009, FVTR_FUNC_13
_RE_TEST_CASE_ID = re.compile(
    r"\b(FVT[SR]?_[A-Z]+_\d+)\b",
    re.IGNORECASE,
)

# Requirement IDs: HW-IRS_DIM_VI_275, HW-IRS_PAM_93
_RE_REQUIREMENT_ID = re.compile(
    r"\b(HW-IRS_\w+)\b",
    re.IGNORECASE,
)

# Component / item numbers: 7HA-02944-AAAA, 7HA 02944 AAAA
_RE_COMPONENT_ID = re.compile(
    r"\b(7HA[\s-]\d{5}[\s-][A-Z]{4})\b",
    re.IGNORECASE,
)

# Cross-reference tags (bracket form): [HWADD:TOP:0012]
_RE_CROSS_REF_BRACKET = re.compile(
    r"\[([A-Z][A-Z0-9_:]+(?::[A-Z0-9_]+)+)\]",
)

# Cross-reference tags (bare colon form): HWADD:TOP:0012
_RE_CROSS_REF_BARE = re.compile(
    r"\b([A-Z]{2,}(?::[A-Z0-9_]+){1,})\b",
)


# ── Scope-signal patterns ───────────────────────────────────────
# These indicate query intent is scoped to a document type, module,
# or system — but are NOT structured identifiers.

# Known doc types (exact match, case-insensitive).
_DOC_TYPES = {"FVTR", "HWIRS", "HW-IRS"}
_DOC_TYPE_CANONICAL = {
    "FVTR": "FVTR",
    "HWIRS": "HwIRS",
    "HW-IRS": "HwIRS",
}

# Known module names (exact match, case-insensitive).
# Values must match what appears in ChromaDB metadata.
_MODULE_NAMES = {
    "DIM-V": "DIM-V",
    "DIM-VI": "DIM-V",      # alias
    "DIM-NV": "DIM-NV",
    "DIM": "DIM",
    "PAM": "PAM",
}

# Known system names.
_SYSTEM_NAMES = {"TOP"}

# Regex to find module-name-like tokens in the query.
_RE_MODULE_TOKEN = re.compile(
    r"\b(DIM-(?:V|VI|NV)|DIM|PAM)\b",
    re.IGNORECASE,
)

# Regex to find doc-type-like tokens in the query.
_RE_DOC_TYPE_TOKEN = re.compile(
    r"\b(FVTR|HwIRS|HW-IRS|HWIRS)\b",
    re.IGNORECASE,
)

# Regex to find system-name tokens.
_RE_SYSTEM_TOKEN = re.compile(
    r"\b(TOP)\b",
    re.IGNORECASE,
)


# ── Stopwords for remainder classification ──────────────────────
# If stripping IDs and scope tokens leaves only these, the query is
# considered identifier-only (EXACT_LOOKUP).

_STOPWORDS = frozenset({
    "a", "an", "the", "is", "are", "was", "were", "be", "been",
    "being", "have", "has", "had", "do", "does", "did", "will",
    "would", "could", "should", "may", "might", "shall", "can",
    "of", "in", "to", "for", "with", "on", "at", "from", "by",
    "about", "as", "into", "through", "during", "before", "after",
    "and", "but", "or", "nor", "not", "no", "so", "yet",
    "what", "which", "who", "whom", "where", "when", "how", "why",
    "this", "that", "these", "those", "it", "its",
    "all", "each", "every", "both", "few", "more", "most", "some",
    "any", "other", "such",
    "me", "my", "i", "you", "your", "we", "our", "they", "their",
    "show", "get", "find", "give", "list", "tell",
    "?", "!", ".", ",", ":", ";",
})


# ── Public API ──────────────────────────────────────────────────


def analyze_query(query: str) -> QueryAnalysis:
    """Parse *query* and return a ``QueryAnalysis`` with routing strategy.

    Parameters
    ----------
    query:
        Raw user query string.

    Returns
    -------
    QueryAnalysis
        Populated with detected identifiers, scope filters, semantic
        remainder, and the chosen ``QueryStrategy``.
    """
    analysis = QueryAnalysis(original_query=query)

    # ── 1. Extract structured identifiers ───────────────────────
    remaining = query

    # Test-case IDs
    for m in _RE_TEST_CASE_ID.finditer(query):
        analysis.test_case_ids.append(m.group(1))
    remaining = _RE_TEST_CASE_ID.sub(" ", remaining)

    # Requirement IDs
    for m in _RE_REQUIREMENT_ID.finditer(query):
        analysis.requirement_ids.append(m.group(1))
    remaining = _RE_REQUIREMENT_ID.sub(" ", remaining)

    # Component IDs
    for m in _RE_COMPONENT_ID.finditer(query):
        # Normalise spaces to hyphens.
        analysis.component_ids.append(m.group(1).replace(" ", "-"))
    remaining = _RE_COMPONENT_ID.sub(" ", remaining)

    # Cross-references (bracket form first, then bare).
    for m in _RE_CROSS_REF_BRACKET.finditer(query):
        analysis.cross_references.append(m.group(1))
    remaining = _RE_CROSS_REF_BRACKET.sub(" ", remaining)

    for m in _RE_CROSS_REF_BARE.finditer(remaining):
        ref = m.group(1)
        if ref not in analysis.cross_references:
            analysis.cross_references.append(ref)
    remaining = _RE_CROSS_REF_BARE.sub(" ", remaining)

    # Deduplicate.
    analysis.test_case_ids = list(dict.fromkeys(analysis.test_case_ids))
    analysis.requirement_ids = list(dict.fromkeys(analysis.requirement_ids))
    analysis.component_ids = list(dict.fromkeys(analysis.component_ids))
    analysis.cross_references = list(dict.fromkeys(analysis.cross_references))

    # ── 2. Extract scope signals ────────────────────────────────
    scope_filters: dict[str, str] = {}

    # Doc type
    doc_type_m = _RE_DOC_TYPE_TOKEN.search(remaining)
    if doc_type_m:
        raw = doc_type_m.group(1).upper().replace(" ", "")
        canonical = _DOC_TYPE_CANONICAL.get(raw)
        if canonical:
            scope_filters["doc_type"] = canonical
        remaining = remaining[:doc_type_m.start()] + " " + remaining[doc_type_m.end():]

    # Module name
    module_m = _RE_MODULE_TOKEN.search(remaining)
    if module_m:
        raw = module_m.group(1).upper()
        canonical = _MODULE_NAMES.get(raw)
        if canonical:
            scope_filters["module_name"] = canonical
        remaining = remaining[:module_m.start()] + " " + remaining[module_m.end():]

    # System name (only add if no other scope found yet — "TOP" is
    # very common and low-signal when doc_type or module is known).
    if not scope_filters:
        system_m = _RE_SYSTEM_TOKEN.search(remaining)
        if system_m:
            # Don't filter by system in ChromaDB (all docs are TOP),
            # but do note it as a scope signal.
            remaining = remaining[:system_m.start()] + " " + remaining[system_m.end():]
            # Mark as scoped even though no filter needed.
            scope_filters["_system"] = "TOP"

    analysis.scope_filters = {k: v for k, v in scope_filters.items()
                              if not k.startswith("_")}

    # ── 3. Compute semantic remainder ───────────────────────────
    remainder_words = remaining.lower().split()
    meaningful_words = [w for w in remainder_words
                        if w not in _STOPWORDS and len(w) > 1]
    analysis.semantic_remainder = " ".join(remaining.split()).strip()

    # ── 4. Determine routing strategy ───────────────────────────
    has_ids = analysis.has_identifiers
    has_scope = bool(scope_filters)
    has_meaningful_text = len(meaningful_words) > 0

    if has_ids and not has_meaningful_text:
        # Only identifiers, no natural language → direct lookup.
        analysis.strategy = QueryStrategy.EXACT_LOOKUP
    elif has_ids or has_scope:
        # Identifiers or scope signals + natural language → scoped search.
        analysis.strategy = QueryStrategy.SCOPED_SEMANTIC
    else:
        # Pure natural language → broad search.
        analysis.strategy = QueryStrategy.UNCONSTRAINED

    logger.debug(
        "Query analyzed: strategy=%s ids=%d scope=%s remainder=%r",
        analysis.strategy.value,
        sum(len(lst) for lst in [
            analysis.test_case_ids, analysis.requirement_ids,
            analysis.component_ids, analysis.cross_references,
        ]),
        analysis.scope_filters or "(none)",
        analysis.semantic_remainder[:60],
    )

    return analysis
