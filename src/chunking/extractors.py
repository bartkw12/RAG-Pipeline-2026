"""Field extractors — pull structured metadata from raw content-block text.

Each extractor targets a specific ``ChunkType`` and returns a plain
dictionary whose keys match the corresponding ``ChunkMetadata`` field
names.  The chunk-assembly layer merges these dicts into metadata
objects.

Two generic helpers — ``extract_cross_references`` and
``extract_component_ids`` — apply to **all** block types.
"""

from __future__ import annotations

import re


# ── Regex patterns ──────────────────────────────────────────────

# Bold-field pattern:  **Field:** value (captures the rest of the line)
def _bold_field(name: str) -> re.Pattern[str]:
    """Build a regex that captures the value after ``**<name>:** ``."""
    return re.compile(
        rf"\*\*{re.escape(name)}:\*\*\s*(.+?)(?:\n|$)",
        re.IGNORECASE,
    )

_RE_TEST_CASE_ID = _bold_field("Test case")
_RE_TEST_NAME    = _bold_field("Test")
_RE_TEST_ITEM    = _bold_field("Test Item")
_RE_TEST_RESULT  = _bold_field("Result")
_RE_DATE         = _bold_field("Date")
_RE_TESTER       = _bold_field("Test carried out by")
_RE_VERIFIER     = _bold_field("Test verified by")
_RE_FAIL_CRIT    = _bold_field("Failure criteria")

# Traceability line — may contain multiple bracket IDs:
# **Traceability:** [HW-IRS_PAM_93][HW-IRS_PAM_98] [FVTSR_PAM_0009]
_RE_TRACEABILITY_LINE = _bold_field("Traceability")

# Reference line:
# **Reference:** HW-IRS_DIM_VI_392, HW-IRS_DIM_VI_393, ...
# **Reference:** Standard Test
# Possibly multi-line, so we capture more generously.
_RE_REFERENCE_BLOCK = re.compile(
    r"\*\*Reference:\*\*\s*([\s\S]*?)(?=\n\*\*|\Z)",
    re.IGNORECASE,
)

# Bracket-style ID: [IDENTIFIER] where IDENTIFIER contains uppercase
# letters, digits, underscores, and colons.
_RE_BRACKET_ID = re.compile(r"\[([A-Z][A-Z0-9_:]+(?::[A-Z0-9_]+)*)\]")

# HW-IRS requirement identifiers anywhere in text.
_RE_REQUIREMENT_ID = re.compile(r"HW-IRS_\w+")

# DOORS metadata tag:
# [Category: Requirement | Allocation: HW | Priority: Mandatory | Safety: No | Verification: Test]
_RE_CATEGORY     = re.compile(r"\[Category:\s*([^|\]]+)")
_RE_ALLOCATION   = re.compile(r"Allocation:\s*([^|\]]+)")
_RE_PRIORITY     = re.compile(r"Priority:\s*([^|\]]+)")
_RE_SAFETY       = re.compile(r"Safety:\s*([^|\]]+)")
_RE_VERIFICATION = re.compile(r"Verification:\s*([^|\]]+)")

# Thales item number: 7HA-02941-ABAA  or  7HA 02941 ABAA
_RE_COMPONENT_ID = re.compile(r"7HA[\s-]\d{5}[\s-][A-Z]{4}")

# Pipe-table row detector.
_RE_TABLE_ROW = re.compile(r"^\|.*\|", re.MULTILINE)

# Figure / VLM reference detector.
_RE_FIGURE = re.compile(
    r"\[VLM\s*-|"
    r"\[Figure\s*[—–-]|"
    r"^Figure\s+\d+:",
    re.IGNORECASE | re.MULTILINE,
)

# IDs that appear in **Reference:** values (HW-IRS IDs, doc refs, etc.)
# Note: OCR/parser sometimes breaks "HW-IRS_" across lines as "HW- IRS_".
_RE_REF_ID = re.compile(r"HW-\s*IRS_\w+|[A-Z][A-Z0-9_]{2,}")


# ── Test-case extractor ────────────────────────────────────────


def extract_test_case_fields(text: str) -> dict:
    """Extract structured fields from a ``TEST_CASE`` block.

    Returns a dict whose keys match ``ChunkMetadata`` field names.
    Missing fields are omitted from the dict (not set to ``None``).
    """
    fields: dict = {}

    _set_match(fields, "test_case_id", _RE_TEST_CASE_ID, text)
    _set_match(fields, "test_name", _RE_TEST_NAME, text)
    _set_match(fields, "test_item", _RE_TEST_ITEM, text)
    _set_match(fields, "test_result", _RE_TEST_RESULT, text)
    _set_match(fields, "date", _RE_DATE, text)
    _set_match(fields, "tester", _RE_TESTER, text)
    _set_match(fields, "verifier", _RE_VERIFIER, text)
    _set_match(fields, "failure_criteria", _RE_FAIL_CRIT, text)

    # Traceability IDs — extract bracket IDs from the traceability line.
    trace_m = _RE_TRACEABILITY_LINE.search(text)
    if trace_m:
        trace_text = trace_m.group(0)
        ids = _RE_BRACKET_ID.findall(trace_text)
        if ids:
            fields["traceability_ids"] = ids

    # Reference IDs — extract meaningful identifiers from the reference block.
    ref_m = _RE_REFERENCE_BLOCK.search(text)
    if ref_m:
        ref_text = ref_m.group(1)
        # Also try bracket IDs first (e.g. [SAC_GUIDE], [CD_PAM]).
        bracket_ids = _RE_BRACKET_ID.findall(ref_text)
        # Then loose IDs (HW-IRS_DIM_291, etc.).
        loose_ids = _RE_REF_ID.findall(ref_text)
        # Normalise line-broken "HW- IRS_" → "HW-IRS_".
        loose_ids = [re.sub(r"^HW-\s+IRS_", "HW-IRS_", r) for r in loose_ids]
        # Combine, deduplicate, filter generics.
        ref_ids = list(dict.fromkeys(bracket_ids + loose_ids))
        ref_ids = [r for r in ref_ids if not _is_generic_word(r)]
        if ref_ids:
            fields["reference_ids"] = ref_ids

    return fields


# ── Requirement extractor ───────────────────────────────────────


def extract_requirement_fields(text: str) -> dict:
    """Extract structured fields from a ``REQUIREMENT`` block.

    Returns a dict whose keys match ``ChunkMetadata`` field names.
    """
    fields: dict = {}

    # All HW-IRS identifiers in the block.
    req_ids = _RE_REQUIREMENT_ID.findall(text)
    if req_ids:
        fields["requirement_ids"] = sorted(set(req_ids))

    # DOORS metadata tag fields.
    _set_match(fields, "category", _RE_CATEGORY, text)
    _set_match(fields, "allocation", _RE_ALLOCATION, text)
    _set_match(fields, "priority", _RE_PRIORITY, text)
    _set_match(fields, "safety", _RE_SAFETY, text)
    _set_match(fields, "verification_method", _RE_VERIFICATION, text)

    # is_background flag
    if fields.get("category", "").lower() == "background":
        fields["is_background"] = True

    # Traceability IDs from **Traceability:** line.
    trace_m = _RE_TRACEABILITY_LINE.search(text)
    if trace_m:
        trace_text = trace_m.group(0)
        ids = _RE_BRACKET_ID.findall(trace_text)
        if ids:
            fields["traceability_ids"] = ids

    return fields


# ── Generic extractors (all block types) ────────────────────────


def extract_cross_references(text: str) -> list[str]:
    """Return all bracket-style document references found in *text*.

    Matches patterns like ``[HWADD:TOP:0014]``, ``[CD_PAM]``,
    ``[HW-IRS_DIM_VI_275]``, ``[FVTSR_PAM_0002]``, etc.

    Returns a sorted, deduplicated list of the inner ID strings
    (without brackets).
    """
    return sorted(set(_RE_BRACKET_ID.findall(text)))


def extract_component_ids(text: str) -> list[str]:
    """Return all Thales 7HA item numbers found in *text*.

    Matches both ``7HA-02941-ABAA`` and ``7HA 02941 ABAA`` formats.
    Results are normalised to hyphen-separated form.
    """
    raw = _RE_COMPONENT_ID.findall(text)
    # Normalise spaces to hyphens for consistency.
    normalised = [r.replace(" ", "-") for r in raw]
    return sorted(set(normalised))


def detect_embedded_tables(text: str) -> bool:
    """Return ``True`` if *text* contains Markdown pipe-table rows."""
    return bool(_RE_TABLE_ROW.search(text))


def detect_embedded_figures(text: str) -> bool:
    """Return ``True`` if *text* contains a figure reference or VLM tag."""
    return bool(_RE_FIGURE.search(text))


# ── Private helpers ─────────────────────────────────────────────

# Words that match _RE_REF_ID but aren't real document references.
_GENERIC_WORDS = frozenset({
    "CMCS", "HWCC", "HEM", "ECN", "RoHS", "WEEE", "ESD",
    "RAM", "LED", "FAST", "BOM", "DII", "DIM", "PAM", "TOP",
    "NOT", "THE", "AND", "FOR", "ALL", "TEST", "NONE",
    "STANDARD", "ANALYSIS", "INSPECTION", "DEMONSTRATION",
    "NA", "TBD", "TBC", "YES",
    "NO_OUT", "LINKS",  # sentinel from [NO_OUT-LINKS: ...]
})


def _is_generic_word(token: str) -> bool:
    """Return True if *token* is a common acronym, not a doc reference."""
    return token.upper() in _GENERIC_WORDS


def _set_match(
    fields: dict,
    key: str,
    pattern: re.Pattern[str],
    text: str,
) -> None:
    """Search *text* with *pattern*; if matched, set ``fields[key]``."""
    m = pattern.search(text)
    if m:
        value = m.group(1).strip()
        if value:
            fields[key] = value
