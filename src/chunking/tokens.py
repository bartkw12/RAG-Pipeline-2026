"""Token counting utility — thin wrapper around ``tiktoken``.

Provides a single ``count_tokens`` function used by the chunking
pipeline to measure chunk sizes against the configured budgets.

The ``tiktoken.Encoding`` object is cached at module level so the
(relatively expensive) BPE initialisation happens only once per
process, regardless of how many chunks are counted.
"""

from __future__ import annotations

import tiktoken

# ── Module-level encoding cache ─────────────────────────────────
# Keyed by encoding name so switching models mid-process works.
_cache: dict[str, tiktoken.Encoding] = {}


def _get_encoding(encoding_name: str) -> tiktoken.Encoding:
    """Return a cached ``tiktoken.Encoding`` instance."""
    if encoding_name not in _cache:
        _cache[encoding_name] = tiktoken.get_encoding(encoding_name)
    return _cache[encoding_name]


def count_tokens(text: str, encoding_name: str = "cl100k_base") -> int:
    """Count the number of tokens in *text*.

    Parameters
    ----------
    text:
        The string to tokenise.
    encoding_name:
        Name of the ``tiktoken`` encoding.  Defaults to
        ``"cl100k_base"`` (GPT-4 / Ada-002 family).

    Returns
    -------
    int
        Token count.  Returns ``0`` for empty or whitespace-only input.
    """
    if not text or not text.strip():
        return 0
    enc = _get_encoding(encoding_name)
    return len(enc.encode(text))
