"""Document selection — resolves *which* files to ingest.

Supports three input modes with strict precedence (most explicit wins):

1. **CLI paths/globs**  — user passes explicit file paths or glob patterns.
2. **Manifest file**    — user points to a JSON manifest with include/exclude rules.
3. **Drop folder**      — default: scan ``INPUT_DIR`` recursively for supported files.

Only **one** mode is active per run.  If ``--paths`` is given, the input folder
and any manifest are ignored.  If ``--manifest`` is given, the input folder is
ignored.  This avoids surprises where explicit selections silently pull in
extra files.

This module is pure logic — no printing, no file mutations, no side effects.
It returns a ``SelectionResult`` that the pipeline can act on.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Iterable

from config.paths import INPUT_DIR, MANIFEST_DEFAULT, SUPPORTED_EXTENSIONS

logger = logging.getLogger(__name__)

# ── Public types ────────────────────────────────────────────────


class SelectionMode(str, Enum):
    """How the documents were selected."""

    CLI = "cli"              # explicit paths / globs from CLI args
    MANIFEST = "manifest"    # loaded from a manifest JSON file
    DROP_FOLDER = "drop_folder"  # scanned from INPUT_DIR


@dataclass(frozen=True)
class SelectionResult:
    """Outcome of document selection.

    Attributes:
        mode:       Which selection method was used.
        files:      Resolved, deduplicated, sorted list of file paths to ingest.
        skipped:    Files that were found but excluded (wrong extension, etc.).
        warnings:   Human-readable messages about issues encountered.
    """

    mode: SelectionMode
    files: list[Path]
    skipped: list[Path] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
