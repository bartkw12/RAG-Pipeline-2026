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

# ── Internal helpers ────────────────────────────────────────────


def _is_supported(path: Path) -> bool:
    """Check whether a file has a supported extension."""
    return path.suffix.lower() in SUPPORTED_EXTENSIONS


def _resolve_and_validate(path: Path) -> Path | None:
    """Resolve a path and return it only if it's an existing file."""
    resolved = path.expanduser().resolve()
    if resolved.is_file():
        return resolved
    return None

def _expand_globs(patterns: Iterable[str], root: Path | None = None) -> set[Path]:
    """Expand a list of glob patterns into resolved file paths.

    If *root* is given, patterns are treated as relative to *root*.
    Otherwise they are treated as-is (absolute or cwd-relative).
    """
    out: set[Path] = set()
    for pattern in patterns:
        base = root or Path(".")
        # Path.glob expects a relative pattern
        try:
            for match in base.glob(pattern):
                resolved = match.resolve()
                if resolved.is_file():
                    out.add(resolved)
        except (OSError, ValueError) as exc:
            logger.warning("Glob pattern '%s' failed: %s", pattern, exc)
    return out

def _filter_supported(
    paths: Iterable[Path],
) -> tuple[list[Path], list[Path]]:
    """Split paths into (supported, skipped) and sort each list."""
    supported: list[Path] = []
    skipped: list[Path] = []
    for p in paths:
        if _is_supported(p):
            supported.append(p)
        else:
            skipped.append(p)
    return sorted(set(supported)), sorted(set(skipped))

# ── Selection strategies ────────────────────────────────────────


# ── Selection strategies ────────────────────────────────────────


def _select_from_cli(cli_paths: list[str]) -> SelectionResult:
    """Resolve explicit CLI paths and glob patterns."""
    warnings: list[str] = []
    candidates: set[Path] = set()

    for raw in cli_paths:
        p = Path(raw)

        # If it looks like a glob (contains * or ?), expand it
        if any(c in raw for c in ("*", "?")):
            expanded = _expand_globs([raw])
            if not expanded:
                warnings.append(f"Glob pattern '{raw}' matched no files.")
            candidates |= expanded
        else:
            resolved = _resolve_and_validate(p)
            if resolved:
                candidates.add(resolved)
            else:
                warnings.append(f"Path not found or not a file: '{raw}'")

    supported, skipped = _filter_supported(candidates)

    if skipped:
        exts = ", ".join(sorted({p.suffix for p in skipped}))
        warnings.append(
            f"{len(skipped)} file(s) skipped — unsupported extension(s): {exts}. "
            f"Supported: {', '.join(sorted(SUPPORTED_EXTENSIONS))}"
        )

    return SelectionResult(
        mode=SelectionMode.CLI,
        files=supported,
        skipped=skipped,
        warnings=warnings,
    )

def _select_from_manifest(manifest_path: Path) -> SelectionResult:
    """Load a manifest JSON and resolve the files it describes.

    Manifest format::

        {
            "roots": ["./input"],
            "include": ["**/*.pdf", "**/*.docx"],
            "exclude": ["**/~$*.docx", "**/draft_*"],
            "files":  ["C:/Engineering/Special/spec-1234.pdf"]
        }

    All keys are optional.  ``roots`` defaults to ``INPUT_DIR``.
    ``include`` defaults to all supported extensions.
    """
    pass



