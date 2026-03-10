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

from ..config.paths import INPUT_DIR, MANIFEST_DEFAULT, SUPPORTED_EXTENSIONS

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
    warnings: list[str] = []

    if not manifest_path.exists():
        warnings.append(f"Manifest not found: '{manifest_path}'")
        return SelectionResult(
            mode=SelectionMode.MANIFEST,
            files=[],
            warnings=warnings,
        )

    try:
        data = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError) as exc:
        warnings.append(f"Failed to read manifest '{manifest_path}': {exc}")
        return SelectionResult(
            mode=SelectionMode.MANIFEST,
            files=[],
            warnings=warnings,
        )

    # Parse manifest fields
    roots = [Path(r).expanduser().resolve() for r in data.get("roots", [])]
    if not roots:
        roots = [INPUT_DIR]

    include_patterns: list[str] = data.get("include", [])
    exclude_patterns: list[str] = data.get("exclude", [])
    explicit_files: list[str] = data.get("files", [])

    # Validate roots exist
    for root in roots:
        if not root.is_dir():
            warnings.append(f"Manifest root directory does not exist: '{root}'")

    # 1. Collect explicit files
    candidates: set[Path] = set()
    for f in explicit_files:
        resolved = _resolve_and_validate(Path(f))
        if resolved:
            candidates.add(resolved)
        else:
            warnings.append(f"Manifest explicit file not found: '{f}'")

    # 2. Expand include patterns (or default to all supported types)
    if include_patterns:
        for root in roots:
            if root.is_dir():
                candidates |= _expand_globs(include_patterns, root=root)
    else:
        # Default: include all supported extensions under each root
        default_patterns = [f"**/*{ext}" for ext in sorted(SUPPORTED_EXTENSIONS)]
        for root in roots:
            if root.is_dir():
                candidates |= _expand_globs(default_patterns, root=root)

    # 3. Apply exclude patterns
    excluded: set[Path] = set()
    for root in roots:
        if root.is_dir():
            excluded |= _expand_globs(exclude_patterns, root=root)

    candidates -= excluded

    if excluded:
        logger.debug("Manifest excludes removed %d file(s).", len(excluded))

    supported, skipped = _filter_supported(candidates)

    if skipped:
        exts = ", ".join(sorted({p.suffix for p in skipped}))
        warnings.append(
            f"{len(skipped)} file(s) skipped — unsupported extension(s): {exts}."
        )

    return SelectionResult(
        mode=SelectionMode.MANIFEST,
        files=supported,
        skipped=skipped,
        warnings=warnings,
    )


def _select_from_drop_folder(input_dir: Path) -> SelectionResult:
    """Recursively scan the input directory for supported files.

    Ignores the ``processed/`` subdirectory so already-ingested files that
    were moved there aren't re-discovered.
    """
    warnings: list[str] = []

    if not input_dir.is_dir():
        warnings.append(f"Input directory does not exist: '{input_dir}'")
        return SelectionResult(
            mode=SelectionMode.DROP_FOLDER,
            files=[],
            warnings=warnings,
        )

    processed_dir = (input_dir / "processed").resolve()
    candidates: list[Path] = []

    for p in input_dir.rglob("*"):
        resolved = p.resolve()
        # Skip anything inside the processed/ subfolder
        if resolved == processed_dir or _is_child_of(resolved, processed_dir):
            continue
        if resolved.is_file() and _is_supported(resolved):
            candidates.append(resolved)

    files = sorted(set(candidates))

    if not files:
        warnings.append(
            f"No supported files found in '{input_dir}'. "
            f"Supported extensions: {', '.join(sorted(SUPPORTED_EXTENSIONS))}"
        )

    return SelectionResult(
        mode=SelectionMode.DROP_FOLDER,
        files=files,
        warnings=warnings,
    )


def _is_child_of(child: Path, parent: Path) -> bool:
    """Return True if *child* is strictly inside *parent*."""
    try:
        child.relative_to(parent)
        return True
    except ValueError:
        return False


# ── Main entry point ────────────────────────────────────────────


def select_documents(
    *,
    cli_paths: list[str] | None = None,
    manifest_path: Path | None = None,
    input_dir: Path | None = None,
) -> SelectionResult:
    """Select documents for ingestion using strict precedence.

    Priority (highest first):

    1. ``cli_paths`` — explicit file paths / glob patterns from CLI.
    2. ``manifest_path`` — path to a manifest JSON file.
    3. Drop folder scan of ``input_dir`` (defaults to ``INPUT_DIR``).

    Only **one** mode is active per call.  If ``cli_paths`` is provided,
    the manifest and drop folder are ignored entirely.

    Parameters
    ----------
    cli_paths:
        List of file paths or glob patterns (e.g. ``["*.pdf", "D:/specs/doc.docx"]``).
    manifest_path:
        Path to a JSON manifest file.  Falls back to ``MANIFEST_DEFAULT``
        only if the caller explicitly requests manifest mode (i.e. this
        parameter is ``MANIFEST_DEFAULT`` or a real path — not auto-detected).
    input_dir:
        Override the default input directory for drop-folder mode.

    Returns
    -------
    SelectionResult
        Contains the resolved file list, selection mode, any skipped files,
        and warning messages.
    """
    # ── Mode 1: CLI paths (highest priority) ────────────────────
    if cli_paths:
        logger.info("Selection mode: CLI (%d path(s)/pattern(s) provided).",
                     len(cli_paths))
        return _select_from_cli(cli_paths)

    # ── Mode 2: Manifest ────────────────────────────────────────
    if manifest_path is not None:
        logger.info("Selection mode: manifest ('%s').", manifest_path)
        return _select_from_manifest(manifest_path)

    # ── Mode 3: Drop folder (default) ──────────────────────────
    folder = input_dir or INPUT_DIR
    logger.info("Selection mode: drop folder ('%s').", folder)
    return _select_from_drop_folder(folder)
