"""Ingestion pipeline — orchestrates document selection, tracking, and parsing.

This is the central entry point that wires together:

* **select.py**   → decides *which* files to ingest (CLI / manifest / drop folder)
* **registry.py** → checks duplicates and tracks what's been ingested
* **parser.py**   → converts raw documents to normalised output (stub for now)

Typical invocation (from CLI or programmatically)::

    from ingestion.pipeline import run

    summary = run(cli_paths=["D:/specs/*.pdf"])
    print(summary)
"""

from __future__ import annotations

import logging
import shutil
from dataclasses import dataclass, field
from pathlib import Path

from config.paths import INPUT_DIR, PROCESSED_DIR, ensure_dirs
from ingestion.registry import (
    CheckResult,
    FileStatus,
    IngestionRegistry,
)
from ingestion.select import SelectionMode, SelectionResult, select_documents

logger = logging.getLogger(__name__)

# ── Result types ────────────────────────────────────────────────


@dataclass
class FileOutcome:
    """What happened to a single file during the pipeline run."""

    path: Path
    status: FileStatus
    doc_id: str
    message: str


@dataclass
class PipelineSummary:
    """Aggregate result of a full pipeline run."""

    selection_mode: SelectionMode
    total_selected: int = 0
    new_ingested: int = 0
    re_ingested: int = 0
    skipped_unchanged: int = 0
    skipped_unsupported: int = 0
    failed: int = 0
    outcomes: list[FileOutcome] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    def __str__(self) -> str:
        lines = [
            "",
            "═" * 56,
            "  Ingestion Summary",
            "═" * 56,
            f"  Selection mode : {self.selection_mode.value}",
            f"  Files selected : {self.total_selected}",
            f"  New ingested   : {self.new_ingested}",
            f"  Re-ingested    : {self.re_ingested}",
            f"  Skipped (dup)  : {self.skipped_unchanged}",
            f"  Skipped (ext)  : {self.skipped_unsupported}",
            f"  Failed         : {self.failed}",
            "═" * 56,
        ]
        if self.warnings:
            lines.append("  Warnings:")
            for w in self.warnings:
                lines.append(f"    - {w}")
            lines.append("═" * 56)
        return "\n".join(lines)

# ── File movement ───────────────────────────────────────────────


def _is_child_of(child: Path, parent: Path) -> bool:
    """Return True if *child* is strictly inside *parent*."""
    try:
        child.relative_to(parent)
        return True
    except ValueError:
        return False


def _move_to_processed(file_path: Path) -> None:
    """Move a file from the input folder into ``input/processed/``,
    preserving its subdirectory structure relative to ``INPUT_DIR``.

    Only acts on files that live inside ``INPUT_DIR``.  Files provided
    via CLI or manifest pointing elsewhere are never moved.
    """
    resolved = file_path.resolve()
    input_resolved = INPUT_DIR.resolve()

    if not _is_child_of(resolved, input_resolved):
        return  # file is external — leave it alone

    # Preserve relative subdirectory structure
    relative = resolved.relative_to(input_resolved)
    dest = PROCESSED_DIR / relative

    # Handle name collision (append counter)
    if dest.exists():
        stem = dest.stem
        suffix = dest.suffix
        counter = 1
        while dest.exists():
            dest = dest.with_name(f"{stem}_{counter}{suffix}")
            counter += 1

    dest.parent.mkdir(parents=True, exist_ok=True)
    shutil.move(str(resolved), str(dest))
    logger.info("Moved '%s' → '%s'", resolved.name, dest)

# ── Parser stub ─────────────────────────────────────────────────


def _parse_document(path: Path) -> bool:
    """Parse a single document.

    This is a **stub** — it will be replaced with real parsing logic
    (PDF extraction, DOCX conversion, etc.) in the next step.

    Returns True on success, False on failure.
    """
    logger.info("Parsing '%s' … (stub — no real parsing yet)", path.name)
    # TODO: dispatch to appropriate parser based on extension
    #   .pdf  → ingestion.parsers.pdf
    #   .docx → ingestion.parsers.docx
    return True

# ── Main pipeline ───────────────────────────────────────────────


def run(
    *,
    cli_paths: list[str] | None = None,
    manifest_path: Path | None = None,
    input_dir: Path | None = None,
    dry_run: bool = False,
    force: bool = False,
) -> PipelineSummary:
    """Run the ingestion pipeline end-to-end.

    Parameters
    ----------
    cli_paths:
        Explicit file paths / glob patterns (highest priority).
    manifest_path:
        Path to a manifest JSON (second priority).
    input_dir:
        Override the default input directory for drop-folder mode.
    dry_run:
        If True, report what *would* happen without actually ingesting
        or moving any files.
    force:
        If True, re-ingest all selected files regardless of the registry.

    Returns
    -------
    PipelineSummary
        Counts and per-file outcomes for the entire run.
    """
    # ── 1. Ensure directories exist ─────────────────────────────
    ensure_dirs()

    # ── 2. Select documents ─────────────────────────────────────
    selection: SelectionResult = select_documents(
        cli_paths=cli_paths,
        manifest_path=manifest_path,
        input_dir=input_dir,
    )

    summary = PipelineSummary(
        selection_mode=selection.mode,
        total_selected=len(selection.files),
        skipped_unsupported=len(selection.skipped),
        warnings=list(selection.warnings),
    )

    if not selection.files:
        summary.warnings.append("No files to ingest.")
        logger.warning("No files to ingest.")
        return summary

    # Log selection info
    logger.info(
        "Selected %d file(s) via %s mode.",
        len(selection.files),
        selection.mode.value,
    )
    for w in selection.warnings:
        logger.warning(w)


