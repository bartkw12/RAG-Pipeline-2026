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


