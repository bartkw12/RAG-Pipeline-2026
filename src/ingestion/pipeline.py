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


