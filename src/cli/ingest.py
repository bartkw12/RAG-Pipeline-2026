"""CLI entrypoint for the ingestion pipeline.

Usage examples::

    # Drop-folder mode (default) — ingest everything in input/
    python -m src.cli.ingest

    # CLI mode — explicit files or globs
    python -m src.cli.ingest --paths "D:/specs/*.pdf" report.docx

    # Manifest mode
    python -m src.cli.ingest --manifest manifest.json

    # Dry run — see what would happen without doing anything
    python -m src.cli.ingest --dry-run

    # Force re-ingest everything, ignore duplicate detection
    python -m src.cli.ingest --force

    # Override the default input directory
    python -m src.cli.ingest --input-dir "D:/Engineering/Docs"

    # Combine flags
    python -m src.cli.ingest --paths "*.pdf" --dry-run --verbose
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path


def _build_parser() -> argparse.ArgumentParser:
    """Build the argument parser for the ingest command."""
    parser = argparse.ArgumentParser(
        prog="ingest",
        description=(
            "Ingest engineering documents into the RAG pipeline.\n\n"
            "Documents can be selected in three ways (highest priority first):\n"
            "  1. --paths   Explicit file paths or glob patterns\n"
            "  2. --manifest  A JSON manifest with include/exclude rules\n"
            "  3. (default)   Scan the input/ drop folder"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    # ── Selection mode arguments (mutually exclusive intent) ────
    selection = parser.add_argument_group("document selection")
    selection.add_argument(
        "--paths",
        nargs="+",
        metavar="PATH",
        help=(
            "One or more file paths or glob patterns to ingest. "
            'Example: --paths "D:/specs/*.pdf" report.docx'
        ),
    )
    selection.add_argument(
        "--manifest",
        type=Path,
        metavar="FILE",
        help="Path to a JSON manifest file with include/exclude rules.",
    )
    selection.add_argument(
        "--input-dir",
        type=Path,
        metavar="DIR",
        help="Override the default input directory for drop-folder mode.",
    )

