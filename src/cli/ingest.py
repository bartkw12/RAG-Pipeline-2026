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

    # ── Behaviour flags ─────────────────────────────────────────
    behaviour = parser.add_argument_group("behaviour")
    behaviour.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be ingested without actually doing it.",
    )
    behaviour.add_argument(
        "--force",
        action="store_true",
        help="Re-ingest all files, ignoring the duplicate registry.",
    )

    # ── Output control ──────────────────────────────────────────
    output = parser.add_argument_group("output")
    output.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Enable verbose (DEBUG-level) logging.",
    )
    output.add_argument(
        "--quiet", "-q",
        action="store_true",
        help="Suppress all output except errors.",
    )

    # ── Parsing options ─────────────────────────────────────────
    parsing = parser.add_argument_group("parsing")
    parsing.add_argument(
        "--ocr",
        action="store_true",
        help="Enable OCR (EasyOCR) for scanned or image-heavy documents.",
    )
    parsing.add_argument(
        "--vlm",
        action="store_true",
        help="Enable the Vision-Language Model pipeline for richer extraction.",
    )
    parsing.add_argument(
        "--vlm-backend",
        choices=["azure", "local"],
        default="azure",
        help='VLM backend to use (default: "azure").',
    )
    parsing.add_argument(
        "--table-mode",
        choices=["accurate", "fast"],
        default="accurate",
        help='Table extraction mode (default: "accurate").',
    )
    parsing.add_argument(
        "--no-strip-headers",
        action="store_true",
        help="Keep page headers and footers in the output.",
    )
    parsing.add_argument(
        "--no-strip-toc",
        action="store_true",
        help="Keep the table-of-contents in the output.",
    )
    parsing.add_argument(
        "--keep-images",
        action="store_true",
        help="Keep image placeholders instead of stripping them.",
    )

    return parser


def _configure_logging(*, verbose: bool = False, quiet: bool = False) -> None:
    """Set up logging for the CLI session."""
    if quiet:
        level = logging.ERROR
    elif verbose:
        level = logging.DEBUG
    else:
        level = logging.INFO

    logging.basicConfig(
        level=level,
        format="%(levelname)-8s  %(message)s",
        handlers=[logging.StreamHandler(sys.stderr)],
    )

def main(argv: list[str] | None = None) -> int:
    """Parse arguments and run the ingestion pipeline.

    Returns 0 on success, 1 if no files were found, 2 on errors.
    """
    parser = _build_parser()
    args = parser.parse_args(argv)

    # Validate: --paths and --manifest are mutually exclusive
    if args.paths and args.manifest:
        parser.error("--paths and --manifest cannot be used together. Pick one.")

    _configure_logging(verbose=args.verbose, quiet=args.quiet)

    # Import pipeline here (after logging is configured) so log messages
    # emitted during module-level code are captured properly.
    from ..ingestion.pipeline import run

    summary = run(
        cli_paths=args.paths,
        manifest_path=args.manifest,
        input_dir=args.input_dir,
        dry_run=args.dry_run,
        force=args.force,
    )

    # Print summary to stdout (not via logging, so --quiet still shows it)
    if not args.quiet:
        print(summary)

    # Exit code
    if summary.failed > 0:
        return 2
    if summary.total_selected == 0:
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
