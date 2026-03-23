"""CLI entrypoint for standalone re-chunking.

Re-chunks cached Markdown files without re-parsing the original documents.

Usage examples::

    # Re-chunk all cached Markdown files
    python -m src.cli.chunk

    # Re-chunk a specific document by its doc_id
    python -m src.cli.chunk --doc-id f4b48797a6a2...

    # Re-chunk with verbose logging
    python -m src.cli.chunk --verbose
"""

from __future__ import annotations

import argparse
import logging
import sys


def _build_parser() -> argparse.ArgumentParser:
    """Build the argument parser for the chunk command."""
    parser = argparse.ArgumentParser(
        prog="chunk",
        description=(
            "Re-chunk cached Markdown files without re-parsing.\n\n"
            "By default, all files in cache/markdown/ are chunked.\n"
            "Use --doc-id to target a specific document."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    parser.add_argument(
        "--doc-id",
        metavar="ID",
        help="SHA-256 doc_id of a specific document to re-chunk.",
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
    """Parse arguments and run the chunking pipeline.

    Returns 0 on success, 1 if no files were found, 2 on errors.
    """
    parser = _build_parser()
    args = parser.parse_args(argv)

    _configure_logging(verbose=args.verbose, quiet=args.quiet)

    from ..chunking.pipeline import chunk_all, chunk_document_file

    if args.doc_id:
        # Single document mode
        try:
            result = chunk_document_file(args.doc_id)
            print(
                f"Chunked {args.doc_id[:12]}... -> "
                f"{len(result.chunks)} chunks"
            )
            return 0
        except FileNotFoundError as e:
            logging.error(str(e))
            return 1
        except Exception:
            logging.exception("Chunking failed for %s", args.doc_id[:12])
            return 2
    else:
        # All documents mode
        results = chunk_all()
        if not results:
            print("No Markdown files found to chunk.")
            return 1

        total_chunks = sum(len(r.chunks) for r in results)
        print(
            f"Chunked {len(results)} document(s), "
            f"{total_chunks} total chunks."
        )
        return 0


if __name__ == "__main__":
    sys.exit(main())
