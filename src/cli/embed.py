"""CLI entrypoint for embedding chunked documents.

Embeds tier-3 (atomic) chunks into a persistent ChromaDB vector store
using Azure OpenAI Ada-002.

Usage examples::

    # Embed all chunked documents
    python -m src.cli.embed

    # Embed a specific document by its doc_id
    python -m src.cli.embed --doc-id f4b48797a6a2...

    # Force re-embedding (delete + re-insert)
    python -m src.cli.embed --force

    # Embed with verbose logging
    python -m src.cli.embed --verbose
"""

from __future__ import annotations

import argparse
import logging
import sys


def _build_parser() -> argparse.ArgumentParser:
    """Build the argument parser for the embed command."""
    parser = argparse.ArgumentParser(
        prog="embed",
        description=(
            "Embed chunked documents into ChromaDB.\n\n"
            "By default, all chunk files in cache/chunk/ are embedded.\n"
            "Use --doc-id to target a specific document.\n"
            "Use --force to re-embed everything from scratch."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    parser.add_argument(
        "--doc-id",
        metavar="ID",
        help="SHA-256 doc_id of a specific document to embed.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help=(
            "Delete existing vectors for the target document(s) "
            "and re-embed from scratch."
        ),
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
    """Parse arguments and run the embedding pipeline.

    Returns 0 on success, 1 if no files were found, 2 on errors.
    """
    parser = _build_parser()
    args = parser.parse_args(argv)

    _configure_logging(verbose=args.verbose, quiet=args.quiet)

    from ..embedding.pipeline import embed_all, embed_document

    if args.doc_id:
        # Single document mode
        try:
            result = embed_document(args.doc_id, force=args.force)
            print(
                f"Embedded {args.doc_id[:12]}... → "
                f"{result.embedded} embedded, "
                f"{result.skipped} skipped, "
                f"{result.failed} failed  "
                f"[{result.total_chunks} tier-3 chunks]"
            )
            if result.errors:
                for err in result.errors:
                    logging.error(err)
                return 2
            return 0
        except FileNotFoundError as e:
            logging.error(str(e))
            return 1
        except Exception:
            logging.exception("Embedding failed for %s", args.doc_id[:12])
            return 2
    else:
        # All documents mode
        results = embed_all(force=args.force)
        if not results:
            print("No chunk files found to embed.")
            return 1

        total_embedded = sum(r.embedded for r in results)
        total_skipped = sum(r.skipped for r in results)
        total_failed = sum(r.failed for r in results)
        total_chunks = sum(r.total_chunks for r in results)

        print(
            f"Embedded {len(results)} document(s): "
            f"{total_embedded} embedded, "
            f"{total_skipped} skipped, "
            f"{total_failed} failed  "
            f"[{total_chunks} tier-3 chunks]"
        )

        if total_failed:
            for r in results:
                for err in r.errors:
                    logging.error(err)
            return 2
        return 0


if __name__ == "__main__":
    sys.exit(main())
