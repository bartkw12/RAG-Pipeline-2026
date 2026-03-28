"""CLI entrypoint for interactive RAG chat (REPL mode).

Start a persistent session that keeps the embedding store loaded
across queries.  Supports runtime output toggling via slash commands.

Usage examples::

    # Start interactive chat with defaults
    python -m src.cli chat

    # Start in JSON output mode
    python -m src.cli chat --json

    # Use a specific model and reranker
    python -m src.cli chat --model gpt-5-nano --reranker cross-encoder

    # Verbose diagnostics
    python -m src.cli chat -v
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="chat",
        description=(
            "Interactive RAG chat session.\n\n"
            "Keeps the embedding store loaded across queries so you\n"
            "can ask multiple questions without cold-start overhead.\n"
            "Type '/help' inside the session for runtime commands."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    # ── Generation options ──────────────────────────────────────
    gen = parser.add_argument_group("generation options")
    gen.add_argument(
        "--model",
        default="gpt-5-mini",
        metavar="NAME",
        help="Azure OpenAI deployment name (default: gpt-5-mini).",
    )
    gen.add_argument(
        "--temperature",
        type=float,
        default=1.0,
        metavar="T",
        help="Sampling temperature (default: 1.0; reasoning models only support 1).",
    )
    gen.add_argument(
        "--max-tokens",
        type=int,
        default=16_384,
        metavar="N",
        dest="max_output_tokens",
        help="Max completion tokens incl. reasoning (default: 16384).",
    )

    # ── Retrieval options ───────────────────────────────────────
    retrieval = parser.add_argument_group("retrieval options")
    retrieval.add_argument(
        "--top-k",
        type=int,
        default=8,
        metavar="N",
        help="Number of chunks after re-ranking (default: 8).",
    )
    retrieval.add_argument(
        "--broad",
        type=int,
        default=20,
        metavar="N",
        help="Number of hybrid search candidates (default: 20).",
    )
    retrieval.add_argument(
        "--reranker",
        choices=["cross-encoder", "llm", "none"],
        default="none",
        help="Re-ranker to use (default: none).",
    )
    retrieval.add_argument(
        "--no-abbreviations",
        action="store_true",
        help="Disable abbreviation expansion.",
    )

    # ── Output format ───────────────────────────────────────────
    output = parser.add_argument_group("output format")
    output.add_argument(
        "--json",
        action="store_true",
        dest="json_output",
        help="Start in JSON output mode (toggleable at runtime with /json).",
    )
    output.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Show detailed diagnostics (DEBUG logging).",
    )
    output.add_argument(
        "--quiet", "-q",
        action="store_true",
        help="Suppress all output except errors and answers.",
    )

    return parser


def _configure_logging(*, verbose: bool = False, quiet: bool = False) -> None:
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


# ── Banner ──────────────────────────────────────────────────────

_SLASH_HELP = (
    "/json   — toggle JSON output mode\n"
    "/text   — force human-readable output\n"
    "/help   — show this list\n"
    "/clear  — clear the terminal\n"
    "/stats  — show query count and cumulative time\n"
    "exit    — quit the session"
)


def _print_banner(model: str, vector_count: int) -> None:
    """Print a one-time startup banner with session info."""
    print(
        f"\nUC37 RAG Chat ({model})  —  {vector_count} vectors loaded\n"
        f"Type a question, or /help for commands.  'exit' to quit.\n"
        f"{'━' * 60}"
    )


# ── Placeholder — remaining functions added in subsequent steps ─


def main(argv: list[str] | None = None) -> int:
    """Parse arguments and enter the interactive REPL loop."""
    parser = _build_parser()
    args = parser.parse_args(argv)
    _configure_logging(verbose=args.verbose, quiet=args.quiet)

    # Eagerly load the embedding store so we can show vector count
    from ..embedding.store import EmbeddingStore
    store = EmbeddingStore()
    _print_banner(args.model, store.count())

    # TODO: _handle_slash_command, REPL loop
    print("REPL loop not yet implemented.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
