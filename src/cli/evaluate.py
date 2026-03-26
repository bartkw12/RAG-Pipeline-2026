"""CLI entrypoint for retrieval evaluation.

Run the retrieval pipeline against the golden test set and display
stratified IR metrics (Precision@k, Recall@k, MRR, nDCG, Hit Rate).

Usage examples::

    # Default evaluation — stratified table
    python -m src.cli.evaluate

    # Custom golden set path
    python -m src.cli.evaluate --golden tests/golden_retrieval.json

    # Per-query breakdown
    python -m src.cli.evaluate --verbose

    # Compare rerankers
    python -m src.cli.evaluate --reranker none
    python -m src.cli.evaluate --reranker cross-encoder
    python -m src.cli.evaluate --reranker llm

    # JSON output for scripting
    python -m src.cli.evaluate --json

    # Only evaluate a single query type
    python -m src.cli.evaluate --type exact_lookup
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="evaluate",
        description=(
            "Evaluate the retrieval pipeline against a golden test set.\n\n"
            "Computes Precision@k, Recall@k, MRR, nDCG, and Hit Rate\n"
            "stratified by query type, with an overall aggregate."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    parser.add_argument(
        "--golden",
        default="tests/golden_retrieval.json",
        metavar="PATH",
        help="Path to the golden test set JSON (default: tests/golden_retrieval.json).",
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
        "--max-tokens",
        type=int,
        default=4000,
        metavar="N",
        help="Token budget for context window (default: 4000).",
    )
    retrieval.add_argument(
        "--no-expand",
        action="store_true",
        help="Disable abbreviation expansion.",
    )

    # ── Filter options ──────────────────────────────────────────
    filt = parser.add_argument_group("filter options")
    filt.add_argument(
        "--type",
        dest="query_type",
        metavar="TYPE",
        help=(
            "Only evaluate queries of this type "
            "(e.g. exact_lookup, scoped_semantic)."
        ),
    )

    # ── Output format ───────────────────────────────────────────
    output = parser.add_argument_group("output format")
    output.add_argument(
        "--json",
        action="store_true",
        dest="json_output",
        help="Output structured JSON instead of formatted tables.",
    )
    output.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Show per-query breakdown before the stratified table.",
    )
    output.add_argument(
        "--quiet", "-q",
        action="store_true",
        help="Suppress pipeline log output.",
    )

    return parser


def main(argv: list[str] | None = None) -> int:
    """CLI entrypoint — parse args, run evaluation, display results."""
    parser = _build_parser()
    args = parser.parse_args(argv)

    # Logging
    if args.quiet:
        logging.basicConfig(level=logging.ERROR, format="%(message)s")
    elif args.verbose:
        logging.basicConfig(level=logging.DEBUG, format="%(message)s")
    else:
        logging.basicConfig(level=logging.WARNING, format="%(message)s")

    # Validate golden path
    golden_path = Path(args.golden)
    if not golden_path.exists():
        print(f"Error: golden test set not found: {golden_path}", file=sys.stderr)
        return 1

    # ── Optional type filter ────────────────────────────────────
    # If --type is given we filter queries before evaluation.
    filtered_data: dict | None = None
    if args.query_type:
        raw = json.loads(golden_path.read_text(encoding="utf-8"))
        original_count = len(raw["queries"])
        raw["queries"] = [
            q for q in raw["queries"]
            if q.get("query_type") == args.query_type
        ]
        if not raw["queries"]:
            print(
                f"Error: no queries of type '{args.query_type}' "
                f"in {golden_path} ({original_count} total queries)",
                file=sys.stderr,
            )
            return 1
        filtered_data = raw

    # ── Build config ────────────────────────────────────────────
    from ..retrieval.models import RetrievalConfig

    config = RetrievalConfig(
        top_k_broad=args.broad,
        top_k_final=args.top_k,
        reranker_type=args.reranker,
        max_context_tokens=args.max_tokens,
        expand_abbreviations=not args.no_expand,
    )

    # ── Run evaluation ──────────────────────────────────────────
    from ..retrieval.evaluate import evaluate

    if filtered_data is not None:
        # Write filtered golden to a temp file so evaluate() reads it
        import tempfile
        import os
        fd, tmp_path = tempfile.mkstemp(suffix=".json")
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as f:
                json.dump(filtered_data, f)
            report = evaluate(tmp_path, config)
        finally:
            os.unlink(tmp_path)
    else:
        report = evaluate(golden_path, config)

    # ── Display ─────────────────────────────────────────────────
    if args.json_output:
        print(json.dumps(report.to_dict(), indent=2))
    else:
        print()
        report.print_table(per_query=args.verbose)
        print()

    return 0


if __name__ == "__main__":
    sys.exit(main())
