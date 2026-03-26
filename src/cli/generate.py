"""CLI entrypoint for grounded generation.

Run a query through the full retrieval → generation pipeline and
display the grounded answer with citations, confidence, and
verification diagnostics.

Usage examples::

    # Basic question — human-readable output
    python -m src.cli generate "What are the DIM-V thermal test results?"

    # JSON output for agents / scripting
    python -m src.cli generate "What is FVTR_OPT_01?" --json

    # Override model and temperature
    python -m src.cli generate "PAM safety requirements" --model gpt-5-nano --temperature 0.0

    # Adjust retrieval settings
    python -m src.cli generate "HW-IRS_DIM_VI_275 verification" --top-k 5 --reranker none

    # Verbose output (shows retrieval diagnostics)
    python -m src.cli generate "DIM-V optical tests" -v
"""

from __future__ import annotations

import argparse
import json
import logging
import sys


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="generate",
        description=(
            "Run a query through the retrieval → generation pipeline.\n\n"
            "Displays a grounded answer with source citations, confidence\n"
            "assessment, and structural verification.  Use --json for\n"
            "machine-readable output."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    parser.add_argument(
        "query",
        help="The question to answer (natural language or identifier).",
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
        default=0.1,
        metavar="T",
        help="Sampling temperature (default: 0.1).",
    )
    gen.add_argument(
        "--max-tokens",
        type=int,
        default=2048,
        metavar="N",
        dest="max_output_tokens",
        help="Maximum generation tokens (default: 2048).",
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
        help="Output structured JSON instead of formatted text.",
    )
    output.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Show detailed diagnostics (DEBUG logging).",
    )
    output.add_argument(
        "--quiet", "-q",
        action="store_true",
        help="Suppress all output except errors and the answer.",
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


def main(argv: list[str] | None = None) -> int:
    """Parse arguments and run the generation pipeline.

    Returns 0 on success, 1 on generation error, 2 on retrieval error.
    """
    parser = _build_parser()
    args = parser.parse_args(argv)

    _configure_logging(verbose=args.verbose, quiet=args.quiet)

    from ..generation.models import GenerationConfig
    from ..generation.pipeline import generate
    from ..retrieval.models import RetrievalConfig

    retrieval_config = RetrievalConfig(
        top_k_broad=args.broad,
        top_k_final=args.top_k,
        reranker_type=args.reranker,
        expand_abbreviations=not args.no_abbreviations,
    )

    generation_config = GenerationConfig(
        model=args.model,
        temperature=args.temperature,
        max_output_tokens=args.max_output_tokens,
    )

    try:
        result = generate(args.query, retrieval_config, generation_config)
    except Exception:
        logging.exception("Pipeline failed")
        return 2

    if result.error:
        if "Retrieval failed" in result.error:
            if not args.quiet:
                print(f"Error: {result.error}", file=sys.stderr)
            return 2
        if not args.quiet:
            print(f"Error: {result.error}", file=sys.stderr)
        return 1

    if args.json_output:
        print(json.dumps(result.to_dict(), indent=2, ensure_ascii=False))
    else:
        print(result.to_text())

    return 0


if __name__ == "__main__":
    sys.exit(main())
