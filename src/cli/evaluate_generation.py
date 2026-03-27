"""CLI entrypoint for generation evaluation.

Run the full retrieval → generation pipeline against a golden test set
and display stratified generation-quality metrics (Fact Recall,
Abstention Accuracy, Citation Coverage, Verification Rate, etc.).

Usage examples::

    # Default evaluation — stratified table
    python -m src.cli evaluate-generation

    # Custom golden set path
    python -m src.cli evaluate-generation --golden tests/golden_generation.json

    # Per-query breakdown
    python -m src.cli evaluate-generation --verbose

    # JSON output for scripting / saving
    python -m src.cli evaluate-generation --json

    # Only evaluate a single query type
    python -m src.cli evaluate-generation --type abstention

    # Override model
    python -m src.cli evaluate-generation --model gpt-5-nano

    # Save JSON results to file
    python -m src.cli evaluate-generation --json --output tests/results/evaluation/run.json
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="evaluate-generation",
        description=(
            "Evaluate the generation pipeline against a golden test set.\n\n"
            "Computes Fact Recall, Abstention Accuracy, Citation Coverage,\n"
            "Verification Rate, Confidence Alignment, and Doc-ID\n"
            "Precision/Recall — stratified by query type."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    parser.add_argument(
        "--golden",
        default="tests/golden_generation.json",
        metavar="PATH",
        help=(
            "Path to the golden test set JSON "
            "(default: tests/golden_generation.json)."
        ),
    )

    # ── Generation options ──────────────────────────────────────
    gen = parser.add_argument_group("generation options")
    gen.add_argument(
        "--model",
        default="gpt-5-mini",
        help="Azure OpenAI deployment name (default: gpt-5-mini).",
    )
    gen.add_argument(
        "--temperature",
        type=float,
        default=1.0,
        metavar="T",
        help="Sampling temperature (default: 1.0).",
    )
    gen.add_argument(
        "--max-tokens",
        type=int,
        default=2048,
        metavar="N",
        dest="max_output_tokens",
        help="Max output tokens (default: 2048).",
    )
    gen.add_argument(
        "--reasoning-effort",
        default=None,
        metavar="LEVEL",
        help="Reasoning effort for GPT-5 family (low/medium/high).",
    )

    # ── Retrieval options ───────────────────────────────────────
    ret = parser.add_argument_group("retrieval options")
    ret.add_argument(
        "--top-k",
        type=int,
        default=8,
        metavar="N",
        help="Number of chunks after re-ranking (default: 8).",
    )
    ret.add_argument(
        "--broad",
        type=int,
        default=20,
        metavar="N",
        help="Number of hybrid search candidates (default: 20).",
    )
    ret.add_argument(
        "--reranker",
        choices=["cross-encoder", "llm", "none"],
        default="cross-encoder",
        help="Re-ranker to use (default: cross-encoder).",
    )
    ret.add_argument(
        "--max-context-tokens",
        type=int,
        default=4000,
        metavar="N",
        help="Token budget for context window (default: 4000).",
    )
    ret.add_argument(
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
            "(e.g. exact_lookup, abstention, adversarial_edge)."
        ),
    )

    # ── Evaluation options ──────────────────────────────────────
    evl = parser.add_argument_group("evaluation options")
    evl.add_argument(
        "--semantic",
        action="store_true",
        dest="semantic_facts",
        help=(
            "Use an LLM-as-judge (GPT-5-nano) for semantic fact "
            "matching instead of strict substring matching."
        ),
    )

    # ── Output options ──────────────────────────────────────────
    output = parser.add_argument_group("output options")
    output.add_argument(
        "--json",
        action="store_true",
        dest="json_output",
        help="Output structured JSON instead of formatted tables.",
    )
    output.add_argument(
        "--output", "-o",
        metavar="PATH",
        dest="output_path",
        help="Save JSON results to a file (implies --json).",
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
        print(
            f"Error: golden test set not found: {golden_path}",
            file=sys.stderr,
        )
        return 1

    # ── Build configs ───────────────────────────────────────────
    from ..retrieval.models import RetrievalConfig
    from ..generation.models import GenerationConfig

    ret_config = RetrievalConfig(
        top_k_broad=args.broad,
        top_k_final=args.top_k,
        reranker_type=args.reranker,
        max_context_tokens=args.max_context_tokens,
        expand_abbreviations=not args.no_expand,
    )

    gen_config = GenerationConfig(
        model=args.model,
        temperature=args.temperature,
        max_output_tokens=args.max_output_tokens,
        reasoning_effort=args.reasoning_effort,
    )

    # ── Run evaluation ──────────────────────────────────────────
    from ..generation.evaluate import evaluate_generation

    report = evaluate_generation(
        golden_path,
        retrieval_config=ret_config,
        generation_config=gen_config,
        query_type_filter=args.query_type,
        semantic_facts=args.semantic_facts,
    )

    # ── Display / save ──────────────────────────────────────────
    if args.output_path:
        # Save to file (always JSON)
        out_path = Path(args.output_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(
            json.dumps(report.to_dict(), indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        print(f"Results saved to {out_path}")
        if not args.json_output:
            # Also print the table to stdout
            print()
            report.print_table(per_query=args.verbose)
            print()
    elif args.json_output:
        print(json.dumps(report.to_dict(), indent=2, ensure_ascii=False))
    else:
        print()
        report.print_table(per_query=args.verbose)
        print()

    return 0


if __name__ == "__main__":
    sys.exit(main())
