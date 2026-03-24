"""CLI entrypoint for interactive retrieval testing.

Run a query through the full retrieval pipeline and display the
assembled context, scores, timings, and diagnostics.

Usage examples::

    # Basic query — shows context window
    python -m src.cli.retrieve "DIM-V thermal test results"

    # Exact lookup
    python -m src.cli.retrieve "FVTR_MECH_01"

    # Use LLM reranker with verbose output
    python -m src.cli.retrieve "HW-IRS_DIM_VI_275 verification" --reranker llm -v

    # Raw ranked chunks (skip context assembly display)
    python -m src.cli.retrieve "PAM optical tests" --no-context

    # JSON output for scripting
    python -m src.cli.retrieve "What is thermal protection?" --json

    # Adjust candidate counts
    python -m src.cli.retrieve "DIM-V safety" --top-k 5 --broad 15
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="retrieve",
        description=(
            "Run a query through the retrieval pipeline and display results.\n\n"
            "Shows the assembled context window by default.  Use --json for\n"
            "structured output, or --no-context for raw ranked chunks."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    parser.add_argument(
        "query",
        help="The query string (natural language or identifier).",
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

    # ── Output format ───────────────────────────────────────────
    output = parser.add_argument_group("output format")
    output.add_argument(
        "--json",
        action="store_true",
        dest="json_output",
        help="Output structured JSON instead of formatted text.",
    )
    output.add_argument(
        "--no-context",
        action="store_true",
        help="Show raw ranked chunks instead of assembled context.",
    )
    output.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Show detailed diagnostics (scores, timings, scope).",
    )
    output.add_argument(
        "--quiet", "-q",
        action="store_true",
        help="Suppress all output except errors.",
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


def _print_text_output(result: object, *, verbose: bool, no_context: bool) -> None:
    """Pretty-print retrieval results to stdout."""
    from ..retrieval.models import RetrievalResult
    assert isinstance(result, RetrievalResult)

    analysis = result.query_analysis

    # ── Header ──────────────────────────────────────────────────
    print(f"Query:     {analysis.original_query}")
    print(f"Strategy:  {result.strategy}")

    if analysis.expanded_query and analysis.expanded_query != analysis.original_query:
        print(f"Expanded:  {analysis.expanded_query}")
    if analysis.scope_filters:
        print(f"Scope:     {analysis.scope_filters}")
    if analysis.has_identifiers:
        ids = []
        if analysis.test_case_ids:
            ids.extend(analysis.test_case_ids)
        if analysis.requirement_ids:
            ids.extend(analysis.requirement_ids)
        if analysis.component_ids:
            ids.extend(analysis.component_ids)
        if analysis.cross_references:
            ids.extend(analysis.cross_references)
        print(f"IDs:       {', '.join(ids)}")

    if verbose:
        print(f"Remainder: {analysis.semantic_remainder!r}")
        timings_str = "  ".join(f"{k}={v:.3f}s" for k, v in result.timings.items())
        print(f"Timings:   {timings_str}")

    print()

    # ── Raw chunks mode ─────────────────────────────────────────
    if no_context:
        print(f"Ranked chunks ({len(result.scored_chunks)}):")
        print("-" * 70)
        for i, sc in enumerate(result.scored_chunks, 1):
            print(f"\n[{i}]  {sc.chunk_id}")
            print(f"     type={sc.chunk_type}  doc_id={sc.doc_id[:12]}…")
            print(f"     score={sc.score:.4f}  vec={sc.vector_score:.3f}  "
                  f"bm25={sc.bm25_score:.3f}  rerank={sc.rerank_score:.3f}")
            if verbose:
                meta_keys = ["module_name", "doc_type", "heading", "test_case_id"]
                meta_parts = [f"{k}={sc.metadata.get(k)}" for k in meta_keys if sc.metadata.get(k)]
                if meta_parts:
                    print(f"     {', '.join(meta_parts)}")
            # Show first 200 chars of text.
            preview = sc.text[:200].replace("\n", " ")
            if len(sc.text) > 200:
                preview += "…"
            print(f"     {preview}")
        return

    # ── Context window mode (default) ───────────────────────────
    ctx = result.context
    print(f"Context: {len(ctx.sections)} section(s), {ctx.total_tokens} tokens, "
          f"{len(ctx.doc_ids)} doc(s)")
    print("=" * 70)

    for sec in ctx.sections:
        print(f"\n### {sec.section_heading}  [{sec.doc_type or '?'}]")
        if verbose:
            print(f"    section_number={sec.section_number}  "
                  f"hint={sec.content_type_hint}  tokens={sec.token_count}")

        if sec.preamble:
            print(f"\n{sec.preamble}")

        for chunk in sec.child_chunks:
            print()
            if verbose:
                print(f"  [{chunk.chunk_id[:16]}…  score={chunk.score:.4f}  "
                      f"type={chunk.chunk_type}]")
            print(chunk.text)

    print("\n" + "=" * 70)
    if verbose:
        print(f"Chunk IDs: {', '.join(cid[:12] + '…' for cid in ctx.chunk_ids)}")
        print(f"Doc IDs:   {', '.join(did[:12] + '…' for did in ctx.doc_ids)}")


def _print_json_output(result: object) -> None:
    """Output retrieval results as structured JSON."""
    from ..retrieval.models import RetrievalResult
    assert isinstance(result, RetrievalResult)

    analysis = result.query_analysis
    output = {
        "query": analysis.original_query,
        "strategy": result.strategy,
        "expanded_query": analysis.expanded_query,
        "scope_filters": analysis.scope_filters,
        "identifiers": {
            "test_case_ids": analysis.test_case_ids,
            "requirement_ids": analysis.requirement_ids,
            "component_ids": analysis.component_ids,
            "cross_references": analysis.cross_references,
        },
        "timings": result.timings,
        "context": {
            "total_tokens": result.context.total_tokens,
            "chunk_ids": result.context.chunk_ids,
            "doc_ids": result.context.doc_ids,
            "sections": [
                {
                    "heading": sec.section_heading,
                    "section_number": sec.section_number,
                    "doc_type": sec.doc_type,
                    "content_type_hint": sec.content_type_hint,
                    "token_count": sec.token_count,
                    "preamble": sec.preamble,
                    "chunks": [
                        {
                            "chunk_id": ch.chunk_id,
                            "chunk_type": ch.chunk_type,
                            "score": ch.score,
                            "vector_score": ch.vector_score,
                            "bm25_score": ch.bm25_score,
                            "rerank_score": ch.rerank_score,
                            "text": ch.text,
                        }
                        for ch in sec.child_chunks
                    ],
                }
                for sec in result.context.sections
            ],
        },
        "scored_chunks": [
            {
                "chunk_id": sc.chunk_id,
                "doc_id": sc.doc_id,
                "chunk_type": sc.chunk_type,
                "score": sc.score,
                "vector_score": sc.vector_score,
                "bm25_score": sc.bm25_score,
                "rerank_score": sc.rerank_score,
            }
            for sc in result.scored_chunks
        ],
        "prompt_text": result.context.to_prompt_text(),
    }

    print(json.dumps(output, indent=2, ensure_ascii=False))


def main(argv: list[str] | None = None) -> int:
    """Parse arguments and run the retrieval pipeline.

    Returns 0 on success, 1 if no results, 2 on errors.
    """
    parser = _build_parser()
    args = parser.parse_args(argv)

    _configure_logging(verbose=args.verbose, quiet=args.quiet)

    from ..retrieval.models import RetrievalConfig
    from ..retrieval.pipeline import retrieve

    config = RetrievalConfig(
        top_k_broad=args.broad,
        top_k_final=args.top_k,
        reranker_type=args.reranker,
        max_context_tokens=args.max_tokens,
        expand_abbreviations=not args.no_expand,
    )

    try:
        t0 = time.perf_counter()
        result = retrieve(args.query, config)
        wall = time.perf_counter() - t0
    except Exception:
        logging.exception("Retrieval failed")
        return 2

    if not result.scored_chunks and not result.context.sections:
        if not args.quiet:
            print("No results found.", file=sys.stderr)
        return 1

    if args.json_output:
        _print_json_output(result)
    else:
        _print_text_output(result, verbose=args.verbose, no_context=args.no_context)
        if not args.quiet:
            print(f"\nCompleted in {wall:.2f}s", file=sys.stderr)

    return 0


if __name__ == "__main__":
    sys.exit(main())
