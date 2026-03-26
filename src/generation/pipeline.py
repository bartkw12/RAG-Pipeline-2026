"""Generation pipeline orchestrator — single entry point for query → grounded answer.

Wires together all generation components:

1. **Retrieval** — fetch relevant context via the retrieval pipeline.
2. **Empty-context guard** — abstain without LLM call if no evidence.
3. **Prompt construction** — build grounded prompt with source manifest.
4. **LLM call** — structured JSON output via Azure OpenAI.
5. **Citation resolution** — map source IDs back to real chunks.
6. **Verification** — deterministic structural-groundedness check.
7. **Confidence** — hybrid score from retrieval + verification signals.

Usage::

    from src.generation.pipeline import generate

    result = generate("What are the DIM-V thermal test results?")
    print(result.to_text())   # human-readable
    print(result.to_dict())   # JSON for agents
"""

from __future__ import annotations

import logging
import time

from ..retrieval.models import RetrievalConfig, RetrievalResult
from ..retrieval.pipeline import retrieve
from .client import generate_completion
from .confidence import compute_system_confidence
from .models import (
    Claim,
    ConfidenceLevel,
    GenerationConfig,
    GenerationResult,
)
from .prompt import build_prompt, resolve_citations
from .verification import verify_generation

logger = logging.getLogger(__name__)

# ── Canned abstention answer ────────────────────────────────────

_ABSTENTION_ANSWER = (
    "I don't have enough information in the provided documents to "
    "answer this question."
)


# ── Public API ──────────────────────────────────────────────────


def generate(
    query: str,
    retrieval_config: RetrievalConfig | None = None,
    generation_config: GenerationConfig | None = None,
) -> GenerationResult:
    """Run the full retrieval → generation pipeline.

    Parameters
    ----------
    query:
        The user's natural-language question.
    retrieval_config:
        Retrieval configuration override.  Uses defaults when ``None``.
    generation_config:
        Generation configuration override.  Uses defaults when ``None``.

    Returns
    -------
    GenerationResult
    """
    gen_cfg = generation_config or GenerationConfig()
    total_t0 = time.perf_counter()

    # ── 1. Retrieve ─────────────────────────────────────────────
    try:
        retrieval_result = retrieve(query, retrieval_config)
    except Exception as exc:
        logger.error("Retrieval failed: %s", exc)
        return GenerationResult(
            query=query,
            model=gen_cfg.model,
            error=f"Retrieval failed: {exc}",
            total_elapsed_s=time.perf_counter() - total_t0,
        )

    # ── 2. Generate from retrieved context ──────────────────────
    result = generate_from_context(query, retrieval_result, gen_cfg)
    result.total_elapsed_s = time.perf_counter() - total_t0
    return result


def generate_from_context(
    query: str,
    retrieval_result: RetrievalResult,
    config: GenerationConfig | None = None,
) -> GenerationResult:
    """Generate a grounded answer from pre-existing retrieval output.

    Skips the retrieval step — useful for testing, benchmarking, and
    agents that manage their own retrieval.

    Parameters
    ----------
    query:
        The user's natural-language question.
    retrieval_result:
        A ``RetrievalResult`` from the retrieval pipeline.
    config:
        Generation configuration override.  Uses defaults when ``None``.

    Returns
    -------
    GenerationResult
    """
    cfg = config or GenerationConfig()
    context = retrieval_result.context

    # ── Empty-context guard ─────────────────────────────────────
    if not context.sections:
        logger.info("No context sections — abstaining without LLM call.")
        return GenerationResult(
            answer=_ABSTENTION_ANSWER,
            abstained=True,
            model_confidence=ConfidenceLevel.LOW,
            system_confidence=ConfidenceLevel.LOW,
            confidence_components={
                "retrieval_support": 0.0,
                "citation_coverage": 1.0,
                "verification_pass": 1.0,
            },
            query=query,
            strategy=retrieval_result.strategy,
            retrieval_result=retrieval_result,
            model=cfg.model,
        )

    # ── 3. Build prompt ─────────────────────────────────────────
    messages, source_manifest = build_prompt(query, context)

    # ── 4. Call LLM ─────────────────────────────────────────────
    gen_t0 = time.perf_counter()
    try:
        raw_response, usage = generate_completion(messages, cfg)
    except RuntimeError as exc:
        logger.error("Generation failed: %s", exc)
        return GenerationResult(
            query=query,
            strategy=retrieval_result.strategy,
            retrieval_result=retrieval_result,
            model=cfg.model,
            error=f"Generation failed: {exc}",
        )
    gen_elapsed = time.perf_counter() - gen_t0

    # ── 5. Resolve citations ────────────────────────────────────
    citations = resolve_citations(raw_response, source_manifest)

    # ── 6. Verify ───────────────────────────────────────────────
    verification = verify_generation(raw_response, source_manifest, citations)

    # ── 7. Compute confidence ───────────────────────────────────
    model_conf_str = raw_response.get("confidence", "LOW")
    try:
        model_conf = ConfidenceLevel(model_conf_str)
    except ValueError:
        model_conf = ConfidenceLevel.LOW

    system_conf, conf_components = compute_system_confidence(
        model_conf,
        retrieval_result.scored_chunks,
        verification,
    )

    # ── 8. Assemble result ──────────────────────────────────────
    claims = [
        Claim(
            statement=c.get("statement", ""),
            source_ids=c.get("source_ids", []),
        )
        for c in raw_response.get("claims", [])
    ]

    return GenerationResult(
        answer=raw_response.get("answer", ""),
        claims=claims,
        citations=citations,
        abstained=raw_response.get("abstained", False),
        partial=raw_response.get("partial", False),
        unanswered_aspects=raw_response.get("unanswered_aspects", []),
        contradictions_noted=raw_response.get("contradictions_noted", False),
        model_confidence=model_conf,
        model_confidence_reasoning=raw_response.get("confidence_reasoning", ""),
        system_confidence=system_conf,
        confidence_components=conf_components,
        verification=verification,
        query=query,
        strategy=retrieval_result.strategy,
        retrieval_result=retrieval_result,
        model=cfg.model,
        usage=usage,
        elapsed_s=gen_elapsed,
    )
