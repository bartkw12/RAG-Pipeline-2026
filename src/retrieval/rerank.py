"""Pluggable re-ranking — narrow hybrid search candidates to the top-k.

Three implementations:

* **CrossEncoderReranker** — local cross-encoder model
  (``cross-encoder/ms-marco-MiniLM-L-6-v2``) via ``sentence-transformers``.
  Fast on CPU (~80–150 ms for 20 candidates).
* **LLMReranker** — Azure GPT-4.1-mini scores each candidate's
  relevance in a single prompt and returns a JSON array.
* **NoReranker** — passthrough baseline; returns chunks unchanged.

Usage::

    from src.retrieval.rerank import get_reranker

    reranker = get_reranker("cross-encoder")
    reranked = reranker.rerank(query, chunks, top_k=8)
"""

from __future__ import annotations

import json
import logging
import re
from abc import ABC, abstractmethod

from openai import AzureOpenAI

from ..config.env_config import load_azure_vlm_config
from .models import ScoredChunk

logger = logging.getLogger(__name__)


# ── Abstract base ───────────────────────────────────────────────


class Reranker(ABC):
    """Base class for all re-ranker implementations."""

    @abstractmethod
    def rerank(
        self,
        query: str,
        chunks: list[ScoredChunk],
        top_k: int = 8,
    ) -> list[ScoredChunk]:
        """Score and re-order *chunks* by relevance to *query*.

        Parameters
        ----------
        query:
            The user's (optionally expanded) query string.
        chunks:
            Candidate chunks from hybrid search — typically ~20.
        top_k:
            Maximum number of chunks to return after re-ranking.

        Returns
        -------
        list[ScoredChunk]
            The top *top_k* chunks sorted by ``rerank_score``
            descending.  Each chunk's ``rerank_score`` and ``score``
            are updated in-place.
        """


# ── Cross-encoder re-ranker ────────────────────────────────────


class CrossEncoderReranker(Reranker):
    """Re-rank using a local cross-encoder model.

    Model: ``cross-encoder/ms-marco-MiniLM-L-6-v2`` (22 M params).
    Loaded lazily on first call and cached for the process lifetime.
    CPU-only; no GPU required.
    """

    _MODEL_NAME = "cross-encoder/ms-marco-MiniLM-L-6-v2"

    def __init__(self) -> None:
        self._model: object | None = None  # lazy-loaded CrossEncoder

    def _load_model(self) -> object:
        if self._model is None:
            try:
                from sentence_transformers import CrossEncoder
            except ImportError as e:
                raise ImportError(
                    "sentence-transformers is required for CrossEncoderReranker. "
                    "Install it with: pip install sentence-transformers>=2.2.0"
                ) from e

            logger.info("Loading cross-encoder model: %s", self._MODEL_NAME)
            self._model = CrossEncoder(self._MODEL_NAME)
            logger.info("Cross-encoder model loaded.")
        return self._model

    def rerank(
        self,
        query: str,
        chunks: list[ScoredChunk],
        top_k: int = 8,
    ) -> list[ScoredChunk]:
        if not chunks:
            return []

        model = self._load_model()

        # Build (query, document) pairs for scoring.
        pairs = [[query, chunk.text] for chunk in chunks]

        # Score all pairs in a single batch.
        from sentence_transformers import CrossEncoder

        assert isinstance(model, CrossEncoder)
        raw_scores: list[float] = model.predict(pairs).tolist()  # type: ignore[union-attr]

        # Normalise scores to [0, 1] via min-max.
        min_s = min(raw_scores) if raw_scores else 0.0
        max_s = max(raw_scores) if raw_scores else 1.0
        span = max_s - min_s if max_s != min_s else 1.0

        for chunk, raw in zip(chunks, raw_scores):
            normalised = (raw - min_s) / span
            chunk.rerank_score = normalised
            chunk.score = normalised  # rerank score becomes the primary score

        # Sort descending and trim.
        chunks.sort(key=lambda c: c.rerank_score, reverse=True)
        result = chunks[:top_k]

        logger.info(
            "CrossEncoder re-ranked %d → %d chunks. "
            "Top score: %.3f, bottom score: %.3f",
            len(pairs), len(result),
            result[0].rerank_score if result else 0.0,
            result[-1].rerank_score if result else 0.0,
        )
        return result


# ── LLM re-ranker ──────────────────────────────────────────────


class LLMReranker(Reranker):
    """Re-rank using Azure GPT-4.1-mini for relevance scoring.

    Sends all candidates in a single prompt and asks the LLM to
    return a JSON array of ``{"index": i, "score": 0-10}`` objects.
    """

    def __init__(
        self,
        config: dict[str, str] | None = None,
        model: str | None = None,
    ) -> None:
        cfg = config or load_azure_vlm_config(section="OpenAI")
        self._client = AzureOpenAI(
            azure_endpoint=cfg["endpoint"],
            api_key=cfg["api_key"],
            api_version=cfg["api_version"],
        )
        self._model = model or "gpt-4.1-mini"

    def rerank(
        self,
        query: str,
        chunks: list[ScoredChunk],
        top_k: int = 8,
    ) -> list[ScoredChunk]:
        if not chunks:
            return []

        # Build the candidate list for the prompt.
        candidate_lines: list[str] = []
        for i, chunk in enumerate(chunks):
            # Truncate very long chunks to avoid prompt bloat.
            text_preview = chunk.text[:800] if len(chunk.text) > 800 else chunk.text
            candidate_lines.append(f"[{i}] {text_preview}")

        candidates_block = "\n\n".join(candidate_lines)

        prompt = (
            "You are a document relevance judge for engineering verification reports.\n\n"
            f"**Query:** {query}\n\n"
            f"**Candidates:**\n{candidates_block}\n\n"
            "Rate each candidate's relevance to the query on a scale of 0 to 10 "
            "(10 = highly relevant, 0 = irrelevant).\n\n"
            "Return ONLY a JSON array of objects with keys \"index\" (int) and "
            "\"score\" (number 0-10). Example:\n"
            '[{"index": 0, "score": 8}, {"index": 1, "score": 3}]\n\n'
            "Return scores for ALL candidates."
        )

        try:
            response = self._client.chat.completions.create(
                model=self._model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.0,
                max_tokens=1000,
            )
            content = response.choices[0].message.content or ""
        except Exception as e:
            logger.warning("LLM reranker call failed: %s — falling back to RRF order.", e)
            return chunks[:top_k]

        # Parse the JSON response.
        scores = self._parse_scores(content, len(chunks))

        for i, chunk in enumerate(chunks):
            s = scores.get(i, 0.0)
            chunk.rerank_score = s / 10.0  # normalise to [0, 1]
            chunk.score = chunk.rerank_score

        chunks.sort(key=lambda c: c.rerank_score, reverse=True)
        result = chunks[:top_k]

        logger.info(
            "LLM re-ranked %d → %d chunks. "
            "Top score: %.1f/10, bottom score: %.1f/10",
            len(chunks), len(result),
            result[0].rerank_score * 10 if result else 0.0,
            result[-1].rerank_score * 10 if result else 0.0,
        )
        return result

    @staticmethod
    def _parse_scores(content: str, n_chunks: int) -> dict[int, float]:
        """Extract index→score mapping from LLM JSON response."""
        # Try to find a JSON array in the response.
        match = re.search(r"\[.*\]", content, re.DOTALL)
        if not match:
            logger.warning("LLM reranker returned no JSON array: %s", content[:200])
            return {}

        try:
            items = json.loads(match.group())
        except json.JSONDecodeError:
            logger.warning("LLM reranker returned invalid JSON: %s", content[:200])
            return {}

        scores: dict[int, float] = {}
        for item in items:
            if not isinstance(item, dict):
                continue
            idx = item.get("index")
            score = item.get("score")
            if isinstance(idx, int) and 0 <= idx < n_chunks and isinstance(score, (int, float)):
                scores[idx] = float(max(0.0, min(10.0, score)))

        return scores


# ── No-op re-ranker ─────────────────────────────────────────────


class NoReranker(Reranker):
    """Passthrough — returns candidates in their original RRF order."""

    def rerank(
        self,
        query: str,
        chunks: list[ScoredChunk],
        top_k: int = 8,
    ) -> list[ScoredChunk]:
        result = chunks[:top_k]
        logger.info("NoReranker: passed through %d of %d chunks.", len(result), len(chunks))
        return result


# ── Factory ─────────────────────────────────────────────────────

# Singleton cache for expensive-to-construct rerankers.
_reranker_cache: dict[str, Reranker] = {}


def get_reranker(reranker_type: str = "cross-encoder") -> Reranker:
    """Return a ``Reranker`` instance for the given type.

    Parameters
    ----------
    reranker_type:
        One of ``"cross-encoder"``, ``"llm"``, or ``"none"``.

    Returns
    -------
    Reranker

    Raises
    ------
    ValueError
        If *reranker_type* is not recognised.
    """
    key = reranker_type.lower().strip()

    if key in _reranker_cache:
        return _reranker_cache[key]

    if key == "cross-encoder":
        reranker: Reranker = CrossEncoderReranker()
    elif key == "llm":
        reranker = LLMReranker()
    elif key == "none":
        reranker = NoReranker()
    else:
        raise ValueError(
            f"Unknown reranker_type={reranker_type!r}. "
            f"Choose from: 'cross-encoder', 'llm', 'none'."
        )

    _reranker_cache[key] = reranker
    return reranker
