"""Azure OpenAI embedding API wrapper with batching and retry.

Calls the Azure OpenAI ``/embeddings`` endpoint via the ``openai`` SDK.
Handles batching (max 20 texts per request) and exponential-backoff
retry for transient errors (429 / 5xx).
"""

from __future__ import annotations

import logging
import time
from typing import Sequence

from openai import AzureOpenAI, APIError, RateLimitError, APITimeoutError, InternalServerError

from ..config.env_config import load_azure_embedding_config

logger = logging.getLogger(__name__)

# ── Tunables ────────────────────────────────────────────────────
_BATCH_SIZE: int = 20          # max texts per API call
_MAX_RETRIES: int = 3          # retry attempts on transient errors
_INITIAL_BACKOFF: float = 2.0  # seconds; doubles each retry
_EXPECTED_DIMS: int = 1536     # Ada-002 output dimensionality


# ── Public API ──────────────────────────────────────────────────


def embed_texts(
    texts: Sequence[str],
    *,
    config: dict[str, str] | None = None,
) -> list[list[float]]:
    """Embed a sequence of texts using Azure OpenAI Ada-002.

    Parameters
    ----------
    texts:
        Strings to embed.  May be any length; the function handles
        batching internally.
    config:
        Azure credential dict with keys ``endpoint``, ``api_key``,
        ``model``, ``api_version``.  Loaded from the project config
        file when not supplied.

    Returns
    -------
    list[list[float]]
        One 1536-dim vector per input text, in the same order.

    Raises
    ------
    RuntimeError
        After exhausting retries or on an unexpected API error.
    ValueError
        If the API returns vectors with an unexpected dimensionality.
    """
    if not texts:
        return []

    cfg = config or load_azure_embedding_config()

    client = AzureOpenAI(
        azure_endpoint=cfg["endpoint"],
        api_key=cfg["api_key"],
        api_version=cfg["api_version"],
    )

    all_vectors: list[list[float]] = []
    total = len(texts)

    for start in range(0, total, _BATCH_SIZE):
        batch = texts[start : start + _BATCH_SIZE]
        batch_num = start // _BATCH_SIZE + 1
        logger.debug(
            "Embedding batch %d  (%d–%d of %d)",
            batch_num, start + 1, min(start + _BATCH_SIZE, total), total,
        )

        vectors = _embed_batch(client, cfg["model"], batch)
        all_vectors.extend(vectors)

    logger.info("Embedded %d text(s) → %d vectors.", total, len(all_vectors))
    return all_vectors


# ── Internals ───────────────────────────────────────────────────


def _embed_batch(
    client: AzureOpenAI,
    model: str,
    texts: Sequence[str],
) -> list[list[float]]:
    """Call the embeddings endpoint for a single batch with retry."""
    backoff = _INITIAL_BACKOFF

    for attempt in range(1, _MAX_RETRIES + 1):
        try:
            response = client.embeddings.create(
                input=list(texts),
                model=model,
            )

            vectors = [item.embedding for item in response.data]

            # Validate dimensionality on first vector.
            if vectors and len(vectors[0]) != _EXPECTED_DIMS:
                raise ValueError(
                    f"Expected {_EXPECTED_DIMS}-dim vectors, "
                    f"got {len(vectors[0])}-dim from model '{model}'."
                )

            return vectors

        except (RateLimitError, APITimeoutError, InternalServerError) as exc:
            if attempt == _MAX_RETRIES:
                raise RuntimeError(
                    f"Embedding failed after {_MAX_RETRIES} retries: {exc}"
                ) from exc
            logger.warning(
                "Transient error (attempt %d/%d): %s — retrying in %.1fs",
                attempt, _MAX_RETRIES, exc, backoff,
            )
            time.sleep(backoff)
            backoff *= 2

        except APIError as exc:
            raise RuntimeError(
                f"Embedding API error: {exc}"
            ) from exc

    # Should not be reached, but satisfies the type checker.
    raise RuntimeError("Embedding failed: retries exhausted.")
