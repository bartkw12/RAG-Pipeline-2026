"""Azure OpenAI chat-completion client with structured output and retry.

Calls the Azure OpenAI ``/chat/completions`` endpoint via the ``openai``
SDK with JSON-schema structured output.  Handles exponential-backoff
retry for transient errors (429 / 5xx).
"""

from __future__ import annotations

import json
import logging
import time
from typing import Any

from openai import (
    APIError,
    APITimeoutError,
    AzureOpenAI,
    InternalServerError,
    RateLimitError,
)

from ..config.env_config import load_azure_generation_config
from .models import GenerationConfig
from .prompt import get_response_schema

logger = logging.getLogger(__name__)

# ── Tunables ────────────────────────────────────────────────────
_MAX_RETRIES: int = 3
_INITIAL_BACKOFF: float = 2.0  # seconds; doubles each retry


# ── Public API ──────────────────────────────────────────────────


def generate_completion(
    messages: list[dict[str, str]],
    config: GenerationConfig | None = None,
) -> tuple[dict[str, Any], dict[str, int]]:
    """Send a chat-completion request with structured JSON output.

    Parameters
    ----------
    messages:
        OpenAI-format message list (system + user) from
        ``build_prompt()``.
    config:
        Generation configuration.  Uses defaults when ``None``.

    Returns
    -------
    tuple[dict, dict]
        ``(parsed_response, usage)`` where *parsed_response* is the
        model's JSON output conforming to the generation schema and
        *usage* contains ``prompt_tokens``, ``completion_tokens``,
        ``total_tokens``.

    Raises
    ------
    RuntimeError
        After exhausting retries or on a non-retryable API error.
    """
    cfg = config or GenerationConfig()
    creds = load_azure_generation_config(
        section=cfg.config_section,
        model_key=_model_key_for(cfg.model),
    )

    client = AzureOpenAI(
        azure_endpoint=creds["endpoint"],
        api_key=creds["api_key"],
        api_version=creds["api_version"],
    )

    schema = get_response_schema()

    # Build keyword arguments for the completion call.
    kwargs: dict[str, Any] = {
        "model": creds["model"],
        "messages": messages,
        "temperature": cfg.temperature,
        "max_completion_tokens": cfg.max_output_tokens,
        "response_format": {
            "type": "json_schema",
            "json_schema": {
                "name": "generation_response",
                "strict": True,
                "schema": schema,
            },
        },
    }

    # Optional reasoning-effort control (GPT-5 family).
    if cfg.reasoning_effort is not None:
        kwargs["reasoning_effort"] = cfg.reasoning_effort

    return _call_with_retry(client, kwargs)


# ── Internals ───────────────────────────────────────────────────


# Map deployment names back to config model keys.
_MODEL_KEY_MAP: dict[str, str] = {
    "gpt-5-mini": "GPT-5-mini",
    "gpt-5-nano": "GPT-5-nano",
}


def _model_key_for(model: str) -> str:
    """Resolve a deployment name to the config ``models`` dict key."""
    return _MODEL_KEY_MAP.get(model, model)


def _call_with_retry(
    client: AzureOpenAI,
    kwargs: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, int]]:
    """Execute the chat-completion call with exponential-backoff retry."""
    backoff = _INITIAL_BACKOFF

    for attempt in range(1, _MAX_RETRIES + 1):
        try:
            response = client.chat.completions.create(**kwargs)

            content = response.choices[0].message.content or ""
            parsed: dict[str, Any] = json.loads(content)

            usage: dict[str, int] = {
                "prompt_tokens": response.usage.prompt_tokens if response.usage else 0,
                "completion_tokens": response.usage.completion_tokens if response.usage else 0,
                "total_tokens": response.usage.total_tokens if response.usage else 0,
            }

            logger.info(
                "Generation complete: %d prompt + %d completion tokens.",
                usage["prompt_tokens"],
                usage["completion_tokens"],
            )
            return parsed, usage

        except (RateLimitError, APITimeoutError, InternalServerError) as exc:
            if attempt == _MAX_RETRIES:
                raise RuntimeError(
                    f"Generation failed after {_MAX_RETRIES} retries: {exc}"
                ) from exc
            logger.warning(
                "Transient error (attempt %d/%d): %s — retrying in %.1fs",
                attempt, _MAX_RETRIES, exc, backoff,
            )
            time.sleep(backoff)
            backoff *= 2

        except APIError as exc:
            # Non-retryable API error.
            raise RuntimeError(f"Generation API error: {exc}") from exc

        except (json.JSONDecodeError, IndexError, AttributeError) as exc:
            # Malformed response — not retryable.
            raise RuntimeError(
                f"Failed to parse generation response: {exc}"
            ) from exc

    # Should not be reached, but satisfies the type checker.
    raise RuntimeError("Generation failed: retries exhausted.")
