"""OpenAI adapter — concrete implementation of the LLM port."""

from __future__ import annotations

import json
import logging
import time
from collections.abc import Callable
from functools import wraps
from typing import Any, TypeVar

from pydantic import BaseModel

from stem_agent.core.config import StemAgentConfig

logger = logging.getLogger(__name__)

T = TypeVar("T")

# Exponential backoff: 1s → 2s → 4s → 8s between the four attempts. Four
# attempts is a sweet spot for OpenAI rate-limit bursts — any longer and
# the user gives up waiting, any shorter and a transient 429 kills a run.
_RETRY_DELAYS = (1.0, 2.0, 4.0, 8.0)


def _retryable_openai_errors() -> tuple[type[BaseException], ...]:
    """Resolve the OpenAI exception classes we retry on.

    Imported lazily so tests can stub ``openai`` without pulling the real
    package into their import graph.
    """
    import openai  # local import; caller already depends on openai SDK

    return (openai.RateLimitError, openai.APITimeoutError, openai.APIConnectionError)


def _with_retry(func: Callable[..., T]) -> Callable[..., T]:
    """Retry the wrapped call up to four times on transient OpenAI errors.

    Built manually so we don't take a tenacity dependency just for this.
    The sleep durations come from ``_RETRY_DELAYS`` and can be
    monkey-patched in tests to keep them fast.
    """

    @wraps(func)
    def wrapper(self: OpenAIAdapter, *args: Any, **kwargs: Any) -> T:
        retryable = _retryable_openai_errors()
        last_exc: BaseException | None = None
        for attempt, delay in enumerate(_RETRY_DELAYS, start=1):
            try:
                return func(self, *args, **kwargs)
            except retryable as exc:
                last_exc = exc
                if attempt == len(_RETRY_DELAYS):
                    break
                logger.warning(
                    "OpenAI call failed (attempt %d/%d): %s — retrying in %.1fs",
                    attempt,
                    len(_RETRY_DELAYS),
                    exc,
                    delay,
                )
                self._sleep(delay)
        assert last_exc is not None
        raise last_exc

    return wrapper


class OpenAIAdapter:
    """LLM adapter backed by the OpenAI API.

    Satisfies LLMPort via structural subtyping — no inheritance needed.
    Calls are wrapped in a small exponential-backoff retry for
    ``RateLimitError``, ``APITimeoutError`` and ``APIConnectionError``;
    other errors propagate immediately so callers see real failures fast.
    """

    def __init__(self, config: StemAgentConfig) -> None:
        self._config = config
        self._client: Any = None
        self.last_usage: dict[str, int] | None = None

    def _get_client(self) -> Any:
        """Lazy-initialize the OpenAI client with the configured timeout."""
        if self._client is None:
            from openai import OpenAI

            self._client = OpenAI(
                api_key=self._config.openai_api_key,
                timeout=self._config.request_timeout,
            )
        return self._client

    def _sleep(self, seconds: float) -> None:
        """Indirection around ``time.sleep`` so tests can patch the delay."""
        time.sleep(seconds)

    def _record_usage(self, response: Any) -> None:
        """Capture ``response.usage`` into ``last_usage`` if the SDK returned it."""
        usage = getattr(response, "usage", None)
        if usage is None:
            self.last_usage = None
            return
        self.last_usage = {
            "prompt_tokens": getattr(usage, "prompt_tokens", 0) or 0,
            "completion_tokens": getattr(usage, "completion_tokens", 0) or 0,
            "total_tokens": getattr(usage, "total_tokens", 0) or 0,
        }

    @_with_retry
    def generate(self, prompt: str, *, model: str | None = None) -> str:
        """Generate a free-form text response via OpenAI chat completions."""
        client = self._get_client()
        response = client.chat.completions.create(
            model=model or self._config.planning_model,
            messages=[{"role": "user", "content": prompt}],
            temperature=self._config.temperature,
        )
        self._record_usage(response)
        return response.choices[0].message.content or ""

    @_with_retry
    def structured_generate(
        self,
        prompt: str,
        response_model: type[BaseModel],
        *,
        model: str | None = None,
    ) -> BaseModel:
        """Generate a response conforming to a Pydantic model schema.

        Instructs the model to return JSON matching the schema, then
        validates and parses with Pydantic.
        """
        schema = response_model.model_json_schema()
        schema_str = json.dumps(schema, indent=2)

        structured_prompt = (
            f"{prompt}\n\n"
            f"Respond ONLY with valid JSON matching this schema:\n"
            f"```json\n{schema_str}\n```\n"
            f"Do not include any text outside the JSON."
        )

        client = self._get_client()
        response = client.chat.completions.create(
            model=model or self._config.planning_model,
            messages=[{"role": "user", "content": structured_prompt}],
            temperature=self._config.temperature,
            response_format={"type": "json_object"},
        )
        self._record_usage(response)

        raw = response.choices[0].message.content or "{}"
        parsed = json.loads(raw)
        return response_model.model_validate(parsed)
