"""Deterministic LLM double; same Protocol as production adapters, no network.

Used by tests as a fixture and by the ablation CLI command as the only
sensible sampler when there's no API key. The class is intentionally
small: substring-routed responses with token-count estimates that
populate ``last_usage`` so journal events look like a real run.
"""

from __future__ import annotations

import json
from typing import Any

from pydantic import BaseModel


class FakeLLM:
    """LLMPort-compatible test double with substring-keyed responses."""

    def __init__(
        self,
        responses: dict[str, str] | None = None,
        structured_responses: dict[str, dict[str, Any]] | None = None,
        default_response: str = '{"issues": [], "summary": "No issues found.", "is_clean": true}',
    ) -> None:
        self._responses = responses or {}
        self._structured_responses = structured_responses or {}
        self._default_response = default_response
        self.calls: list[dict[str, Any]] = []
        self.last_usage: dict[str, int] | None = None

    @staticmethod
    def _fake_usage(prompt: str, completion: str) -> dict[str, int]:
        prompt_tokens = max(1, len(prompt) // 4)
        completion_tokens = max(1, len(completion) // 4)
        return {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": prompt_tokens + completion_tokens,
        }

    def generate(self, prompt: str, *, model: str | None = None) -> str:
        self.calls.append({"method": "generate", "prompt": prompt, "model": model})
        for key, response in self._responses.items():
            if key in prompt:
                self.last_usage = self._fake_usage(prompt, response)
                return response
        self.last_usage = self._fake_usage(prompt, self._default_response)
        return self._default_response

    def structured_generate(
        self,
        prompt: str,
        response_model: type[BaseModel],
        *,
        model: str | None = None,
    ) -> BaseModel:
        self.calls.append(
            {
                "method": "structured_generate",
                "prompt": prompt,
                "model": model,
                "response_model": response_model.__name__,
            }
        )
        for key, data in self._structured_responses.items():
            if key in prompt:
                self.last_usage = self._fake_usage(prompt, json.dumps(data))
                return response_model.model_validate(data)
        fallback = self._structured_responses.get("default", {})
        self.last_usage = self._fake_usage(prompt, json.dumps(fallback))
        return response_model.model_validate(fallback)
