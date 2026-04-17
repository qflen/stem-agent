"""OpenAI adapter — concrete implementation of the LLM port."""

from __future__ import annotations

import json
from typing import Any

from pydantic import BaseModel

from stem_agent.core.config import StemAgentConfig


class OpenAIAdapter:
    """LLM adapter backed by the OpenAI API.

    Satisfies LLMPort via structural subtyping — no inheritance needed.
    """

    def __init__(self, config: StemAgentConfig) -> None:
        self._config = config
        self._client: Any = None
        self.last_usage: dict[str, int] | None = None

    def _get_client(self) -> Any:
        """Lazy-initialize the OpenAI client."""
        if self._client is None:
            from openai import OpenAI

            self._client = OpenAI(api_key=self._config.openai_api_key)
        return self._client

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
