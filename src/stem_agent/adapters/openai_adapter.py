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

    def _get_client(self) -> Any:
        """Lazy-initialize the OpenAI client."""
        if self._client is None:
            from openai import OpenAI

            self._client = OpenAI(api_key=self._config.openai_api_key)
        return self._client

    def generate(self, prompt: str, *, model: str | None = None) -> str:
        """Generate a free-form text response via OpenAI chat completions."""
        client = self._get_client()
        response = client.chat.completions.create(
            model=model or self._config.planning_model,
            messages=[{"role": "user", "content": prompt}],
            temperature=self._config.temperature,
        )
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

        raw = response.choices[0].message.content or "{}"
        parsed = json.loads(raw)
        return response_model.model_validate(parsed)
