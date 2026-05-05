"""LLM port; the protocol that any language model adapter must satisfy."""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from pydantic import BaseModel


@runtime_checkable
class LLMPort(Protocol):
    """Structural interface for language model interactions.

    Uses Protocol (structural subtyping) over ABC (nominal subtyping)
    because any object with matching methods satisfies the contract;
    no inheritance required. This mirrors the extension-point philosophy:
    the core depends on shape, not lineage.

    Adapters expose a ``last_usage`` attribute refreshed after each call
    with ``prompt_tokens``, ``completion_tokens`` and ``total_tokens`` so
    callers can journal cost without plumbing return-value changes through
    every phase.
    """

    last_usage: dict[str, int] | None

    def generate(self, prompt: str, *, model: str | None = None) -> str:
        """Generate a free-form text response.

        Args:
            prompt: The full prompt to send to the model.
            model: Optional model override. If None, use the adapter's default.

        Returns:
            The model's text response.
        """
        ...

    def structured_generate(
        self,
        prompt: str,
        response_model: type[BaseModel],
        *,
        model: str | None = None,
    ) -> BaseModel:
        """Generate a response conforming to a Pydantic model schema.

        Args:
            prompt: The full prompt to send to the model.
            response_model: Pydantic model class defining the expected response structure.
            model: Optional model override.

        Returns:
            An instance of response_model populated from the model's response.
        """
        ...
