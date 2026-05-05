"""Filesystem-backed prompt archive; captures every composed prompt by hash.

When a run is launched with ``--store-prompts`` the CLI wraps the LLM
adapter in ``PromptArchivingLLM``, a transparent decorator over
``LLMPort``. Every prompt that flows through ``generate`` /
``structured_generate`` is hashed with the same 16-character SHA-256
prefix the journal records, and the body is written to
``<dir>/<hash>.txt`` exactly once. The ``replay`` CLI command reads the
archive back so reviewers can see *the actual prompt body* a journal
``LLM_CALL`` event refers to.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from pydantic import BaseModel

from stem_agent.core.journal import EvolutionJournal
from stem_agent.ports.llm import LLMPort


class PromptArchivingLLM:
    """LLMPort decorator that archives prompts before delegating."""

    def __init__(self, inner: LLMPort, prompts_dir: Path | str) -> None:
        self._inner = inner
        self._dir = Path(prompts_dir)
        self._dir.mkdir(parents=True, exist_ok=True)

    @property
    def last_usage(self) -> dict[str, int] | None:
        return getattr(self._inner, "last_usage", None)

    def generate(self, prompt: str, *, model: str | None = None) -> str:
        self._archive(prompt)
        return self._inner.generate(prompt, model=model)

    def structured_generate(
        self,
        prompt: str,
        response_model: type[BaseModel],
        *,
        model: str | None = None,
    ) -> Any:
        self._archive(prompt)
        return self._inner.structured_generate(prompt, response_model, model=model)

    def _archive(self, prompt: str) -> None:
        digest = EvolutionJournal.hash_prompt(prompt)
        path = self._dir / f"{digest}.txt"
        if not path.exists():
            path.write_text(prompt, encoding="utf-8")
