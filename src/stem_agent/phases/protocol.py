"""Phase protocol — the contract every differentiation phase must satisfy.

Phases are pluggable strategies: the stem agent core iterates over phases
without knowing their internals. A new domain could swap in entirely
different phase implementations.
"""

from __future__ import annotations

from typing import Any, Protocol, runtime_checkable

from stem_agent.core.journal import EvolutionJournal
from stem_agent.ports.llm import LLMPort


@runtime_checkable
class PhaseProtocol(Protocol):
    """Every differentiation phase implements this interface.

    The core calls execute() and uses the returned dict as input
    to the next phase. The journal is passed for self-documentation.
    """

    @property
    def name(self) -> str:
        """Human-readable phase name for logging and journal entries."""
        ...

    def execute(
        self,
        context: dict[str, Any],
        llm: LLMPort,
        journal: EvolutionJournal,
    ) -> dict[str, Any]:
        """Execute this phase of differentiation.

        Args:
            context: Accumulated state from previous phases.
            llm: Language model adapter for any LLM calls this phase needs.
            journal: Evolution journal for recording decisions and events.

        Returns:
            Updated context dict with this phase's contributions.
        """
        ...
