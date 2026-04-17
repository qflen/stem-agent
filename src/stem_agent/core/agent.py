"""StemAgent — the undifferentiated core that orchestrates differentiation.

This is the main entry point for the differentiation process. It manages
the state machine, phases, and journal, driving the agent from
UNDIFFERENTIATED through to SPECIALIZED (or FAILED).
"""

from __future__ import annotations

from datetime import UTC
from typing import Any

from rich.console import Console

from stem_agent.adapters.json_storage import JsonStorageAdapter
from stem_agent.capabilities.registry import CapabilityRegistry, build_default_registry
from stem_agent.core.config import StemAgentConfig
from stem_agent.core.journal import EvolutionJournal
from stem_agent.core.state_machine import (
    AgentState,
    GuardFailedError,
    StateMachine,
)
from stem_agent.evaluation.benchmark import (
    parse_review_response,
)
from stem_agent.evaluation.fixtures.code_samples import BenchmarkSample
from stem_agent.phases.planning import PlanningPhase
from stem_agent.phases.sensing import SensingPhase
from stem_agent.phases.specialization import SpecializationPhase, SpecializedAgentConfig
from stem_agent.phases.validation import ValidationPhase, diagnose_failure
from stem_agent.ports.llm import LLMPort

console = Console()


class StemAgent:
    """The undifferentiated agent core that drives self-specialization.

    Orchestrates the lifecycle: UNDIFFERENTIATED → SENSING →
    DIFFERENTIATING → VALIDATING → SPECIALIZED (with rollback support).
    """

    def __init__(
        self,
        config: StemAgentConfig,
        llm: LLMPort,
        registry: CapabilityRegistry | None = None,
        corpus: list[BenchmarkSample] | None = None,
    ) -> None:
        self._config = config
        self._llm = llm
        self._journal = EvolutionJournal()
        self._state_machine = StateMachine(journal=self._journal)
        self._registry = registry or build_default_registry()
        self._corpus = corpus
        self._context: dict[str, Any] = {
            "planning_model": config.planning_model,
            "execution_model": config.execution_model,
            "f1_threshold": config.f1_threshold,
            "improvement_required": config.improvement_required,
            "max_rollback_attempts": config.max_rollback_attempts,
        }
        self._agent_config: SpecializedAgentConfig | None = None
        self._storage = JsonStorageAdapter(config.journal_dir)

    @property
    def state(self) -> AgentState:
        return self._state_machine.state

    @property
    def journal(self) -> EvolutionJournal:
        return self._journal

    @property
    def agent_config(self) -> SpecializedAgentConfig | None:
        return self._agent_config

    def differentiate(self, domain: str = "code_quality_analysis") -> bool:
        """Run the full differentiation process.

        Args:
            domain: The domain signal to specialize for.

        Returns:
            True if specialization succeeded, False if it failed.
        """
        self._context["domain"] = domain

        console.print(f"\n[bold blue]Starting differentiation for domain: {domain}[/bold blue]\n")

        # Phase 1: SENSING
        self._transition(AgentState.SENSING)
        console.print("[dim]Phase 1: Sensing domain...[/dim]")
        sensing = SensingPhase()
        self._context = sensing.execute(self._context, self._llm, self._journal)
        console.print("[green]  Sensing complete.[/green]")

        # Enter differentiation loop (DIFFERENTIATING → VALIDATING → SPECIALIZED or ROLLBACK)
        while True:
            # Phase 2+3: DIFFERENTIATING (planning + specialization)
            self._transition(AgentState.DIFFERENTIATING)
            console.print("[dim]Phase 2: Planning specialization...[/dim]")
            planning = PlanningPhase(registry=self._registry)
            self._context = planning.execute(self._context, self._llm, self._journal)
            console.print("[green]  Planning complete.[/green]")

            console.print("[dim]Phase 3: Specializing...[/dim]")
            specialization = SpecializationPhase(registry=self._registry)
            self._context = specialization.execute(self._context, self._llm, self._journal)
            console.print("[green]  Specialization complete.[/green]")

            # Phase 4: VALIDATING
            self._transition(AgentState.VALIDATING)
            console.print("[dim]Phase 4: Validating against benchmark...[/dim]")
            validation = ValidationPhase(corpus=self._corpus)
            self._context = validation.execute(self._context, self._llm, self._journal)
            console.print("[green]  Validation complete.[/green]")

            # Try to graduate to SPECIALIZED
            guard_context = {
                **self._context,
                "rollback_count": self._state_machine.rollback_count,
            }

            try:
                self._state_machine.transition(AgentState.SPECIALIZED, guard_context)
                self._agent_config = self._context["agent_config"]
                console.print(
                    "\n[bold green]Differentiation successful! "
                    "Agent is now SPECIALIZED.[/bold green]\n"
                )
                self._save_journal()
                return True

            except GuardFailedError as e:
                console.print(f"[yellow]  Guard failed: {e}[/yellow]")

                # Check rollback budget
                try:
                    self._state_machine.transition(AgentState.ROLLBACK, guard_context)
                except GuardFailedError:
                    # Rollback budget exhausted
                    console.print(
                        "\n[bold red]Rollback budget exhausted. "
                        "Differentiation failed.[/bold red]\n"
                    )
                    self._state_machine.transition(AgentState.FAILED, guard_context)
                    self._save_journal()
                    return False

                # Diagnose and adjust
                console.print("[dim]  Rolling back — diagnosing failure...[/dim]")
                adjustments = diagnose_failure(self._context, self._journal)
                self._context["rollback_adjustments"] = adjustments
                for adj in adjustments:
                    console.print(f"[yellow]    Adjustment: {adj}[/yellow]")
                console.print()

    def review(self, code: str) -> dict[str, Any]:
        """Review code using the specialized agent.

        Args:
            code: Python source code to review.

        Returns:
            Structured review result.

        Raises:
            RuntimeError: If the agent has not been specialized.
        """
        if self._agent_config is None:
            raise RuntimeError("Agent has not been specialized. Run differentiate() first.")

        if self.state == AgentState.SPECIALIZED:
            self._state_machine.transition(AgentState.EXECUTING)

        prompt = f"{self._agent_config.system_prompt}\n\n## Code to Review\n```python\n{code}\n```"
        raw_response = self._llm.generate(prompt, model=self._agent_config.model)
        result = parse_review_response(raw_response)
        return result.model_dump()

    def _transition(self, target: AgentState) -> None:
        """Execute a state transition with logging."""
        self._state_machine.transition(target, self._context)

    def _save_journal(self) -> None:
        """Persist the evolution journal."""
        from datetime import datetime

        timestamp = datetime.now(UTC).strftime("%Y%m%d_%H%M%S")
        key = f"journal_{timestamp}"
        self._storage.save(key, self._journal.to_dict())
        console.print(f"[dim]Journal saved: {key}[/dim]")
