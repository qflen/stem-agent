"""Specialization phase; self-assembly from the plan.

Executes the specialization plan by composing the agent's system prompt
from capability fragments and domain insights, configuring the multi-pass
review pipeline, and wiring tool adapters.
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field

from stem_agent.capabilities.prompt_library import (
    UNDIFFERENTIATED_PROMPT,
    compose_system_prompt,
)
from stem_agent.capabilities.registry import CapabilityRegistry, build_default_registry
from stem_agent.core.journal import EvolutionJournal
from stem_agent.phases.planning import SpecializationPlan
from stem_agent.ports.llm import LLMPort

_ROLLBACK_FULL_DEPTH = 3
_ROLLBACK_MARKER = "IMPORTANT adjustments based on prior evaluation"


def _rollback_history(context: dict[str, Any]) -> list[dict[str, Any]]:
    """Return the canonical history list, building one from legacy adjustments if needed."""
    history = context.get("rollback_history")
    if history:
        return list(history)
    adjustments = context.get("rollback_adjustments")
    if adjustments:
        return [{"attempt_idx": 0, "adjustments": list(adjustments), "summary": ""}]
    return []


def _render_rollback_history(history: list[dict[str, Any]]) -> str:
    """Render history with K=3 most recent in full and older as one-line summaries.

    The bounded-by-K rule keeps the composed prompt from growing linearly
    in the number of rollbacks: at attempt 6, the prompt is only ~3
    attempts' worth of full text plus three short summary lines.
    """
    full = history[-_ROLLBACK_FULL_DEPTH:]
    collapsed = history[: max(0, len(history) - _ROLLBACK_FULL_DEPTH)]

    lines = ["", f"{_ROLLBACK_MARKER}:"]
    for entry in collapsed:
        summary = entry.get("summary") or "no summary"
        lines.append(f"- earlier attempt {entry.get('attempt_idx', '?')}: {summary}")
    for entry in full:
        idx = entry.get("attempt_idx", "?")
        for adj in entry.get("adjustments", []):
            lines.append(f"- attempt {idx}: {adj}")
    return "\n".join(lines) + "\n"


class SpecializedAgentConfig(BaseModel):
    """The fully assembled configuration for the specialized agent."""

    system_prompt: str = Field(
        description="The composed system prompt for code review",
    )
    baseline_prompt: str = Field(
        default=UNDIFFERENTIATED_PROMPT,
        description="The undifferentiated baseline prompt for comparison",
    )
    review_passes: list[str] = Field(
        default_factory=list,
        description="Ordered list of review pass names",
    )
    capabilities: list[str] = Field(
        default_factory=list,
        description="Names of active capabilities",
    )
    model: str = Field(
        default="gpt-4o",
        description="Model to use for review execution",
    )


class SpecializationPhase:
    """Self-assembly: builds the specialized agent from the plan."""

    def __init__(self, registry: CapabilityRegistry | None = None) -> None:
        self._registry = registry or build_default_registry()

    @property
    def name(self) -> str:
        return "specialization"

    def execute(
        self,
        context: dict[str, Any],
        llm: LLMPort,
        journal: EvolutionJournal,
    ) -> dict[str, Any]:
        """Assemble the specialized agent from the plan.

        Args:
            context: Must contain 'specialization_plan' from planning phase.
            llm: LLM adapter (not used directly, but available for refinement).
            journal: For logging each assembly step.

        Returns:
            Context updated with 'agent_config' key.
        """
        plan: SpecializationPlan = context["specialization_plan"]
        registry = context.get("registry", self._registry)

        history = _rollback_history(context)
        if history:
            latest = history[-1]
            journal.log_decision(
                phase=self.name,
                decision=(
                    f"Applying {len(latest['adjustments'])} rollback adjustments "
                    f"(history depth={len(history)})"
                ),
                reasoning="; ".join(latest["adjustments"]),
            )

        domain_insights = plan.domain_insights_for_prompt
        if history:
            domain_insights += _render_rollback_history(history)

        generated_fragments = context.get("generated_fragments") or {}
        system_prompt = compose_system_prompt(
            capability_names=plan.selected_capabilities,
            domain_insights=domain_insights,
            extra_fragments=generated_fragments,
        )

        if generated_fragments:
            journal.log_decision(
                phase=self.name,
                decision=(
                    f"Spliced {len(generated_fragments)} generated fragment(s) into the prompt"
                ),
                reasoning=(
                    "capability_generation produced: "
                    + ", ".join(sorted(generated_fragments.keys()))
                ),
            )

        journal.log_decision(
            phase=self.name,
            decision="Composed system prompt",
            reasoning=f"Assembled from {len(plan.selected_capabilities)} capability fragments "
            f"+ domain insights ({len(domain_insights)} chars) "
            f"+ {len(generated_fragments)} generated fragment(s)",
        )

        # Log each capability addition
        for cap_name in plan.selected_capabilities:
            cap = registry.get(cap_name)
            if cap:
                journal.log_capability_added(
                    capability=cap_name,
                    reason=f"Wired into review pipeline: {cap.description}",
                )

        # Configure review passes
        pass_names = [rp.pass_name for rp in plan.review_passes]
        journal.log_decision(
            phase=self.name,
            decision=f"Configured {len(pass_names)} review passes: {pass_names}",
            reasoning="Multi-pass architecture discovered during planning; "
            "each pass focuses on a specific issue category for higher precision",
        )

        active_capabilities = list(plan.selected_capabilities)
        for gen_name in generated_fragments:
            if gen_name not in active_capabilities:
                active_capabilities.append(gen_name)

        agent_config = SpecializedAgentConfig(
            system_prompt=system_prompt,
            review_passes=pass_names,
            capabilities=active_capabilities,
            model=context.get("execution_model", "gpt-4o"),
        )

        journal.log_phase_result(
            phase=self.name,
            result={
                "prompt_length": len(system_prompt),
                "pass_count": len(pass_names),
                "capabilities": plan.selected_capabilities,
            },
        )

        context["agent_config"] = agent_config
        return context
