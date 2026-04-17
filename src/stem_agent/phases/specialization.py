"""Specialization phase — self-assembly from the plan.

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

        # Apply rollback adjustments if any
        adjustments = context.get("rollback_adjustments", [])
        if adjustments:
            journal.log_decision(
                phase=self.name,
                decision=f"Applying {len(adjustments)} rollback adjustments",
                reasoning="; ".join(adjustments),
            )

        # Compose system prompt from selected capabilities
        domain_insights = plan.domain_insights_for_prompt

        # If there were rollback adjustments, add them as extra guidance
        if adjustments:
            domain_insights += "\n\nIMPORTANT adjustments based on prior evaluation:\n"
            for adj in adjustments:
                domain_insights += f"- {adj}\n"

        system_prompt = compose_system_prompt(
            capability_names=plan.selected_capabilities,
            domain_insights=domain_insights,
        )

        journal.log_decision(
            phase=self.name,
            decision="Composed system prompt",
            reasoning=f"Assembled from {len(plan.selected_capabilities)} capability fragments "
            f"+ domain insights ({len(domain_insights)} chars)",
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
            reasoning="Multi-pass architecture discovered during planning — "
            "each pass focuses on a specific issue category for higher precision",
        )

        agent_config = SpecializedAgentConfig(
            system_prompt=system_prompt,
            review_passes=pass_names,
            capabilities=plan.selected_capabilities,
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
