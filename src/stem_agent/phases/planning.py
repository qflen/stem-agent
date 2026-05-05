"""Planning phase; capability selection and architecture decisions.

Based on domain knowledge from sensing, the agent generates a
specialization plan: which capabilities to acquire, what architecture
to use, and what evaluation criteria to apply.
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field

from stem_agent.capabilities.registry import (
    Capability,
    CapabilityRegistry,
    build_default_registry,
)
from stem_agent.core.journal import EvolutionJournal
from stem_agent.core.priors import NEUTRAL_WEIGHT, weight_capabilities
from stem_agent.phases.sensing import DomainKnowledge
from stem_agent.ports.llm import LLMPort
from stem_agent.ports.storage import StoragePort


def _rank_capabilities(
    capabilities: list[Capability],
    tool_fit: dict[str, int],
    prior_weights: dict[str, float] | None = None,
) -> list[Capability]:
    """Sort capabilities by combined tool-fit score and cross-run prior.

    Score = (1 + tool_fit_match) * prior_weight, descending; ties keep
    the registry's original order so the output is deterministic when
    both signals are absent.
    """
    weights = prior_weights or {}

    def score(cap: Capability) -> float:
        tool_score = sum(tool_fit.get(tag, 0) for tag in cap.tags)
        prior = weights.get(cap.name, NEUTRAL_WEIGHT)
        return -((1 + tool_score) * prior)

    indexed = list(enumerate(capabilities))
    indexed.sort(key=lambda pair: (score(pair[1]), pair[0]))
    return [cap for _, cap in indexed]


class ReviewPassConfig(BaseModel):
    """Configuration for a single pass in the multi-pass review pipeline."""

    pass_name: str
    focus_area: str
    capability_name: str
    priority: int = 0


class SpecializationPlan(BaseModel):
    """The agent's plan for how to specialize itself."""

    selected_capabilities: list[str] = Field(
        default_factory=list,
        description="Names of capabilities to acquire from the registry",
    )
    review_passes: list[ReviewPassConfig] = Field(
        default_factory=list,
        description="Ordered multi-pass review pipeline configuration",
    )
    evaluation_criteria: dict[str, float] = Field(
        default_factory=dict,
        description="Metric name → minimum threshold for validation",
    )
    domain_insights_for_prompt: str = Field(
        default="",
        description="Domain-specific insights to embed in the system prompt",
    )
    reasoning: str = Field(
        default="",
        description="Why this plan was chosen",
    )


PLANNING_PROMPT_TEMPLATE = """\
You are helping an AI agent plan its specialization for {domain}.

The agent has learned the following about the domain:
- Review strategies: {strategies}
- Issue taxonomy: {taxonomy}
- Tools/techniques: {tools}
- Key insights: {insights}
- Tool-fit hits over probe slice: {tool_fit}

Available capabilities the agent can select (ordered by tool-fit relevance):
{capabilities}

Design a specialization plan:

1. **Selected capabilities**: Which capabilities should the agent acquire? \
   Choose from the available list by name.
2. **Review passes**: Design a multi-pass review pipeline. Each pass focuses \
   on one aspect. Suggest 3-5 passes ordered by priority. \
   Each pass should map to a capability name.
3. **Evaluation criteria**: What metric thresholds should the agent meet? \
   Use standard metrics: f1_threshold, precision_min, recall_min.
4. **Domain insights**: What key insights from sensing should be embedded \
   in the agent's system prompt? Write 2-3 sentences.
5. **Reasoning**: Explain why this plan was chosen over alternatives.

Respond with JSON:
{{
  "selected_capabilities": ["cap1", "cap2", ...],
  "review_passes": [
    {{"pass_name": "...", "focus_area": "...", "capability_name": "...", "priority": 1}},
    ...
  ],
  "evaluation_criteria": {{"f1_threshold": 0.6, "precision_min": 0.5}},
  "domain_insights_for_prompt": "...",
  "reasoning": "..."
}}"""


class PlanningPhase:
    """Capability selection and architecture planning based on domain knowledge."""

    def __init__(self, registry: CapabilityRegistry | None = None) -> None:
        self._registry = registry or build_default_registry()

    @property
    def name(self) -> str:
        return "planning"

    @staticmethod
    def _prior_weights_from(context: dict[str, Any]) -> dict[str, float]:
        storage: StoragePort | None = context.get("storage")
        domain: str = context.get("domain") or context.get("domain_knowledge").domain_name
        if storage is None or not domain:
            return {}
        try:
            return weight_capabilities(domain, storage)
        except Exception:
            return {}

    def execute(
        self,
        context: dict[str, Any],
        llm: LLMPort,
        journal: EvolutionJournal,
    ) -> dict[str, Any]:
        """Generate a specialization plan from domain knowledge.

        Args:
            context: Must contain 'domain_knowledge' from sensing phase.
            llm: LLM adapter for plan generation.
            journal: For logging planning decisions.

        Returns:
            Context updated with 'specialization_plan' key.
        """
        knowledge: DomainKnowledge = context["domain_knowledge"]
        prior_weights = self._prior_weights_from(context)
        ranked_caps = _rank_capabilities(
            self._registry.list_all(), knowledge.tool_fit, prior_weights
        )

        cap_descriptions = "\n".join(f"- {c.name}: {c.description}" for c in ranked_caps)
        tool_fit_summary = (
            ", ".join(f"{k}={v}" for k, v in sorted(knowledge.tool_fit.items()))
            if knowledge.tool_fit
            else "(none)"
        )

        prompt = PLANNING_PROMPT_TEMPLATE.format(
            domain=knowledge.domain_name,
            strategies=", ".join(knowledge.review_strategies[:5]),
            taxonomy=str(knowledge.issue_taxonomy),
            tools=", ".join(knowledge.tool_categories[:5]),
            insights=", ".join(knowledge.key_insights[:3]),
            capabilities=cap_descriptions,
            tool_fit=tool_fit_summary,
        )

        prompt_hash = EvolutionJournal.hash_prompt(prompt)

        plan = llm.structured_generate(
            prompt, SpecializationPlan, model=context.get("planning_model")
        )

        usage = getattr(llm, "last_usage", None)
        journal.log_llm_call(
            phase=self.name,
            model=context.get("planning_model", "default"),
            prompt_hash=prompt_hash,
            token_count=usage.get("total_tokens") if usage else None,
        )

        # Validate selected capabilities exist in registry
        valid_caps = []
        for cap_name in plan.selected_capabilities:
            if self._registry.get(cap_name) is not None:
                valid_caps.append(cap_name)
                journal.log_capability_added(
                    capability=cap_name,
                    reason=f"Selected during planning for {knowledge.domain_name}",
                )
            else:
                journal.log_decision(
                    phase=self.name,
                    decision=f"Skipped unknown capability: {cap_name}",
                    reasoning="Capability not found in registry; LLM hallucinated a name",
                )

        plan = plan.model_copy(update={"selected_capabilities": valid_caps})

        # Ensure review passes reference valid capabilities
        valid_passes = []
        for rp in plan.review_passes:
            if rp.capability_name in valid_caps or rp.capability_name == "severity_ranking":
                valid_passes.append(rp)
                if rp.capability_name not in valid_caps:
                    valid_caps.append(rp.capability_name)
                    plan = plan.model_copy(update={"selected_capabilities": valid_caps})

        plan = plan.model_copy(update={"review_passes": valid_passes})

        journal.log_decision(
            phase=self.name,
            decision=f"Selected {len(valid_caps)} capabilities, {len(valid_passes)} passes",
            reasoning=plan.reasoning,
        )

        journal.log_phase_result(
            phase=self.name,
            result={
                "selected_capabilities": valid_caps,
                "pass_count": len(valid_passes),
                "evaluation_criteria": plan.evaluation_criteria,
            },
        )

        context["specialization_plan"] = plan
        context["registry"] = self._registry
        return context
