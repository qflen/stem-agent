"""Sensing phase — domain reconnaissance via LLM.

The stem agent receives a domain signal (e.g., "code_quality_analysis")
and queries the LLM to build a structured understanding of the domain.
This is the first signal the undifferentiated agent reads to begin
its specialization journey.
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field

from stem_agent.core.journal import EvolutionJournal
from stem_agent.ports.llm import LLMPort


class DomainKnowledge(BaseModel):
    """Structured knowledge about a problem domain, discovered via LLM."""

    domain_name: str = ""
    review_strategies: list[str] = Field(
        default_factory=list,
        description="How experts approach this type of analysis",
    )
    issue_taxonomy: dict[str, list[str]] = Field(
        default_factory=dict,
        description="Categories of issues → specific issue types",
    )
    tool_categories: list[str] = Field(
        default_factory=list,
        description="Categories of tools and techniques used",
    )
    output_format_patterns: list[str] = Field(
        default_factory=list,
        description="What structured, actionable output looks like",
    )
    key_insights: list[str] = Field(
        default_factory=list,
        description="Non-obvious insights about the domain",
    )


SENSING_PROMPT_TEMPLATE = """\
You are helping an AI agent understand the domain of {domain}.

The agent needs to specialize itself for this domain. Provide comprehensive, \
expert-level knowledge about how this type of work is done.

Describe:
1. **Review strategies**: How do experts approach {domain}? What is their \
   typical workflow? Do they use multiple passes? What do they look at first?
2. **Issue taxonomy**: What categories of issues exist? For each category, \
   list 3-5 specific issue types.
3. **Tools and techniques**: What tools and techniques are commonly used? \
   Think about static analysis, pattern matching, AST analysis, etc.
4. **Output format**: What does a structured, actionable review output look like? \
   What information should each finding include?
5. **Key insights**: What non-obvious things distinguish expert analysis from \
   naive analysis? What are common false positives to avoid?

Respond with a JSON object matching this structure:
{{
  "domain_name": "{domain}",
  "review_strategies": ["strategy1", "strategy2", ...],
  "issue_taxonomy": {{
    "category1": ["issue_type1", "issue_type2", ...],
    "category2": [...]
  }},
  "tool_categories": ["tool1", "tool2", ...],
  "output_format_patterns": ["pattern1", "pattern2", ...],
  "key_insights": ["insight1", "insight2", ...]
}}

Be thorough and specific. This knowledge will directly shape how the agent \
configures itself."""


class SensingPhase:
    """Domain reconnaissance: queries LLM to build structured domain knowledge."""

    @property
    def name(self) -> str:
        return "sensing"

    def execute(
        self,
        context: dict[str, Any],
        llm: LLMPort,
        journal: EvolutionJournal,
    ) -> dict[str, Any]:
        """Query the LLM for domain knowledge and parse into structured form.

        Args:
            context: Must contain 'domain' key with the domain signal string.
            llm: LLM adapter for domain reconnaissance.
            journal: For logging the sensing results.

        Returns:
            Context updated with 'domain_knowledge' key.
        """
        domain = context.get("domain", "code_quality_analysis")
        prompt = SENSING_PROMPT_TEMPLATE.format(domain=domain)

        journal.log_decision(
            phase=self.name,
            decision=f"Sensing domain: {domain}",
            reasoning="Initial signal received; querying LLM for structured domain knowledge",
        )

        prompt_hash = EvolutionJournal.hash_prompt(prompt)

        # Use structured generation to get a validated DomainKnowledge object
        knowledge = llm.structured_generate(
            prompt, DomainKnowledge, model=context.get("planning_model")
        )

        journal.log_llm_call(
            phase=self.name,
            model=context.get("planning_model", "default"),
            prompt_hash=prompt_hash,
        )

        # Ensure domain_name is set
        if not knowledge.domain_name:
            knowledge = knowledge.model_copy(update={"domain_name": domain})

        journal.log_phase_result(
            phase=self.name,
            result={
                "domain_name": knowledge.domain_name,
                "strategy_count": len(knowledge.review_strategies),
                "taxonomy_categories": list(knowledge.issue_taxonomy.keys()),
                "tool_count": len(knowledge.tool_categories),
                "insight_count": len(knowledge.key_insights),
            },
        )

        context["domain_knowledge"] = knowledge
        return context
