"""Sensing phase; domain reconnaissance grounded in probe samples.

The undifferentiated agent receives a domain signal and queries the LLM
to build a structured understanding of the domain. The prompt is no
longer generic: it embeds the bodies of four unlabelled ``probe``
samples drawn from the corpus partition, plus the deterministic output
of ``analyze_structure`` and ``scan_patterns`` over them as a
``tool_fit`` histogram. The agent has to answer "what would catch the
issues *in these specific snippets*?", which forces the resulting
``DomainKnowledge`` to be calibrated to the corpus rather than to a
generic mental model of code review.
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field

from stem_agent.capabilities.tools import analyze_structure, scan_patterns
from stem_agent.core.journal import EvolutionJournal
from stem_agent.evaluation.fixtures.code_samples import (
    BenchmarkSample,
    CorpusPartition,
)
from stem_agent.ports.llm import LLMPort

_PROBE_BODY_LIMIT = 600


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
    tool_fit: dict[str, int] = Field(
        default_factory=dict,
        description=(
            "Hit-count by deterministic-tool category over the probe slice; "
            "filled in by the agent (not the LLM) so planning can bias "
            "capability selection toward tools that actually fire on this corpus."
        ),
    )


def _format_probe_body(sample: BenchmarkSample) -> str:
    body = sample.code.strip()
    if len(body) > _PROBE_BODY_LIMIT:
        body = body[:_PROBE_BODY_LIMIT] + "\n# ... (truncated)"
    return body


def compute_tool_fit(samples: list[BenchmarkSample]) -> dict[str, int]:
    """Run analyze_structure and scan_patterns on each probe and aggregate hits.

    ``security`` counts the number of pattern-scanner matches; ``structure``
    counts the number of probes whose AST exceeds the long-function or
    deep-nesting heuristics. The shape mirrors the canonical issue
    categories so planning can match capability tags directly against it.
    """
    counts: dict[str, int] = {}
    for sample in samples:
        metrics = analyze_structure(sample.code)
        if metrics is not None and (
            metrics.max_function_length > 30 or metrics.max_nesting_depth > 3
        ):
            counts["structure"] = counts.get("structure", 0) + 1
        patterns = scan_patterns(sample.code)
        if patterns:
            counts["security"] = counts.get("security", 0) + len(patterns)
    return counts


def _build_probe_block(samples: list[BenchmarkSample]) -> str:
    if not samples:
        return "(no probe samples available)"
    lines: list[str] = []
    for index, sample in enumerate(samples, start=1):
        metrics = analyze_structure(sample.code)
        patterns = scan_patterns(sample.code)
        summary = (
            f"functions={metrics.function_count} "
            f"max_len={metrics.max_function_length} "
            f"max_depth={metrics.max_nesting_depth}"
            if metrics is not None
            else "ast=parse_failed"
        )
        pattern_summary = (
            ", ".join(p.pattern_description for p in patterns) if patterns else "no scanner hits"
        )
        lines.append(
            f"### probe_{index}; {sample.sample_id}\n"
            f"static_signal: {summary}; patterns: {pattern_summary}\n"
            f"```python\n{_format_probe_body(sample)}\n```"
        )
    return "\n\n".join(lines)


SENSING_PROMPT_TEMPLATE = """\
You are helping an AI agent understand the domain of {domain}.

The agent has run two deterministic tools; an AST analyser and a regex
pattern scanner; over four unlabelled probe snippets from the corpus.
The aggregated hit-count by tool category is:

{tool_fit_summary}

Below are the probe snippets the agent will rely on as ground for its
strategy choice:

{probe_block}

Given these probes and these tool-hit signals, design the agent's
specialization plan:

1. **Review strategies**: which approaches catch the most issues that
   *these* snippets exemplify? Order by impact for this corpus.
2. **Issue taxonomy**: name 3-5 categories with 3-5 specific issues each
   that the corpus hints at.
3. **Tool categories**: which tools are likely to fire most often here?
   Match them to the static signals above.
4. **Output format patterns**: what does an actionable review look like
   for this domain?
5. **Key insights**: non-obvious things that separate expert analysis
   from naive analysis on *this* kind of code.

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

Be specific. This knowledge directly shapes the agent's capability set."""


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
        """Query the LLM with probe-grounded context for structured domain knowledge.

        Reads ``context["partition"]`` (a ``CorpusPartition``) so the
        prompt embeds the probe samples and the deterministic tool
        signals computed over them. Falls back to the generic prompt
        only when no partition is available; that path is preserved for
        unit tests that exercise sensing in isolation.
        """
        domain = context.get("domain", "code_quality_analysis")
        partition_obj: CorpusPartition | None = context.get("partition")
        probe_samples = list(partition_obj.probe) if partition_obj else []
        tool_fit = compute_tool_fit(probe_samples)
        tool_fit_summary = (
            ", ".join(f"{cat}: {hits}" for cat, hits in sorted(tool_fit.items()))
            if tool_fit
            else "(no probe-level signals fired)"
        )
        probe_block = _build_probe_block(probe_samples)

        prompt = SENSING_PROMPT_TEMPLATE.format(
            domain=domain,
            tool_fit_summary=tool_fit_summary,
            probe_block=probe_block,
        )

        journal.log_decision(
            phase=self.name,
            decision=f"Sensing domain: {domain}",
            reasoning=(f"probe_count={len(probe_samples)}, tool_fit={tool_fit_summary}"),
        )

        prompt_hash = EvolutionJournal.hash_prompt(prompt)

        knowledge = llm.structured_generate(
            prompt, DomainKnowledge, model=context.get("planning_model")
        )

        usage = getattr(llm, "last_usage", None)
        journal.log_llm_call(
            phase=self.name,
            model=context.get("planning_model", "default"),
            prompt_hash=prompt_hash,
            token_count=usage.get("total_tokens") if usage else None,
        )

        if not knowledge.domain_name:
            knowledge = knowledge.model_copy(update={"domain_name": domain})
        knowledge = knowledge.model_copy(update={"tool_fit": tool_fit})

        journal.log_phase_result(
            phase=self.name,
            result={
                "domain_name": knowledge.domain_name,
                "strategy_count": len(knowledge.review_strategies),
                "taxonomy_categories": list(knowledge.issue_taxonomy.keys()),
                "tool_count": len(knowledge.tool_categories),
                "insight_count": len(knowledge.key_insights),
                "tool_fit": tool_fit,
                "probe_count": len(probe_samples),
            },
        )

        context["domain_knowledge"] = knowledge
        return context
