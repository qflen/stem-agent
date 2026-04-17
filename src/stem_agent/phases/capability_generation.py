"""Capability generation — the stem-cell move.

The registry-only path is a capability composer: it picks from a
pre-authored catalog. The stem metaphor asks for more — the agent
should be able to propose a capability it does not yet have. That is
what this phase does.

Between sensing and the first planning pass, the agent queries the LLM
for ONE new capability that targets a gap in the existing registry. The
proposal carries three things: a ``name``, a ``prompt_fragment`` to
splice into the composed system prompt, and — optionally — a short
Python ``check`` function used as a deterministic sanity net at review
time. The Python code is validated through the sandbox in
``capabilities/sandbox.py`` before it can reach the registry:

* rejected proposals do not mutate the registry; a ``DECISION`` event
  explains why, and the rest of differentiation continues on the pure
  composition path;
* accepted proposals are inserted into the registry with
  ``origin="generated"`` so the planning phase can select them like any
  other capability, and their fragment is carried in the context for
  ``compose_system_prompt`` to splice in.

The phase is deliberately one-shot: it does not re-run on rollback. The
rollback loop steers the *prompt*, not the capability catalog.
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field, ValidationError

from stem_agent.capabilities.registry import (
    Capability,
    CapabilityCategory,
    CapabilityRegistry,
    build_default_registry,
)
from stem_agent.capabilities.sandbox import SandboxResult, run_in_sandbox
from stem_agent.core.journal import EvolutionJournal
from stem_agent.phases.sensing import DomainKnowledge
from stem_agent.ports.llm import LLMPort


class ProposedCapability(BaseModel):
    """A capability proposal the agent surfaces during generation.

    ``validator_code`` may be ``None`` when no static check is practical;
    in that case admission only requires the prompt fragment. When it is
    present, it must define ``def check(code: str) -> list[str]`` and
    pass the sandbox.
    """

    name: str = Field(description="snake_case identifier, unique within the registry")
    description: str = Field(description="one-line description of what this pass catches")
    prompt_fragment: str = Field(
        description="Markdown-formatted review pass to splice into the system prompt",
    )
    validator_code: str | None = Field(
        default=None,
        description=(
            "Optional Python source defining `def check(code: str) -> list[str]:`. "
            "Only `re`, `ast`, `string` imports are permitted; the code is run in a "
            "sandboxed subprocess before being accepted."
        ),
    )


_GENERATION_PROMPT_TEMPLATE = """\
You are helping an AI agent propose ONE new review capability it does not already have.

Domain: {domain}
Key insights:
{insights}
Issue taxonomy: {taxonomy}

The agent already has these capabilities:
{existing}

Propose exactly ONE new capability that closes a gap the existing set does not
cover. The new capability must be specific to this domain and must not overlap
with the existing list.

Respond with JSON matching this schema:
{{
  "name": "<snake_case identifier>",
  "description": "<one-line description>",
  "prompt_fragment": "## <Named Pass>\\n- <specific instruction>\\n- <specific instruction>\\n...",
  "validator_code": "<optional: a Python function `def check(code: str) -> list[str]` \
that returns a list of short string descriptions when the code exhibits the problem \
and [] otherwise. Only `re`, `ast`, `string` imports are permitted. Do not open files, \
do not call subprocesses, do not import os/sys/io/subprocess/socket/pickle. Use null \
if no static check is practical.>"
}}
"""


def _format_existing(registry: CapabilityRegistry) -> str:
    return "\n".join(f"- {cap.name}: {cap.description}" for cap in registry.list_all())


def _format_insights(knowledge: DomainKnowledge) -> str:
    return "\n".join(f"- {line}" for line in knowledge.key_insights[:5]) or "- (none)"


class CapabilityGenerationPhase:
    """Propose, validate, and admit one brand-new capability per run."""

    def __init__(self, registry: CapabilityRegistry | None = None) -> None:
        self._registry = registry or build_default_registry()

    @property
    def name(self) -> str:
        return "capability_generation"

    def execute(
        self,
        context: dict[str, Any],
        llm: LLMPort,
        journal: EvolutionJournal,
    ) -> dict[str, Any]:
        """Ask the LLM for one new capability and admit it if it validates.

        Failure modes at every step fall back to composition-only: the
        registry is left untouched, a ``DECISION`` event records the
        reason, and differentiation continues.
        """
        knowledge: DomainKnowledge = context["domain_knowledge"]
        registry = context.get("registry") or self._registry

        prompt = _GENERATION_PROMPT_TEMPLATE.format(
            domain=knowledge.domain_name or context.get("domain", ""),
            insights=_format_insights(knowledge),
            taxonomy=str(knowledge.issue_taxonomy),
            existing=_format_existing(registry),
        )
        prompt_hash = EvolutionJournal.hash_prompt(prompt)

        try:
            proposal = llm.structured_generate(
                prompt, ProposedCapability, model=context.get("planning_model")
            )
        except (ValidationError, ValueError, KeyError) as exc:
            journal.log_decision(
                phase=self.name,
                decision="No capability proposed — falling back to composition-only",
                reasoning=f"LLM did not return a valid proposal ({type(exc).__name__})",
            )
            context["registry"] = registry
            return context

        usage = getattr(llm, "last_usage", None)
        journal.log_llm_call(
            phase=self.name,
            model=context.get("planning_model", "default"),
            prompt_hash=prompt_hash,
            token_count=usage.get("total_tokens") if usage else None,
        )

        if not isinstance(proposal, ProposedCapability):
            journal.log_decision(
                phase=self.name,
                decision="Proposal rejected — wrong model type",
                reasoning=f"expected ProposedCapability, got {type(proposal).__name__}",
            )
            context["registry"] = registry
            return context

        if registry.get(proposal.name) is not None:
            journal.log_decision(
                phase=self.name,
                decision=f"Proposal rejected: '{proposal.name}' already in registry",
                reasoning="capability names are unique; LLM duplicated an existing entry",
            )
            context["registry"] = registry
            return context

        sandbox_result: SandboxResult | None = None
        if proposal.validator_code:
            sandbox_result = run_in_sandbox(proposal.validator_code)
            if not sandbox_result.ok:
                journal.log_decision(
                    phase=self.name,
                    decision=f"Proposal '{proposal.name}' rejected by sandbox",
                    reasoning=sandbox_result.error,
                )
                context["registry"] = registry
                return context

        capability = Capability(
            name=proposal.name,
            category=CapabilityCategory.DETECTION,
            description=proposal.description,
            prompt_fragment=proposal.prompt_fragment,
            tags=frozenset({"generated"}),
            origin="generated",
            validator_code=proposal.validator_code,
        )
        try:
            registry.register(capability)
        except ValueError as exc:
            journal.log_decision(
                phase=self.name,
                decision=f"Proposal '{proposal.name}' rejected on registration",
                reasoning=str(exc),
            )
            context["registry"] = registry
            return context

        admission_note = "sandbox validation" if sandbox_result else "prompt-only review"
        journal.log_capability_added(
            capability=proposal.name,
            reason=f"generated: admitted after {admission_note}",
        )
        journal.log_decision(
            phase=self.name,
            decision=f"Admitted generated capability: {proposal.name}",
            reasoning=(
                f"prompt fragment={len(proposal.prompt_fragment)} chars, "
                f"validator={'yes' if proposal.validator_code else 'none'}"
            ),
        )

        generated_fragments = dict(context.get("generated_fragments") or {})
        generated_fragments[proposal.name] = proposal.prompt_fragment
        context["generated_fragments"] = generated_fragments
        context["registry"] = registry

        journal.log_phase_result(
            phase=self.name,
            result={
                "admitted": proposal.name,
                "origin": "generated",
                "has_validator": proposal.validator_code is not None,
            },
        )

        return context
