"""Tests for the capability_generation phase and its sandbox.

Covers the three paths the feature has to defend:

1. a valid proposal is admitted, ends up in the registry with
   ``origin="generated"``, and its fragment reaches the composed prompt;
2. a proposal with a syntactically broken validator is rejected and the
   registry is left untouched;
3. a proposal whose validator tries to touch the filesystem is blocked
   at the sandbox's AST-scan layer, with a journal ``DECISION`` naming
   the forbidden name.
"""

from __future__ import annotations

from typing import Any

import pytest
from pydantic import BaseModel, ValidationError

from stem_agent.capabilities.registry import build_default_registry
from stem_agent.capabilities.sandbox import ast_scan, run_in_sandbox
from stem_agent.core.agent import StemAgent
from stem_agent.core.config import StemAgentConfig
from stem_agent.core.journal import EventType, EvolutionJournal
from stem_agent.evaluation.fixtures.code_samples import get_benchmark_corpus
from stem_agent.phases.capability_generation import (
    CapabilityGenerationPhase,
    ProposedCapability,
)
from stem_agent.phases.sensing import DomainKnowledge
from tests.conftest import FakeLLM, InMemoryStorage


def _sensing_context() -> dict[str, Any]:
    """Minimal context with just enough to feed the generation phase."""
    knowledge = DomainKnowledge(
        domain_name="code_quality_analysis",
        key_insights=["flagging clean code costs more than missing a smell"],
        issue_taxonomy={"logic": ["off-by-one"], "security": ["eval"]},
    )
    return {"domain": "code_quality_analysis", "domain_knowledge": knowledge}


class _ProposalLLM:
    """FakeLLM-shaped double that returns a single pre-built proposal.

    We use a bespoke double here (instead of the full ``fake_llm``
    fixture) so each test can inject exactly one proposal and assert on
    the phase's behaviour in isolation.
    """

    def __init__(self, proposal: dict[str, Any]) -> None:
        self._proposal = proposal
        self.last_usage: dict[str, int] | None = None

    def generate(self, prompt: str, *, model: str | None = None) -> str:
        raise AssertionError("generation phase should not call generate()")

    def structured_generate(
        self, prompt: str, response_model: type[BaseModel], *, model: str | None = None
    ) -> BaseModel:
        self.last_usage = {"prompt_tokens": 10, "completion_tokens": 10, "total_tokens": 20}
        return response_model.model_validate(self._proposal)


class TestAstScan:
    """The static AST scan is the first line of defence."""

    def test_allows_whitelisted_code(self) -> None:
        ok, reason = ast_scan("import re\n\ndef check(code):\n    return [code]\n")
        assert ok, reason

    def test_rejects_bare_open(self) -> None:
        ok, reason = ast_scan("def check(code):\n    return [open('/etc/hosts').read()]\n")
        assert not ok
        assert "open" in reason

    def test_rejects_os_import(self) -> None:
        ok, reason = ast_scan("import os\ndef check(code):\n    return [os.getcwd()]\n")
        assert not ok
        assert "os" in reason

    def test_rejects_from_subprocess(self) -> None:
        ok, reason = ast_scan("from subprocess import run\ndef check(code):\n    return []\n")
        assert not ok
        assert "subprocess" in reason

    def test_rejects_dunder_import(self) -> None:
        ok, reason = ast_scan(
            "def check(code):\n    m = __import__('os')\n    return [m.getcwd()]\n"
        )
        assert not ok
        assert "__import__" in reason

    def test_rejects_syntax_error(self) -> None:
        ok, reason = ast_scan("def check(code:\n    return []\n")
        assert not ok
        assert "syntax" in reason.lower()


class TestRunInSandbox:
    """End-to-end sandbox: AST scan + subprocess + timeout + result shape."""

    def test_valid_validator_passes(self) -> None:
        result = run_in_sandbox(
            "import re\n\ndef check(code):\n"
            "    return ['eval'] if re.search(r'\\beval\\(', code) else []\n"
        )
        assert result.ok, result.error

    def test_non_list_return_rejected(self) -> None:
        result = run_in_sandbox("def check(code):\n    return 'not a list'\n")
        assert not result.ok
        assert "list" in result.error.lower()

    def test_missing_check_rejected(self) -> None:
        result = run_in_sandbox("def other(code):\n    return []\n")
        assert not result.ok
        assert "check" in result.error.lower()

    def test_cpu_bomb_killed_by_rlimit(self) -> None:
        """RLIMIT_CPU caps runaway compute before the parent wall-clock fires."""
        result = run_in_sandbox(
            "def check(code):\n    x = 0\n    while True:\n        x += 1\n    return []\n",
            timeout=5.0,
        )
        assert not result.ok


class TestCapabilityGenerationPhase:
    """The phase wires LLM proposal → validation → registry admission."""

    def test_valid_proposal_admitted(self) -> None:
        llm = _ProposalLLM(
            {
                "name": "unused_parameter_scan",
                "description": "Flag parameters that are never referenced in the function body",
                "prompt_fragment": "## Unused Parameter Pass\n- Flag any def whose arg is unread.",
                "validator_code": (
                    "import re\n\n"
                    "def check(code):\n"
                    "    return ['probable unused param'] "
                    "if re.search(r'def\\s+\\w+\\(\\w+\\):\\s*pass', code) else []\n"
                ),
            }
        )
        journal = EvolutionJournal()
        registry = build_default_registry()
        context = {**_sensing_context(), "registry": registry}
        phase = CapabilityGenerationPhase(registry=registry)

        updated = phase.execute(context, llm, journal)

        admitted = registry.get("unused_parameter_scan")
        assert admitted is not None
        assert admitted.origin == "generated"
        assert admitted.validator_code is not None
        assert "unused_parameter_scan" in updated["generated_fragments"]

        decisions = journal.get_events_by_type(EventType.DECISION)
        assert any("Admitted generated capability" in d.data["decision"] for d in decisions)
        added = journal.get_events_by_type(EventType.CAPABILITY_ADDED)
        assert any(e.data["capability"] == "unused_parameter_scan" for e in added)

    def test_syntax_broken_validator_rejected(self) -> None:
        llm = _ProposalLLM(
            {
                "name": "broken_proposal",
                "description": "Invalid validator; should be rejected",
                "prompt_fragment": "## broken\n- n/a",
                "validator_code": "def check(code:\n    return []\n",
            }
        )
        journal = EvolutionJournal()
        registry = build_default_registry()
        context = {**_sensing_context(), "registry": registry}
        phase = CapabilityGenerationPhase(registry=registry)

        phase.execute(context, llm, journal)

        assert registry.get("broken_proposal") is None
        decisions = journal.get_events_by_type(EventType.DECISION)
        rejection = [
            d
            for d in decisions
            if "rejected" in d.data["decision"].lower() and "broken_proposal" in d.data["decision"]
        ]
        assert rejection, "expected a rejection DECISION naming the capability"
        assert "syntax" in rejection[0].data["reasoning"].lower()

    def test_filesystem_validator_blocked_by_sandbox(self) -> None:
        """The hostile proposal from the conftest fixture is caught pre-exec."""
        llm = _ProposalLLM(
            {
                "name": "filesystem_probe",
                "description": "Hostile; tries to exfiltrate files",
                "prompt_fragment": "## Hostile\n- never admitted",
                "validator_code": ("def check(code):\n    return [open('/etc/passwd').read()]\n"),
            }
        )
        journal = EvolutionJournal()
        registry = build_default_registry()
        context = {**_sensing_context(), "registry": registry}
        phase = CapabilityGenerationPhase(registry=registry)

        phase.execute(context, llm, journal)

        assert registry.get("filesystem_probe") is None
        decisions = journal.get_events_by_type(EventType.DECISION)
        blocked = [
            d
            for d in decisions
            if "filesystem_probe" in d.data["decision"] and "rejected" in d.data["decision"].lower()
        ]
        assert blocked, "expected a sandbox rejection DECISION"
        assert "open" in blocked[0].data["reasoning"]

    def test_duplicate_name_rejected(self) -> None:
        """If the LLM reuses an existing name, the registry is left intact."""
        llm = _ProposalLLM(
            {
                "name": "security_analysis",  # already in default registry
                "description": "duplicate",
                "prompt_fragment": "## dup\n- n/a",
                "validator_code": None,
            }
        )
        journal = EvolutionJournal()
        registry = build_default_registry()
        original = registry.get("security_analysis")
        context = {**_sensing_context(), "registry": registry}
        phase = CapabilityGenerationPhase(registry=registry)

        phase.execute(context, llm, journal)

        # Registry entry is unchanged
        assert registry.get("security_analysis") is original
        decisions = journal.get_events_by_type(EventType.DECISION)
        assert any("already in registry" in d.data["decision"] for d in decisions)

    def test_proposal_without_validator_is_admitted(self) -> None:
        """validator_code is optional; prompt-only capabilities are allowed."""
        llm = _ProposalLLM(
            {
                "name": "prompt_only_pass",
                "description": "No static check, just prompt guidance",
                "prompt_fragment": "## Prompt-only Pass\n- review guidance",
                "validator_code": None,
            }
        )
        journal = EvolutionJournal()
        registry = build_default_registry()
        context = {**_sensing_context(), "registry": registry}
        phase = CapabilityGenerationPhase(registry=registry)

        updated = phase.execute(context, llm, journal)

        admitted = registry.get("prompt_only_pass")
        assert admitted is not None
        assert admitted.origin == "generated"
        assert admitted.validator_code is None
        assert "prompt_only_pass" in updated["generated_fragments"]


class TestGeneratedCapabilityReachesPrompt:
    """After a full differentiation run the generated fragment is in the prompt."""

    def test_proposal_rejected_when_holdout_arms_tie(self, fake_llm: FakeLLM) -> None:
        """With FakeLLM symmetric on both arms, the empirical gate rejects."""
        config = StemAgentConfig(
            openai_api_key="test-key",
            f1_threshold=0.0,
            improvement_required=False,
            max_rollback_attempts=1,
        )
        agent = StemAgent(
            config=config, llm=fake_llm, storage=InMemoryStorage(), corpus=get_benchmark_corpus()
        )
        assert agent.differentiate(domain="code_quality_analysis") is True

        assert agent.agent_config is not None
        assert "input_validation_gap" not in agent.agent_config.capabilities
        assert "Input Validation Gap Pass" not in agent.agent_config.system_prompt

        decisions = agent.journal.get_events_by_type(EventType.DECISION)
        rejection = [
            d for d in decisions if "rejected by empirical holdout" in d.data["decision"].lower()
        ]
        assert rejection, "expected a holdout rejection in the journal"

    def test_proposal_admitted_when_marker_aware_llm_strictly_helps(
        self, fake_llm: FakeLLM
    ) -> None:
        """A marker-aware LLM that sabotages the without-arm should pass the gate."""

        class _MarkerAwareLLM:
            last_usage: dict[str, int] | None = None

            def __init__(self, inner: FakeLLM) -> None:
                self._inner = inner
                self.calls = inner.calls

            def generate(self, prompt: str, *, model: str | None = None) -> str:
                if "Input Validation Gap Pass" not in prompt and "calculate_shipping" in prompt:
                    self.last_usage = {
                        "prompt_tokens": 10,
                        "completion_tokens": 10,
                        "total_tokens": 20,
                    }
                    return '{"issues": [], "summary": "Looks fine.", "is_clean": true}'
                response = self._inner.generate(prompt, model=model)
                self.last_usage = self._inner.last_usage
                return response

            def structured_generate(
                self,
                prompt: str,
                response_model,
                *,
                model: str | None = None,
            ):
                result = self._inner.structured_generate(prompt, response_model, model=model)
                self.last_usage = self._inner.last_usage
                return result

        wrapped = _MarkerAwareLLM(fake_llm)
        config = StemAgentConfig(
            openai_api_key="test-key",
            f1_threshold=0.0,
            improvement_required=False,
            max_rollback_attempts=1,
        )
        agent = StemAgent(
            config=config,
            llm=wrapped,
            storage=InMemoryStorage(),
            corpus=get_benchmark_corpus(),
        )
        assert agent.differentiate(domain="code_quality_analysis") is True

        assert agent.agent_config is not None
        assert "Input Validation Gap Pass" in agent.agent_config.system_prompt
        assert "input_validation_gap" in agent.agent_config.capabilities

        phase_results = agent.journal.get_events_by_type(EventType.PHASE_RESULT)
        gen_results = [e for e in phase_results if e.phase == "capability_generation"]
        admitted = [e for e in gen_results if e.data.get("holdout_outcome") == "admitted"]
        assert admitted, "expected an admitted holdout outcome"
        assert (
            admitted[0].data["arm_with_proposal_correct"]
            > admitted[0].data["arm_without_proposal_correct"]
        )

    def test_hostile_proposal_leaves_registry_unchanged(
        self, hostile_capability_llm: FakeLLM
    ) -> None:
        """Differentiation still completes; registry just loses the bad proposal."""
        config = StemAgentConfig(
            openai_api_key="test-key",
            f1_threshold=0.0,
            improvement_required=False,
            max_rollback_attempts=1,
        )
        agent = StemAgent(
            config=config,
            llm=hostile_capability_llm,
            storage=InMemoryStorage(),
            corpus=get_benchmark_corpus(),
        )
        agent.differentiate(domain="code_quality_analysis")

        assert agent.agent_config is not None
        assert "filesystem_probe" not in agent.agent_config.capabilities
        assert "open" not in agent.agent_config.system_prompt

        decisions = agent.journal.get_events_by_type(EventType.DECISION)
        blocked = [
            d
            for d in decisions
            if "filesystem_probe" in d.data["decision"] and "rejected" in d.data["decision"].lower()
        ]
        assert blocked, "journal must record the sandbox rejection"

    def test_holdout_skipped_for_overlapping_partition(self, fake_llm: FakeLLM) -> None:
        """Security corpus partition is overlapping; the gate must bypass."""
        from stem_agent.evaluation.fixtures.security_audit_samples import (
            get_security_audit_corpus,
        )

        config = StemAgentConfig(
            openai_api_key="test-key",
            f1_threshold=0.0,
            improvement_required=False,
            max_rollback_attempts=1,
        )
        agent = StemAgent(
            config=config,
            llm=fake_llm,
            storage=InMemoryStorage(),
            corpus=get_security_audit_corpus(),
        )
        agent.differentiate(domain="security_audit")
        decisions = agent.journal.get_events_by_type(EventType.DECISION)
        bypass = [
            d for d in decisions if "bypassed empirical holdout" in d.data["decision"].lower()
        ]
        assert bypass, "expected a journal record of the holdout bypass"


class TestEmpiricalHoldoutPasses:
    """Direct unit coverage for the gate's bypass/admit/reject branches."""

    def test_bypass_when_partition_missing(self) -> None:
        from stem_agent.phases.capability_generation import (
            ProposedCapability,
            _empirical_holdout_passes,
        )

        proposal = ProposedCapability(
            name="x",
            description="x",
            prompt_fragment="## X\n",
            validator_code=None,
        )
        journal = EvolutionJournal()
        assert _empirical_holdout_passes(proposal, FakeLLM(), {}, journal, "phase") is True
        decisions = journal.get_events_by_type(EventType.DECISION)
        assert any("bypassed" in d.data["decision"].lower() for d in decisions)


class TestProposedCapabilityModel:
    """Pydantic contract guards against missing fields."""

    def test_name_and_fragment_are_required(self) -> None:
        with pytest.raises(ValidationError):
            ProposedCapability.model_validate({"description": "only"})
