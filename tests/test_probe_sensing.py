"""Probe-grounded sensing; the prompt must show probes and their tool signals.

Pins the contract that ``SensingPhase`` consumes a partition's probe
slice, embeds the bodies in the prompt, and produces a ``DomainKnowledge``
with a ``tool_fit`` histogram that ``PlanningPhase`` can use to bias
capability ordering. The histogram is computed deterministically; the
LLM's only input is the prompt; so we can pin the resulting numbers
exactly.
"""

from __future__ import annotations

from stem_agent.capabilities.registry import build_default_registry
from stem_agent.core.journal import EvolutionJournal
from stem_agent.evaluation.fixtures.code_samples import (
    BenchmarkSample,
    get_benchmark_corpus,
    partition,
)
from stem_agent.phases.planning import PlanningPhase, _rank_capabilities
from stem_agent.phases.sensing import (
    DomainKnowledge,
    SensingPhase,
    compute_tool_fit,
)
from tests.conftest import FakeLLM


def _partition_seed_zero():
    return partition(get_benchmark_corpus(), seed=0)


class TestComputeToolFit:
    def test_no_samples_returns_empty_dict(self) -> None:
        assert compute_tool_fit([]) == {}

    def test_security_pattern_counts(self) -> None:
        sample = BenchmarkSample(
            sample_id="probe_sec",
            description="eval and hardcoded",
            code='API_KEY = "sk-abc"\ndef f(x):\n    return eval(x)\n',
            issue_categories=["security"],
        )
        fit = compute_tool_fit([sample])
        assert fit.get("security", 0) >= 1

    def test_structure_only_counts_once_per_probe(self) -> None:
        long_fn = "def big():\n" + "    x = 1\n" * 40
        sample = BenchmarkSample(
            sample_id="probe_struct",
            description="long function",
            code=long_fn,
            issue_categories=["structure"],
        )
        fit = compute_tool_fit([sample])
        assert fit.get("structure", 0) == 1


class TestSensingPromptEmbedsProbes:
    def test_prompt_includes_probe_bodies(self, fake_llm: FakeLLM) -> None:
        partition_obj = _partition_seed_zero()
        context = {
            "domain": "code_quality_analysis",
            "partition": partition_obj,
        }
        journal = EvolutionJournal()
        SensingPhase().execute(context, fake_llm, journal)
        last_prompt = fake_llm.calls[-1]["prompt"]
        for sample in partition_obj.probe:
            first_line = sample.code.splitlines()[0]
            assert first_line in last_prompt

    def test_phase_result_records_tool_fit(self, fake_llm: FakeLLM) -> None:
        partition_obj = _partition_seed_zero()
        context = {
            "domain": "code_quality_analysis",
            "partition": partition_obj,
        }
        journal = EvolutionJournal()
        SensingPhase().execute(context, fake_llm, journal)
        from stem_agent.core.journal import EventType

        results = journal.get_events_by_type(EventType.PHASE_RESULT)
        sensing_result = next(r for r in results if r.phase == "sensing")
        assert "tool_fit" in sensing_result.data
        assert isinstance(sensing_result.data["tool_fit"], dict)

    def test_domain_knowledge_carries_tool_fit(self, fake_llm: FakeLLM) -> None:
        partition_obj = _partition_seed_zero()
        context = {
            "domain": "code_quality_analysis",
            "partition": partition_obj,
        }
        journal = EvolutionJournal()
        result = SensingPhase().execute(context, fake_llm, journal)
        knowledge: DomainKnowledge = result["domain_knowledge"]
        # The deterministic compute_tool_fit must override whatever the LLM said.
        assert knowledge.tool_fit == compute_tool_fit(list(partition_obj.probe))


class TestPlanningCapabilityRanking:
    def test_no_tool_fit_preserves_order(self) -> None:
        registry = build_default_registry()
        original = registry.list_all()
        assert _rank_capabilities(original, {}) == original

    def test_security_heavy_fit_bumps_security_capability(self) -> None:
        registry = build_default_registry()
        ranked = _rank_capabilities(registry.list_all(), {"security": 5})
        names = [c.name for c in ranked]
        # security_analysis carries the "security" tag; should rank ahead of
        # untagged-for-security siblings like style_consistency.
        assert names.index("security_analysis") < names.index("style_consistency")

    def test_planning_prompt_includes_tool_fit_summary(self, fake_llm: FakeLLM) -> None:
        knowledge = DomainKnowledge(
            domain_name="code_quality_analysis",
            review_strategies=["multi-pass"],
            issue_taxonomy={"logic": ["off-by-one"]},
            tool_categories=["AST"],
            key_insights=["specificity"],
            tool_fit={"security": 4, "structure": 2},
        )
        context: dict = {
            "domain_knowledge": knowledge,
            "planning_model": "test-model",
        }
        journal = EvolutionJournal()
        PlanningPhase().execute(context, fake_llm, journal)
        last_prompt = fake_llm.calls[-1]["prompt"]
        assert "Tool-fit hits over probe slice" in last_prompt
        assert "security=4" in last_prompt
        assert "structure=2" in last_prompt
