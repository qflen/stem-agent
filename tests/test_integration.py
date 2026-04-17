"""Integration tests — the crown jewel.

Full differentiation pipeline with FakeLLM → deterministic, fast, no API calls.
Verifies the agent transitions through all states, the journal contains
expected events, and the specialized agent produces correct output structure.
"""

from __future__ import annotations

import pytest

from stem_agent.core.agent import StemAgent
from stem_agent.core.config import StemAgentConfig
from stem_agent.core.journal import EventType
from stem_agent.core.state_machine import AgentState
from stem_agent.evaluation.fixtures.code_samples import get_benchmark_corpus
from tests.conftest import FakeLLM


@pytest.fixture
def config() -> StemAgentConfig:
    return StemAgentConfig(
        openai_api_key="test-key",
        f1_threshold=0.3,  # Lower threshold for testing with FakeLLM
        improvement_required=False,  # FakeLLM may not always improve over baseline
        max_rollback_attempts=3,
    )


class TestFullDifferentiationPipeline:
    """End-to-end differentiation: UNDIFFERENTIATED → SPECIALIZED."""

    def test_successful_differentiation(self, config: StemAgentConfig, fake_llm: FakeLLM) -> None:
        corpus = get_benchmark_corpus()
        agent = StemAgent(config=config, llm=fake_llm, corpus=corpus)

        assert agent.state == AgentState.UNDIFFERENTIATED
        success = agent.differentiate(domain="code_quality_analysis")

        assert success is True
        assert agent.state == AgentState.SPECIALIZED

    def test_transitions_through_all_states(
        self, config: StemAgentConfig, fake_llm: FakeLLM
    ) -> None:
        corpus = get_benchmark_corpus()
        agent = StemAgent(config=config, llm=fake_llm, corpus=corpus)
        agent.differentiate(domain="code_quality_analysis")

        journal = agent.journal
        transitions = journal.get_events_by_type(EventType.STATE_TRANSITION)

        states_visited = set()
        for t in transitions:
            states_visited.add(t.data["from"])
            states_visited.add(t.data["to"])

        # Must have visited these core states
        assert "undifferentiated" in states_visited
        assert "sensing" in states_visited
        assert "differentiating" in states_visited
        assert "validating" in states_visited
        assert "specialized" in states_visited

    def test_journal_contains_expected_event_types(
        self, config: StemAgentConfig, fake_llm: FakeLLM
    ) -> None:
        corpus = get_benchmark_corpus()
        agent = StemAgent(config=config, llm=fake_llm, corpus=corpus)
        agent.differentiate(domain="code_quality_analysis")

        journal = agent.journal

        assert len(journal.get_events_by_type(EventType.STATE_TRANSITION)) > 0
        assert len(journal.get_events_by_type(EventType.DECISION)) > 0
        assert len(journal.get_events_by_type(EventType.LLM_CALL)) > 0
        assert len(journal.get_events_by_type(EventType.METRIC_MEASUREMENT)) > 0
        assert len(journal.get_events_by_type(EventType.PHASE_RESULT)) > 0
        assert len(journal.get_events_by_type(EventType.CAPABILITY_ADDED)) > 0


class TestSpecializedAgentOutput:
    """After specialization, the agent produces structured review results."""

    def test_review_returns_structured_result(
        self, config: StemAgentConfig, fake_llm: FakeLLM
    ) -> None:
        corpus = get_benchmark_corpus()
        agent = StemAgent(config=config, llm=fake_llm, corpus=corpus)
        agent.differentiate(domain="code_quality_analysis")

        code = "def foo():\n    return 1 + 1"
        result = agent.review(code)

        assert "issues" in result
        assert "summary" in result
        assert "is_clean" in result

    def test_review_before_differentiation_raises(
        self, config: StemAgentConfig, fake_llm: FakeLLM
    ) -> None:
        agent = StemAgent(config=config, llm=fake_llm)
        with pytest.raises(RuntimeError, match="not been specialized"):
            agent.review("def foo(): pass")


class TestRollbackMechanism:
    """Rollback triggers when validation fails and agent adapts."""

    def test_rollback_triggered_on_poor_results(self, poor_fake_llm: FakeLLM) -> None:
        """When FakeLLM produces poor results, rollback should trigger."""
        config = StemAgentConfig(
            openai_api_key="test-key",
            f1_threshold=0.95,  # Impossibly high threshold → forces rollback
            improvement_required=True,
            max_rollback_attempts=2,
        )
        corpus = get_benchmark_corpus()
        agent = StemAgent(config=config, llm=poor_fake_llm, corpus=corpus)

        # Should fail after exhausting rollback budget
        success = agent.differentiate(domain="code_quality_analysis")
        assert success is False
        assert agent.state == AgentState.FAILED

    def test_rollback_logged_in_journal(self, poor_fake_llm: FakeLLM) -> None:
        config = StemAgentConfig(
            openai_api_key="test-key",
            f1_threshold=0.95,
            improvement_required=True,
            max_rollback_attempts=1,
        )
        corpus = get_benchmark_corpus()
        agent = StemAgent(config=config, llm=poor_fake_llm, corpus=corpus)
        agent.differentiate(domain="code_quality_analysis")

        journal = agent.journal
        rollback_events = journal.get_events_by_type(EventType.ROLLBACK_REASON)
        guard_failures = journal.get_events_by_type(EventType.GUARD_FAILURE)

        assert len(rollback_events) > 0
        assert len(guard_failures) > 0

    def test_max_rollback_attempts_respected(self, poor_fake_llm: FakeLLM) -> None:
        """Agent stops after max_rollback_attempts, not infinitely."""
        config = StemAgentConfig(
            openai_api_key="test-key",
            f1_threshold=0.99,
            improvement_required=True,
            max_rollback_attempts=2,
        )
        corpus = get_benchmark_corpus()
        agent = StemAgent(config=config, llm=poor_fake_llm, corpus=corpus)
        agent.differentiate(domain="code_quality_analysis")

        journal = agent.journal
        transitions = journal.get_events_by_type(EventType.STATE_TRANSITION)
        rollback_transitions = [t for t in transitions if t.data.get("to") == "rollback"]
        # Should not exceed max_rollback_attempts
        assert len(rollback_transitions) <= 2


class TestJournalPersistence:
    """Journal round-trips through serialization."""

    def test_journal_serialization_after_differentiation(
        self, config: StemAgentConfig, fake_llm: FakeLLM
    ) -> None:
        corpus = get_benchmark_corpus()
        agent = StemAgent(config=config, llm=fake_llm, corpus=corpus)
        agent.differentiate(domain="code_quality_analysis")

        from stem_agent.core.journal import EvolutionJournal

        data = agent.journal.to_dict()
        restored = EvolutionJournal.from_dict(data)

        assert len(restored) == len(agent.journal)
        for orig, rest in zip(agent.journal.events, restored.events, strict=True):
            assert orig.event_type == rest.event_type
            assert orig.phase == rest.phase


class TestMultiDomainSpecialization:
    """Proving the pipeline isn't hard-wired to code_quality_analysis."""

    def test_multi_domain_specialization(self, fake_llm: FakeLLM) -> None:
        """Differentiate for two distinct domains and assert the specialisations differ.

        The FakeLLM returns a broad six-capability plan for
        ``code_quality_analysis`` and a narrower two-capability
        (security + severity) plan for ``security_audit``. The agents
        should end up with visibly different prompts and capability sets.

        This test asserts architectural generalisation, not benchmark
        quality — the thresholds are pinned low so FakeLLM's generic
        review responses don't gate the assertion we actually care about.
        """
        from stem_agent.evaluation.fixtures.security_audit_samples import (
            get_security_audit_corpus,
        )

        permissive = StemAgentConfig(
            openai_api_key="test-key",
            f1_threshold=0.0,
            improvement_required=False,
            max_rollback_attempts=1,
        )

        cq_agent = StemAgent(config=permissive, llm=fake_llm, corpus=get_benchmark_corpus())
        assert cq_agent.differentiate(domain="code_quality_analysis") is True
        cq_config = cq_agent.agent_config
        assert cq_config is not None

        sec_agent = StemAgent(config=permissive, llm=fake_llm, corpus=get_security_audit_corpus())
        assert sec_agent.differentiate(domain="security_audit") is True
        sec_config = sec_agent.agent_config
        assert sec_config is not None

        # Security specialisation is strictly narrower than code-quality.
        assert set(sec_config.capabilities) < set(cq_config.capabilities)
        assert "security_analysis" in sec_config.capabilities
        assert "structural_analysis" not in sec_config.capabilities
        assert "performance_analysis" not in sec_config.capabilities

        # And the composed prompts pick up different domain-specific language.
        assert "security" in sec_config.system_prompt.lower()
        assert "out of scope" in sec_config.system_prompt.lower()
        assert sec_config.system_prompt != cq_config.system_prompt


class TestBenchmarkCorpus:
    """The benchmark corpus itself is well-formed."""

    def test_corpus_has_20_samples(self) -> None:
        corpus = get_benchmark_corpus()
        assert len(corpus) == 20

    def test_corpus_distribution(self) -> None:
        """Verify the category distribution matches the specification."""
        corpus = get_benchmark_corpus()
        buggy = [s for s in corpus if not s.is_clean]
        clean = [s for s in corpus if s.is_clean]
        assert len(buggy) == 15
        assert len(clean) == 5

    def test_all_buggy_samples_have_categories(self) -> None:
        corpus = get_benchmark_corpus()
        for sample in corpus:
            if not sample.is_clean:
                assert len(sample.issue_categories) > 0, (
                    f"Sample {sample.sample_id} has no issue categories"
                )

    def test_all_clean_samples_have_no_categories(self) -> None:
        corpus = get_benchmark_corpus()
        for sample in corpus:
            if sample.is_clean:
                assert len(sample.issue_categories) == 0, (
                    f"Clean sample {sample.sample_id} has issue categories"
                )

    def test_all_samples_have_code(self) -> None:
        corpus = get_benchmark_corpus()
        for sample in corpus:
            assert len(sample.code.strip()) > 0, f"Sample {sample.sample_id} has no code"

    def test_sample_ids_are_unique(self) -> None:
        corpus = get_benchmark_corpus()
        ids = [s.sample_id for s in corpus]
        assert len(ids) == len(set(ids))
