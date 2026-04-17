"""Tests for each differentiation phase in isolation with FakeLLM.

Each phase is tested as a standalone unit: given known inputs,
does it produce the expected outputs and journal entries?
"""

from __future__ import annotations

from stem_agent.capabilities.registry import build_default_registry
from stem_agent.core.journal import EventType, EvolutionJournal
from stem_agent.evaluation.benchmark import SampleVerdict
from stem_agent.evaluation.comparator import ComparisonResult
from stem_agent.evaluation.fixtures.code_samples import BenchmarkSample
from stem_agent.evaluation.metrics import ClassificationMetrics
from stem_agent.phases.planning import PlanningPhase, SpecializationPlan
from stem_agent.phases.sensing import DomainKnowledge, SensingPhase
from stem_agent.phases.specialization import SpecializationPhase, SpecializedAgentConfig
from stem_agent.phases.validation import (
    ValidationPhase,
    cross_check_verdicts,
    diagnose_failure,
)
from tests.conftest import FakeLLM


class TestSensingPhase:
    """Sensing phase: queries LLM and parses domain knowledge."""

    def test_produces_domain_knowledge(self, fake_llm: FakeLLM, journal: EvolutionJournal) -> None:
        phase = SensingPhase()
        context = {"domain": "code_quality_analysis"}
        result = phase.execute(context, fake_llm, journal)

        assert "domain_knowledge" in result
        knowledge = result["domain_knowledge"]
        assert isinstance(knowledge, DomainKnowledge)
        assert knowledge.domain_name == "code_quality_analysis"

    def test_domain_knowledge_has_required_fields(
        self, fake_llm: FakeLLM, journal: EvolutionJournal
    ) -> None:
        phase = SensingPhase()
        result = phase.execute({"domain": "code_quality_analysis"}, fake_llm, journal)
        knowledge: DomainKnowledge = result["domain_knowledge"]

        assert len(knowledge.review_strategies) > 0
        assert len(knowledge.issue_taxonomy) > 0
        assert len(knowledge.tool_categories) > 0
        assert len(knowledge.key_insights) > 0

    def test_logs_to_journal(self, fake_llm: FakeLLM, journal: EvolutionJournal) -> None:
        phase = SensingPhase()
        phase.execute({"domain": "code_quality_analysis"}, fake_llm, journal)

        # Should have logged: decision, LLM call, phase result
        decisions = journal.get_events_by_type(EventType.DECISION)
        llm_calls = journal.get_events_by_type(EventType.LLM_CALL)
        results = journal.get_events_by_type(EventType.PHASE_RESULT)

        assert len(decisions) >= 1
        assert len(llm_calls) == 1
        assert len(results) == 1
        assert results[0].phase == "sensing"

    def test_uses_planning_model(self, fake_llm: FakeLLM, journal: EvolutionJournal) -> None:
        phase = SensingPhase()
        phase.execute(
            {"domain": "code_quality_analysis", "planning_model": "gpt-4o-mini"},
            fake_llm,
            journal,
        )

        assert fake_llm.calls[-1]["model"] == "gpt-4o-mini"

    def test_records_token_count_from_llm_usage(
        self, fake_llm: FakeLLM, journal: EvolutionJournal
    ) -> None:
        phase = SensingPhase()
        phase.execute({"domain": "code_quality_analysis"}, fake_llm, journal)

        llm_call = journal.get_events_by_type(EventType.LLM_CALL)[0]
        assert isinstance(llm_call.data["token_count"], int)
        assert llm_call.data["token_count"] > 0

    def test_handles_missing_domain_name(self, journal: EvolutionJournal) -> None:
        """If the LLM doesn't return domain_name, it's filled in from context."""
        llm = FakeLLM(
            structured_responses={
                "default": {
                    "domain_name": "",
                    "review_strategies": ["strategy"],
                    "issue_taxonomy": {"bugs": ["off-by-one"]},
                    "tool_categories": ["ast"],
                    "output_format_patterns": ["json"],
                    "key_insights": ["insight"],
                }
            }
        )
        phase = SensingPhase()
        result = phase.execute({"domain": "my_domain"}, llm, journal)

        assert result["domain_knowledge"].domain_name == "my_domain"


class TestPlanningPhase:
    """Planning phase: produces a specialization plan from domain knowledge."""

    def _make_context_with_knowledge(self) -> dict:
        return {
            "domain_knowledge": DomainKnowledge(
                domain_name="code_quality_analysis",
                review_strategies=["multi-pass", "severity-based"],
                issue_taxonomy={"logic": ["off-by-one"], "security": ["injection"]},
                tool_categories=["AST analysis", "regex"],
                output_format_patterns=["JSON"],
                key_insights=["specificity matters"],
            ),
        }

    def test_produces_specialization_plan(
        self, fake_llm: FakeLLM, journal: EvolutionJournal
    ) -> None:
        phase = PlanningPhase()
        context = self._make_context_with_knowledge()
        result = phase.execute(context, fake_llm, journal)

        assert "specialization_plan" in result
        plan = result["specialization_plan"]
        assert isinstance(plan, SpecializationPlan)

    def test_selected_capabilities_are_valid(
        self, fake_llm: FakeLLM, journal: EvolutionJournal
    ) -> None:
        """All selected capabilities must exist in the registry."""
        registry = build_default_registry()
        phase = PlanningPhase(registry=registry)
        context = self._make_context_with_knowledge()
        result = phase.execute(context, fake_llm, journal)
        plan: SpecializationPlan = result["specialization_plan"]

        for cap_name in plan.selected_capabilities:
            assert registry.get(cap_name) is not None, f"Unknown capability: {cap_name}"

    def test_filters_hallucinated_capabilities(self, journal: EvolutionJournal) -> None:
        """Capabilities not in the registry are filtered out."""
        llm = FakeLLM(
            structured_responses={
                "default": {
                    "selected_capabilities": ["structural_analysis", "hallucinated_cap"],
                    "review_passes": [
                        {
                            "pass_name": "structural",
                            "focus_area": "structure",
                            "capability_name": "structural_analysis",
                            "priority": 1,
                        },
                    ],
                    "evaluation_criteria": {"f1_threshold": 0.6},
                    "domain_insights_for_prompt": "insights",
                    "reasoning": "reason",
                }
            }
        )
        phase = PlanningPhase()
        context = self._make_context_with_knowledge()
        result = phase.execute(context, llm, journal)
        plan: SpecializationPlan = result["specialization_plan"]

        assert "structural_analysis" in plan.selected_capabilities
        assert "hallucinated_cap" not in plan.selected_capabilities

    def test_logs_capability_additions(self, fake_llm: FakeLLM, journal: EvolutionJournal) -> None:
        phase = PlanningPhase()
        context = self._make_context_with_knowledge()
        phase.execute(context, fake_llm, journal)

        cap_events = journal.get_events_by_type(EventType.CAPABILITY_ADDED)
        assert len(cap_events) > 0


class TestSpecializationPhase:
    """Specialization phase: assembles the specialized agent config."""

    def _make_context_with_plan(self) -> dict:
        return {
            "specialization_plan": SpecializationPlan(
                selected_capabilities=[
                    "structural_analysis",
                    "logic_correctness",
                    "security_analysis",
                ],
                review_passes=[
                    {
                        "pass_name": "structural",
                        "focus_area": "structure",
                        "capability_name": "structural_analysis",
                        "priority": 1,
                    },
                    {
                        "pass_name": "logic",
                        "focus_area": "logic",
                        "capability_name": "logic_correctness",
                        "priority": 2,
                    },
                ],
                evaluation_criteria={"f1_threshold": 0.6},
                domain_insights_for_prompt="Multi-pass analysis improves accuracy.",
                reasoning="Standard code review pipeline.",
            ),
            "registry": build_default_registry(),
            "execution_model": "gpt-4o",
        }

    def test_produces_agent_config(self, fake_llm: FakeLLM, journal: EvolutionJournal) -> None:
        phase = SpecializationPhase()
        context = self._make_context_with_plan()
        result = phase.execute(context, fake_llm, journal)

        assert "agent_config" in result
        config = result["agent_config"]
        assert isinstance(config, SpecializedAgentConfig)

    def test_system_prompt_includes_capability_fragments(
        self, fake_llm: FakeLLM, journal: EvolutionJournal
    ) -> None:
        phase = SpecializationPhase()
        context = self._make_context_with_plan()
        result = phase.execute(context, fake_llm, journal)
        config: SpecializedAgentConfig = result["agent_config"]

        # System prompt should contain fragments from selected capabilities
        prompt_lower = config.system_prompt.lower()
        assert "structural analysis" in prompt_lower or "structural" in prompt_lower
        assert "logic correctness" in prompt_lower or "logic" in prompt_lower

    def test_system_prompt_includes_domain_insights(
        self, fake_llm: FakeLLM, journal: EvolutionJournal
    ) -> None:
        phase = SpecializationPhase()
        context = self._make_context_with_plan()
        result = phase.execute(context, fake_llm, journal)
        config: SpecializedAgentConfig = result["agent_config"]

        assert "Multi-pass analysis" in config.system_prompt

    def test_applies_rollback_adjustments(
        self, fake_llm: FakeLLM, journal: EvolutionJournal
    ) -> None:
        """When rollback adjustments exist, they're added to the prompt."""
        phase = SpecializationPhase()
        context = self._make_context_with_plan()
        context["rollback_adjustments"] = [
            "Reduce false positive rate",
            "Add confidence thresholds",
        ]
        result = phase.execute(context, fake_llm, journal)
        config: SpecializedAgentConfig = result["agent_config"]

        assert "Reduce false positive rate" in config.system_prompt
        assert "confidence thresholds" in config.system_prompt

    def test_preserves_review_passes(self, fake_llm: FakeLLM, journal: EvolutionJournal) -> None:
        phase = SpecializationPhase()
        context = self._make_context_with_plan()
        result = phase.execute(context, fake_llm, journal)
        config: SpecializedAgentConfig = result["agent_config"]

        assert len(config.review_passes) == 2
        assert "structural" in config.review_passes


class TestValidationPhase:
    """Validation phase: runs benchmark and computes metrics."""

    def test_produces_comparison_metrics(
        self, fake_llm: FakeLLM, journal: EvolutionJournal, small_corpus
    ) -> None:
        config = SpecializedAgentConfig(
            system_prompt="Review this code.",
            capabilities=["logic_correctness"],
            review_passes=["logic"],
        )

        phase = ValidationPhase(corpus=small_corpus)
        context = {"agent_config": config}
        result = phase.execute(context, fake_llm, journal)

        assert "comparison" in result
        assert "baseline_f1" in result
        assert "specialized_f1" in result
        assert isinstance(result["comparison"], ComparisonResult)

    def test_logs_metrics_to_journal(
        self, fake_llm: FakeLLM, journal: EvolutionJournal, small_corpus
    ) -> None:
        config = SpecializedAgentConfig(system_prompt="Review.", capabilities=[])
        phase = ValidationPhase(corpus=small_corpus)
        phase.execute({"agent_config": config}, fake_llm, journal)

        metric_events = journal.get_events_by_type(EventType.METRIC_MEASUREMENT)
        assert len(metric_events) >= 2  # baseline + specialized

    def test_logs_llm_calls_with_token_counts(
        self, fake_llm: FakeLLM, journal: EvolutionJournal, small_corpus
    ) -> None:
        """Each review call should land in the journal with a non-None token_count."""
        config = SpecializedAgentConfig(system_prompt="Review.", capabilities=[])
        phase = ValidationPhase(corpus=small_corpus)
        phase.execute({"agent_config": config}, fake_llm, journal)

        llm_calls = journal.get_events_by_type(EventType.LLM_CALL)
        # Baseline + specialized, one per sample in the small corpus
        assert len(llm_calls) == 2 * len(small_corpus)
        assert all(isinstance(c.data["token_count"], int) for c in llm_calls)
        assert journal.total_tokens > 0


class TestCrossCheckVerdicts:
    """The deterministic static-analysis cross-check flags two narrow cases."""

    def _verdict(self, sample_id: str, detected: list[str]) -> SampleVerdict:
        return SampleVerdict(
            sample_id=sample_id,
            detected_categories=set(detected),
            is_clean_detected=not detected,
            ground_truth_categories=set(),
            is_clean_truth=False,
        )

    def test_flags_structure_false_positive(self, journal: EvolutionJournal) -> None:
        sample = BenchmarkSample(
            sample_id="tiny",
            description="tiny well-structured function",
            code="def add(a: int, b: int) -> int:\n    return a + b\n",
            issue_categories=[],
            is_clean=True,
        )
        verdict = self._verdict("tiny", ["structure"])
        disagreements = cross_check_verdicts([verdict], [sample], journal)

        assert len(disagreements) == 1
        assert disagreements[0]["kind"] == "llm_flagged_structure_but_ast_clean"
        decisions = journal.get_events_by_type(EventType.DECISION)
        assert any("tiny" in d.data["decision"] for d in decisions)

    def test_flags_missed_security_pattern(self, journal: EvolutionJournal) -> None:
        sample = BenchmarkSample(
            sample_id="hardcoded",
            description="hardcoded key",
            code='API_KEY = "sk-abc123"\n',
            issue_categories=["security"],
        )
        verdict = self._verdict("hardcoded", ["structure"])  # LLM missed security
        disagreements = cross_check_verdicts([verdict], [sample], journal)

        kinds = {d["kind"] for d in disagreements}
        assert "scanner_found_security_pattern_llm_missed" in kinds

    def test_agreement_produces_no_disagreements(self, journal: EvolutionJournal) -> None:
        sample = BenchmarkSample(
            sample_id="clean",
            description="clean code",
            code="def greet(name: str) -> str:\n    return f'Hello, {name}!'\n",
            issue_categories=[],
            is_clean=True,
        )
        verdict = self._verdict("clean", [])  # LLM says clean — matches
        disagreements = cross_check_verdicts([verdict], [sample], journal)

        assert disagreements == []

    def test_unknown_sample_id_is_skipped(self, journal: EvolutionJournal) -> None:
        verdict = self._verdict("missing", ["structure"])
        disagreements = cross_check_verdicts([verdict], [], journal)
        assert disagreements == []


class TestDiagnoseFailure:
    """diagnose_failure produces actionable adjustment suggestions."""

    def test_low_precision_suggests_reducing_false_positives(
        self, journal: EvolutionJournal
    ) -> None:
        context = {
            "comparison": ComparisonResult(
                baseline=ClassificationMetrics(5, 1, 3, 2),
                specialized=ClassificationMetrics(4, 5, 1, 3),  # Worse precision
            ),
        }
        adjustments = diagnose_failure(context, journal)
        assert any("false positive" in a.lower() for a in adjustments)

    def test_low_recall_suggests_improving_detection(self, journal: EvolutionJournal) -> None:
        context = {
            "comparison": ComparisonResult(
                baseline=ClassificationMetrics(5, 1, 3, 2),
                specialized=ClassificationMetrics(3, 0, 5, 4),  # Worse recall
            ),
        }
        adjustments = diagnose_failure(context, journal)
        assert any("detection" in a.lower() for a in adjustments)

    def test_no_improvement_suggests_simplification(self, journal: EvolutionJournal) -> None:
        baseline = ClassificationMetrics(5, 1, 3, 2)
        context = {
            "comparison": ComparisonResult(
                baseline=baseline,
                specialized=baseline,  # Same as baseline
            ),
        }
        adjustments = diagnose_failure(context, journal)
        assert any("simplify" in a.lower() or "baseline" in a.lower() for a in adjustments)

    def test_logs_rollback_reason(self, journal: EvolutionJournal) -> None:
        context = {
            "comparison": ComparisonResult(
                baseline=ClassificationMetrics(5, 1, 3, 2),
                specialized=ClassificationMetrics(3, 3, 2, 4),
            ),
        }
        diagnose_failure(context, journal)

        rollback_events = journal.get_events_by_type(EventType.ROLLBACK_REASON)
        assert len(rollback_events) == 1
        assert rollback_events[0].data["adjustments"]
