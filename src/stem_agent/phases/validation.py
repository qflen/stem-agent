"""Validation phase — benchmark-driven fitness check with regression gates.

Runs the full benchmark suite through both the undifferentiated baseline
and the specialized agent, computes metrics, and evaluates guard conditions.
This is where the QA rigor lives.
"""

from __future__ import annotations

from typing import Any

from stem_agent.core.journal import EvolutionJournal
from stem_agent.evaluation.benchmark import (
    ReviewFunction,
    make_llm_review_fn,
    run_benchmark,
)
from stem_agent.evaluation.comparator import ComparisonResult
from stem_agent.evaluation.fixtures.code_samples import BenchmarkSample
from stem_agent.phases.specialization import SpecializedAgentConfig
from stem_agent.ports.llm import LLMPort


class ValidationPhase:
    """Benchmark-driven validation with regression gates.

    Runs both baseline and specialized agents against the benchmark corpus,
    computes metrics, and returns results for guard evaluation by the
    state machine.
    """

    def __init__(self, corpus: list[BenchmarkSample] | None = None) -> None:
        self._corpus = corpus

    @property
    def name(self) -> str:
        return "validation"

    def execute(
        self,
        context: dict[str, Any],
        llm: LLMPort,
        journal: EvolutionJournal,
    ) -> dict[str, Any]:
        """Run benchmark evaluation and compute comparison metrics.

        Args:
            context: Must contain 'agent_config' from specialization phase.
            llm: LLM adapter for running reviews.
            journal: For logging metrics and results.

        Returns:
            Context updated with 'validation_result', 'comparison',
            'baseline_f1', 'specialized_f1'.
        """
        agent_config: SpecializedAgentConfig = context["agent_config"]

        journal.log_decision(
            phase=self.name,
            decision="Starting benchmark evaluation",
            reasoning="Running both baseline and specialized agents "
            "against the full benchmark corpus to compute comparative metrics",
        )

        # Build review functions
        review_fn: ReviewFunction
        if context.get("review_fn"):
            # Allow injection for testing
            review_fn = context["review_fn"]
        else:
            review_fn = make_llm_review_fn(llm, model=agent_config.model)

        baseline_fn: ReviewFunction
        if context.get("baseline_fn"):
            baseline_fn = context["baseline_fn"]
        else:
            baseline_fn = make_llm_review_fn(llm, model=context.get("planning_model"))

        # Run baseline
        journal.log_decision(
            phase=self.name,
            decision="Running baseline evaluation",
            reasoning="Undifferentiated agent with generic prompt — this is the control",
        )
        baseline_verdicts, baseline_metrics = run_benchmark(
            baseline_fn,
            agent_config.baseline_prompt,
            corpus=self._corpus,
        )

        journal.log_metric(
            phase=self.name,
            metrics={
                "baseline_precision": baseline_metrics.precision,
                "baseline_recall": baseline_metrics.recall,
                "baseline_f1": baseline_metrics.f1,
                "baseline_specificity": baseline_metrics.specificity,
            },
        )

        # Run specialized
        journal.log_decision(
            phase=self.name,
            decision="Running specialized evaluation",
            reasoning="Specialized agent with composed prompt from differentiation",
        )
        specialized_verdicts, specialized_metrics = run_benchmark(
            review_fn,
            agent_config.system_prompt,
            corpus=self._corpus,
        )

        journal.log_metric(
            phase=self.name,
            metrics={
                "specialized_precision": specialized_metrics.precision,
                "specialized_recall": specialized_metrics.recall,
                "specialized_f1": specialized_metrics.f1,
                "specialized_specificity": specialized_metrics.specificity,
            },
        )

        # Compare
        comparison = ComparisonResult(
            baseline=baseline_metrics,
            specialized=specialized_metrics,
        )

        journal.log_phase_result(
            phase=self.name,
            result=comparison.summary(),
        )

        context["validation_result"] = {
            "baseline_verdicts": [
                {
                    "sample_id": v.sample_id,
                    "detected": list(v.detected_categories),
                    "truth": list(v.ground_truth_categories),
                    "clean_detected": v.is_clean_detected,
                    "clean_truth": v.is_clean_truth,
                }
                for v in baseline_verdicts
            ],
            "specialized_verdicts": [
                {
                    "sample_id": v.sample_id,
                    "detected": list(v.detected_categories),
                    "truth": list(v.ground_truth_categories),
                    "clean_detected": v.is_clean_detected,
                    "clean_truth": v.is_clean_truth,
                }
                for v in specialized_verdicts
            ],
        }
        context["comparison"] = comparison
        context["baseline_f1"] = baseline_metrics.f1
        context["specialized_f1"] = specialized_metrics.f1
        return context


def diagnose_failure(
    context: dict[str, Any],
    journal: EvolutionJournal,
) -> list[str]:
    """Analyze validation failure and suggest adjustments for rollback.

    Reads the validation results and comparison to determine what went wrong
    and how the specialization plan should be adjusted.

    Returns:
        List of adjustment strings to apply on the next specialization attempt.
    """
    comparison: ComparisonResult = context["comparison"]
    adjustments: list[str] = []

    # Check what specifically failed
    if comparison.specialized.precision < comparison.baseline.precision:
        adjustments.append(
            "Reduce false positive rate — add instruction to only flag issues "
            "when confident, avoid flagging clean/correct code"
        )

    if comparison.specialized.recall < comparison.baseline.recall:
        adjustments.append(
            "Improve issue detection — ensure all review passes are thorough "
            "and cover the full range of issue types in the taxonomy"
        )

    if comparison.specialized.specificity < comparison.baseline.specificity:
        adjustments.append(
            "Improve clean code recognition — add explicit instruction to verify "
            "that flagged issues are real, not false alarms on correct patterns"
        )

    if comparison.f1_delta <= 0:
        adjustments.append(
            "Specialization did not improve over baseline — simplify the prompt "
            "to reduce confusion and focus on the most impactful review passes"
        )

    if not adjustments:
        adjustments.append(
            "Metrics below threshold but no specific degradation pattern identified — "
            "try more targeted review passes with clearer instructions"
        )

    reason = (
        f"Validation failed: specialized_F1={comparison.specialized.f1:.3f}, "
        f"baseline_F1={comparison.baseline.f1:.3f}, "
        f"delta={comparison.f1_delta:+.3f}"
    )
    journal.log_rollback_reason(reason=reason, adjustments=adjustments)

    return adjustments
