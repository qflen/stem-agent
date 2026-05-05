"""K=3 rollback splicing; bounded-prompt-length contract.

Each rollback attempt's adjustments are spliced into the next composed
prompt; without a cap, the prompt would grow linearly with the rollback
count. ``SpecializationPhase`` keeps the latest 3 attempts in full and
collapses older ones to a one-line summary, so the prompt size is
bounded across long chains.
"""

from __future__ import annotations

from stem_agent.capabilities.registry import build_default_registry
from stem_agent.core.journal import EvolutionJournal
from stem_agent.phases.planning import SpecializationPlan
from stem_agent.phases.specialization import SpecializationPhase
from tests.conftest import FakeLLM


def _plan() -> SpecializationPlan:
    return SpecializationPlan(
        selected_capabilities=["structural_analysis", "logic_correctness"],
        review_passes=[
            {
                "pass_name": "structural",
                "focus_area": "structure",
                "capability_name": "structural_analysis",
                "priority": 1,
            },
        ],
        evaluation_criteria={"f1_threshold": 0.6},
        domain_insights_for_prompt="Multi-pass review.",
        reasoning="Standard pipeline.",
    )


def _context_with_history(history: list[dict]) -> dict:
    return {
        "specialization_plan": _plan(),
        "registry": build_default_registry(),
        "execution_model": "gpt-4o",
        "rollback_history": history,
    }


def _attempt(idx: int, *, full: int = 1, short_summary: bool = True) -> dict:
    """Build a synthetic attempt entry with ``full`` adjustment lines."""
    return {
        "attempt_idx": idx,
        "adjustments": [f"attempt-{idx} adjustment {i}" for i in range(full)],
        "summary": f"attempt-{idx} summary" if short_summary else "",
    }


class TestEmptyHistory:
    def test_no_history_omits_marker(self) -> None:
        context = {
            "specialization_plan": _plan(),
            "registry": build_default_registry(),
            "execution_model": "gpt-4o",
        }
        result = SpecializationPhase().execute(context, FakeLLM(), EvolutionJournal())
        assert "IMPORTANT adjustments" not in result["agent_config"].system_prompt


class TestRecentSplicedInFull:
    def test_one_attempt_full_text_present(self) -> None:
        context = _context_with_history([_attempt(0, full=2)])
        result = SpecializationPhase().execute(context, FakeLLM(), EvolutionJournal())
        prompt = result["agent_config"].system_prompt
        assert "attempt-0 adjustment 0" in prompt
        assert "attempt-0 adjustment 1" in prompt

    def test_three_recent_all_full(self) -> None:
        history = [_attempt(i, full=2) for i in range(3)]
        context = _context_with_history(history)
        prompt = (
            SpecializationPhase()
            .execute(context, FakeLLM(), EvolutionJournal())["agent_config"]
            .system_prompt
        )
        for i in range(3):
            assert f"attempt-{i} adjustment 0" in prompt
            assert f"attempt-{i} adjustment 1" in prompt


class TestOlderCollapsedToSummary:
    def test_attempt_zero_collapses_at_depth_four(self) -> None:
        history = [_attempt(i, full=2) for i in range(4)]
        context = _context_with_history(history)
        prompt = (
            SpecializationPhase()
            .execute(context, FakeLLM(), EvolutionJournal())["agent_config"]
            .system_prompt
        )
        # Earliest attempt collapses: only summary line, not the full adjustments.
        assert "attempt-0 summary" in prompt
        assert "attempt-0 adjustment 0" not in prompt
        # The most recent three are still full.
        for i in (1, 2, 3):
            assert f"attempt-{i} adjustment 0" in prompt


class TestPromptLengthBounded:
    def test_six_attempts_no_runaway_growth(self) -> None:
        """6-attempt prompt must not grow much past the 3-attempt prompt."""
        baseline_history = [_attempt(i, full=2) for i in range(3)]
        baseline_prompt = (
            SpecializationPhase()
            .execute(_context_with_history(baseline_history), FakeLLM(), EvolutionJournal())[
                "agent_config"
            ]
            .system_prompt
        )

        big_history = [_attempt(i, full=2) for i in range(6)]
        big_prompt = (
            SpecializationPhase()
            .execute(_context_with_history(big_history), FakeLLM(), EvolutionJournal())[
                "agent_config"
            ]
            .system_prompt
        )

        # Three additional collapsed summary lines; each ~30 chars. The bound
        # has plenty of headroom; what matters is that it isn't 2× the baseline.
        max_expected = len(baseline_prompt) + 3 * 200
        assert len(big_prompt) <= max_expected, (
            f"prompt size grew unboundedly: baseline={len(baseline_prompt)}, "
            f"6-attempt={len(big_prompt)}, allowed={max_expected}"
        )

    def test_each_attempt_summary_shows_in_collapsed_block(self) -> None:
        history = [_attempt(i, full=2) for i in range(6)]
        context = _context_with_history(history)
        prompt = (
            SpecializationPhase()
            .execute(context, FakeLLM(), EvolutionJournal())["agent_config"]
            .system_prompt
        )
        for i in range(3):
            assert f"earlier attempt {i}" in prompt


class TestLegacyAdjustmentsPath:
    def test_legacy_adjustments_still_spliced(self) -> None:
        """When tests pre-date task 12 and pass rollback_adjustments only, splice still fires."""
        context = {
            "specialization_plan": _plan(),
            "registry": build_default_registry(),
            "execution_model": "gpt-4o",
            "rollback_adjustments": ["legacy: be conservative"],
        }
        prompt = (
            SpecializationPhase()
            .execute(context, FakeLLM(), EvolutionJournal())["agent_config"]
            .system_prompt
        )
        assert "legacy: be conservative" in prompt
        assert "IMPORTANT adjustments" in prompt
