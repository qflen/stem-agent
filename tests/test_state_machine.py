"""Tests for the state machine — transition logic, guards, rollback.

Tests behavior, not structure. A QA automation engineer should read
these and see: "they tested the paths that actually matter."
"""

from __future__ import annotations

import pytest

from stem_agent.core.journal import EvolutionJournal
from stem_agent.core.state_machine import (
    AgentState,
    GuardFailedError,
    InvalidTransitionError,
    StateMachine,
)


class TestValidTransitions:
    """Every permitted transition in the lifecycle works correctly."""

    def test_undifferentiated_to_sensing(self, state_machine: StateMachine) -> None:
        state_machine.transition(AgentState.SENSING)
        assert state_machine.state == AgentState.SENSING

    def test_sensing_to_differentiating(self, state_machine: StateMachine) -> None:
        state_machine.transition(AgentState.SENSING)
        state_machine.transition(AgentState.DIFFERENTIATING)
        assert state_machine.state == AgentState.DIFFERENTIATING

    def test_differentiating_to_validating(self, state_machine: StateMachine) -> None:
        state_machine.transition(AgentState.SENSING)
        state_machine.transition(AgentState.DIFFERENTIATING)
        state_machine.transition(AgentState.VALIDATING)
        assert state_machine.state == AgentState.VALIDATING

    def test_validating_to_specialized_when_guards_pass(self, state_machine: StateMachine) -> None:
        state_machine.transition(AgentState.SENSING)
        state_machine.transition(AgentState.DIFFERENTIATING)
        state_machine.transition(AgentState.VALIDATING)
        state_machine.transition(
            AgentState.SPECIALIZED,
            context={"specialized_f1": 0.8, "baseline_f1": 0.5, "f1_threshold": 0.6},
        )
        assert state_machine.state == AgentState.SPECIALIZED

    def test_specialized_to_executing(self, state_machine: StateMachine) -> None:
        state_machine.transition(AgentState.SENSING)
        state_machine.transition(AgentState.DIFFERENTIATING)
        state_machine.transition(AgentState.VALIDATING)
        state_machine.transition(
            AgentState.SPECIALIZED,
            context={"specialized_f1": 0.8, "baseline_f1": 0.5, "f1_threshold": 0.6},
        )
        state_machine.transition(AgentState.EXECUTING)
        assert state_machine.state == AgentState.EXECUTING

    def test_full_happy_path_records_history(self, state_machine: StateMachine) -> None:
        """The complete lifecycle is recorded in history."""
        state_machine.transition(AgentState.SENSING)
        state_machine.transition(AgentState.DIFFERENTIATING)
        state_machine.transition(AgentState.VALIDATING)
        state_machine.transition(
            AgentState.SPECIALIZED,
            context={"specialized_f1": 0.8, "baseline_f1": 0.5},
        )
        state_machine.transition(AgentState.EXECUTING)

        assert len(state_machine.history) == 5
        assert state_machine.history[0] == (AgentState.UNDIFFERENTIATED, AgentState.SENSING)
        assert state_machine.history[-1] == (AgentState.SPECIALIZED, AgentState.EXECUTING)


class TestInvalidTransitions:
    """Every forbidden transition is rejected."""

    def test_cannot_skip_sensing(self, state_machine: StateMachine) -> None:
        with pytest.raises(InvalidTransitionError) as exc_info:
            state_machine.transition(AgentState.DIFFERENTIATING)
        assert exc_info.value.current == AgentState.UNDIFFERENTIATED
        assert exc_info.value.target == AgentState.DIFFERENTIATING

    def test_cannot_go_backwards_from_sensing(self, state_machine: StateMachine) -> None:
        state_machine.transition(AgentState.SENSING)
        with pytest.raises(InvalidTransitionError):
            state_machine.transition(AgentState.UNDIFFERENTIATED)

    def test_cannot_skip_to_specialized(self, state_machine: StateMachine) -> None:
        state_machine.transition(AgentState.SENSING)
        with pytest.raises(InvalidTransitionError):
            state_machine.transition(AgentState.SPECIALIZED)

    def test_cannot_execute_without_specializing(self, state_machine: StateMachine) -> None:
        state_machine.transition(AgentState.SENSING)
        with pytest.raises(InvalidTransitionError):
            state_machine.transition(AgentState.EXECUTING)

    def test_cannot_rollback_from_sensing(self, state_machine: StateMachine) -> None:
        state_machine.transition(AgentState.SENSING)
        with pytest.raises(InvalidTransitionError):
            state_machine.transition(AgentState.ROLLBACK)


class TestGuardPredicates:
    """Guard predicates enforce quantitative criteria on transitions."""

    def _reach_validating(self, sm: StateMachine) -> None:
        sm.transition(AgentState.SENSING)
        sm.transition(AgentState.DIFFERENTIATING)
        sm.transition(AgentState.VALIDATING)

    def test_f1_below_threshold_blocks_specialization(self, state_machine: StateMachine) -> None:
        self._reach_validating(state_machine)
        with pytest.raises(GuardFailedError) as exc_info:
            state_machine.transition(
                AgentState.SPECIALIZED,
                context={"specialized_f1": 0.3, "baseline_f1": 0.2, "f1_threshold": 0.6},
            )
        assert "f1_above_threshold" in exc_info.value.guard_name

    def test_no_improvement_blocks_specialization(self, state_machine: StateMachine) -> None:
        """Specialized F1 must exceed baseline F1 (regression gate)."""
        self._reach_validating(state_machine)
        with pytest.raises(GuardFailedError) as exc_info:
            state_machine.transition(
                AgentState.SPECIALIZED,
                context={
                    "specialized_f1": 0.7,
                    "baseline_f1": 0.8,
                    "f1_threshold": 0.6,
                    "improvement_required": True,
                },
            )
        assert "improvement_over_baseline" in exc_info.value.guard_name

    def test_improvement_check_can_be_disabled(self, state_machine: StateMachine) -> None:
        """When improvement_required=False, regression is allowed."""
        self._reach_validating(state_machine)
        state_machine.transition(
            AgentState.SPECIALIZED,
            context={
                "specialized_f1": 0.7,
                "baseline_f1": 0.8,
                "f1_threshold": 0.6,
                "improvement_required": False,
            },
        )
        assert state_machine.state == AgentState.SPECIALIZED

    def test_exact_threshold_passes(self, state_machine: StateMachine) -> None:
        """F1 exactly equal to threshold should pass."""
        self._reach_validating(state_machine)
        state_machine.transition(
            AgentState.SPECIALIZED,
            context={"specialized_f1": 0.6, "baseline_f1": 0.5, "f1_threshold": 0.6},
        )
        assert state_machine.state == AgentState.SPECIALIZED


class TestRollback:
    """Rollback mechanism works correctly and respects the budget."""

    def _reach_validating(self, sm: StateMachine) -> None:
        sm.transition(AgentState.SENSING)
        sm.transition(AgentState.DIFFERENTIATING)
        sm.transition(AgentState.VALIDATING)

    def test_rollback_increments_count(self, state_machine: StateMachine) -> None:
        self._reach_validating(state_machine)
        assert state_machine.rollback_count == 0

        state_machine.transition(
            AgentState.ROLLBACK,
            context={"rollback_count": 0, "max_rollback_attempts": 3},
        )
        assert state_machine.rollback_count == 1
        assert state_machine.state == AgentState.ROLLBACK

    def test_rollback_returns_to_differentiating(self, state_machine: StateMachine) -> None:
        self._reach_validating(state_machine)
        state_machine.transition(
            AgentState.ROLLBACK,
            context={"rollback_count": 0, "max_rollback_attempts": 3},
        )
        state_machine.transition(AgentState.DIFFERENTIATING)
        assert state_machine.state == AgentState.DIFFERENTIATING

    def test_max_rollback_enforced(self, state_machine: StateMachine) -> None:
        """Cannot rollback when budget is exhausted."""
        self._reach_validating(state_machine)
        with pytest.raises(GuardFailedError) as exc_info:
            state_machine.transition(
                AgentState.ROLLBACK,
                context={"rollback_count": 3, "max_rollback_attempts": 3},
            )
        assert "rollback_budget_remaining" in exc_info.value.guard_name

    def test_multiple_rollback_cycles(self, journal: EvolutionJournal) -> None:
        """Agent can go through multiple rollback cycles within budget."""
        sm = StateMachine(journal=journal)
        sm.transition(AgentState.SENSING)

        for i in range(3):
            sm.transition(AgentState.DIFFERENTIATING)
            sm.transition(AgentState.VALIDATING)
            sm.transition(
                AgentState.ROLLBACK,
                context={"rollback_count": i, "max_rollback_attempts": 3},
            )

        assert sm.rollback_count == 3

        # Can still differentiate after rollback
        sm.transition(AgentState.DIFFERENTIATING)
        sm.transition(AgentState.VALIDATING)

        # But budget is now exhausted
        with pytest.raises(GuardFailedError):
            sm.transition(
                AgentState.ROLLBACK,
                context={"rollback_count": 3, "max_rollback_attempts": 3},
            )

    def test_rollback_preserves_journal_history(self, journal: EvolutionJournal) -> None:
        """Journal events from before rollback are preserved."""
        sm = StateMachine(journal=journal)
        sm.transition(AgentState.SENSING)
        sm.transition(AgentState.DIFFERENTIATING)
        sm.transition(AgentState.VALIDATING)

        events_before = len(journal)

        sm.transition(
            AgentState.ROLLBACK,
            context={"rollback_count": 0, "max_rollback_attempts": 3},
        )

        # Journal grew — no events were lost
        assert len(journal) > events_before

    def test_failed_state_on_exhausted_budget(self, journal: EvolutionJournal) -> None:
        """Agent transitions to FAILED when rollback budget is exhausted."""
        sm = StateMachine(journal=journal)
        sm.transition(AgentState.SENSING)
        sm.transition(AgentState.DIFFERENTIATING)
        sm.transition(AgentState.VALIDATING)

        # Budget exhausted — can't rollback
        with pytest.raises(GuardFailedError):
            sm.transition(
                AgentState.ROLLBACK,
                context={"rollback_count": 3, "max_rollback_attempts": 3},
            )

        # But can transition to FAILED
        sm.transition(AgentState.FAILED)
        assert sm.state == AgentState.FAILED


class TestReset:
    def test_reset_returns_to_undifferentiated(self, state_machine: StateMachine) -> None:
        state_machine.transition(AgentState.SENSING)
        state_machine.reset()
        assert state_machine.state == AgentState.UNDIFFERENTIATED

    def test_reset_clears_rollback_count(self, state_machine: StateMachine) -> None:
        state_machine.transition(AgentState.SENSING)
        state_machine.transition(AgentState.DIFFERENTIATING)
        state_machine.transition(AgentState.VALIDATING)
        state_machine.transition(
            AgentState.ROLLBACK,
            context={"rollback_count": 0, "max_rollback_attempts": 3},
        )
        state_machine.reset()
        assert state_machine.rollback_count == 0


class TestStateAccessors:
    def test_get_valid_targets_from_undifferentiated(self, state_machine: StateMachine) -> None:
        targets = state_machine.get_valid_targets()
        assert AgentState.SENSING in targets
        assert AgentState.DIFFERENTIATING not in targets

    def test_can_transition_returns_true_for_valid(self, state_machine: StateMachine) -> None:
        assert state_machine.can_transition(AgentState.SENSING) is True

    def test_can_transition_returns_false_for_invalid(self, state_machine: StateMachine) -> None:
        assert state_machine.can_transition(AgentState.EXECUTING) is False
