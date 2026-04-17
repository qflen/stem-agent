"""Property-based tests for StateMachine.

Unit tests pin specific transitions; properties pin the *invariants*
that should hold no matter which sequence a run takes. Hypothesis
searches the space of transition sequences looking for one that
breaks them.
"""

from __future__ import annotations

import contextlib

from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

from stem_agent.core.state_machine import (
    AgentState,
    GuardFailedError,
    InvalidTransitionError,
    StateMachine,
)

_PERMISSIVE_CONTEXT = {
    "specialized_f1": 1.0,
    "baseline_f1": 0.0,
    "f1_threshold": 0.0,
    "improvement_required": False,
    "rollback_count": 0,
    "max_rollback_attempts": 999,
}


def _all_targets() -> list[AgentState]:
    return list(AgentState)


_target_strategy = st.sampled_from(_all_targets())


@given(targets=st.lists(_target_strategy, max_size=30))
@settings(max_examples=200, suppress_health_check=[HealthCheck.function_scoped_fixture])
def test_current_state_matches_last_history_entry(targets: list[AgentState]) -> None:
    """After any sequence of attempted transitions, the current state is
    always the target of the most recent successful one (or UNDIFFERENTIATED
    if none ever succeeded)."""
    sm = StateMachine()
    last_target: AgentState | None = None
    for target in targets:
        try:
            sm.transition(target, _PERMISSIVE_CONTEXT)
        except (InvalidTransitionError, GuardFailedError):
            pass
        else:
            last_target = target

    if last_target is None:
        assert sm.state == AgentState.UNDIFFERENTIATED
        assert sm.history == []
    else:
        assert sm.state == last_target
        assert sm.history[-1][1] == last_target


@given(targets=st.lists(_target_strategy, max_size=30))
@settings(max_examples=200, suppress_health_check=[HealthCheck.function_scoped_fixture])
def test_rollback_count_is_monotonic(targets: list[AgentState]) -> None:
    """rollback_count only ever grows — never drops between transitions."""
    sm = StateMachine()
    prev = sm.rollback_count
    for target in targets:
        with contextlib.suppress(InvalidTransitionError, GuardFailedError):
            sm.transition(target, _PERMISSIVE_CONTEXT)
        assert sm.rollback_count >= prev
        prev = sm.rollback_count


@given(targets=st.lists(_target_strategy, min_size=1, max_size=30))
@settings(max_examples=200, suppress_health_check=[HealthCheck.function_scoped_fixture])
def test_failed_is_terminal(targets: list[AgentState]) -> None:
    """Once the machine enters FAILED, no further transition can succeed."""
    sm = StateMachine()
    entered_failed = False
    for target in targets:
        if entered_failed:
            with contextlib.suppress(InvalidTransitionError, GuardFailedError):
                sm.transition(target, _PERMISSIVE_CONTEXT)
            assert sm.state == AgentState.FAILED
            continue
        with contextlib.suppress(InvalidTransitionError, GuardFailedError):
            sm.transition(target, _PERMISSIVE_CONTEXT)
        if sm.state == AgentState.FAILED:
            entered_failed = True


@given(
    source=st.sampled_from(_all_targets()),
    target=st.sampled_from(_all_targets()),
)
def test_can_transition_matches_table_membership(source: AgentState, target: AgentState) -> None:
    """can_transition(target) is true iff (source, target) is in the table."""
    sm = StateMachine()
    sm._state = source
    expected = (source, target) in StateMachine.TRANSITION_TABLE
    assert sm.can_transition(target) is expected


@given(targets=st.lists(_target_strategy, max_size=30))
@settings(max_examples=200, suppress_health_check=[HealthCheck.function_scoped_fixture])
def test_history_length_equals_successful_transition_count(targets: list[AgentState]) -> None:
    """History grows by exactly one per successful transition attempt."""
    sm = StateMachine()
    successes = 0
    for target in targets:
        try:
            sm.transition(target, _PERMISSIVE_CONTEXT)
        except (InvalidTransitionError, GuardFailedError):
            pass
        else:
            successes += 1
        assert len(sm.history) == successes
