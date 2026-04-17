"""State machine governing the agent lifecycle.

Transitions have guard predicates — quantitative conditions that must be
satisfied before a transition is allowed. This is not decorative: calling
a phase out of order raises InvalidTransitionError.
"""

from __future__ import annotations

import enum
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from stem_agent.core.journal import EvolutionJournal


class AgentState(enum.Enum):
    """States in the agent lifecycle."""

    UNDIFFERENTIATED = "undifferentiated"
    SENSING = "sensing"
    DIFFERENTIATING = "differentiating"
    VALIDATING = "validating"
    ROLLBACK = "rollback"
    SPECIALIZED = "specialized"
    EXECUTING = "executing"
    FAILED = "failed"


class InvalidTransitionError(Exception):
    """Raised when a state transition is not permitted."""

    def __init__(self, current: AgentState, target: AgentState, reason: str = "") -> None:
        self.current = current
        self.target = target
        self.reason = reason
        msg = f"Cannot transition from {current.value} to {target.value}"
        if reason:
            msg += f": {reason}"
        super().__init__(msg)


class GuardFailedError(Exception):
    """Raised when a transition's guard predicate is not satisfied."""

    def __init__(self, transition: str, guard_name: str, details: str = "") -> None:
        self.transition = transition
        self.guard_name = guard_name
        self.details = details
        msg = f"Guard '{guard_name}' failed for transition {transition}"
        if details:
            msg += f": {details}"
        super().__init__(msg)


# Type alias for guard predicates: they receive context and return (passed, reason)
GuardPredicate = Callable[[dict[str, Any]], tuple[bool, str]]


@dataclass
class Transition:
    """A permitted state transition with optional guard predicates."""

    source: AgentState
    target: AgentState
    guards: list[tuple[str, GuardPredicate]] = field(default_factory=list)

    @property
    def name(self) -> str:
        return f"{self.source.value} → {self.target.value}"


def _build_transition_table() -> dict[tuple[AgentState, AgentState], Transition]:
    """Build the permitted transition table.

    Returns:
        Mapping from (source, target) to Transition objects.
    """
    transitions = [
        Transition(AgentState.UNDIFFERENTIATED, AgentState.SENSING),
        Transition(AgentState.SENSING, AgentState.DIFFERENTIATING),
        Transition(AgentState.DIFFERENTIATING, AgentState.VALIDATING),
        Transition(
            AgentState.VALIDATING,
            AgentState.SPECIALIZED,
            guards=[
                ("f1_above_threshold", _guard_f1_threshold),
                ("improvement_over_baseline", _guard_improvement),
            ],
        ),
        Transition(
            AgentState.VALIDATING,
            AgentState.ROLLBACK,
            guards=[
                ("rollback_budget_remaining", _guard_rollback_budget),
            ],
        ),
        Transition(AgentState.ROLLBACK, AgentState.DIFFERENTIATING),
        Transition(AgentState.SPECIALIZED, AgentState.EXECUTING),
        # Failure from validation when rollback budget exhausted
        Transition(AgentState.VALIDATING, AgentState.FAILED),
    ]
    return {(t.source, t.target): t for t in transitions}


def _guard_f1_threshold(ctx: dict[str, Any]) -> tuple[bool, str]:
    """Guard: specialized F1 must meet the configured threshold."""
    f1 = ctx.get("specialized_f1", 0.0)
    threshold = ctx.get("f1_threshold", 0.6)
    passed = f1 >= threshold
    reason = f"F1={f1:.3f} {'≥' if passed else '<'} threshold={threshold:.3f}"
    return passed, reason


def _guard_improvement(ctx: dict[str, Any]) -> tuple[bool, str]:
    """Guard: specialized F1 must exceed baseline F1 (no regression)."""
    if not ctx.get("improvement_required", True):
        return True, "improvement check disabled"
    specialized = ctx.get("specialized_f1", 0.0)
    baseline = ctx.get("baseline_f1", 0.0)
    passed = specialized > baseline
    reason = f"specialized_F1={specialized:.3f} {'>' if passed else '≤'} baseline_F1={baseline:.3f}"
    return passed, reason


def _guard_rollback_budget(ctx: dict[str, Any]) -> tuple[bool, str]:
    """Guard: rollback attempts must not exceed the maximum."""
    attempts = ctx.get("rollback_count", 0)
    max_attempts = ctx.get("max_rollback_attempts", 3)
    passed = attempts < max_attempts
    reason = f"rollback_count={attempts} {'<' if passed else '≥'} max={max_attempts}"
    return passed, reason


class StateMachine:
    """Manages the agent lifecycle with enforced transitions and guard predicates.

    The state machine is not decorative — attempting an invalid transition
    raises an exception. Guards are evaluated with provided context and all
    must pass for a guarded transition to proceed.
    """

    TRANSITION_TABLE = _build_transition_table()

    def __init__(self, journal: EvolutionJournal | None = None) -> None:
        self._state = AgentState.UNDIFFERENTIATED
        self._rollback_count = 0
        self._journal = journal
        self._history: list[tuple[AgentState, AgentState]] = []

    @property
    def state(self) -> AgentState:
        return self._state

    @property
    def rollback_count(self) -> int:
        return self._rollback_count

    @property
    def history(self) -> list[tuple[AgentState, AgentState]]:
        return list(self._history)

    def can_transition(self, target: AgentState) -> bool:
        """Check if a transition to the target state is structurally permitted."""
        return (self._state, target) in self.TRANSITION_TABLE

    def get_valid_targets(self) -> list[AgentState]:
        """Return all states reachable from the current state."""
        return [target for (source, target) in self.TRANSITION_TABLE if source == self._state]

    def transition(self, target: AgentState, context: dict[str, Any] | None = None) -> None:
        """Execute a state transition, evaluating all guard predicates.

        Args:
            target: The desired next state.
            context: Data for guard predicate evaluation (metrics, counts, etc.).

        Raises:
            InvalidTransitionError: If the transition is not in the table.
            GuardFailedError: If any guard predicate fails.
        """
        ctx = context or {}
        key = (self._state, target)
        transition = self.TRANSITION_TABLE.get(key)

        if transition is None:
            raise InvalidTransitionError(self._state, target)

        # Evaluate all guards — every one must pass
        for guard_name, guard_fn in transition.guards:
            passed, reason = guard_fn(ctx)
            if not passed:
                if self._journal is not None:
                    self._journal.log_guard_failure(
                        transition=transition.name,
                        guard=guard_name,
                        reason=reason,
                    )
                raise GuardFailedError(transition.name, guard_name, reason)

        previous = self._state
        self._state = target
        self._history.append((previous, target))

        # Track rollback count
        if target == AgentState.ROLLBACK:
            self._rollback_count += 1

        if self._journal is not None:
            self._journal.log_transition(
                source=previous,
                target=target,
                context=ctx,
            )

    def reset(self) -> None:
        """Reset the state machine to UNDIFFERENTIATED. Preserves history."""
        self._state = AgentState.UNDIFFERENTIATED
        self._rollback_count = 0
