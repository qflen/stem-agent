"""Evolution journal — append-only structured event log.

The journal is the agent's self-model: every state transition, LLM call,
metric measurement, and rollback reason is recorded. During rollback,
the agent reads its own journal to understand what went wrong.

Think of it as the agent's equivalent of a JetBrains inspection report.
"""

from __future__ import annotations

import enum
import hashlib
from datetime import UTC, datetime
from typing import Any

from pydantic import BaseModel, Field


def _sanitize_context(ctx: dict[str, Any]) -> dict[str, Any]:
    """Strip non-serializable objects from context for journal storage."""
    safe = {}
    for key, value in ctx.items():
        if isinstance(value, (str, int, float, bool, type(None))):
            safe[key] = value
        elif isinstance(value, (list, tuple, dict)):
            safe[key] = str(value)[:200]
        # Skip complex objects (CapabilityRegistry, ComparisonResult, etc.)
    return safe


class EventType(enum.Enum):
    """Categories of journal events."""

    STATE_TRANSITION = "state_transition"
    LLM_CALL = "llm_call"
    METRIC_MEASUREMENT = "metric_measurement"
    GUARD_FAILURE = "guard_failure"
    PHASE_RESULT = "phase_result"
    ROLLBACK_REASON = "rollback_reason"
    CAPABILITY_ADDED = "capability_added"
    DECISION = "decision"
    ERROR = "error"


class JournalEvent(BaseModel):
    """A single immutable event in the evolution journal."""

    timestamp: str = Field(default_factory=lambda: datetime.now(UTC).isoformat())
    event_type: EventType
    phase: str = ""
    data: dict[str, Any] = Field(default_factory=dict)

    def model_post_init(self, _context: Any) -> None:
        """Freeze after creation — events are append-only."""


class EvolutionJournal:
    """Append-only event log tracking the agent's differentiation process.

    Events cannot be mutated or removed once recorded. The journal can be
    serialized to and from a dictionary for persistence via the StoragePort.
    """

    def __init__(self) -> None:
        self._events: list[JournalEvent] = []

    @property
    def events(self) -> list[JournalEvent]:
        """Return a copy of all recorded events."""
        return list(self._events)

    def __len__(self) -> int:
        return len(self._events)

    def _append(self, event: JournalEvent) -> None:
        """Append an event. Internal — all public methods delegate here."""
        self._events.append(event)

    def log_transition(
        self,
        source: Any,
        target: Any,
        context: dict[str, Any] | None = None,
    ) -> None:
        """Record a state transition."""
        self._append(
            JournalEvent(
                event_type=EventType.STATE_TRANSITION,
                data={
                    "from": source.value if hasattr(source, "value") else str(source),
                    "to": target.value if hasattr(target, "value") else str(target),
                    "context": _sanitize_context(context or {}),
                },
            )
        )

    def log_llm_call(
        self,
        phase: str,
        model: str,
        prompt_hash: str,
        token_count: int | None = None,
        temperature: float = 0.0,
    ) -> None:
        """Record an LLM call with metadata (not the full prompt, for size)."""
        self._append(
            JournalEvent(
                event_type=EventType.LLM_CALL,
                phase=phase,
                data={
                    "model": model,
                    "prompt_hash": prompt_hash,
                    "token_count": token_count,
                    "temperature": temperature,
                },
            )
        )

    def log_metric(self, phase: str, metrics: dict[str, float]) -> None:
        """Record a metric measurement."""
        self._append(
            JournalEvent(
                event_type=EventType.METRIC_MEASUREMENT,
                phase=phase,
                data=metrics,
            )
        )

    def log_guard_failure(self, transition: str, guard: str, reason: str) -> None:
        """Record a guard predicate failure."""
        self._append(
            JournalEvent(
                event_type=EventType.GUARD_FAILURE,
                data={
                    "transition": transition,
                    "guard": guard,
                    "reason": reason,
                },
            )
        )

    def log_phase_result(self, phase: str, result: dict[str, Any]) -> None:
        """Record the output of a differentiation phase."""
        self._append(
            JournalEvent(
                event_type=EventType.PHASE_RESULT,
                phase=phase,
                data=result,
            )
        )

    def log_rollback_reason(self, reason: str, adjustments: list[str]) -> None:
        """Record why a rollback was triggered and what adjustments were planned."""
        self._append(
            JournalEvent(
                event_type=EventType.ROLLBACK_REASON,
                data={
                    "reason": reason,
                    "adjustments": adjustments,
                },
            )
        )

    def log_capability_added(self, capability: str, reason: str) -> None:
        """Record a capability being added during specialization."""
        self._append(
            JournalEvent(
                event_type=EventType.CAPABILITY_ADDED,
                data={"capability": capability, "reason": reason},
            )
        )

    def log_decision(self, phase: str, decision: str, reasoning: str) -> None:
        """Record an architectural or strategic decision with reasoning."""
        self._append(
            JournalEvent(
                event_type=EventType.DECISION,
                phase=phase,
                data={"decision": decision, "reasoning": reasoning},
            )
        )

    def log_error(self, phase: str, error: str, details: str = "") -> None:
        """Record an error that occurred during a phase."""
        self._append(
            JournalEvent(
                event_type=EventType.ERROR,
                phase=phase,
                data={"error": error, "details": details},
            )
        )

    def get_events_by_type(self, event_type: EventType) -> list[JournalEvent]:
        """Filter events by type."""
        return [e for e in self._events if e.event_type == event_type]

    def get_events_by_phase(self, phase: str) -> list[JournalEvent]:
        """Filter events by phase."""
        return [e for e in self._events if e.phase == phase]

    def to_dict(self) -> dict[str, Any]:
        """Serialize the entire journal for persistence."""
        return {
            "events": [e.model_dump(mode="json") for e in self._events],
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> EvolutionJournal:
        """Reconstruct a journal from persisted data."""
        journal = cls()
        for event_data in data.get("events", []):
            event = JournalEvent(**event_data)
            journal._events.append(event)
        return journal

    @staticmethod
    def hash_prompt(prompt: str) -> str:
        """Create a deterministic hash of a prompt for logging without storing the full text."""
        return hashlib.sha256(prompt.encode()).hexdigest()[:16]
