"""Tests for the evolution journal; append-only event log.

Validates: append-only semantics, serialization round-trips,
and correct metadata capture on LLM calls.
"""

from __future__ import annotations

from stem_agent.core.journal import EventType, EvolutionJournal
from stem_agent.core.state_machine import AgentState


class TestAppendOnlySemantics:
    """Events cannot be mutated or removed once recorded."""

    def test_event_count_increases_monotonically(self, journal: EvolutionJournal) -> None:
        assert len(journal) == 0
        journal.log_decision("test", "decision 1", "reason 1")
        assert len(journal) == 1
        journal.log_decision("test", "decision 2", "reason 2")
        assert len(journal) == 2

    def test_events_property_returns_copy(self, journal: EvolutionJournal) -> None:
        """Modifying the returned list does not affect the journal."""
        journal.log_decision("test", "decision", "reason")
        events = journal.events
        events.clear()
        assert len(journal) == 1

    def test_past_events_unchanged_after_new_append(self, journal: EvolutionJournal) -> None:
        journal.log_decision("phase1", "first", "reason1")
        first_event = journal.events[0]

        journal.log_decision("phase2", "second", "reason2")

        # First event is unchanged
        assert journal.events[0].data == first_event.data
        assert journal.events[0].phase == first_event.phase


class TestSerializationRoundTrip:
    """Journal serializes and deserializes perfectly."""

    def test_empty_journal_round_trips(self) -> None:
        original = EvolutionJournal()
        data = original.to_dict()
        restored = EvolutionJournal.from_dict(data)
        assert len(restored) == 0

    def test_populated_journal_round_trips(self, journal: EvolutionJournal) -> None:
        journal.log_transition(AgentState.UNDIFFERENTIATED, AgentState.SENSING, {})
        journal.log_llm_call("sensing", "gpt-4o-mini", "abc123", token_count=500)
        journal.log_metric("validation", {"f1": 0.75, "precision": 0.8})
        journal.log_decision("planning", "chose multi-pass", "better coverage")
        journal.log_capability_added("security_analysis", "top-3 issue category")
        journal.log_error("specialization", "parse error", "invalid JSON")
        journal.log_guard_failure("v→s", "f1_threshold", "0.3 < 0.6")
        journal.log_rollback_reason("low F1", ["reduce aggressiveness"])

        data = journal.to_dict()
        restored = EvolutionJournal.from_dict(data)

        assert len(restored) == len(journal)

        for original, restored_event in zip(journal.events, restored.events, strict=True):
            assert original.event_type == restored_event.event_type
            assert original.phase == restored_event.phase
            assert original.data == restored_event.data

    def test_event_types_preserved_across_serialization(self, journal: EvolutionJournal) -> None:
        journal.log_transition(AgentState.SENSING, AgentState.DIFFERENTIATING)
        journal.log_metric("test", {"score": 0.9})

        data = journal.to_dict()
        restored = EvolutionJournal.from_dict(data)

        assert restored.events[0].event_type == EventType.STATE_TRANSITION
        assert restored.events[1].event_type == EventType.METRIC_MEASUREMENT


class TestLLMCallMetadata:
    """LLM call events capture the right metadata."""

    def test_captures_model_and_hash(self, journal: EvolutionJournal) -> None:
        journal.log_llm_call(
            phase="sensing",
            model="gpt-4o-mini",
            prompt_hash="a1b2c3d4",
            token_count=1024,
            temperature=0.0,
        )
        event = journal.events[0]
        assert event.data["model"] == "gpt-4o-mini"
        assert event.data["prompt_hash"] == "a1b2c3d4"
        assert event.data["token_count"] == 1024
        assert event.data["temperature"] == 0.0

    def test_prompt_hash_is_deterministic(self) -> None:
        hash1 = EvolutionJournal.hash_prompt("test prompt")
        hash2 = EvolutionJournal.hash_prompt("test prompt")
        assert hash1 == hash2

    def test_different_prompts_produce_different_hashes(self) -> None:
        hash1 = EvolutionJournal.hash_prompt("prompt A")
        hash2 = EvolutionJournal.hash_prompt("prompt B")
        assert hash1 != hash2

    def test_hash_is_truncated_to_16_chars(self) -> None:
        h = EvolutionJournal.hash_prompt("any prompt")
        assert len(h) == 16


class TestEventFiltering:
    """Events can be filtered by type and phase."""

    def test_filter_by_type(self, journal: EvolutionJournal) -> None:
        journal.log_decision("p1", "d1", "r1")
        journal.log_metric("p1", {"f1": 0.5})
        journal.log_decision("p2", "d2", "r2")

        decisions = journal.get_events_by_type(EventType.DECISION)
        assert len(decisions) == 2
        assert all(e.event_type == EventType.DECISION for e in decisions)

    def test_filter_by_phase(self, journal: EvolutionJournal) -> None:
        journal.log_decision("sensing", "d1", "r1")
        journal.log_decision("planning", "d2", "r2")
        journal.log_decision("sensing", "d3", "r3")

        sensing_events = journal.get_events_by_phase("sensing")
        assert len(sensing_events) == 2

    def test_filter_returns_empty_for_no_matches(self, journal: EvolutionJournal) -> None:
        journal.log_decision("sensing", "d1", "r1")
        assert journal.get_events_by_type(EventType.ERROR) == []
        assert journal.get_events_by_phase("nonexistent") == []


class TestTimestamps:
    def test_events_have_timestamps(self, journal: EvolutionJournal) -> None:
        journal.log_decision("test", "decision", "reason")
        assert journal.events[0].timestamp
        # Should be ISO format
        assert "T" in journal.events[0].timestamp


class TestTotalTokens:
    def test_empty_journal_has_zero_tokens(self, journal: EvolutionJournal) -> None:
        assert journal.total_tokens == 0

    def test_sums_across_llm_calls(self, journal: EvolutionJournal) -> None:
        journal.log_llm_call("sensing", "gpt-4o-mini", "h1", token_count=120)
        journal.log_llm_call("planning", "gpt-4o-mini", "h2", token_count=340)
        journal.log_llm_call("validation", "gpt-4o", "h3", token_count=800)
        assert journal.total_tokens == 1260

    def test_ignores_llm_calls_without_token_count(self, journal: EvolutionJournal) -> None:
        journal.log_llm_call("sensing", "gpt-4o-mini", "h1", token_count=100)
        journal.log_llm_call("planning", "gpt-4o-mini", "h2")  # token_count=None
        assert journal.total_tokens == 100

    def test_ignores_non_llm_events(self, journal: EvolutionJournal) -> None:
        journal.log_llm_call("sensing", "gpt-4o-mini", "h1", token_count=50)
        journal.log_decision("planning", "d", "r")
        journal.log_metric("validation", {"f1": 0.9})
        assert journal.total_tokens == 50
