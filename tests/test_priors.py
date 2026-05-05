"""Cross-run priors; graduation-rate weights across saved journals.

Pins: empty store yields empty dict; one-success / mixed-history runs
match the Laplace formula; domain-filter excludes other-domain journals;
malformed journals are skipped without crashing; planning consults the
weights via ``_rank_capabilities``.
"""

from __future__ import annotations

from typing import Any

from stem_agent.capabilities.registry import build_default_registry
from stem_agent.core.priors import (
    NEUTRAL_WEIGHT,
    weight_capabilities,
    weight_for,
)
from stem_agent.phases.planning import _rank_capabilities
from tests.conftest import InMemoryStorage


def _journal_with(domain: str, capabilities: list[str], graduated: bool) -> dict[str, Any]:
    """Build the smallest serialised journal the priors module needs."""
    events: list[dict[str, Any]] = [
        {
            "event_type": "state_transition",
            "phase": "",
            "timestamp": "2026-01-01T00:00:00Z",
            "data": {
                "from": "undifferentiated",
                "to": "sensing",
                "context": {"domain": domain},
            },
        }
    ]
    for cap in capabilities:
        events.append(
            {
                "event_type": "capability_added",
                "phase": "planning",
                "timestamp": "2026-01-01T00:00:01Z",
                "data": {
                    "capability": cap,
                    "reason": f"Selected during planning for {domain}",
                },
            }
        )
    final = "specialized" if graduated else "failed"
    events.append(
        {
            "event_type": "state_transition",
            "phase": "",
            "timestamp": "2026-01-01T00:00:02Z",
            "data": {
                "from": "validating",
                "to": final,
                "context": {"domain": domain},
            },
        }
    )
    return {"events": events}


class TestEmptyStore:
    def test_empty_store_returns_empty(self) -> None:
        assert weight_capabilities("code_quality_analysis", InMemoryStorage()) == {}

    def test_weight_for_unknown_returns_neutral(self) -> None:
        assert weight_for("missing", {}) == NEUTRAL_WEIGHT


class TestSingleHistory:
    def test_one_success_yields_two_thirds(self) -> None:
        storage = InMemoryStorage()
        storage.save(
            "journal_001",
            _journal_with("code_quality_analysis", ["logic_correctness"], graduated=True),
        )
        weights = weight_capabilities("code_quality_analysis", storage)
        # Laplace: (1 + 1) / (1 + 2) = 2/3
        assert abs(weights["logic_correctness"] - (2 / 3)) < 1e-9

    def test_one_failure_yields_one_third(self) -> None:
        storage = InMemoryStorage()
        storage.save(
            "journal_002",
            _journal_with("code_quality_analysis", ["logic_correctness"], graduated=False),
        )
        weights = weight_capabilities("code_quality_analysis", storage)
        # Laplace: (0 + 1) / (1 + 2) = 1/3
        assert abs(weights["logic_correctness"] - (1 / 3)) < 1e-9


class TestMixedHistory:
    def test_three_successes_one_failure(self) -> None:
        storage = InMemoryStorage()
        for i in range(3):
            storage.save(
                f"journal_succ_{i}",
                _journal_with("code_quality_analysis", ["security_analysis"], graduated=True),
            )
        storage.save(
            "journal_fail",
            _journal_with("code_quality_analysis", ["security_analysis"], graduated=False),
        )
        weights = weight_capabilities("code_quality_analysis", storage)
        # Laplace: (3 + 1) / (4 + 2) = 4/6 = 2/3
        assert abs(weights["security_analysis"] - (4 / 6)) < 1e-9

    def test_other_domain_journals_filtered_out(self) -> None:
        storage = InMemoryStorage()
        storage.save(
            "journal_cq",
            _journal_with("code_quality_analysis", ["logic_correctness"], graduated=True),
        )
        storage.save(
            "journal_sec",
            _journal_with("security_audit", ["logic_correctness"], graduated=False),
        )
        weights = weight_capabilities("code_quality_analysis", storage)
        # Only the CQ run counts: (1 + 1) / (1 + 2) = 2/3.
        assert abs(weights["logic_correctness"] - (2 / 3)) < 1e-9


class TestRobustness:
    def test_malformed_journal_skipped(self) -> None:
        storage = InMemoryStorage()
        storage.save(
            "journal_good",
            _journal_with("code_quality_analysis", ["logic_correctness"], graduated=True),
        )
        storage.save("journal_broken", {"events": "not a list"})
        storage.save("journal_empty", {})
        weights = weight_capabilities("code_quality_analysis", storage)
        # Only the good journal counts.
        assert "logic_correctness" in weights

    def test_journals_without_domain_skipped(self) -> None:
        storage = InMemoryStorage()
        storage.save(
            "journal_no_domain",
            {
                "events": [
                    {
                        "event_type": "capability_added",
                        "data": {
                            "capability": "logic_correctness",
                            "reason": "Selected during planning for ???",
                        },
                    }
                ]
            },
        )
        assert weight_capabilities("code_quality_analysis", storage) == {}

    def test_capability_added_without_planning_reason_ignored(self) -> None:
        """Specialization wires already-selected caps with a different reason; skip those."""
        storage = InMemoryStorage()
        events = _journal_with("code_quality_analysis", ["logic_correctness"], graduated=True)[
            "events"
        ]
        events.append(
            {
                "event_type": "capability_added",
                "data": {
                    "capability": "performance_analysis",
                    "reason": "Wired into review pipeline",
                },
            }
        )
        storage.save("journal_mixed", {"events": events})
        weights = weight_capabilities("code_quality_analysis", storage)
        assert "logic_correctness" in weights
        assert "performance_analysis" not in weights


class TestPlanningIntegration:
    def test_priors_promote_capability_in_ranking(self) -> None:
        registry = build_default_registry()
        # Force priors that boost performance_analysis well above the rest.
        priors = {"performance_analysis": 0.95}
        ranked = _rank_capabilities(registry.list_all(), {}, priors)
        assert ranked[0].name == "performance_analysis"

    def test_priors_combine_with_tool_fit(self) -> None:
        registry = build_default_registry()
        # Without priors, security_analysis would lead; the prior pulls it back
        # behind a strongly-positive prior on logic_correctness.
        priors = {"logic_correctness": 0.95, "security_analysis": 0.05}
        ranked = _rank_capabilities(registry.list_all(), {"security": 4}, priors)
        names = [c.name for c in ranked]
        assert names.index("logic_correctness") < names.index("security_analysis")
