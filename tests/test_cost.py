"""Token-cost estimator + budget cap."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from stem_agent.core.journal import EvolutionJournal
from stem_agent.evaluation.cost import (
    BudgetExceededError,
    assert_under_cap,
    cumulative_spend,
    estimate_cost,
)


def _journal_with_calls(calls: list[tuple[str, int]]) -> EvolutionJournal:
    journal = EvolutionJournal()
    for model, tokens in calls:
        journal.log_llm_call(
            phase="validation",
            model=model,
            prompt_hash="0123456789abcdef",
            token_count=tokens,
        )
    return journal


def test_empty_journal_zero_dollars() -> None:
    assert estimate_cost(EvolutionJournal()) == 0.0


def test_known_tokens_known_cost() -> None:
    journal = _journal_with_calls([("gpt-4o-mini", 1000)])
    # 60% input @ $0.15/M, 40% output @ $0.60/M:
    # input = 600 * 0.00015 / 1000 = 0.000090
    # output = 400 * 0.0006 / 1000 = 0.00024
    # Note the price table lives in dollars per 1k tokens.
    cost = estimate_cost(journal)
    expected = (600 / 1000) * 0.00015 + (400 / 1000) * 0.0006
    assert cost == pytest.approx(expected, rel=1e-9)


def test_unknown_model_raises() -> None:
    journal = _journal_with_calls([("imaginary-model", 100)])
    with pytest.raises(KeyError):
        estimate_cost(journal)


def test_calls_without_token_count_skipped() -> None:
    journal = EvolutionJournal()
    journal.log_llm_call(phase="x", model="gpt-4o-mini", prompt_hash="h", token_count=None)
    assert estimate_cost(journal) == 0.0


def test_assert_under_cap_passes_when_below() -> None:
    assert_under_cap(20.0, cap=25.0)


def test_assert_over_cap_raises() -> None:
    with pytest.raises(BudgetExceededError):
        assert_under_cap(26.0, cap=25.0)


def test_cumulative_spend_sums_across_journals(tmp_path: Path) -> None:
    journal_a = _journal_with_calls([("gpt-4o", 1000)])
    journal_b = _journal_with_calls([("gpt-4o-mini", 2000)])
    (tmp_path / "journal_a.json").write_text(json.dumps(journal_a.to_dict()))
    (tmp_path / "journal_b.json").write_text(json.dumps(journal_b.to_dict()))
    total = cumulative_spend(tmp_path)
    assert total == pytest.approx(estimate_cost(journal_a) + estimate_cost(journal_b))


def test_cumulative_spend_handles_missing_dir() -> None:
    assert cumulative_spend(Path("/nonexistent/x/y/z")) == 0.0


def test_scripted_models_cost_zero() -> None:
    journal = _journal_with_calls([("scripted-planner", 1000), ("scripted-executor", 1000)])
    assert estimate_cost(journal) == 0.0
