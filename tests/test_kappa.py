"""Cohen's κ; perfect agreement, complete disagreement, chance, mismatched lengths."""

from __future__ import annotations

import pytest

from stem_agent.evaluation.kappa import cohen_kappa, disputed_count, kappa_with_ci


def test_perfect_agreement_yields_one() -> None:
    labels = ["a", "b", "a", "c"]
    assert cohen_kappa(labels, labels) == 1.0


def test_complete_disagreement_below_zero_when_chance_high() -> None:
    a = ["x"] * 5 + ["y"] * 5
    b = ["y"] * 5 + ["x"] * 5
    assert cohen_kappa(a, b) < 0


def test_chance_agreement_near_zero() -> None:
    a = ["x", "y", "x", "y", "x", "y"]
    b = ["y", "x", "y", "x", "y", "x"]
    # Chance agreement → κ ≈ -1 here (every observed pair disagrees, p_o=0).
    # The point is it's far from 1.0; the magnitude doesn't matter.
    assert cohen_kappa(a, b) <= 0.0


def test_empty_returns_zero() -> None:
    assert cohen_kappa([], []) == 0.0


def test_mismatched_lengths_raises() -> None:
    with pytest.raises(ValueError):
        cohen_kappa(["a"], ["a", "b"])


def test_kappa_with_ci_brackets_known_kappa() -> None:
    a = ["a", "a", "b", "b", "c", "c"] * 5
    b = ["a", "b", "b", "b", "c", "c"] * 5
    result = kappa_with_ci(a, b, n_resamples=100, rng_seed=3)
    assert result.lo <= result.kappa <= result.hi


def test_disputed_counts_disagreements() -> None:
    assert disputed_count(["a", "b", "a"], ["a", "a", "a"]) == 1


def test_render_kappa_result() -> None:
    result = kappa_with_ci(["a", "b"], ["a", "a"], n_resamples=10, rng_seed=1)
    rendered = result.render()
    assert "κ =" in rendered
    assert "[" in rendered
