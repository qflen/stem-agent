"""Bootstrap CI module; pin determinism, edge cases, and the headline formatter."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from stem_agent.evaluation.bootstrap import (
    CI,
    bootstrap_mean,
    bootstrap_metric,
    headline_table,
    pool_seed_journals,
)


class TestBootstrapDeterminism:
    def test_same_seed_same_result(self) -> None:
        flags = [1] * 10 + [0] * 5
        a = bootstrap_metric(flags, n_resamples=50, rng_seed=7)
        b = bootstrap_metric(flags, n_resamples=50, rng_seed=7)
        assert a == b

    def test_different_seed_changes_ci(self) -> None:
        flags = [1] * 10 + [0] * 5
        a = bootstrap_metric(flags, n_resamples=50, rng_seed=7)
        b = bootstrap_metric(flags, n_resamples=50, rng_seed=42)
        assert (a.lo, a.hi) != (b.lo, b.hi)


class TestEdgeCases:
    def test_empty_input_returns_zero_ci(self) -> None:
        result = bootstrap_metric([], n_resamples=10)
        assert result == CI(mean=0.0, lo=0.0, hi=0.0)

    def test_all_correct_pins_at_one(self) -> None:
        result = bootstrap_metric([1] * 20, n_resamples=50)
        assert result.mean == 1.0
        assert result.lo == 1.0
        assert result.hi == 1.0

    def test_all_wrong_pins_at_zero(self) -> None:
        result = bootstrap_metric([0] * 20, n_resamples=50)
        assert result.mean == 0.0
        assert result.hi == 0.0

    def test_single_sample_widest_ci(self) -> None:
        result = bootstrap_metric([1], n_resamples=50)
        assert result.mean == 1.0


class TestRendering:
    def test_render_3_decimals(self) -> None:
        ci = CI(mean=0.7777, lo=0.6, hi=0.85)
        assert ci.render(decimals=3) == "0.778 [0.600, 0.850]"

    def test_render_default_3(self) -> None:
        ci = CI(mean=0.5, lo=0.4, hi=0.6)
        assert "0.500" in ci.render()


class TestMeanInsideCI:
    def test_mean_within_lo_hi_for_balanced_input(self) -> None:
        result = bootstrap_metric([1, 1, 0, 1, 0, 0, 1, 0], n_resamples=200)
        assert result.lo <= result.mean <= result.hi


class TestHeadlineTable:
    def _seed_journal(self, tmp_path: Path, seed: int, specialized: dict[str, float]) -> Path:
        path = tmp_path / f"journal_seed{seed}.json"
        path.write_text(
            json.dumps(
                {
                    "events": [
                        {
                            "event_type": "phase_result",
                            "phase": "validation",
                            "timestamp": "2026-04-21T00:00:00Z",
                            "data": {
                                "baseline": {
                                    "precision": 0.0,
                                    "recall": 0.0,
                                    "f1": 0.0,
                                    "specificity": 0.0,
                                },
                                "specialized": specialized,
                                "delta": {},
                            },
                        }
                    ]
                }
            )
        )
        return path

    def test_pool_seed_journals_returns_per_metric_lists(self, tmp_path: Path) -> None:
        self._seed_journal(
            tmp_path,
            0,
            {"precision": 0.5, "recall": 0.7, "f1": 0.583, "specificity": 0.4},
        )
        self._seed_journal(
            tmp_path,
            1,
            {"precision": 0.6, "recall": 0.9, "f1": 0.72, "specificity": 0.5},
        )
        pooled = pool_seed_journals(sorted(tmp_path.glob("journal_*.json")))
        assert pooled["f1"] == [0.583, 0.72]
        assert pooled["recall"] == [0.7, 0.9]

    def test_headline_table_includes_metric_names(self, tmp_path: Path) -> None:
        for s in range(3):
            self._seed_journal(
                tmp_path,
                s,
                {"precision": 0.6, "recall": 0.9, "f1": 0.72, "specificity": 0.4},
            )
        text = headline_table(tmp_path, n_resamples=50)
        for metric in ("precision", "recall", "f1", "specificity"):
            assert metric in text

    def test_headline_table_handles_empty_dir(self, tmp_path: Path) -> None:
        text = headline_table(tmp_path, n_resamples=50)
        assert "no journals" in text


class TestSeedLevelBootstrap:
    """``bootstrap_mean`` is what feeds the headline."""

    def test_three_identical_values_yield_zero_width_ci(self) -> None:
        result = bootstrap_mean([0.7, 0.7, 0.7], n_resamples=200)
        assert result.mean == pytest.approx(0.7)
        assert result.lo == pytest.approx(0.7)
        assert result.hi == pytest.approx(0.7)

    def test_spread_yields_nontrivial_ci(self) -> None:
        result = bootstrap_mean([0.5, 0.7, 0.9], n_resamples=200, rng_seed=1)
        assert result.mean == pytest.approx((0.5 + 0.7 + 0.9) / 3)
        assert result.lo < result.mean
        assert result.hi > result.mean

    def test_empty_returns_zero_ci(self) -> None:
        result = bootstrap_mean([], n_resamples=10)
        assert result == CI(mean=0.0, lo=0.0, hi=0.0)


@pytest.mark.parametrize("alpha,n", [(0.05, 50), (0.10, 50), (0.05, 200)])
def test_alpha_widens_or_narrows_ci(alpha: float, n: int) -> None:
    flags = [1] * 5 + [0] * 5
    result = bootstrap_metric(flags, n_resamples=n, alpha=alpha, rng_seed=1)
    assert 0.0 <= result.lo <= result.mean <= result.hi <= 1.0
