"""Smoke + correctness for the with/without capability-generation ablation.

Pins the wall-clock budget the writeup quotes (<500ms) and asserts the two
arms differ on at least one metric; without that gap the ablation isn't
saying anything.
"""

from __future__ import annotations

import time

from stem_agent.evaluation.ablation import AblationGrid, run_ablation


def test_grid_returns_two_named_arms() -> None:
    grid = run_ablation()
    assert isinstance(grid, AblationGrid)
    assert grid.with_gen.arm == "with-gen"
    assert grid.without_gen.arm == "without-gen"


def test_grid_has_four_metrics() -> None:
    grid = run_ablation()
    assert len(grid.with_gen.values()) == 4
    assert len(grid.without_gen.values()) == 4


def test_runs_under_500ms() -> None:
    start = time.perf_counter()
    run_ablation()
    elapsed = time.perf_counter() - start
    assert elapsed < 0.5, f"ablation took {elapsed:.3f}s; over the 500ms budget"


def test_render_is_a_two_row_table() -> None:
    grid = run_ablation()
    text = grid.render()
    lines = text.splitlines()
    assert len(lines) >= 4  # header + divider + 2 rows
    assert "with-gen" in text
    assert "without-gen" in text
    assert "precision" in text
    assert "specificity" in text


def test_arms_differ_on_at_least_one_metric() -> None:
    """Without a measurable gap the ablation is not telling us anything."""
    grid = run_ablation()
    assert grid.with_gen.values() != grid.without_gen.values()


def test_runs_deterministically_across_invocations() -> None:
    one = run_ablation()
    two = run_ablation()
    assert one.with_gen.values() == two.with_gen.values()
    assert one.without_gen.values() == two.without_gen.values()
