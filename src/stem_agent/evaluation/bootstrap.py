"""Bootstrap CIs over per-sample correctness, pooled across seeds.

The headline numbers in the writeup are means across three live seeds
plus 95% percentile-method confidence intervals. ``bootstrap_metric``
implements the resample, ``pool_seed_journals`` reads the per-sample
verdicts a journal records, and ``headline_table`` renders the
``mean [lo, hi]`` row format the writeup quotes. Every operation is
deterministic given a seed, and the runtime is bounded by the
``n_resamples`` parameter; production uses 1000, tests use 50 to keep
the suite under 10s.
"""

from __future__ import annotations

import json
import random
from dataclasses import dataclass
from pathlib import Path

_METRICS = ("precision", "recall", "f1", "specificity")


@dataclass(frozen=True)
class CI:
    mean: float
    lo: float
    hi: float

    def render(self, decimals: int = 3) -> str:
        return f"{self.mean:.{decimals}f} [{self.lo:.{decimals}f}, {self.hi:.{decimals}f}]"


def bootstrap_metric(
    correctness: list[int],
    *,
    n_resamples: int = 1000,
    alpha: float = 0.05,
    rng_seed: int = 0,
) -> CI:
    """Percentile-method 95% CI over per-sample 0/1 correctness flags."""
    if not correctness:
        return CI(mean=0.0, lo=0.0, hi=0.0)
    rng = random.Random(rng_seed)
    n = len(correctness)
    means: list[float] = []
    for _ in range(n_resamples):
        sample = [correctness[rng.randrange(n)] for _ in range(n)]
        means.append(sum(sample) / n)
    means.sort()
    lo_index = int((alpha / 2) * n_resamples)
    hi_index = min(n_resamples - 1, int((1 - alpha / 2) * n_resamples))
    overall_mean = sum(correctness) / n
    return CI(mean=overall_mean, lo=means[lo_index], hi=means[hi_index])


def bootstrap_mean(
    values: list[float],
    *,
    n_resamples: int = 1000,
    alpha: float = 0.05,
    rng_seed: int = 0,
) -> CI:
    """Percentile-method 95% CI over the *mean* of seed-level metric values.

    Used by ``headline_table`` to summarise N seed-level F1/precision/recall
    point estimates into a ``mean [lo, hi]`` row. With N=3 the resampling
    distribution is necessarily coarse; the CI reflects exactly the
    seed-to-seed variance the writeup intends to report.
    """
    if not values:
        return CI(mean=0.0, lo=0.0, hi=0.0)
    rng = random.Random(rng_seed)
    n = len(values)
    means: list[float] = []
    for _ in range(n_resamples):
        sample = [values[rng.randrange(n)] for _ in range(n)]
        means.append(sum(sample) / n)
    means.sort()
    lo_index = int((alpha / 2) * n_resamples)
    hi_index = min(n_resamples - 1, int((1 - alpha / 2) * n_resamples))
    overall_mean = sum(values) / n
    return CI(mean=overall_mean, lo=means[lo_index], hi=means[hi_index])


def _seed_metric_values(seed_paths: list[Path]) -> dict[str, list[float]]:
    """Pull the per-seed specialized metric values from each journal."""
    pooled: dict[str, list[float]] = {m: [] for m in _METRICS}
    for path in seed_paths:
        blob = json.loads(path.read_text())
        events = blob.get("events", [])
        for event in events:
            if event.get("event_type") != "phase_result":
                continue
            if event.get("phase") != "validation":
                continue
            data = event.get("data", {})
            specialized = data.get("specialized") if isinstance(data, dict) else None
            if not isinstance(specialized, dict):
                continue
            for metric in _METRICS:
                value = specialized.get(metric)
                if isinstance(value, int | float):
                    pooled[metric].append(float(value))
            break
    return pooled


def pool_seed_journals(seed_paths: list[Path]) -> dict[str, list[float]]:
    """Pool the per-seed specialized metric values across journals.

    Returns a dict mapping metric name → list of seed-level values
    (one per journal). Used by ``headline_table`` and the writeup
    update; tests can also call it directly.
    """
    return _seed_metric_values(seed_paths)


def headline_table(seeds_dir: Path, *, n_resamples: int = 1000) -> str:
    """Render the ``mean [lo, hi]`` headline row per metric for the writeup."""
    seed_paths = sorted(seeds_dir.glob("journal_seed*.json"))
    if not seed_paths:
        return f"(no journals in {seeds_dir})"
    pooled = _seed_metric_values(seed_paths)
    rows = [
        f"| {'metric':<11} | {'mean':>6} | {'95% CI':>20} |",
        f"|{'-' * 13}|{'-' * 8}|{'-' * 22}|",
    ]
    for metric in _METRICS:
        ci = bootstrap_mean(pooled[metric], n_resamples=n_resamples)
        rows.append(f"| {metric:<11} | {ci.mean:6.3f} | [{ci.lo:6.3f}, {ci.hi:6.3f}] |")
    return "\n".join(rows)
