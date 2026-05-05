"""Cohen's κ; agreement between the corpus labels and a second-rater pass.

Used by the adjudicator script (``scripts/adjudicate.py``) to score how
well an independent ``gpt-4o-mini`` re-labelling agrees with the
hand-authored ground truth. The bootstrap CI is computed over a
percentile resample of the per-sample agreement; production runs use
1000 resamples, tests use 50.
"""

from __future__ import annotations

import random
from dataclasses import dataclass


@dataclass(frozen=True)
class KappaResult:
    kappa: float
    lo: float
    hi: float

    def render(self, decimals: int = 3) -> str:
        return f"κ = {self.kappa:.{decimals}f} [{self.lo:.{decimals}f}, {self.hi:.{decimals}f}]"


def cohen_kappa(rater_a: list[str], rater_b: list[str]) -> float:
    """Compute κ between two parallel-aligned label sequences."""
    if len(rater_a) != len(rater_b):
        raise ValueError("rater inputs must align by sample")
    if not rater_a:
        return 0.0
    n = len(rater_a)
    labels = sorted(set(rater_a) | set(rater_b))
    p_o = sum(1 for a, b in zip(rater_a, rater_b, strict=True) if a == b) / n
    p_e = 0.0
    for label in labels:
        p_a = sum(1 for x in rater_a if x == label) / n
        p_b = sum(1 for x in rater_b if x == label) / n
        p_e += p_a * p_b
    if p_e >= 1.0:
        return 1.0 if p_o >= 1.0 else 0.0
    return (p_o - p_e) / (1 - p_e)


def kappa_with_ci(
    rater_a: list[str],
    rater_b: list[str],
    *,
    n_resamples: int = 1000,
    alpha: float = 0.05,
    rng_seed: int = 0,
) -> KappaResult:
    if not rater_a:
        return KappaResult(0.0, 0.0, 0.0)
    rng = random.Random(rng_seed)
    n = len(rater_a)
    samples: list[float] = []
    for _ in range(n_resamples):
        idx = [rng.randrange(n) for _ in range(n)]
        a_sample = [rater_a[i] for i in idx]
        b_sample = [rater_b[i] for i in idx]
        samples.append(cohen_kappa(a_sample, b_sample))
    samples.sort()
    lo_index = int((alpha / 2) * n_resamples)
    hi_index = min(n_resamples - 1, int((1 - alpha / 2) * n_resamples))
    return KappaResult(
        kappa=cohen_kappa(rater_a, rater_b),
        lo=samples[lo_index],
        hi=samples[hi_index],
    )


def disputed_count(rater_a: list[str], rater_b: list[str]) -> int:
    """Number of samples where the two raters disagree."""
    return sum(1 for a, b in zip(rater_a, rater_b, strict=True) if a != b)
