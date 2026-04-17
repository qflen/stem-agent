"""Before/after statistical comparison between baseline and specialized agents."""

from __future__ import annotations

from dataclasses import dataclass

from stem_agent.evaluation.metrics import ClassificationMetrics


@dataclass(frozen=True)
class ComparisonResult:
    """Side-by-side comparison of baseline vs. specialized agent performance."""

    baseline: ClassificationMetrics
    specialized: ClassificationMetrics

    @property
    def precision_delta(self) -> float:
        return self.specialized.precision - self.baseline.precision

    @property
    def recall_delta(self) -> float:
        return self.specialized.recall - self.baseline.recall

    @property
    def f1_delta(self) -> float:
        return self.specialized.f1 - self.baseline.f1

    @property
    def specificity_delta(self) -> float:
        return self.specialized.specificity - self.baseline.specificity

    @property
    def improved(self) -> bool:
        """Whether the specialized agent outperforms baseline on F1."""
        return self.specialized.f1 > self.baseline.f1

    def summary(self) -> dict[str, dict[str, float]]:
        """Return a summary dict suitable for display or journaling."""
        return {
            "baseline": {
                "precision": round(self.baseline.precision, 4),
                "recall": round(self.baseline.recall, 4),
                "f1": round(self.baseline.f1, 4),
                "specificity": round(self.baseline.specificity, 4),
            },
            "specialized": {
                "precision": round(self.specialized.precision, 4),
                "recall": round(self.specialized.recall, 4),
                "f1": round(self.specialized.f1, 4),
                "specificity": round(self.specialized.specificity, 4),
            },
            "delta": {
                "precision": round(self.precision_delta, 4),
                "recall": round(self.recall_delta, 4),
                "f1": round(self.f1_delta, 4),
                "specificity": round(self.specificity_delta, 4),
            },
        }
