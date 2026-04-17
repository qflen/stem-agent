"""Evaluation metrics — precision, recall, F1, specificity.

Handles edge cases a QA engineer would check: empty inputs,
perfect scores, zero detection, all false positives.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ClassificationMetrics:
    """Confusion-matrix-derived metrics for issue detection."""

    true_positives: int
    false_positives: int
    true_negatives: int
    false_negatives: int

    @property
    def precision(self) -> float:
        """What fraction of detected issues are real issues."""
        denom = self.true_positives + self.false_positives
        if denom == 0:
            return 0.0
        return self.true_positives / denom

    @property
    def recall(self) -> float:
        """What fraction of real issues were detected."""
        denom = self.true_positives + self.false_negatives
        if denom == 0:
            return 0.0
        return self.true_positives / denom

    @property
    def f1(self) -> float:
        """Harmonic mean of precision and recall."""
        p = self.precision
        r = self.recall
        if p + r == 0:
            return 0.0
        return 2 * (p * r) / (p + r)

    @property
    def specificity(self) -> float:
        """What fraction of clean samples were correctly identified as clean."""
        denom = self.true_negatives + self.false_positives
        if denom == 0:
            return 0.0
        return self.true_negatives / denom

    @property
    def accuracy(self) -> float:
        """Overall correct classification rate."""
        total = (
            self.true_positives + self.false_positives + self.true_negatives + self.false_negatives
        )
        if total == 0:
            return 0.0
        return (self.true_positives + self.true_negatives) / total


def compute_metrics(
    detected_categories: list[set[str]],
    ground_truth_categories: list[set[str]],
    is_clean_detected: list[bool],
    is_clean_truth: list[bool],
) -> ClassificationMetrics:
    """Compute classification metrics across a benchmark corpus.

    For each sample, we compare detected issue categories against ground-truth
    categories. A detected category that exists in ground truth is a TP.
    A detected category not in ground truth is a FP. A ground truth category
    not detected is a FN. Clean samples correctly identified as clean are TNs.

    Args:
        detected_categories: Per-sample sets of detected issue categories.
        ground_truth_categories: Per-sample sets of expected issue categories.
        is_clean_detected: Per-sample: did the agent declare the code clean?
        is_clean_truth: Per-sample: is the code actually clean?

    Returns:
        Aggregated classification metrics.
    """
    if not detected_categories:
        return ClassificationMetrics(0, 0, 0, 0)

    tp = fp = tn = fn = 0

    for detected, truth, det_clean, truth_clean in zip(
        detected_categories,
        ground_truth_categories,
        is_clean_detected,
        is_clean_truth,
        strict=True,
    ):
        if truth_clean:
            # Clean sample
            if det_clean:
                tn += 1  # Correctly identified as clean
            else:
                fp += len(detected)  # Each spurious issue is a false positive
        else:
            # Sample with known issues
            matched = detected & truth
            tp += len(matched)
            fp += len(detected - truth)
            fn += len(truth - detected)

    return ClassificationMetrics(
        true_positives=tp,
        false_positives=fp,
        true_negatives=tn,
        false_negatives=fn,
    )
