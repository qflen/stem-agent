"""Tests for evaluation metrics; edge cases a QA engineer would check.

Perfect detection, zero detection, all false positives, empty input,
single sample; each boundary condition is verified.
"""

from __future__ import annotations

from stem_agent.evaluation.metrics import ClassificationMetrics, compute_metrics


class TestClassificationMetrics:
    """Direct tests on the ClassificationMetrics dataclass."""

    def test_perfect_detection(self) -> None:
        """All issues detected, no false positives → F1 = 1.0."""
        m = ClassificationMetrics(
            true_positives=10, false_positives=0, true_negatives=5, false_negatives=0
        )
        assert m.precision == 1.0
        assert m.recall == 1.0
        assert m.f1 == 1.0

    def test_zero_detection(self) -> None:
        """No issues detected at all → F1 = 0.0."""
        m = ClassificationMetrics(
            true_positives=0, false_positives=0, true_negatives=5, false_negatives=10
        )
        assert m.precision == 0.0
        assert m.recall == 0.0
        assert m.f1 == 0.0

    def test_all_false_positives(self) -> None:
        """Every detection is wrong → precision = 0.0."""
        m = ClassificationMetrics(
            true_positives=0, false_positives=10, true_negatives=0, false_negatives=5
        )
        assert m.precision == 0.0
        assert m.f1 == 0.0

    def test_no_true_negatives(self) -> None:
        """No clean samples in corpus → specificity = 0.0 (no data, not failure)."""
        m = ClassificationMetrics(
            true_positives=5, false_positives=2, true_negatives=0, false_negatives=1
        )
        assert m.specificity == 0.0

    def test_perfect_specificity(self) -> None:
        """All clean samples correctly identified → specificity = 1.0."""
        m = ClassificationMetrics(
            true_positives=5, false_positives=0, true_negatives=5, false_negatives=1
        )
        assert m.specificity == 1.0

    def test_all_zeros(self) -> None:
        """All counts zero → all metrics = 0.0, no division by zero."""
        m = ClassificationMetrics(0, 0, 0, 0)
        assert m.precision == 0.0
        assert m.recall == 0.0
        assert m.f1 == 0.0
        assert m.specificity == 0.0
        assert m.accuracy == 0.0

    def test_accuracy(self) -> None:
        m = ClassificationMetrics(
            true_positives=8, false_positives=2, true_negatives=4, false_negatives=1
        )
        assert m.accuracy == (8 + 4) / (8 + 2 + 4 + 1)

    def test_balanced_metrics(self) -> None:
        """F1 is the harmonic mean of precision and recall."""
        m = ClassificationMetrics(
            true_positives=6, false_positives=4, true_negatives=3, false_negatives=2
        )
        p = 6 / (6 + 4)  # 0.6
        r = 6 / (6 + 2)  # 0.75
        expected_f1 = 2 * p * r / (p + r)
        assert abs(m.f1 - expected_f1) < 1e-10


class TestComputeMetrics:
    """Tests for the compute_metrics aggregation function."""

    def test_single_buggy_sample_detected(self) -> None:
        """One sample with one category, correctly detected."""
        result = compute_metrics(
            detected_categories=[{"logic"}],
            ground_truth_categories=[{"logic"}],
            is_clean_detected=[False],
            is_clean_truth=[False],
        )
        assert result.true_positives == 1
        assert result.false_positives == 0
        assert result.false_negatives == 0

    def test_single_clean_sample_detected(self) -> None:
        """One clean sample correctly identified as clean."""
        result = compute_metrics(
            detected_categories=[set()],
            ground_truth_categories=[set()],
            is_clean_detected=[True],
            is_clean_truth=[True],
        )
        assert result.true_negatives == 1
        assert result.false_positives == 0

    def test_false_positive_on_clean_code(self) -> None:
        """Agent flags issues on clean code → false positives."""
        result = compute_metrics(
            detected_categories=[{"logic", "security"}],
            ground_truth_categories=[set()],
            is_clean_detected=[False],
            is_clean_truth=[True],
        )
        assert result.false_positives == 2
        assert result.true_negatives == 0

    def test_missed_category_counts_as_false_negative(self) -> None:
        """Agent detects some categories but misses others."""
        result = compute_metrics(
            detected_categories=[{"logic"}],
            ground_truth_categories=[{"logic", "security"}],
            is_clean_detected=[False],
            is_clean_truth=[False],
        )
        assert result.true_positives == 1
        assert result.false_negatives == 1

    def test_hallucinated_category_counts_as_false_positive(self) -> None:
        """Agent detects a category that doesn't exist in ground truth."""
        result = compute_metrics(
            detected_categories=[{"logic", "performance"}],
            ground_truth_categories=[{"logic"}],
            is_clean_detected=[False],
            is_clean_truth=[False],
        )
        assert result.true_positives == 1
        assert result.false_positives == 1

    def test_empty_input(self) -> None:
        """No samples → all zeros, no crash."""
        result = compute_metrics([], [], [], [])
        assert result.true_positives == 0
        assert result.f1 == 0.0

    def test_multiple_samples(self) -> None:
        """Aggregation across several samples."""
        result = compute_metrics(
            detected_categories=[{"logic"}, set(), {"security"}],
            ground_truth_categories=[{"logic"}, set(), {"security", "logic"}],
            is_clean_detected=[False, True, False],
            is_clean_truth=[False, True, False],
        )
        # Sample 0: TP=1, Sample 1: TN=1, Sample 2: TP=1 FN=1
        assert result.true_positives == 2
        assert result.true_negatives == 1
        assert result.false_negatives == 1
        assert result.false_positives == 0
