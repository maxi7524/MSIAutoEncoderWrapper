"""Tests for generic classification-head metrics, focused on mask handling.

``unobserved_label_policy: unlabelled`` (this repository's actual multi-label
configuration — see ``PixelDataset._target_sample`` and
``23_08_26_architecture_predictive/report/methodology.md`` §2) produces a target
*availability* mask shaped one flag per (sample, class) pair, not one flag per
sample. These tests pin that shape down explicitly, since the previous
implementation (``mask.reshape(-1)`` then row-indexing) silently mishandled it.
"""

from __future__ import annotations

import numpy as np
import pytest

from msi_autoencoder_wrapper.analysis.autoencoder.heads.metrics import (
    evaluate_head,
    per_class_metrics,
)
from msi_autoencoder_wrapper.utils.exceptions import ValidationError


def _logits_favoring_targets(targets: np.ndarray, magnitude: float = 6.0) -> np.ndarray:
    """Build logits that reproduce ``targets`` exactly at ``threshold=0.5``."""
    return np.where(targets > 0, magnitude, -magnitude)


class TestPerClassMetricsMaskShapes:
    def test_per_sample_by_class_mask_all_true_matches_no_mask(self) -> None:
        rng = np.random.default_rng(0)
        targets = (rng.random((20, 3)) > 0.7).astype(np.float32)
        probabilities = rng.random((20, 3))
        mask = np.ones_like(targets, dtype=bool)

        with_mask = per_class_metrics(probabilities, targets, 0.5, mask)
        without_mask = per_class_metrics(probabilities, targets, 0.5, None)

        assert with_mask == without_mask

    def test_per_class_mask_excludes_only_the_masked_entries(self) -> None:
        # Class 0: sample 0's true label is 1 but masked out (unlabelled) — an
        # all-positive-prediction model must not be penalized for missing it.
        targets = np.array([[1.0, 0.0], [0.0, 1.0], [1.0, 0.0]])
        probabilities = np.array([[0.9, 0.9], [0.1, 0.9], [0.9, 0.1]])
        mask = np.array(
            [[False, True], [True, True], [True, True]],
        )

        records = per_class_metrics(probabilities, targets, 0.5, mask)

        # Class 0 available rows: sample 1 (true=0, pred=0), sample 2 (true=1, pred=1).
        assert records[0]["positive_samples"] == 1.0
        assert records[0]["precision"] == 1.0
        assert records[0]["recall"] == 1.0
        # Class 1 available rows: all three, matches the unmasked case.
        assert records[1]["positive_samples"] == 1.0

    def test_class_with_zero_available_positives_reports_nan_average_precision(
        self,
    ) -> None:
        targets = np.zeros((5, 1))
        probabilities = np.full((5, 1), 0.3)

        records = per_class_metrics(probabilities, targets, 0.5)

        assert records[0]["positive_samples"] == 0.0
        assert np.isnan(records[0]["average_precision"])

    def test_one_flag_per_sample_mask_is_broadcast_across_classes(self) -> None:
        targets = np.array([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]])
        probabilities = np.array([[0.9, 0.1], [0.1, 0.9], [0.9, 0.9]])
        row_mask = np.array([True, True, False])

        with_row_mask = per_class_metrics(probabilities, targets, 0.5, row_mask)
        restricted = per_class_metrics(probabilities[:2], targets[:2], 0.5)

        assert with_row_mask == restricted

    def test_class_with_zero_available_positives_reports_nan_roc_auc(self) -> None:
        targets = np.zeros((5, 1))
        probabilities = np.full((5, 1), 0.3)

        records = per_class_metrics(probabilities, targets, 0.5)

        assert np.isnan(records[0]["roc_auc"])

    def test_class_with_zero_available_negatives_reports_nan_roc_auc(self) -> None:
        targets = np.ones((5, 1))
        probabilities = np.full((5, 1), 0.7)

        records = per_class_metrics(probabilities, targets, 0.5)

        assert records[0]["positive_samples"] == 5.0
        assert np.isnan(records[0]["roc_auc"])

    def test_roc_auc_matches_sklearn_for_a_scoreable_class(self) -> None:
        from sklearn.metrics import roc_auc_score

        rng = np.random.default_rng(2)
        targets = (rng.random((30, 1)) > 0.5).astype(np.float32)
        probabilities = rng.random((30, 1))

        records = per_class_metrics(probabilities, targets, 0.5)

        expected = roc_auc_score(targets[:, 0], probabilities[:, 0])
        assert records[0]["roc_auc"] == pytest.approx(expected)

    def test_incompatible_mask_shape_raises(self) -> None:
        targets = np.zeros((4, 3))
        probabilities = np.zeros((4, 3))
        bad_mask = np.ones((4, 5), dtype=bool)

        with pytest.raises(ValidationError):
            per_class_metrics(probabilities, targets, 0.5, bad_mask)


class TestEvaluateHeadMultiLabelMaskShapes:
    def test_per_class_mask_does_not_crash_and_matches_manual_computation(self) -> None:
        rng = np.random.default_rng(1)
        targets = (rng.random((30, 4)) > 0.6).astype(np.float32)
        logits = _logits_favoring_targets(targets)
        mask = np.ones_like(targets, dtype=bool)
        mask[0, :] = False  # one fully-unlabelled sample, entry-wise

        result = evaluate_head(logits, targets, "multi_label", mask, threshold=0.5)

        # Predictions reproduce targets everywhere they are scored, so a perfect
        # classifier's micro/macro F1 and precision/recall must all be 1.0.
        assert result["micro_f1"] == pytest.approx(1.0)
        assert result["macro_f1"] == pytest.approx(1.0)
        assert result["macro_precision"] == pytest.approx(1.0)
        assert result["macro_recall"] == pytest.approx(1.0)
        assert result["micro_precision"] == pytest.approx(1.0)
        assert result["micro_recall"] == pytest.approx(1.0)
        assert result["hamming_loss"] == pytest.approx(0.0)
        # Extreme, target-separating logits give every scoreable class perfect
        # ranking too, so macro roc_auc must also be 1.0.
        assert result["roc_auc"] == pytest.approx(1.0)

    def test_macro_precision_and_recall_are_the_per_class_mean_not_micro_pooled(
        self,
    ) -> None:
        # Class 0: 1 true positive, model predicts it -> precision=recall=1.0.
        # Class 1: 3 true positives, model predicts none -> precision=recall=0.0.
        # Macro (mean over classes) must differ from micro (pooled over samples).
        targets = np.array(
            [[1.0, 1.0], [0.0, 1.0], [0.0, 1.0], [0.0, 0.0]],
        )
        logits = np.array(
            [[6.0, -6.0], [-6.0, -6.0], [-6.0, -6.0], [-6.0, -6.0]],
        )

        result = evaluate_head(logits, targets, "multi_label", threshold=0.5)

        assert result["macro_precision"] == pytest.approx(0.5)
        assert result["macro_recall"] == pytest.approx(0.5)
        assert result["micro_precision"] == pytest.approx(1.0)
        assert result["micro_recall"] == pytest.approx(0.25)

    def test_masked_entries_do_not_affect_the_result(self) -> None:
        targets = np.array([[1.0, 0.0], [0.0, 1.0], [1.0, 0.0]])
        # Logits are wrong only where masked out, so a correct implementation
        # ignores them entirely and still reports a perfect score.
        logits = np.array([[6.0, 6.0], [-6.0, 6.0], [6.0, -6.0]])
        mask = np.array([[True, False], [True, True], [True, True]])

        result = evaluate_head(logits, targets, "multi_label", mask, threshold=0.5)

        assert result["micro_f1"] == pytest.approx(1.0)
        assert result["hamming_loss"] == pytest.approx(0.0)

    def test_hamming_loss_baseline_matches_an_always_negative_predictor(self) -> None:
        rng = np.random.default_rng(3)
        targets = (rng.random((40, 5)) > 0.85).astype(np.float32)  # sparse positives
        logits = rng.normal(size=(40, 5))  # arbitrary — baseline must not depend on these

        result = evaluate_head(logits, targets, "multi_label", threshold=0.5)
        always_negative_predictions = np.zeros_like(targets, dtype=bool)
        expected = float(np.mean(targets.astype(bool) != always_negative_predictions))

        assert result["hamming_loss_baseline_positive_rate"] == pytest.approx(expected)
        assert result["hamming_loss_baseline_positive_rate"] == pytest.approx(float(targets.mean()))

    def test_no_available_entries_raises(self) -> None:
        targets = np.zeros((3, 2))
        logits = np.zeros((3, 2))
        mask = np.zeros((3, 2), dtype=bool)

        with pytest.raises(ValidationError):
            evaluate_head(logits, targets, "multi_label", mask)

    def test_single_label_row_mask_still_row_filters(self) -> None:
        logits = np.array([[5.0, -5.0], [-5.0, 5.0], [5.0, -5.0]])
        targets = np.array([1, 1, 0])  # first two rows disagree with the model
        mask = np.array([False, False, True])

        result = evaluate_head(logits, targets, "single_label", mask)

        assert result["accuracy"] == pytest.approx(1.0)
