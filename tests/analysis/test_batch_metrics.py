"""Agreement of the vectorized head metrics with the scikit-learn reference.

The batch implementation exists only to be faster, so every test here compares it
against :mod:`...heads.metrics`, which stays the definition of correctness.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from msi_autoencoder_wrapper.analysis.autoencoder.heads.batch_metrics import (
    evaluate_head_batch,
    per_class_metrics_batch,
)
from msi_autoencoder_wrapper.analysis.autoencoder.heads.metrics import (
    evaluate_head,
    per_class_metrics,
    probabilities_from_logits,
)

TOLERANCE = 1e-6

DEVICES = ["cpu"] + (["cuda"] if torch.cuda.is_available() else [])


def _reference(logits, targets, mask, threshold=0.5):
    return evaluate_head(logits, targets, "multi_label", mask, threshold)


def _assert_matches(reference: dict, produced: dict) -> None:
    assert set(produced) == set(reference)
    for key, expected in reference.items():
        actual = produced[key]
        if np.isnan(expected):
            assert np.isnan(actual), f"{key}: expected nan, got {actual}"
        else:
            assert actual == pytest.approx(expected, abs=TOLERANCE), key


@pytest.mark.parametrize("device", DEVICES)
class TestEvaluateHeadBatchAgreement:
    """Aggregate metrics must equal the scikit-learn reference."""

    def test_dense_multilabel_case(self, device: str) -> None:
        generator = np.random.default_rng(0)
        logits = generator.normal(scale=2.0, size=(200, 40)).astype(np.float32)
        targets = (generator.random((200, 40)) > 0.8).astype(np.float32)
        mask = np.ones_like(targets)

        _assert_matches(
            _reference(logits, targets, mask), evaluate_head_batch(logits, targets, mask, device=device)
        )

    def test_sparse_labels_like_the_real_task(self, device: str) -> None:
        # Several hundred classes, most of them rare: the regime the sweep evaluates.
        generator = np.random.default_rng(1)
        logits = generator.normal(scale=3.0, size=(300, 150)).astype(np.float32)
        targets = (generator.random((300, 150)) > 0.97).astype(np.float32)
        mask = (generator.random((300, 150)) > 0.1).astype(np.float32)

        _assert_matches(
            _reference(logits, targets, mask), evaluate_head_batch(logits, targets, mask, device=device)
        )

    def test_classes_without_positives_stay_undefined_not_zero(self, device: str) -> None:
        # A class with no available positive has undefined average precision; it must
        # be skipped by the mean rather than counted as zero.
        generator = np.random.default_rng(2)
        logits = generator.normal(size=(80, 6)).astype(np.float32)
        targets = np.zeros((80, 6), dtype=np.float32)
        targets[:, :3] = (generator.random((80, 3)) > 0.7).astype(np.float32)
        mask = np.ones_like(targets)

        _assert_matches(
            _reference(logits, targets, mask), evaluate_head_batch(logits, targets, mask, device=device)
        )

    def test_classes_without_negatives_leave_roc_undefined(self, device: str) -> None:
        generator = np.random.default_rng(3)
        logits = generator.normal(size=(50, 4)).astype(np.float32)
        targets = np.ones((50, 4), dtype=np.float32)
        targets[:, 0] = (generator.random(50) > 0.5).astype(np.float32)
        mask = np.ones_like(targets)

        _assert_matches(
            _reference(logits, targets, mask), evaluate_head_batch(logits, targets, mask, device=device)
        )

    def test_saturating_logits_produce_ties_that_must_be_grouped(self, device: str) -> None:
        # A logistic saturates to exactly 1.0 and 0.0 in float32, so real sweeps do
        # contain tied scores; ungrouped ties would silently shift both ranking metrics.
        generator = np.random.default_rng(4)
        logits = generator.normal(scale=60.0, size=(120, 20)).astype(np.float32)
        targets = (generator.random((120, 20)) > 0.7).astype(np.float32)
        mask = np.ones_like(targets)
        probabilities = probabilities_from_logits(logits, "multi_label")
        assert np.any(probabilities == 1.0) and np.any(probabilities == 0.0)

        _assert_matches(
            _reference(logits, targets, mask), evaluate_head_batch(logits, targets, mask, device=device)
        )

    def test_heavily_tied_scores(self, device: str) -> None:
        # Extreme case: only three distinct scores, so every group is large.
        generator = np.random.default_rng(5)
        logits = generator.choice([-2.0, 0.0, 2.0], size=(90, 12)).astype(np.float32)
        targets = (generator.random((90, 12)) > 0.6).astype(np.float32)
        mask = np.ones_like(targets)

        _assert_matches(
            _reference(logits, targets, mask), evaluate_head_batch(logits, targets, mask, device=device)
        )

    def test_per_sample_mask_is_broadcast_like_the_reference(self, device: str) -> None:
        generator = np.random.default_rng(6)
        logits = generator.normal(size=(70, 9)).astype(np.float32)
        targets = (generator.random((70, 9)) > 0.75).astype(np.float32)
        mask = (generator.random(70) > 0.2).astype(np.float32)

        _assert_matches(
            _reference(logits, targets, mask), evaluate_head_batch(logits, targets, mask, device=device)
        )

    def test_absent_mask_scores_every_entry(self, device: str) -> None:
        generator = np.random.default_rng(7)
        logits = generator.normal(size=(60, 8)).astype(np.float32)
        targets = (generator.random((60, 8)) > 0.7).astype(np.float32)

        _assert_matches(
            _reference(logits, targets, None), evaluate_head_batch(logits, targets, None, device=device)
        )

    @pytest.mark.parametrize("threshold", [0.2, 0.5, 0.9])
    def test_threshold_metrics_follow_the_threshold(self, device: str, threshold: float) -> None:
        generator = np.random.default_rng(8)
        logits = generator.normal(scale=1.5, size=(100, 15)).astype(np.float32)
        targets = (generator.random((100, 15)) > 0.8).astype(np.float32)
        mask = np.ones_like(targets)

        _assert_matches(
            _reference(logits, targets, mask, threshold),
            evaluate_head_batch(logits, targets, mask, threshold, device=device),
        )

    def test_no_available_entry_is_rejected(self, device: str) -> None:
        logits = np.zeros((10, 3), dtype=np.float32)
        targets = np.zeros((10, 3), dtype=np.float32)
        mask = np.zeros((10, 3), dtype=np.float32)

        with pytest.raises(ValueError):
            evaluate_head_batch(logits, targets, mask, device=device)


@pytest.mark.parametrize("device", DEVICES)
class TestPerClassAgreement:
    """Per-class records must equal the scikit-learn reference class by class."""

    def test_records_match_column_by_column(self, device: str) -> None:
        generator = np.random.default_rng(9)
        logits = generator.normal(scale=2.0, size=(150, 25)).astype(np.float32)
        targets = (generator.random((150, 25)) > 0.9).astype(np.float32)
        mask = (generator.random((150, 25)) > 0.15).astype(np.float32)

        probabilities = probabilities_from_logits(logits, "multi_label")
        expected = per_class_metrics(probabilities, targets, 0.5, mask)
        produced = per_class_metrics_batch(logits, targets, mask, 0.5, device=device)

        assert len(produced) == len(expected)
        for reference_record, actual_record in zip(expected, produced):
            assert set(actual_record) == set(reference_record)
            for key, value in reference_record.items():
                if np.isnan(value):
                    assert np.isnan(actual_record[key]), key
                else:
                    assert actual_record[key] == pytest.approx(value, abs=TOLERANCE), key


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_cuda_and_cpu_agree() -> None:
    """The device must not change the result."""
    generator = np.random.default_rng(10)
    logits = generator.normal(scale=2.0, size=(200, 60)).astype(np.float32)
    targets = (generator.random((200, 60)) > 0.9).astype(np.float32)
    mask = (generator.random((200, 60)) > 0.1).astype(np.float32)

    on_cpu = evaluate_head_batch(logits, targets, mask, device="cpu")
    on_cuda = evaluate_head_batch(logits, targets, mask, device="cuda")

    for key, value in on_cpu.items():
        if np.isnan(value):
            assert np.isnan(on_cuda[key]), key
        else:
            assert on_cuda[key] == pytest.approx(value, abs=1e-5), key
