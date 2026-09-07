"""Activation and masking conventions for auxiliary molecular heads."""

import numpy as np
import pytest

from msi_autoencoder_wrapper.analysis.autoencoder.heads.metrics import evaluate_head, probabilities_from_logits


def test_three_state_metrics_use_positive_probability():
    logits = np.array([[[0., 1000., -1000.], [1000., 0., -1000.]],
                       [[1000., 0., -1000.], [0., 1000., -1000.]]])
    targets = np.eye(2)
    probabilities = probabilities_from_logits(logits, "multi_label")
    np.testing.assert_allclose(probabilities, targets, rtol=0, atol=1e-12)
    assert evaluate_head(logits, targets, "multi_label")["micro_f1"] == 1.


def test_composition_evaluation_uses_log_counts_and_excludes_unknown_entries():
    targets = np.array([[2., 0.], [4., 1.]])
    logits = np.array([[np.log(2.), 1000.], [np.log(4.), 1000.]])
    mask = np.array([[True, False], [True, False]])
    result = evaluate_head(logits, targets, "regression", mask)
    assert result["available_entries"] == 2
    assert result["log_count_rmse"] == pytest.approx(0, abs=1e-12)
    unavailable = evaluate_head(logits, targets, "regression", np.zeros_like(mask))
    assert unavailable["available_entries"] == 0
    assert np.isnan(unavailable["log_count_mae"])
