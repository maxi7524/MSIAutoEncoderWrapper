"""Threshold-free ranking and matched-class generalization regressions."""

import numpy as np
import pandas as pd
import pytest
from scipy.special import expit

from msi_autoencoder_wrapper.analysis.autoencoder.heads.metrics import probabilities_from_logits
from msi_autoencoder_wrapper.analysis.autoencoder.heads.predictive_comparison import (
    generalization_gaps, positive_scores, ranking_tables,
)


def test_ce_positive_log_odds_matches_positive_probability():
    logits = np.array([[[1., 2., 3.], [-2., 1., 0.]], [[0., -1., 2.], [0., 0., 0.]]])
    scores = positive_scores(logits)
    assert scores.shape == (2, 2)
    assert scores.dtype == np.float64
    np.testing.assert_allclose(expit(scores), probabilities_from_logits(logits, "multi_label"), rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(positive_scores(logits + 10000), scores, rtol=1e-11, atol=1e-11)


def test_binary_ranking_preserves_extreme_logit_order_and_masks():
    # Saturating sigmoid would make the first two scores equal and lower AP.
    logits = np.array([[1001.], [1000.], [999.], [2000.]])
    y = np.array([[1], [0], [0], [0]])
    states = np.array([[1], [0], [2], [-1]])
    tables = ranking_tables(logits, y, states, np.array([3]), ("ion",))
    per_class = tables["per_class"]
    np.testing.assert_allclose(per_class.average_precision, 1., rtol=0, atol=1e-12)
    np.testing.assert_allclose(per_class.roc_auc, 1., rtol=0, atol=1e-12)
    assert per_class.set_index("population").loc["annotation_retrieval", "negatives"] == 2
    assert per_class.set_index("population").loc["operational_pn", "negatives"] == 1


def test_undefined_and_untrained_classes_are_visible_but_not_ranked():
    y = np.array([[1, 0, 1], [1, 0, 0]])
    states = np.where(y, 1, 0)
    result = ranking_tables(np.zeros_like(y), y, states, np.array([2, 0, 0]), ("all-positive", "absent", "untrained"))
    assert not result["per_class"].eligible.any()
    assert result["prediction"].value.isna().all()
    assert (result["prediction"].classes == 0).all()


def test_ties_use_standard_average_precision():
    y = np.array([[1], [0], [0], [1]])
    result = ranking_tables(np.zeros((4, 1)), y, np.where(y, 1, 0), np.array([5]), ("ion",))
    np.testing.assert_allclose(result["per_class"].average_precision, .5, atol=1e-12, rtol=0)
    np.testing.assert_allclose(result["per_class"].roc_auc, .5, atol=1e-12, rtol=0)


@pytest.mark.parametrize("logits", [np.zeros((3,)), np.zeros((3, 2, 2)), np.array([[np.inf]])])
def test_invalid_scores_rejected(logits):
    with pytest.raises(ValueError):
        positive_scores(logits)


def test_generalization_uses_common_classes_not_difference_of_macro_means():
    rows = []
    for split, values in [("train", [.9, .1]), ("validation", [.7, np.nan]), ("test", [.6, np.nan])]:
        for c, value in enumerate(values):
            rows.append(dict(model_id="run", population="annotation_retrieval", split=split, class_index=c,
                             eligible=np.isfinite(value), average_precision=value, roc_auc=value))
    gaps = generalization_gaps(pd.DataFrame(rows))
    assert set(gaps.classes) == {1}
    np.testing.assert_allclose(gaps.query("held_out == 'validation'").value, .2, atol=1e-12, rtol=0)
