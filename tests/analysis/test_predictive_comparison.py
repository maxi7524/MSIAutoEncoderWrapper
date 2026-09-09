"""Threshold-free ranking and matched-class generalization regressions."""

import numpy as np
import pandas as pd
import pytest
import torch
from scipy.special import expit
from sklearn.metrics import average_precision_score, roc_auc_score

from msi_autoencoder_wrapper.analysis.autoencoder.heads.metrics import probabilities_from_logits
from msi_autoencoder_wrapper.analysis.autoencoder.heads.predictive_comparison import (
    class_agreement, generalization_gaps, pooled_ranking, positive_scores, ranking_tables,
    score_histograms, state_separation,
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


def test_pooled_ranking_matches_the_reference_implementation_including_ties():
    generator = np.random.default_rng(11)
    for size, rounding in ((400, None), (400, 1), (37, 1)):
        labels = (generator.random(size) < .15).astype(int)
        scores = generator.normal(size=size)
        if rounding is not None:
            # Rounding forces score ties, which the threshold grouping must absorb.
            scores = np.round(scores, rounding)
        pooled = pooled_ranking(labels, scores)
        assert pooled["average_precision"] == pytest.approx(average_precision_score(labels, scores), abs=1e-12)
        assert pooled["roc_auc"] == pytest.approx(roc_auc_score(labels, scores), abs=1e-12)
        assert pooled["entries"] == size
        assert pooled["precision_at_recall"].shape == pooled["tpr_at_fpr"].shape


def test_pooled_ranking_is_undefined_without_both_classes():
    for labels in (np.zeros(6), np.ones(6)):
        pooled = pooled_ranking(labels, np.arange(6.))
        assert np.isnan(pooled["average_precision"]) and np.isnan(pooled["roc_auc"])
        assert pooled["entries"] == 6
        assert np.isnan(pooled["precision_at_recall"]).all()


def test_pooled_metrics_reach_the_aggregate_and_curves_use_a_shared_grid():
    generator = np.random.default_rng(5)
    targets = (generator.random((60, 3)) < .3).astype(int)
    states = np.where(targets, 1, generator.integers(0, 2, size=targets.shape) * 2)
    tables = ranking_tables(generator.normal(size=targets.shape), targets, states,
                            targets.sum(axis=0), ("a", "b", "c"))
    aggregate = tables["prediction"].set_index(["population", "scope", "metric"])
    for population in ("annotation_retrieval", "operational_pn"):
        key = (population, "train_supported")
        assert np.isfinite(aggregate.loc[(*key, "micro_average_precision"), "value"])
        assert np.isfinite(aggregate.loc[(*key, "micro_roc_auc"), "value"])
        # The pooled entry count is a denominator, not a class count.
        assert aggregate.loc[(*key, "micro_average_precision"), "entries"] > 0
    curves = tables["ranking_curves"]
    assert set(curves.curve) == {"precision_recall", "roc"}
    assert curves.groupby(["population", "curve"]).x.apply(tuple).nunique() == 1


def test_state_separation_reports_the_expected_direction_and_undefined_pairs():
    # Class 0: scores ordered P above U above N, the direction the evidence rule predicts.
    scores = np.array([[3.], [2.], [1.], [0.]])
    states = np.array([[1], [2], [2], [0]])
    frame = state_separation(scores, states, np.array([1]), ("ion",)).iloc[0]
    assert frame.positive_vs_uncertain_auc == pytest.approx(1.)
    assert frame.negative_vs_uncertain_auc == pytest.approx(0.)
    assert frame.P_entries == 1 and frame.U_entries == 2 and frame.N_entries == 1
    assert frame.negative_share_of_unannotated == pytest.approx(1 / 3)
    without_uncertain = state_separation(scores, np.array([[1], [0], [0], [0]]), np.array([1]), ("ion",)).iloc[0]
    assert np.isnan(without_uncertain.positive_vs_uncertain_auc)
    assert np.isnan(without_uncertain.negative_vs_uncertain_auc)


def test_score_histograms_share_edges_and_retain_clipped_entries():
    scores = np.array([[-100.], [0.], [100.]])
    states = np.array([[1], [2], [0]])
    frame = score_histograms(scores, states, np.array([1]), bins=4)
    assert set(frame.state) == {"P", "N", "U"}
    assert set(frame.scope) == {"all", "rare", "medium", "frequent"}
    edges = frame.groupby("quantity").left_edge.apply(tuple)
    assert edges.nunique() == 2  # One shared edge set per binned quantity.
    log_odds = frame.query("quantity == 'positive_log_odds' and scope == 'all'").set_index("state")
    assert log_odds.loc["P", "below_range"].max() == 1  # -100 is clipped, not dropped.
    assert log_odds.loc["N", "above_range"].max() == 1
    assert frame.query("quantity == 'positive_log_odds' and scope == 'all'")["count"].sum() == 3


def test_class_agreement_pairs_only_classes_both_conditions_can_rank():
    rows = []
    for label, values in (("A", [.9, .4, .1]), ("B", [.8, .5, np.nan])):
        for index, value in enumerate(values):
            rows.append(dict(model_id=f"{label}-0", label=label, split="validation", population="annotation_retrieval",
                             class_index=index, class_name=f"ion{index}", train_positives=10, frequency="medium",
                             prevalence=.1, eligible=np.isfinite(value), average_precision=value, roc_auc=value))
    paired = class_agreement(pd.DataFrame(rows))
    assert len(paired) == 2  # The class undefined for B is excluded.
    assert set(paired.class_name) == {"ion0", "ion1"}
    np.testing.assert_allclose(paired.difference, paired.value_left - paired.value_right)
    assert set(paired.left) == {"A"} and set(paired.right) == {"B"}


@pytest.mark.parametrize("device", ["cpu"] + (["cuda"] if torch.cuda.is_available() else []))
def test_batched_per_class_ranking_reproduces_the_scikit_learn_reference(device):
    # Unbounded log odds, heavy imbalance, ties, and classes that are undefined in one
    # population but not the other: the conditions the batched kernels must survive.
    generator = np.random.default_rng(23)
    targets = (generator.random((300, 12)) < .08).astype(int)
    targets[:, 0] = 0  # No positive anywhere.
    states = np.where(targets, 1, generator.integers(0, 2, size=targets.shape) * 2)
    logits = np.round(generator.normal(scale=6.0, size=targets.shape), 2)  # Rounding forces ties.
    logits[:, 3] += 500.0  # A class whose scores sit far outside any probability range.
    tables = ranking_tables(logits, targets, states, targets.sum(axis=0),
                            tuple(f"ion{index}" for index in range(targets.shape[1])), device=device)
    per_class = tables["per_class"].set_index(["population", "class_index"])
    masks = {"annotation_retrieval": states != -1, "operational_pn": (states == 1) | (states == 0)}
    for population, mask in masks.items():
        for column in range(targets.shape[1]):
            row = per_class.loc[(population, column)]
            y, s = targets[mask[:, column], column], logits[mask[:, column], column]
            if not row.eligible:
                assert np.isnan(row.average_precision) and np.isnan(row.roc_auc)
                continue
            assert row.average_precision == pytest.approx(average_precision_score(y, s), abs=1e-5)
            assert row.roc_auc == pytest.approx(roc_auc_score(y, s), abs=1e-5)
