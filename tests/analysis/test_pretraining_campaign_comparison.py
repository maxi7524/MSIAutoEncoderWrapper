"""Known-answer checks of the paired comparisons of pretraining variants with the baselines."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from scipy import stats

from msi_autoencoder_wrapper.analysis.autoencoder.experiments.pretraining_campaign_comparison import (
    angle_between,
    calibration_table,
    factorial_effects,
    logits_from_probabilities,
    metric_direction,
    paired_improvements,
    pairwise_discrimination,
    rank_variants,
    ridge_probe,
    summarize,
    true_label_ranks,
    variant_contrasts,
)


def _metrics(rows: list[tuple]) -> pd.DataFrame:
    """Metric table from ``(variant, repetition, metric, value)`` rows on one axis and stage."""
    return pd.DataFrame([{"axis": "a", "stage": "s", "variant": variant, "repetition": repetition,
                          "family": "f", "metric": metric, "evaluation": "test", "ranking": "none", "value": value}
                         for variant, repetition, metric, value in rows])


def test_improvements_are_paired_by_repetition_and_signed_by_metric_direction() -> None:
    metrics = _metrics([("v", 0, "masserstein", 3.0), ("v", 1, "masserstein", 5.0),
                        ("v", 0, "average_precision", 0.6), ("v", 1, "average_precision", 0.4)])
    reference = _metrics([("baseline", 0, "masserstein", 4.0), ("baseline", 1, "masserstein", 4.0),
                          ("baseline", 0, "average_precision", 0.5), ("baseline", 1, "average_precision", 0.5)])
    paired = paired_improvements(metrics, reference).set_index(["metric", "repetition"])

    ## Lower Masserstein and higher AP are both positive improvements
    assert paired.loc[("masserstein", 0), "improvement"] == pytest.approx(1.0)
    assert paired.loc[("masserstein", 1), "improvement"] == pytest.approx(-1.0)
    assert paired.loc[("average_precision", 0), "improvement"] == pytest.approx(0.1)
    assert paired.loc[("masserstein", 0), "relative_improvement"] == pytest.approx(0.25)
    ## Regression: a lower false-positive lift (less partner confusion) is an improvement
    assert metric_direction("false_positive_lift") == -1
    assert metric_direction("discrimination") == 1
    with pytest.raises(ValueError, match="more than one value"):
        paired_improvements(metrics, pd.concat([reference, reference]))


def test_summary_reports_student_interval_and_sign_counts() -> None:
    frame = pd.DataFrame({"group": ["x"] * 3, "improvement": [1.0, 2.0, 3.0]})
    summary = summarize(frame, ["group"]).iloc[0]
    half = stats.t.ppf(0.975, 2) * 1.0 / np.sqrt(3)

    assert (summary.n, summary["mean"], summary.sd) == (3, 2.0, 1.0)
    assert summary.ci_low == pytest.approx(2.0 - half)
    assert summary.ci_high == pytest.approx(2.0 + half)
    assert (summary.positive, summary.negative) == (3, 0)
    ## A single repetition has no interval
    assert np.isnan(summarize(frame.iloc[:1], ["group"]).iloc[0].ci_low)


def test_variants_are_ranked_by_mean_rank_with_tie_breaker() -> None:
    summary = pd.DataFrame({"variant": ["p", "q", "p", "q"], "family": ["r", "r", "h", "h"],
                            "metric": ["masserstein", "masserstein", "average_precision", "average_precision"],
                            "ranking": ["none", "none", "ar", "ar"], "evaluation": "test", "mean": [0.2, 0.1, 0.01, 0.05]})
    selection = [{"family": "r", "metric": "masserstein"}, {"family": "h", "metric": "average_precision",
                                                           "ranking": "ar"}]
    ranking = rank_variants(summary, selection, evaluation="test",
                            tie_breaker={"family": "h", "metric": "average_precision"})

    ## Equal mean ranks (1.5 each); q has the better AP improvement
    assert ranking.variant.tolist() == ["q", "p"]
    assert ranking.mean_rank.tolist() == [1.5, 1.5]


def test_contrasts_average_each_side_per_repetition_and_skip_missing_variants() -> None:
    metrics = _metrics([("a", 0, "average_precision", 0.5), ("b", 0, "average_precision", 0.3),
                        ("c", 0, "average_precision", 0.1), ("a", 1, "average_precision", 0.6),
                        ("b", 1, "average_precision", 0.2), ("c", 1, "average_precision", 0.2)])
    contrasts = variant_contrasts(metrics, {"a_vs_bc": {"plus": ["a"], "minus": ["b", "c"]},
                                            "missing": {"plus": ["z"], "minus": ["a"]}}).set_index("repetition")

    assert set(contrasts.contrast) == {"a_vs_bc"}
    assert contrasts.loc[0, "improvement"] == pytest.approx(0.5 - 0.2)
    assert contrasts.loc[1, "improvement"] == pytest.approx(0.6 - 0.2)


def test_factorial_effects_recover_main_effects_and_interaction() -> None:
    ## y = 2 overlap + 1 rare + 0.5 overlap*rare; schedule has no effect
    factors, rows = {}, []
    for schedule in ("joint", "staged"):
        for overlap in (0, 1):
            for rare in (0, 1):
                name = f"{schedule}_{overlap}{rare}"
                factors[name] = {"schedule": schedule, "overlap": overlap, "rare": rare}
                rows.append((name, 0, "average_precision", 2.0 * overlap + rare + 0.5 * overlap * rare))
    factors["single"] = {"components": "single"}
    rows.append(("single", 0, "average_precision", 100.0))
    effects = factorial_effects(_metrics(rows), factors, {"schedule": ["joint", "staged"], "overlap": [0, 1],
                                                          "rare": [0, 1]}).set_index("term").effect

    ## Variants outside the design (single) are ignored
    assert effects["overlap"] == pytest.approx(2.25)
    assert effects["rare"] == pytest.approx(1.25)
    assert effects["schedule"] == pytest.approx(0.0)
    assert effects["overlap:rare"] == pytest.approx(0.25)
    assert effects["schedule:overlap"] == pytest.approx(0.0)


def test_calibration_is_zero_for_calibrated_scores_and_brier_matches_definition() -> None:
    probabilities = np.full((4, 2), 0.25)
    labels = np.array([[1, 0], [0, 0], [0, 1], [0, 0]])
    table, summary = calibration_table(probabilities, labels)

    assert summary["expected_calibration_error"] == pytest.approx(0.0)
    assert summary["brier_score"] == pytest.approx(np.mean((0.25 - labels) ** 2))
    assert table.entries.sum() == 8
    assert table.set_index("bin").loc[2, "positive_rate"] == pytest.approx(0.25)


def test_true_label_ranks_give_best_rank_to_ties_and_nan_to_unannotated() -> None:
    ranks = true_label_ranks(np.array([[3.0, 1.0, 2.0], [1.0, 1.0, 0.0]]),
                             np.array([[0, 1, 1], [1, 1, 0]]))

    assert np.isnan(ranks[0, 0]) and ranks[0, 1] == 3 and ranks[0, 2] == 2
    assert ranks[1, 0] == 1 and ranks[1, 1] == 1


def test_pairwise_discrimination_pools_both_directions_and_measures_partner_confusion() -> None:
    scores = np.array([[2.0, 1.0], [0.0, 1.0], [0.0, 3.0], [0.5, -1.0]])
    labels = np.array([[1, 0], [1, 0], [0, 1], [0, 0]])
    table = pairwise_discrimination(scores, labels, np.ones_like(labels), np.array([[0, 1]])).iloc[0]

    assert table.pixels == 3
    assert table.discrimination == pytest.approx(2.0 / 3.0)
    assert table.margin == pytest.approx((1.0 - 1.0 + 3.0) / 3.0)
    assert table.false_positive_rate_when_partner_present == pytest.approx(2.0 / 3.0)
    assert table.false_positive_rate_when_both_absent == pytest.approx(0.5)
    assert table.false_positive_lift == pytest.approx(2.0 / 3.0 - 0.5)


def test_ridge_probe_decodes_linear_targets_and_ignores_scale() -> None:
    generator = np.random.default_rng(0)
    features = generator.normal(size=(400, 3)) * np.array([1.0, 50.0, 0.01])
    targets = np.stack([features[:, 0] > 0, features[:, 2] > 0], axis=1).astype(float)
    evaluation = generator.normal(size=(200, 3)) * np.array([1.0, 50.0, 0.01])
    scores = ridge_probe(features, targets, [evaluation])[0]

    assert scores.shape == (200, 2)
    assert np.corrcoef(scores[:, 0], evaluation[:, 0])[0, 1] > 0.9
    assert np.corrcoef(scores[:, 1], evaluation[:, 2])[0, 1] > 0.9


def test_angles_and_probability_logits() -> None:
    assert angle_between(np.array([[1.0, 0.0], [1.0, 1.0]]), np.array([[0.0, 2.0], [2.0, 2.0]])) == pytest.approx(
        [90.0, 0.0])
    probabilities = np.array([0.1, 0.5, 0.9])
    assert 1.0 / (1.0 + np.exp(-logits_from_probabilities(probabilities))) == pytest.approx(probabilities)
    assert np.isfinite(logits_from_probabilities(np.array([0.0, 1.0]))).all()
