"""Paired-seed inference, duplicate protection and validation-only selection."""

import numpy as np
import pandas as pd
import pytest

from msi_autoencoder_wrapper.analysis.autoencoder.experiments.predictive_reports import (
    baseline_contrasts, condition_order, condition_summary, decision_table, experimental_units,
    paired_comparisons, resolve_shortlist,
)


def _inventory():
    return pd.DataFrame([
        dict(model_id=f"{condition}-{seed}-{duplicate}", source="campaign", role="candidate", condition=condition,
             label=condition, repetition=seed, initialization_seed=100 + seed, training_seed=200 + seed,
             data_contract="data", backbone_contract="backbone", training_contract="training")
        for condition in ("A", "B") for seed in (0, 1) for duplicate in range(2 if condition == "A" else 1)
    ])


def test_duplicate_conditions_do_not_inflate_seed_count_and_actual_seeds_must_match():
    inventory = _inventory()
    frame = inventory[["model_id", "condition", "label", "repetition"]].copy()
    frame["value"] = np.where(frame.condition == "A", .8, .5)
    frame["metric"] = "ap"
    units = experimental_units(frame, inventory, ["metric"])
    assert len(units) == 4
    assert set(units.query("condition == 'A'").task_count) == {2}
    differences, summary = paired_comparisons(units, ["metric"])
    assert len(differences) == 2
    assert summary.iloc[0].pairs == 2
    assert summary.iloc[0].mean_difference == pytest.approx(.3, abs=1e-12)
    changed = inventory.copy()
    changed.loc[changed.condition == "B", "training_seed"] += 1000
    differences, summary = paired_comparisons(experimental_units(frame, changed, ["metric"]), ["metric"])
    assert differences.empty
    assert summary.iloc[0].pairs == 0


def test_test_scores_cannot_change_selection_and_incomplete_seeds_are_ineligible():
    inventory = _inventory()
    predictions, reconstruction = [], []
    for row in inventory.to_dict("records"):
        for split in ("train", "validation", "test"):
            for population in ("annotation_retrieval", "operational_pn"):
                predictions.append({**row, "split": split, "population": population, "scope": "train_supported",
                                    "metric": "average_precision", "value": .8 if row["condition"] == "A" else .6})
        reconstruction.append({**row, "split": "validation", "metric": "masserstein", "statistic": "mean", "value": .1})
    predictions, reconstruction = pd.DataFrame(predictions), pd.DataFrame(reconstruction)
    first = decision_table(predictions, reconstruction, inventory, expected_seeds=2).set_index("condition")
    assert first.loc["A", "validation_rank"] == 1
    assert first.loc["A", "validation_pareto"]
    predictions.loc[predictions.split == "test", "value"] = -1000
    second = decision_table(predictions, reconstruction, inventory, expected_seeds=2).set_index("condition")
    pd.testing.assert_series_equal(first.validation_rank, second.validation_rank)
    pd.testing.assert_series_equal(first.validation_pareto, second.validation_pareto)
    assert not decision_table(predictions, reconstruction, inventory, expected_seeds=5).eligible.any()


def test_condition_order_puts_the_baseline_first_and_groups_loss_families():
    inventory = pd.DataFrame([
        dict(model_id="1", source="new", role="candidate", condition="c1", label="vpu (VariationalPULoss)",
             family="VariationalPULoss", weight_mode="none"),
        dict(model_id="2", source="new", role="candidate", condition="c2", label="bce_global (Weighted)",
             family="Weighted", weight_mode="global"),
        dict(model_id="3", source="new", role="candidate", condition="c3", label="bce_per_class (Weighted)",
             family="Weighted", weight_mode="per_class"),
        dict(model_id="4", source="old", role="baseline", condition="c4", label="balanced (Balanced)",
             family="Balanced", weight_mode="none"),
    ])
    order = condition_order(inventory)
    assert order[0] == "balanced (Balanced)"
    assert order.index("bce_global (Weighted)") + 1 == order.index("bce_per_class (Weighted)")
    assert set(order) == set(inventory.label)


def test_baseline_contrasts_orient_every_row_as_candidate_minus_reference():
    inventory = pd.DataFrame([dict(source="new", role="candidate"), dict(source="old", role="baseline")])
    contrasts = pd.DataFrame([
        # The baseline appears on the left in one row and on the right in the other.
        dict(left="cand", right="base", left_source="new", right_source="old", left_condition="c", right_condition="b",
             pairs=5, mean_difference=.04, median_difference=.03, positive_pairs=4, ci_low=.01, ci_high=.07, metric="ap"),
        dict(left="base", right="cand", left_source="old", right_source="new", left_condition="b", right_condition="c",
             pairs=5, mean_difference=-.04, median_difference=-.03, positive_pairs=1, ci_low=-.07, ci_high=-.01, metric="ap"),
        dict(left="cand", right="other", left_source="new", right_source="new", left_condition="c", right_condition="d",
             pairs=5, mean_difference=.1, median_difference=.1, positive_pairs=5, ci_low=.05, ci_high=.15, metric="ap"),
    ])
    oriented = baseline_contrasts(contrasts, inventory)
    assert len(oriented) == 2  # The candidate-versus-candidate comparison is dropped.
    assert set(oriented.label) == {"cand"} and set(oriented.reference) == {"base"}
    assert oriented.mean_difference.tolist() == pytest.approx([.04, .04])
    assert oriented.ci_low.tolist() == pytest.approx([.01, .01])
    assert oriented.ci_high.tolist() == pytest.approx([.07, .07])
    assert oriented.positive_pairs.tolist() == [4, 4]


def test_shortlist_prefers_settings_rejects_unknown_labels_and_falls_back_to_the_ranking():
    order = ["a", "b", "c", "d"]
    decision = pd.DataFrame({"label": order, "eligible": [True, True, True, False],
                             "validation_rank": [3., 1., 2., np.nan]})
    assert resolve_shortlist({"shortlist": ["c", "a"]}, order) == ["c", "a"]
    with pytest.raises(ValueError, match="not present"):
        resolve_shortlist({"shortlist": ["z"]}, order)
    assert resolve_shortlist({}, order, decision=decision, size=2) == ["b", "c"]
    assert resolve_shortlist({}, order, size=2) == ["a", "b"]


def test_condition_summary_keeps_the_seed_count_next_to_the_mean():
    units = pd.DataFrame([
        dict(source="new", role="candidate", condition="c", label="cand", metric="ap", value=value)
        for value in (.4, .6)
    ])
    summary = condition_summary(units, ["metric"]).iloc[0]
    assert summary["mean"] == pytest.approx(.5)
    assert summary.minimum == pytest.approx(.4) and summary.maximum == pytest.approx(.6)
    assert summary.seeds == 2


def test_decision_table_reports_pooled_metrics_without_letting_them_select():
    inventory = _inventory()
    predictions, reconstruction = [], []
    for row in inventory.to_dict("records"):
        for split in ("train", "validation", "test"):
            for population in ("annotation_retrieval", "operational_pn"):
                base = {**row, "split": split, "population": population, "scope": "train_supported"}
                predictions.append({**base, "metric": "average_precision", "value": .8 if row["condition"] == "A" else .6})
                # The pooled quantity deliberately contradicts the macro ordering.
                predictions.append({**base, "metric": "micro_average_precision", "value": .1 if row["condition"] == "A" else .9})
        reconstruction.append({**row, "split": "validation", "metric": "masserstein", "statistic": "mean", "value": .1})
    table = decision_table(pd.DataFrame(predictions), pd.DataFrame(reconstruction), inventory,
                           expected_seeds=2).set_index("condition")
    assert "micro_validation_annotation_retrieval" in table
    assert table.loc["A", "micro_validation_annotation_retrieval"] == pytest.approx(.1)
    # Selection still follows the macro quantity.
    assert table.loc["A", "validation_rank"] == 1
