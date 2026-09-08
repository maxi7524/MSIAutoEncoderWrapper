"""Paired-seed inference, duplicate protection and validation-only selection."""

import numpy as np
import pandas as pd
import pytest

from msi_autoencoder_wrapper.analysis.autoencoder.experiments.predictive_reports import (
    decision_table, experimental_units, paired_comparisons,
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
