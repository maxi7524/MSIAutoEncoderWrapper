"""Class sets targeted by the synthetic-pretraining quotas, derived from manifests and annotations."""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np

from msi_autoencoder_wrapper.analysis.autoencoder.experiments.pretraining_campaign_synthetic import (
    TOKEN_KINDS,
    collision_pairs,
    exposure_table,
    rare_reference,
)

KIND = {name: code for code, name in TOKEN_KINDS.items()}


def test_exposure_counts_tokens_and_rows_per_population_class_and_kind() -> None:
    manifest = SimpleNamespace(
        requested_target_indices=np.array([[0, 0, 2], [1, -1, -1]]),
        component_kinds=np.array([[KIND["base_class"], KIND["rare_bonus"], KIND["base_class"]],
                                  [KIND["overlap_bonus"], KIND["blank"], 0]]))
    table = exposure_table({"mix": manifest}, class_count=3).set_index(["class_index", "kind"])

    assert table.loc[(0, "base_class"), "tokens"] == 1
    assert table.loc[(0, "rare_bonus"), "tokens"] == 1
    assert table.loc[(2, "base_class"), "rows"] == 1
    assert table.loc[(1, "overlap_bonus"), "tokens"] == 1
    ## Blank tokens carry no class and padding is not a kind
    assert "blank" not in table.index.get_level_values("kind")


def test_collision_pairs_follow_bins_within_training_pixels() -> None:
    ## Pixels 10 and 11 (train) and 12 (not train); identities map to classes a, b, c
    index = SimpleNamespace(
        annotation_identities=[("a",), ("b",), ("c",), ("unknown",)],
        spectrum_ids=np.array([10, 11, 12]),
        spectrum_offsets=np.array([0, 4, 6, 8]),
        annotation_indices=np.array([0, 1, 2, 3, 0, 1, 0, 2]),
        coordinate_indices=np.array([5, 5, 7, 5, 5, 6, 3, 3]))
    pairs, counts = collision_pairs(index, np.array([10, 11]), ("a", "b", "c"), feature_count=10)

    ## Pixel 10: a and b share bin 5 (the unknown identity is ignored); pixel 11: a and b in
    ## different bins; pixel 12 is not a training pixel, so a-c never collide
    assert pairs[["class_a", "class_b"]].to_numpy().tolist() == [[0, 1]]
    assert pairs.co_annotations.tolist() == [1] and pairs.pixels.tolist() == [1]
    assert counts.tolist() == [2, 2, 1]


def test_rare_reference_takes_the_least_annotated_eligible_quarter() -> None:
    counts = np.array([5, 1, 9, 1, 7, 3, 2, 8])
    eligible = np.array([0, 1, 2, 3, 4, 5, 6])
    ## ceil(0.25 * 7) = 2 classes; ties at count 1 are broken by class index
    assert rare_reference(counts, eligible, 0.25).tolist() == [1, 3]
    assert rare_reference(counts, eligible, 0.5).tolist() == [1, 3, 5, 6]
