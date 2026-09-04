"""Tests for model-dataset train, validation, and test partitions."""

from __future__ import annotations

import pytest
import torch

from msi_autoencoder_wrapper.models.datasets.base_dataset import MSIBaseDataset
from msi_autoencoder_wrapper.models.datasets.base_dataset import RawMSIBaseDataset
from msi_autoencoder_wrapper.models.datasets.splitting import DatasetSplitter
from msi_autoencoder_wrapper.models.datasets.multilabel_sampling import (
    select_proportional_multilabel_indices,
    split_proportional_multilabel_indices,
)
from msi_autoencoder_wrapper.models.datasets.subsetting import DatasetSubsetter
from msi_autoencoder_wrapper.utils.exceptions import ValidationError


class SplitDataset(MSIBaseDataset):
    """Small dataset exposing stable IDs, groups, targets, and masks."""

    def __init__(self, split=None):
        super().__init__(split=split)

    def _source_length(self):
        return 20

    def _get_source_item(self, index):
        return index, torch.tensor([float(index)])

    def _get_source_sample_id(self, index):
        return {"image_key": f"image-{index // 5}", "spectrum_id": index % 5}

    def _get_source_split_group(self, index, group_fields, **kwargs):
        assert group_fields == ["image_key"]
        return index // 5

    def _get_source_split_target(self, index, target_field, **kwargs):
        assert target_field == "condition"
        return index % 2

    def _get_source_split_mask(self, index, mask, **kwargs):
        assert mask == "tissue"
        return index < 10

    def _source_subset_groups(self, source_indices, group_fields=None, **kwargs):
        assert group_fields == ["image_key"]
        return [
            self._get_source_sample_id(index)["image_key"]
            for index in source_indices
        ]


class RawSplitDataset(RawMSIBaseDataset):
    """Small raw dataset used to verify source-index mapping once."""

    def _source_length(self):
        return 10

    def _get_source_item(self, source_index):
        return source_index

    def _get_raw_source_item(self, source_index):
        return {"source_index": source_index}


class UnevenGroupedDataset(MSIBaseDataset):
    """Dataset with large, indivisible groups used to test split balancing."""

    group_sizes = (20, 17, 8, 7, 6, 6, 5, 5, 4, 4, 4, 4, 4, 3, 3, 4)
    groups = tuple(
        group
        for group, size in enumerate(group_sizes)
        for _ in range(size)
    )

    def _source_length(self):
        return len(self.groups)

    def _get_source_item(self, index):
        return index

    def _get_source_split_group(self, index, **kwargs):
        del kwargs
        return self.groups[index]


class MultiLabelSplitDataset(MSIBaseDataset):
    """Synthetic multi-label pixels distributed equally across source images."""

    group_size = 20
    image_count = 3
    groups = tuple(image_index for image_index in range(3) for _ in range(20))
    positive_labels = tuple(
        frozenset({index % 3})
        for index in range(60)
    )

    def _source_length(self):
        return len(self.groups)

    def _get_source_item(self, index):
        return index

    def get_multilabel_split_data(self, indices, group_fields, **kwargs):
        del kwargs
        assert group_fields == ["image_key"]
        return (
            [self.groups[index] for index in indices],
            [self.positive_labels[index] for index in indices],
        )


def test_random_split_is_owned_by_dataset_and_reproducible() -> None:
    config = {
        "strategy": "random",
        "seed": 7,
        "fractions": {"train": 0.6, "validation": 0.2, "test": 0.2},
    }
    first = SplitDataset(split=config).create_partitions()
    second = SplitDataset(split=config).create_partitions()

    assert first.manifest.assignments == second.manifest.assignments
    assert sum(len(dataset) for _, dataset in first.items()) == 20


def test_grouped_split_never_separates_an_image() -> None:
    partitions = DatasetSplitter.split(
        SplitDataset(),
        {
            "strategy": "grouped",
            "seed": 4,
            "fractions": {"train": 0.5, "validation": 0.0, "test": 0.5},
            "parameters": {"group_fields": ["image_key"]},
        },
    )
    assignments = partitions.manifest.assignments
    locations = {}
    for split_name, sample_ids in assignments.items():
        for sample_id in sample_ids:
            locations.setdefault(sample_id["image_key"], set()).add(split_name)
    assert all(len(values) == 1 for values in locations.values())


def test_grouped_split_balances_uneven_groups_by_requested_fraction() -> None:
    """Large groups stay in train when smaller groups can fit evaluation targets."""
    partitions = DatasetSplitter.split(
        UnevenGroupedDataset(),
        {
            "strategy": "grouped",
            "seed": 42,
            "fractions": {"train": 0.8, "validation": 0.1, "test": 0.1},
            "parameters": {"group_fields": ["dataset_id"]},
        },
    )
    sizes = {
        name: 0 if partition is None else len(partition)
        for name, partition in partitions.items()
    }

    assert abs(sizes["train"] - 80) <= 4
    assert abs(sizes["validation"] - 10) <= 4
    assert abs(sizes["test"] - 10) <= 4


def test_single_target_and_mask_stratification_preserve_both_classes() -> None:
    for strategy, parameters in (
        ("target_stratified", {"target_field": "condition"}),
        ("mask_stratified", {"mask": "tissue"}),
    ):
        partitions = DatasetSplitter.split(
            SplitDataset(),
            {
                "strategy": strategy,
                "seed": 3,
                "fractions": {"train": 0.5, "validation": 0.0, "test": 0.5},
                "parameters": parameters,
            },
        )
        assert len(partitions.train) == 10
        assert len(partitions.test) == 10


def test_predefined_split_uses_stable_sample_ids() -> None:
    dataset = SplitDataset()
    train_ids = [dataset.get_sample_id(index) for index in range(1, 20)]
    test_id = dataset.get_sample_id(0)
    partitions = DatasetSplitter.split(
        dataset,
        {
            "strategy": "predefined",
            "fractions": {"train": 0.95, "validation": 0.0, "test": 0.05},
            "assignments": {
                "train": train_ids,
                "validation": [],
                "test": [test_id],
            },
        },
    )
    assert partitions.manifest.assignments["test"] == (test_id,)


def test_stratified_subset_is_reproducible_and_preserves_groups() -> None:
    """The virtual subset selects the same proportional groups for one seed."""
    dataset = SplitDataset()
    first = DatasetSubsetter.select_indices(
        source_length=dataset._source_length(),
        group_provider=dataset._source_subset_groups,
        config={
            "fraction": 0.5,
            "seed": 11,
            "method": "stratified_random",
            "parameters": {"group_fields": ["image_key"]},
        },
    )
    second = DatasetSubsetter.select_indices(
        source_length=dataset._source_length(),
        group_provider=dataset._source_subset_groups,
        config={
            "fraction": 0.5,
            "seed": 11,
            "method": "stratified_random",
            "parameters": {"group_fields": ["image_key"]},
        },
    )

    assert first == second
    assert len(first) == 10
    assert {
        dataset.get_sample_id(index)["image_key"]
        for index in first
    } == {
        f"image-{index}" for index in range(4)
    }


def test_base_dataset_maps_public_indices_once_for_every_public_accessor() -> None:
    """Selection changes public indices without wrapping the dataset object."""
    dataset = SplitDataset().subset(
        {"fraction": 0.5, "seed": 9, "method": "random"}
    )

    assert isinstance(dataset, SplitDataset)
    assert len(dataset) == 10
    source_index = dataset[0][0]
    assert dataset.get_sample_id(0) == {
        "image_key": f"image-{source_index // 5}",
        "spectrum_id": source_index % 5,
    }
    assert dataset.get_split_target(0, target_field="condition") == source_index % 2


def test_raw_dataset_maps_public_indices_before_raw_loading() -> None:
    """Raw access uses the same selection mapping as dense item access."""
    dataset = RawSplitDataset().subset(
        {"fraction": 0.4, "seed": 3, "method": "random"}
    )

    assert dataset.get_raw_item(0)["source_index"] == dataset[0]


def test_proportional_multilabel_subset_preserves_image_and_positive_ratios() -> None:
    """Selection retains every positive label while honoring image quotas."""
    groups = ["large"] * 20 + ["small"] * 10
    labels = [frozenset({0}) for _ in range(20)] + [
        frozenset({1}) if index < 5 else frozenset({0})
        for index in range(10)
    ]

    selected = select_proportional_multilabel_indices(
        groups,
        labels,
        fraction=0.4,
        seed=11,
        minimum_positive_count=1,
    )

    assert len(selected) == 12
    assert sum(groups[index] == "large" for index in selected) == 8
    assert sum(groups[index] == "small" for index in selected) == 4
    assert {label for index in selected for label in labels[index]} == {0, 1}


def test_proportional_multilabel_subset_treats_class_prevalence_as_a_soft_target() -> None:
    """Image quotas remain feasible when two labels share a small image."""
    groups = ["concentrated"] * 6 + ["other"] * 4
    labels = [frozenset({0})] * 3 + [frozenset({1})] * 3 + [frozenset()] * 4

    selected = select_proportional_multilabel_indices(
        groups,
        labels,
        fraction=0.5,
        seed=19,
        minimum_positive_count=1,
    )

    # At this fraction, each label's rounded independent target would be two,
    # but their four required positions cannot fit into the image quota of
    # three. Coverage is hard; prevalence is optimized by the random fill.
    assert len(selected) == 5
    assert sum(groups[index] == "concentrated" for index in selected) == 3
    assert {label for index in selected for label in labels[index]} == {0, 1}


def test_proportional_multilabel_subset_is_registered_in_subsetter() -> None:
    """The public subset method forwards grouping parameters and sparse labels."""
    groups = ["image-a"] * 8 + ["image-b"] * 4
    labels = [frozenset({index % 2}) for index in range(12)]

    def multilabel_provider(indices, group_fields):
        assert group_fields == ["image_key"]
        return (
            [groups[index] for index in indices],
            [labels[index] for index in indices],
        )

    selected = DatasetSubsetter.select_indices(
        source_length=len(groups),
        group_provider=lambda indices, **kwargs: ["unused" for _ in indices],
        multilabel_provider=multilabel_provider,
        config={
            "fraction": 0.5,
            "seed": 5,
            "method": "proportional_multilabel",
            "parameters": {
                "group_fields": ["image_key"],
                "minimum_positive_count": 1,
            },
        },
    )

    assert len(selected) == 6
    assert sum(groups[index] == "image-a" for index in selected) == 4
    assert sum(groups[index] == "image-b" for index in selected) == 2
    assert {label for index in selected for label in labels[index]} == {0, 1}


def test_proportional_multilabel_split_preserves_each_image_and_class() -> None:
    """Every positive class is available in train, validation, and test."""
    dataset = MultiLabelSplitDataset()
    partitions = DatasetSplitter.split(
        dataset,
        {
            "strategy": "proportional_multilabel",
            "seed": 17,
            "fractions": {"train": 0.8, "validation": 0.1, "test": 0.1},
            "parameters": {
                "group_fields": ["image_key"],
                "minimum_positive_per_split": 1,
            },
        },
    )

    split_indices = {
        name: [] if partition is None else list(partition.indices)
        for name, partition in partitions.items()
    }
    for values in split_indices.values():
        assert len(values) > 0
        assert {
            label for index in values for label in dataset.positive_labels[index]
        } == {0, 1, 2}
    assert [
        sum(label in dataset.positive_labels[index] for index in split_indices["train"])
        for label in range(3)
    ] == [16, 16, 16]
    assert [
        sum(label in dataset.positive_labels[index] for index in split_indices["validation"])
        for label in range(3)
    ] == [2, 2, 2]
    assert [
        sum(label in dataset.positive_labels[index] for index in split_indices["test"])
        for label in range(3)
    ] == [2, 2, 2]
    for image_index in range(dataset.image_count):
        assert (
            sum(dataset.groups[index] == image_index for index in split_indices["train"])
            == 16
        )
        assert (
            sum(
                dataset.groups[index] == image_index
                for index in split_indices["validation"]
            )
            == 2
        )
        assert (
            sum(dataset.groups[index] == image_index for index in split_indices["test"])
            == 2
        )


def test_proportional_multilabel_split_rejects_uncoverable_class() -> None:
    """A class with insufficient source support cannot silently disappear."""
    with pytest.raises(ValidationError, match="cannot cover every split"):
        split_proportional_multilabel_indices(
            groups=["image"] * 10,
            positive_labels=[frozenset({0})] * 2 + [frozenset()] * 8,
            fractions={"train": 0.8, "validation": 0.1, "test": 0.1},
            seed=0,
            minimum_positive_per_split=1,
        )
