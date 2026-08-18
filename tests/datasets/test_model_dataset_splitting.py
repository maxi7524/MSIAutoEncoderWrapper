"""Tests for model-dataset train, validation, and test partitions."""

from __future__ import annotations

import torch

from msi_autoencoder_wrapper.models.datasets.base_dataset import MSIBaseDataset
from msi_autoencoder_wrapper.models.datasets.base_dataset import RawMSIBaseDataset
from msi_autoencoder_wrapper.models.datasets.splitting import DatasetSplitter
from msi_autoencoder_wrapper.models.datasets.subsetting import DatasetSubsetter


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
