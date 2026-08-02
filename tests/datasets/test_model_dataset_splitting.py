"""Tests for model-dataset train, validation, and test partitions."""

from __future__ import annotations

import torch

from msi_autoencoder_wrapper.models.datasets.base_dataset import MSIBaseDataset
from msi_autoencoder_wrapper.models.datasets.splitting import DatasetSplitter


class SplitDataset(MSIBaseDataset):
    """Small dataset exposing stable IDs, groups, targets, and masks."""

    def __init__(self, split=None):
        super().__init__(split=split)

    def __len__(self):
        return 20

    def __getitem__(self, index):
        return index, torch.tensor([float(index)])

    def get_sample_id(self, index):
        return {"image_key": f"image-{index // 5}", "spectrum_id": index % 5}

    def get_split_group(self, index, group_fields, **kwargs):
        assert group_fields == ["image_key"]
        return index // 5

    def get_split_target(self, index, target_field, **kwargs):
        assert target_field == "condition"
        return index % 2

    def get_split_mask(self, index, mask, **kwargs):
        assert mask == "tissue"
        return index < 10


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

