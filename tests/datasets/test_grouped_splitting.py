"""Tests for leakage-safe dataset-level splitting."""

from __future__ import annotations

from msi_dataset_manager.operations import grouped_dataset_split


def test_grouped_split_never_separates_patient_datasets() -> None:
    """All datasets belonging to one patient stay in the same partition."""
    records = [
        {
            "source": "metaspace",
            "dataset_id": f"dataset-{index}",
            "metadata": {"patient_id": f"patient-{index // 2}"},
        }
        for index in range(10)
    ]

    first = grouped_dataset_split(
        records,
        group_fields=["patient_id"],
        validation_fraction=0.2,
        test_fraction=0.2,
        seed=42,
    )
    second = grouped_dataset_split(
        records,
        group_fields=["patient_id"],
        validation_fraction=0.2,
        test_fraction=0.2,
        seed=42,
    )

    assert first == second
    patient_partitions = {}
    for partition, partition_records in first.items():
        for record in partition_records:
            patient = record["metadata"]["patient_id"]
            patient_partitions.setdefault(patient, set()).add(partition)
    assert all(len(partitions) == 1 for partitions in patient_partitions.values())


def test_missing_group_metadata_does_not_group_unrelated_datasets() -> None:
    """Unknown patients receive per-dataset fallback groups."""
    records = [
        {"source": "metaspace", "dataset_id": f"dataset-{index}", "metadata": {}}
        for index in range(4)
    ]

    split = grouped_dataset_split(
        records,
        group_fields=["patient_id"],
        validation_fraction=0.25,
        test_fraction=0.25,
        seed=1,
    )

    assert {name: len(values) for name, values in split.items()} == {
        "train": 2,
        "validation": 1,
        "test": 1,
    }
