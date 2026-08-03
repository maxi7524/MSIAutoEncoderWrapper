"""Deterministic dataset-level splits that keep related samples together."""

from __future__ import annotations

import random
from collections import defaultdict
from typing import Any, Dict, List, Mapping, Sequence, Tuple

from ...utils.exceptions import raise_validation_error


def grouped_dataset_split(
    records: Sequence[Mapping[str, Any]],
    *,
    group_fields: Sequence[str],
    validation_fraction: float = 0.15,
    test_fraction: float = 0.15,
    seed: int = 0,
) -> Dict[str, List[Dict[str, Any]]]:
    """Split catalog records without separating a patient or experiment group.

    :param records: Dataset catalog records. Group fields are read first from
        each top-level record and then from its ``metadata`` mapping.
    :type records: Sequence[Mapping[str, Any]]
    :param group_fields: Metadata fields defining one independent group, for
        example ``("patient_id",)`` or ``("experiment_id", "replicate")``.
    :type group_fields: Sequence[str]
    :param validation_fraction: Approximate fraction assigned to validation.
    :type validation_fraction: float
    :param test_fraction: Approximate fraction assigned to testing.
    :type test_fraction: float
    :param seed: Deterministic group-shuffling seed.
    :type seed: int
    :return: Mapping with ``train``, ``validation``, and ``test`` record lists.
    :rtype: Dict[str, List[Dict[str, Any]]]
    :raises ValidationError: If fractions or group fields are invalid.

    Missing group values do not combine unrelated datasets. Each such record
    receives a unique fallback key based on its source and dataset identifier.
    The function groups datasets, not spectra or pixels.
    """
    if not group_fields:
        raise_validation_error("DatasetSplit", "At least one group field is required.")
    if validation_fraction < 0 or test_fraction < 0:
        raise_validation_error("DatasetSplit", "Split fractions cannot be negative.")
    if validation_fraction + test_fraction >= 1:
        raise_validation_error(
            "DatasetSplit",
            "validation_fraction + test_fraction must be less than one.",
        )

    groups: Dict[Tuple[Any, ...], List[Dict[str, Any]]] = defaultdict(list)
    for position, source_record in enumerate(records):
        record = dict(source_record)
        metadata = record.get("metadata", {})
        metadata = metadata if isinstance(metadata, Mapping) else {}
        group_values = tuple(
            record.get(field, metadata.get(field))
            for field in group_fields
        )
        if any(value is None or value == "" for value in group_values):
            group_values = (
                "__ungrouped__",
                record.get("source", ""),
                record.get("dataset_id", position),
            )
        groups[group_values].append(record)

    group_items = list(groups.values())
    random.Random(seed).shuffle(group_items)
    total_records = len(records)
    validation_target = round(total_records * validation_fraction)
    test_target = round(total_records * test_fraction)
    result: Dict[str, List[Dict[str, Any]]] = {
        "train": [],
        "validation": [],
        "test": [],
    }
    for group in group_items:
        if len(result["test"]) < test_target:
            destination = "test"
        elif len(result["validation"]) < validation_target:
            destination = "validation"
        else:
            destination = "train"
        result[destination].extend(group)
    return result
