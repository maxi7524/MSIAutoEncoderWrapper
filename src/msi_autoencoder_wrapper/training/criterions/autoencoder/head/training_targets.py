"""Training-partition target statistics for multi-label objectives."""

from __future__ import annotations

from typing import Any

import torch
from torch.utils.data import Subset

from .....data import TargetBatch
from .....utils.exceptions import raise_incompatible_interface_error


def collect_training_multilabel_targets(
    dataset: Any,
    target_field: str,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return train-only targets and aligned per-class availability masks.

    :param dataset: Active dataset owning the configured partitions.
    :type dataset: Any
    :param target_field: Multi-label target field to collect.
    :type target_field: str
    :return: Float targets and Boolean masks with shape ``(N_train, C)``.
    :rtype: tuple[torch.Tensor, torch.Tensor]
    :raises IncompatibleInterfaceError: If the dataset does not expose the
        configured target in a supported batch representation.
    """
    partitions_getter = getattr(dataset, "create_partitions", None)
    train = partitions_getter().train if callable(partitions_getter) else dataset
    owner = train.dataset if isinstance(train, Subset) else train
    indices = list(train.indices) if isinstance(train, Subset) else list(range(len(train)))
    bulk_getter = getattr(owner, "get_target_batch", None)
    if callable(bulk_getter):
        target_batch = bulk_getter(indices)
    else:
        target_batch = _collate_legacy_targets(train, target_field)
    if target_field not in target_batch.values or target_field not in target_batch.masks:
        raise_incompatible_interface_error(
            "HeadCriterion",
            f"Training dataset does not expose target '{target_field}'.",
        )
    targets = target_batch.values[target_field].to(dtype=torch.float32)
    masks = target_batch.masks[target_field].to(dtype=torch.bool)
    if targets.ndim != 2:
        raise_incompatible_interface_error(
            "HeadCriterion", "Multi-label training targets must have shape [N, C]."
        )
    if masks.shape == targets.shape[:1]:
        masks = masks.unsqueeze(1).expand_as(targets)
    if masks.shape != targets.shape:
        raise_incompatible_interface_error(
            "HeadCriterion", "Multi-label training masks must have shape [N] or [N, C]."
        )
    return targets, masks


def _collate_legacy_targets(dataset: Any, target_field: str) -> TargetBatch:
    values = []
    masks = []
    for index in range(len(dataset)):
        sample = dataset[index]
        if len(sample) < 4:
            raise_incompatible_interface_error(
                "HeadCriterion", "Training dataset samples have no target dictionaries."
            )
        values.append(sample[2][target_field])
        masks.append(sample[3][target_field])
    return TargetBatch(
        values={target_field: torch.stack(values)},
        masks={target_field: torch.stack(masks)},
        schemas={},
    )
