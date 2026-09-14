"""Mask conventions for positive-unlabelled objectives."""

from __future__ import annotations

from collections.abc import Mapping

import torch

from .batches import SpectrumBatch


SIMULATED_NEGATIVE_MASK_SUFFIX = "__simulated_negative"


def simulated_negative_mask_key(target_field: str) -> str:
    """Return the auxiliary mask key for one binary target field.

    :param target_field: Name of the binary target field.
    :type target_field: str
    :return: Key stored in ``TargetBatch.masks``.
    :rtype: str
    """
    return f"{target_field}{SIMULATED_NEGATIVE_MASK_SUFFIX}"


def resolve_simulated_negative_mask(
    batch_data: tuple[torch.Tensor, ...] | SpectrumBatch,
    target_field: str,
    targets: torch.Tensor,
    availability: torch.Tensor,
    class_indices: tuple[int, ...] | None = None,
) -> torch.Tensor:
    """Resolve the declared ``N_sim`` mask aligned with selected head columns.

    :param batch_data: Legacy batch tuple or typed dense batch.
    :type batch_data: tuple[torch.Tensor, ...] | SpectrumBatch
    :param target_field: Binary target field used by the head.
    :type target_field: str
    :param targets: Selected binary target matrix with shape ``(B, C)``.
    :type targets: torch.Tensor
    :param availability: Selected availability matrix with shape ``(B, C)``.
    :type availability: torch.Tensor
    :param class_indices: Optional original columns selected for the head.
    :type class_indices: tuple[int, ...] | None
    :return: Reliable-negative mask with shape ``(B, C)``.
    :rtype: torch.Tensor

    REMARK: Missing auxiliary masks mean that a dataset has no reliable
    simulated negatives. A target value of zero is never inferred as ``N_sim``.
    """
    masks: Mapping[str, torch.Tensor]
    if isinstance(batch_data, SpectrumBatch):
        masks = batch_data.targets.masks
    else:
        masks = batch_data[3]
    raw_mask = masks.get(simulated_negative_mask_key(target_field))
    if raw_mask is None:
        return torch.zeros_like(availability, dtype=torch.bool)  # (B, C)
    resolved = raw_mask.to(device=targets.device, dtype=torch.bool)
    if class_indices is not None and resolved.ndim > 1:
        selection = torch.as_tensor(class_indices, device=targets.device)
        resolved = resolved.index_select(-1, selection)
    if resolved.shape == targets.shape[:1]:
        resolved = resolved.unsqueeze(1).expand_as(targets)  # (B, C)
    if resolved.shape != targets.shape:
        raise ValueError("The simulated-negative mask must have shape [B] or [B, C].")
    return resolved & availability & (targets <= 0.5)  # (B, C)
