"""Shared P/U/N_sim mask resolution for molecular head criteria."""

from __future__ import annotations

import torch

from .....data.supervision_masks import resolve_simulated_negative_mask


def supervised_pu_masks(
    batch_data,
    target_field: str,
    targets: torch.Tensor,
    availability: torch.Tensor,
    class_indices: tuple[int, ...] | None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return mutually exclusive positive, unlabelled, and ``N_sim`` masks.

    :param batch_data: Batch carrying target values and mask mappings.
    :param target_field: Molecular target field.
    :param targets: Binary target values, shape ``(B, C)``.
    :param availability: Target availability mask, shape ``(B, C)``.
    :param class_indices: Optional original target columns selected by the head.
    :return: Positive, unlabelled, and simulated-negative masks, each ``(B, C)``.

    REMARK: ``N_sim`` is declared only by its auxiliary mask. Its target value
    remains zero, therefore absent annotations remain unlabelled by default.
    """
    if availability.shape == targets.shape[:1]:
        availability = availability.unsqueeze(1).expand_as(targets)  # (B, C)
    if availability.shape != targets.shape:
        raise ValueError("Target availability must have shape (B,) or (B, C).")
    simulated_negative = resolve_simulated_negative_mask(
        batch_data,
        target_field,
        targets,
        availability,
        class_indices,
    )  # (B, C)
    positives = availability & (targets > 0.5)  # (B, C)
    unlabelled = availability & ~positives & ~simulated_negative  # (B, C)
    return positives, unlabelled, simulated_negative
