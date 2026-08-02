"""Metrics defined between paired representation-space embeddings."""

from __future__ import annotations

import torch
import torch.nn.functional as F

from ...utils.exceptions import raise_validation_error


def info_nce(
    original: torch.Tensor,
    augmented: torch.Tensor,
    *,
    temperature: float,
) -> torch.Tensor:
    """Return InfoNCE cross entropy for both directions of every pair."""
    if original.ndim != 2 or original.shape != augmented.shape:
        raise_validation_error(
            "InfoNCE", "original and augmented projections must share [B, D]."
        )
    if temperature <= 0:
        raise_validation_error("InfoNCE", "temperature must be positive.")
    batch_size = original.shape[0]
    representations = torch.cat(
        [F.normalize(original, dim=1), F.normalize(augmented, dim=1)], dim=0
    )
    similarities = representations @ representations.T / temperature
    rows = torch.arange(2 * batch_size, device=original.device)
    pair_indices = (rows + batch_size) % (2 * batch_size)
    positives = similarities[rows, pair_indices].unsqueeze(1)
    excluded = torch.eye(
        2 * batch_size, device=original.device, dtype=torch.bool
    )
    excluded[rows, pair_indices] = True
    negatives = similarities[~excluded].view(2 * batch_size, -1)
    logits = torch.cat([positives, negatives], dim=1)
    labels = torch.zeros(2 * batch_size, device=original.device, dtype=torch.long)
    return F.cross_entropy(logits, labels, reduction="none")
