"""Differentiable sample-level classification metrics."""

from __future__ import annotations

import torch
import torch.nn.functional as F


def cross_entropy(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    """Return unreduced cross entropy with one value per sample."""
    return F.cross_entropy(logits, targets.long(), reduction="none")


def binary_cross_entropy(
    logits: torch.Tensor,
    targets: torch.Tensor,
) -> torch.Tensor:
    """Return BCE averaged over classes independently per sample."""
    values = F.binary_cross_entropy_with_logits(
        logits,
        targets.to(dtype=logits.dtype),
        reduction="none",
    )
    return values.flatten(start_dim=1).mean(dim=1)
