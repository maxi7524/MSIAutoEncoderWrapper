"""Masked cross-entropy objective for named single-label heads."""

from __future__ import annotations

from typing import Any, Dict, Tuple

import torch
import torch.nn as nn

from ...autoencoder_base_criterions import MSIHeadCriterion
from ...criterions_manager import CriterionsManager


@CriterionsManager.register_criterion("autoencoder", "head", "MaskedCrossEntropyLoss")
class MSIMaskedCrossEntropyLoss(MSIHeadCriterion):
    """Evaluate a named head only where its dataset target is available."""

    def __init__(
        self,
        head_id: str,
        target_field: str,
        reduction: str = "mean",
    ) -> None:
        super().__init__(head_id=head_id, target_field=target_field)
        self.loss_fn = nn.CrossEntropyLoss(reduction=reduction)
        self._config = {
            "head_id": head_id,
            "target_field": target_field,
            "reduction": reduction,
        }

    def forward(
        self,
        model_outputs: Dict[str, torch.Tensor],
        batch_data: Tuple[torch.Tensor, ...],
        **kwargs: Any,
    ) -> torch.Tensor:
        """Return masked cross-entropy for the configured head."""
        del kwargs
        logits, targets, mask = self.head_batch(model_outputs, batch_data)
        if not bool(mask.any()):
            return logits.sum() * 0.0
        return self.loss_fn(logits[mask], targets[mask].long())
