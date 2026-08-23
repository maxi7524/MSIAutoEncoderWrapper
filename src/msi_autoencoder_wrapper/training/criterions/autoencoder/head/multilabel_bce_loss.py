"""Binary cross-entropy objective for multi-label molecular heads."""

from __future__ import annotations

from typing import Any, Dict, Tuple

import torch
import torch.nn.functional as F

from ...autoencoder_base_criterions import MSIHeadCriterion
from ...criterions_manager import CriterionsManager
from .....utils.exceptions import raise_validation_error


@CriterionsManager.register_criterion("autoencoder", "head", "MultiLabelBCELoss")
class MSIMultiLabelBCELoss(MSIHeadCriterion):
    """Compare molecular logits with a multi-hot dataset target.

    :param head_id: Model head identifier used in the output mapping.
    :type head_id: str
    :param target_field: Dataset target dictionary key.
    :type target_field: str
    :param reduction: PyTorch BCE reduction mode.
    :type reduction: str
    """

    def __init__(
        self,
        head_id: str,
        target_field: str,
        reduction: str = "mean",
    ) -> None:
        super().__init__(head_id=head_id, target_field=target_field)
        if reduction not in {"mean", "sum"}:
            raise_validation_error(
                "MultiLabelBCELoss", "reduction must be 'mean' or 'sum'."
            )
        self.reduction = reduction
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
        """Return BCE loss for the configured head and target field."""
        del kwargs
        logits, targets, mask = self.multilabel_batch(model_outputs, batch_data)
        if not bool(mask.any()):
            return logits.sum() * 0.0
        values = F.binary_cross_entropy_with_logits(
            logits,
            targets,
            reduction="none",
        )
        selected = values[mask]
        return selected.sum() if self.reduction == "sum" else selected.mean()
