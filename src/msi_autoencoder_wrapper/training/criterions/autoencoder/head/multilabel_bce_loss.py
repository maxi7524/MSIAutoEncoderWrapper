"""Binary cross-entropy objective for multi-label molecular heads."""

from __future__ import annotations

from typing import Any, Dict, Tuple

import torch
import torch.nn as nn

from ...autoencoder_base_criterions import MSIHeadCriterion
from ...criterions_manager import CriterionsManager
from .....utils.exceptions import raise_incompatible_interface_error


@CriterionsManager.register_criterion("autoencoder", "head", "MultiLabelBCELoss")
class MSIMultiLabelBCELoss(MSIHeadCriterion):
    """Compare molecular logits with a multi-hot dataset target.

    :param head_name: Model head name used in the output mapping.
    :type head_name: str
    :param target_field: Dataset target dictionary key.
    :type target_field: str
    :param reduction: PyTorch BCE reduction mode.
    :type reduction: str
    """

    def __init__(
        self,
        head_name: str = "molecule",
        target_field: str = "molecule",
        reduction: str = "mean",
    ) -> None:
        super().__init__(head_name=head_name)
        self.target_field = target_field
        self.loss_fn = nn.BCEWithLogitsLoss(reduction=reduction)
        self._config = {
            "head_name": head_name,
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
        if self.output_key not in model_outputs:
            raise_incompatible_interface_error(
                "MultiLabelBCELoss", f"Missing model output '{self.output_key}'."
            )
        if len(batch_data) < 3 or self.target_field not in batch_data[2]:
            raise_incompatible_interface_error(
                "MultiLabelBCELoss",
                f"Missing dataset target '{self.target_field}'.",
            )
        logits = model_outputs[self.output_key]
        targets = batch_data[2][self.target_field].to(logits.device, dtype=logits.dtype)
        return self.loss_fn(logits, targets)
