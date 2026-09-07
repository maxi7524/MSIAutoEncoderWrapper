"""Composition regression for individual synthetic ion components."""

import torch
import torch.nn.functional as F

from ...autoencoder_base_criterions import MSIHeadCriterion
from ...criterions_manager import CriterionsManager


@CriterionsManager.register_criterion("autoencoder", "head", "ElementCountLoss")
class ElementCountLoss(MSIHeadCriterion):
    """Regress log-transformed atom counts where the generator supplies a target.

    :param head_id: Composition head identifier.
    :param target_field: Regression target, normally ``element_counts``.
    :param class_indices: Optional element-column selection.

    REMARK: ``log1p`` limits domination by abundant elements. A softplus maps
    raw outputs to nonnegative log-count predictions; this is not an abundance model.
    """

    def __init__(self, head_id: str, target_field: str, class_indices=None) -> None:
        super().__init__(head_id, target_field, class_indices)
        self._config = dict(head_id=head_id, target_field=target_field, class_indices=self.class_indices)

    def forward(self, model_outputs, batch_data, **kwargs):
        """Return masked Smooth L1 regression in log-count space."""
        logits, targets, mask = self.head_batch(model_outputs, batch_data)
        if mask.ndim == 1:
            mask = mask.unsqueeze(1).expand_as(targets)  # (B, E)
        if logits.shape != targets.shape or mask.shape != targets.shape:
            raise ValueError("Composition predictions and masks must match (B, E) targets.")
        if not bool(mask.any()):
            return logits.sum() * 0.0
        if bool((targets[mask] < 0).any()):
            raise ValueError("Atom counts cannot be negative.")
        return F.smooth_l1_loss(F.softplus(logits[mask]), torch.log1p(targets[mask]))  # ()
