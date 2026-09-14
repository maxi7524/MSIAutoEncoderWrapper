"""Strict P/N_sim and observed P/N_sim/U objectives."""

from __future__ import annotations

import torch
import torch.nn.functional as F

from .supervision_masks import supervised_pu_masks
from ...autoencoder_base_criterions import MSIHeadCriterion
from ...criterions_manager import CriterionsManager


@CriterionsManager.register_criterion("autoencoder", "head", "BCEPNLoss")
class BCEPNLoss(MSIHeadCriterion):
    """Apply binary BCE only to labelled positives and declared ``N_sim``.

    Unlabelled entries never receive a BCE target in this criterion.
    """

    def forward(self, model_outputs, batch_data, **kwargs):
        """Return the P/N_sim binary cross-entropy.

        :return: Scalar loss.
        :rtype: torch.Tensor
        """
        del kwargs
        logits, targets, availability = self.multilabel_batch(model_outputs, batch_data)
        positives, _, simulated_negative = supervised_pu_masks(
            batch_data,
            self.target_field,
            targets,
            availability,
            self.class_indices,
        )
        selected = positives | simulated_negative  # (B, C)
        if not bool(selected.any()):
            return logits.sum() * 0.0  # ()
        labels = positives.to(dtype=logits.dtype)  # (B, C)
        values = F.binary_cross_entropy_with_logits(logits, labels, reduction="none")  # (B, C)
        return values[selected].mean()  # ()


@CriterionsManager.register_criterion(
    "autoencoder", "head", "ObservedPNUSCrossEntropyLoss"
)
class ObservedPNUSCrossEntropyLoss(MSIHeadCriterion):
    """Predict observed N_sim/P/U provenance with output order N_sim, P, U.

    This is an observed-state baseline rather than a latent biological-label
    objective. Every available target entry has one categorical label.
    """

    def forward(self, model_outputs, batch_data, **kwargs):
        """Return the available-entry categorical cross-entropy.

        :return: Scalar loss.
        :rtype: torch.Tensor
        """
        del kwargs
        logits, targets, availability = self.head_batch(model_outputs, batch_data)
        if logits.shape != (*targets.shape, 3):
            raise ValueError("Observed P/N_sim/U logits must have shape (B, C, 3).")
        if availability.shape == targets.shape[:1]:
            availability = availability.unsqueeze(1).expand_as(targets)  # (B, C)
        positives, unlabelled, simulated_negative = supervised_pu_masks(
            batch_data,
            self.target_field,
            targets.to(logits),
            availability.to(device=logits.device, dtype=torch.bool),
            self.class_indices,
        )
        labels = torch.where(
            positives,
            torch.ones_like(targets, dtype=torch.long),
            torch.where(
                unlabelled,
                torch.full_like(targets, 2, dtype=torch.long),
                torch.zeros_like(targets, dtype=torch.long),
            ),
        )  # (B, C)
        selected = positives | unlabelled | simulated_negative  # (B, C)
        if not bool(selected.any()):
            return logits.sum() * 0.0  # ()
        return F.cross_entropy(logits[selected], labels[selected])  # ()
