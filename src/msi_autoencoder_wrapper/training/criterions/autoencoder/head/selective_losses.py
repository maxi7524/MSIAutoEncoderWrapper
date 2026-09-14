"""Strict supervised-PN objectives for SelectiveNet and Deep Gambler heads."""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F

from .supervision_masks import supervised_pu_masks
from ...criterions_manager import CriterionsManager
from ...autoencoder_base_criterions import MSIHeadCriterion


def _pn_labels(
    positives: torch.Tensor,
    simulated_negative: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return P/N_sim selection and binary labels from supervision masks."""
    selected = positives | simulated_negative  # (B, C)
    return selected, positives.to(dtype=torch.float32)


@CriterionsManager.register_criterion("autoencoder", "head", "SelectivePNLoss")
class SelectivePNLoss(MSIHeadCriterion):
    """SelectiveNet loss on reliable P/N_sim labels only.

    The three output channels are classifier, selection, and auxiliary logits.
    Unlabelled entries do not enter this strict supervised implementation and
    are never used as reject targets.
    """

    def __init__(
        self,
        head_id: str,
        target_field: str,
        class_indices: tuple[int, ...] | list[int] | None = None,
        target_coverage: float = 0.8,
        coverage_weight: float = 32.0,
        selective_weight: float = 0.5,
    ) -> None:
        super().__init__(head_id, target_field, class_indices)
        if not 0 < target_coverage <= 1:
            raise ValueError("target_coverage must belong to (0, 1].")
        if not math.isfinite(coverage_weight) or coverage_weight < 0:
            raise ValueError("coverage_weight must be finite and nonnegative.")
        if not 0 <= selective_weight <= 1:
            raise ValueError("selective_weight must belong to [0, 1].")
        self.target_coverage = float(target_coverage)
        self.coverage_weight = float(coverage_weight)
        self.selective_weight = float(selective_weight)
        self._config = {
            "head_id": head_id,
            "target_field": target_field,
            "class_indices": self.class_indices,
            "target_coverage": self.target_coverage,
            "coverage_weight": self.coverage_weight,
            "selective_weight": self.selective_weight,
        }

    def forward(self, model_outputs, batch_data, **kwargs):
        """Return SelectiveNet selective risk plus auxiliary classification loss."""
        del kwargs
        output, targets, mask = self.head_batch(model_outputs, batch_data)
        if output.shape != (*targets.shape, 3):
            raise ValueError("Selective head output must have shape (B, C, 3).")
        positives, _, simulated_negative = supervised_pu_masks(
            batch_data,
            self.target_field,
            targets,
            mask,
            self.class_indices,
        )
        selected, labels = _pn_labels(positives, simulated_negative)
        if not bool(selected.any()):
            return output.sum() * 0.0
        classifier_logits = output[..., 0]  # (B, C)
        selection_logits = output[..., 1]  # (B, C)
        auxiliary_logits = output[..., 2]  # (B, C)
        selected_labels = labels.to(output)[selected]  # (N,)
        selected_classification = F.binary_cross_entropy_with_logits(
            classifier_logits[selected], selected_labels, reduction="none"
        )  # (N,)
        acceptance = selection_logits[selected].sigmoid()  # (N,)
        coverage = acceptance.mean()  # ()
        selective_risk = (acceptance * selected_classification).sum() / acceptance.sum().clamp_min(
            torch.finfo(output.dtype).eps
        )  # ()
        coverage_penalty = (self.target_coverage - coverage).clamp_min(0.0).square()  # ()
        selective_loss = selective_risk + self.coverage_weight * coverage_penalty  # ()
        auxiliary_loss = F.binary_cross_entropy_with_logits(
            auxiliary_logits[selected], selected_labels
        )  # ()
        return (
            self.selective_weight * selective_loss
            + (1.0 - self.selective_weight) * auxiliary_loss
        )  # ()


@CriterionsManager.register_criterion("autoencoder", "head", "DeepGamblerPNLoss")
class DeepGamblerPNLoss(MSIHeadCriterion):
    """Original-style Deep Gambler risk on reliable P/N_sim labels only.

    Output order is ``negative, positive, reserve``. The reserve channel is a
    learned abstention decision; no training entry receives reserve as a label.
    """

    def __init__(
        self,
        head_id: str,
        target_field: str,
        class_indices: tuple[int, ...] | list[int] | None = None,
        reward: float = 1.5,
    ) -> None:
        super().__init__(head_id, target_field, class_indices)
        if not math.isfinite(reward) or not 1.0 < reward < 2.0:
            raise ValueError("Binary Deep Gambler reward must belong to (1, 2).")
        self.reward = float(reward)
        self._config = {
            "head_id": head_id,
            "target_field": target_field,
            "class_indices": self.class_indices,
            "reward": self.reward,
        }

    def forward(self, model_outputs, batch_data, **kwargs):
        """Return the gambling objective on P/N_sim entries."""
        del kwargs
        output, targets, mask = self.head_batch(model_outputs, batch_data)
        if output.shape != (*targets.shape, 3):
            raise ValueError("Deep Gambler output must have shape (B, C, 3).")
        positives, _, simulated_negative = supervised_pu_masks(
            batch_data,
            self.target_field,
            targets,
            mask,
            self.class_indices,
        )
        selected, labels = _pn_labels(positives, simulated_negative)
        if not bool(selected.any()):
            return output.sum() * 0.0
        probabilities = output.softmax(dim=-1)  # (B, C, 3)
        binary_labels = labels.to(dtype=torch.long, device=output.device)  # (B, C)
        class_probability = probabilities.gather(
            dim=-1,
            index=binary_labels.unsqueeze(-1),
        ).squeeze(-1)  # (B, C)
        reserve_probability = probabilities[..., 2]  # (B, C)
        payoff = self.reward * class_probability[selected] + reserve_probability[selected]  # (N,)
        return -payoff.clamp_min(torch.finfo(output.dtype).tiny).log().mean()  # ()
