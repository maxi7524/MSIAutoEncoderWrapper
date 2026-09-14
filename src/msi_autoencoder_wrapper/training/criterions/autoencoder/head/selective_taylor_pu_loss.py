"""Project-specific Taylor-VPU objectives with SelectiveNet or Deep Gambler reject heads."""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F

from .supervision_masks import supervised_pu_masks
from .taylor_pu_components import simulated_negative_bce, taylor_variational_risk
from ...autoencoder_base_criterions import MSIHeadCriterion
from ...criterions_manager import CriterionsManager


def _multilabel_targets_and_mask(
    criterion: MSIHeadCriterion,
    model_outputs,
    batch_data,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return a structured head output with aligned binary targets and mask."""
    output, targets, availability = criterion.head_batch(model_outputs, batch_data)
    targets = targets.to(device=output.device, dtype=output.dtype)  # (B, C)
    if availability.shape == targets.shape[:1]:
        availability = availability.unsqueeze(1).expand_as(targets)  # (B, C)
    if availability.shape != targets.shape:
        raise ValueError("Multi-label head availability must have shape (B,) or (B, C).")
    return output, targets, availability.to(device=output.device, dtype=torch.bool)


class _TaylorPUHybridBase(MSIHeadCriterion):
    """Shared P/U core and separate ``N_sim`` extension for reject hybrids."""

    def __init__(
        self,
        head_id: str,
        target_field: str,
        class_indices: tuple[int, ...] | list[int] | None = None,
        order: int = 2,
        simulated_negative_weight: float = 0.0,
    ) -> None:
        super().__init__(head_id, target_field, class_indices)
        if isinstance(order, bool) or not isinstance(order, int) or order < 1:
            raise ValueError("order must be a positive integer.")
        if not math.isfinite(simulated_negative_weight) or simulated_negative_weight < 0:
            raise ValueError("simulated_negative_weight must be finite and nonnegative.")
        self.order = order
        self.simulated_negative_weight = float(simulated_negative_weight)

    def _taylor_core(
        self,
        classifier_logits: torch.Tensor,
        targets: torch.Tensor,
        availability: torch.Tensor,
        batch_data,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return P/U Taylor risk and the resolved P/N_sim masks."""
        positives, unlabelled, simulated_negative = supervised_pu_masks(
            batch_data,
            self.target_field,
            targets,
            availability,
            self.class_indices,
        )
        loss = taylor_variational_risk(
            classifier_logits, positives, unlabelled, self.order
        )  # ()
        if self.simulated_negative_weight > 0:
            loss = loss + self.simulated_negative_weight * simulated_negative_bce(
                classifier_logits, simulated_negative
            )  # ()
        return loss, positives, simulated_negative


@CriterionsManager.register_criterion(
    "autoencoder", "head", "SelectiveTaylorVariationalPULoss"
)
class SelectiveTaylorVariationalPULoss(_TaylorPUHybridBase):
    """TaylorVPU-N plus a SelectiveNet-style reject regularizer.

    The classifier channel receives Taylor-VPU on P/U and optional BCE on
    ``N_sim``. The selection and auxiliary channels receive only reliable
    P/N_sim labels. This is a project-specific hybrid, not the supervised
    SelectiveNet objective from Geifman and El-Yaniv.

    :param target_coverage: Required mean selection probability on P/N_sim.
    :type target_coverage: float
    :param coverage_weight: Penalty coefficient for missed target coverage.
    :type coverage_weight: float
    :param selective_weight: Mixture coefficient for selective versus
        auxiliary supervised risk.
    :type selective_weight: float
    """

    def __init__(
        self,
        head_id: str,
        target_field: str,
        class_indices: tuple[int, ...] | list[int] | None = None,
        order: int = 2,
        simulated_negative_weight: float = 0.0,
        target_coverage: float = 0.8,
        coverage_weight: float = 32.0,
        selective_weight: float = 0.5,
        reject_weight: float = 1.0,
    ) -> None:
        super().__init__(
            head_id, target_field, class_indices, order, simulated_negative_weight
        )
        if not 0 < target_coverage <= 1:
            raise ValueError("target_coverage must belong to (0, 1].")
        if any(
            not math.isfinite(value) or value < 0
            for value in (coverage_weight, reject_weight)
        ):
            raise ValueError("Selective weights must be finite and nonnegative.")
        if not 0 <= selective_weight <= 1:
            raise ValueError("selective_weight must belong to [0, 1].")
        self.target_coverage = float(target_coverage)
        self.coverage_weight = float(coverage_weight)
        self.selective_weight = float(selective_weight)
        self.reject_weight = float(reject_weight)
        self._config.update(
            target_coverage=self.target_coverage,
            coverage_weight=self.coverage_weight,
            selective_weight=self.selective_weight,
            reject_weight=self.reject_weight,
        )

    def forward(self, model_outputs, batch_data, **kwargs):
        """Return TaylorVPU-N plus SelectiveNet-style reliable-label risk."""
        del kwargs
        output, targets, availability = _multilabel_targets_and_mask(
            self, model_outputs, batch_data
        )
        if output.shape != (*targets.shape, 3):
            raise ValueError("Selective head output must have shape (B, C, 3).")
        classifier_logits = output[..., 0]  # (B, C)
        loss, positives, simulated_negative = self._taylor_core(
            classifier_logits, targets, availability, batch_data
        )
        selected = positives | simulated_negative  # (B, C)
        if not bool(selected.any()) or self.reject_weight == 0:
            return loss

        labels = positives.to(dtype=output.dtype)  # (B, C)
        acceptance = output[..., 1].sigmoid()[selected]  # (N,)
        classifier_risk = F.binary_cross_entropy_with_logits(
            classifier_logits[selected], labels[selected], reduction="none"
        )  # (N,)
        selective_risk = (acceptance * classifier_risk).sum() / acceptance.sum().clamp_min(
            torch.finfo(output.dtype).eps
        )  # ()
        coverage_penalty = (
            self.target_coverage - acceptance.mean()
        ).clamp_min(0.0).square()  # ()
        auxiliary_risk = F.binary_cross_entropy_with_logits(
            output[..., 2][selected], labels[selected]
        )  # ()
        reject_loss = self.selective_weight * (
            selective_risk + self.coverage_weight * coverage_penalty
        ) + (1.0 - self.selective_weight) * auxiliary_risk  # ()
        return loss + self.reject_weight * reject_loss  # ()


@CriterionsManager.register_criterion(
    "autoencoder", "head", "DeepGamblerTaylorVariationalPULoss"
)
class DeepGamblerTaylorVariationalPULoss(_TaylorPUHybridBase):
    """TaylorVPU-N with conditional P/N probability and gambler abstention.

    The PU score is ``p(P | accepted) = p_P / (p_P + p_N)``. Its logit is
    therefore ``positive_logit - negative_logit`` and is independent of the
    reserve channel. The gambler loss is applied only to reliable P/N_sim
    labels. This is a project-specific hybrid, not original Deep Gambler.
    """

    def __init__(
        self,
        head_id: str,
        target_field: str,
        class_indices: tuple[int, ...] | list[int] | None = None,
        order: int = 2,
        simulated_negative_weight: float = 0.0,
        reward: float = 1.5,
        gambling_weight: float = 1.0,
    ) -> None:
        super().__init__(
            head_id, target_field, class_indices, order, simulated_negative_weight
        )
        if not math.isfinite(reward) or not 1.0 < reward < 2.0:
            raise ValueError("Binary Deep Gambler reward must belong to (1, 2).")
        if not math.isfinite(gambling_weight) or gambling_weight < 0:
            raise ValueError("gambling_weight must be finite and nonnegative.")
        self.reward = float(reward)
        self.gambling_weight = float(gambling_weight)
        self._config.update(reward=self.reward, gambling_weight=self.gambling_weight)

    def forward(self, model_outputs, batch_data, **kwargs):
        """Return TaylorVPU-N plus the reliable-label gambling objective."""
        del kwargs
        output, targets, availability = _multilabel_targets_and_mask(
            self, model_outputs, batch_data
        )
        if output.shape != (*targets.shape, 3):
            raise ValueError("Deep Gambler output must have shape (B, C, 3).")
        conditional_positive_logit = output[..., 1] - output[..., 0]  # (B, C)
        loss, positives, simulated_negative = self._taylor_core(
            conditional_positive_logit, targets, availability, batch_data
        )
        selected = positives | simulated_negative  # (B, C)
        if not bool(selected.any()) or self.gambling_weight == 0:
            return loss
        labels = positives.to(dtype=torch.long)  # (B, C)
        probabilities = output.softmax(dim=-1)  # (B, C, 3)
        class_probability = probabilities.gather(
            dim=-1, index=labels.unsqueeze(-1)
        ).squeeze(-1)  # (B, C)
        payoff = self.reward * class_probability[selected] + probabilities[..., 2][selected]  # (N,)
        gambling_loss = -payoff.clamp_min(torch.finfo(output.dtype).tiny).log().mean()  # ()
        return loss + self.gambling_weight * gambling_loss  # ()
