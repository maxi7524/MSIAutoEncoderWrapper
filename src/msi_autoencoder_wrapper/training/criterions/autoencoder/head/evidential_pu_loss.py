"""Evidential Dirichlet supervision combined with the Taylor-VPU core."""

from __future__ import annotations

import math

import torch

from .supervision_masks import supervised_pu_masks
from .taylor_pu_components import taylor_variational_risk
from ...autoencoder_base_criterions import MSIHeadCriterion
from ...criterions_manager import CriterionsManager


def _dirichlet_kl_to_uniform(alpha: torch.Tensor) -> torch.Tensor:
    """Return ``KL(Dir(alpha) || Dir(1))`` for binary Dirichlet vectors.

    :param alpha: Positive concentrations, shape ``(..., 2)``.
    :type alpha: torch.Tensor
    :return: Per-vector KL divergence, shape ``(...)``.
    :rtype: torch.Tensor
    """
    concentration = alpha.sum(dim=-1)  # (...,)
    log_normalizer = torch.lgamma(concentration) - torch.lgamma(alpha).sum(dim=-1)  # (...,)
    uniform_log_normalizer = torch.lgamma(alpha.new_tensor(2.0))  # ()
    expectation = (
        (alpha - 1.0)
        * (torch.digamma(alpha) - torch.digamma(concentration).unsqueeze(-1))
    ).sum(dim=-1)  # (...,)
    return log_normalizer - uniform_log_normalizer + expectation  # (...,)


@CriterionsManager.register_criterion(
    "autoencoder", "head", "EvidentialTaylorVariationalPULoss"
)
class EvidentialTaylorVariationalPULoss(MSIHeadCriterion):
    """TaylorVPU-N with a binary evidential Dirichlet head.

    The Taylor core operates on the Dirichlet mean positive probability over
    P/U. The original evidential classification and KL terms operate only on
    reliable P/N_sim entries. Consequently the head's uncertainty is not an
    observed ``U`` class.

    :param order: Positive Taylor-series truncation order.
    :type order: int
    :param evidential_weight: Weight of reliable P/N_sim evidential risk.
    :type evidential_weight: float
    :param kl_weight: Final multiplier of the evidence-removal KL regularizer.
    :type kl_weight: float
    :param kl_annealing_steps: Number of optimization steps for linear KL
        annealing. Zero enables the final multiplier immediately.
    :type kl_annealing_steps: int
    """

    def __init__(
        self,
        head_id: str,
        target_field: str,
        class_indices: tuple[int, ...] | list[int] | None = None,
        order: int = 2,
        evidential_weight: float = 1.0,
        kl_weight: float = 1.0,
        kl_annealing_steps: int = 1000,
    ) -> None:
        super().__init__(head_id, target_field, class_indices)
        if isinstance(order, bool) or not isinstance(order, int) or order < 1:
            raise ValueError("order must be a positive integer.")
        if any(
            not math.isfinite(value) or value < 0
            for value in (evidential_weight, kl_weight)
        ):
            raise ValueError("Evidential weights must be finite and nonnegative.")
        if (
            isinstance(kl_annealing_steps, bool)
            or not isinstance(kl_annealing_steps, int)
            or kl_annealing_steps < 0
        ):
            raise ValueError("kl_annealing_steps must be a nonnegative integer.")
        self.order = order
        self.evidential_weight = float(evidential_weight)
        self.kl_weight = float(kl_weight)
        self.kl_annealing_steps = kl_annealing_steps
        self.register_buffer("optimization_step", torch.zeros((), dtype=torch.long), persistent=False)
        self._config.update(
            order=order,
            evidential_weight=self.evidential_weight,
            kl_weight=self.kl_weight,
            kl_annealing_steps=kl_annealing_steps,
        )

    @staticmethod
    def concentrations(raw_evidence_logits: torch.Tensor) -> torch.Tensor:
        """Convert raw evidence to valid binary Dirichlet concentrations."""
        return torch.nn.functional.softplus(raw_evidence_logits) + 1.0  # (B, C, 2)

    @staticmethod
    def uncertainty(alpha: torch.Tensor) -> torch.Tensor:
        """Return binary Dirichlet epistemic uncertainty ``2 / sum(alpha)``."""
        return 2.0 / alpha.sum(dim=-1)  # (B, C)

    def forward(self, model_outputs, batch_data, **kwargs):
        """Return Taylor PU risk plus reliable-label evidential supervision."""
        del kwargs
        raw_logits, targets, availability = self.head_batch(model_outputs, batch_data)
        if raw_logits.shape != (*targets.shape, 2):
            raise ValueError("Evidential head output must have shape (B, C, 2).")
        targets = targets.to(raw_logits)  # (B, C)
        if availability.shape == targets.shape[:1]:
            availability = availability.unsqueeze(1).expand_as(targets)  # (B, C)
        if availability.shape != targets.shape:
            raise ValueError("Target availability must have shape (B,) or (B, C).")
        availability = availability.to(device=raw_logits.device, dtype=torch.bool)  # (B, C)
        positives, unlabelled, simulated_negative = supervised_pu_masks(
            batch_data,
            self.target_field,
            targets,
            availability,
            self.class_indices,
        )
        alpha = self.concentrations(raw_logits)  # (B, C, 2)
        probability = alpha[..., 1] / alpha.sum(dim=-1)  # (B, C)
        pu_logits = probability.clamp(
            torch.finfo(probability.dtype).eps,
            1.0 - torch.finfo(probability.dtype).eps,
        ).logit()  # (B, C)
        loss = taylor_variational_risk(pu_logits, positives, unlabelled, self.order)  # ()

        # Original evidential classification and vacuous-evidence KL on P/N_sim.
        reliable = positives | simulated_negative  # (B, C)
        if self.evidential_weight > 0 and bool(reliable.any()):
            labels = torch.stack((1.0 - positives.to(alpha), positives.to(alpha)), dim=-1)  # (B, C, 2)
            concentration = alpha.sum(dim=-1, keepdim=True)  # (B, C, 1)
            mean = alpha / concentration  # (B, C, 2)
            squared_error = (labels - mean).square().sum(dim=-1)  # (B, C)
            variance = (
                alpha * (concentration - alpha) / (concentration.square() * (concentration + 1.0))
            ).sum(dim=-1)  # (B, C)
            evidence_removed = labels + (1.0 - labels) * alpha  # (B, C, 2)
            annealing = 1.0 if self.kl_annealing_steps == 0 else min(
                1.0, float(self.optimization_step.item()) / self.kl_annealing_steps
            )
            evidential = squared_error + variance + (
                self.kl_weight * annealing * _dirichlet_kl_to_uniform(evidence_removed)
            )  # (B, C)
            loss = loss + self.evidential_weight * evidential[reliable].mean()  # ()
        self.optimization_step += 1
        return loss
