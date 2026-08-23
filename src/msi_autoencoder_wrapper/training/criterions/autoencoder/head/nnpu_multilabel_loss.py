"""Non-negative positive-unlabelled risk for multi-label molecular heads."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any, Dict, Tuple

import torch
import torch.nn.functional as F

from ...autoencoder_base_criterions import MSIHeadCriterion
from ...criterions_manager import CriterionsManager
from .....models.datasets.base_dataset import MSIBaseDataset
from .....utils.exceptions import raise_validation_error
from .training_targets import collect_training_multilabel_targets


@CriterionsManager.register_criterion("autoencoder", "head", "NNPUMultiLabelLoss")
class MSINNPUMultiLabelLoss(MSIHeadCriterion):
    """Optimize non-negative PU risk independently for every molecular class.

    Target one denotes a labelled positive. Target zero denotes an unlabelled
    example and is not treated as a verified negative. Sigmoid is incorporated
    numerically through ``softplus`` losses on raw logits.

    :param head_id: Model head identifier.
    :type head_id: str
    :param target_field: Dataset multi-label target key.
    :type target_field: str
    :param prior_method: ``fixed``, ``train_observed``, or
        ``scaled_train_observed``.
    :type prior_method: str
    :param class_prior: Scalar or per-class prior required by ``fixed``.
    :type class_prior: float | Sequence[float] | None
    :param prior_multiplier: Multiplier for ``scaled_train_observed``.
    :type prior_multiplier: float
    :param min_prior: Lower probability bound.
    :type min_prior: float
    :param max_prior: Upper probability bound below one.
    :type max_prior: float
    """

    def __init__(
        self,
        head_id: str,
        target_field: str,
        prior_method: str = "train_observed",
        class_prior: float | Sequence[float] | None = None,
        prior_multiplier: float = 1.0,
        min_prior: float = 1e-4,
        max_prior: float = 0.99,
    ) -> None:
        super().__init__(head_id=head_id, target_field=target_field)
        if prior_method not in {"fixed", "train_observed", "scaled_train_observed"}:
            raise_validation_error(
                "NNPUMultiLabelLoss",
                "prior_method must be 'fixed', 'train_observed', or 'scaled_train_observed'.",
            )
        if prior_method == "fixed" and class_prior is None:
            raise_validation_error(
                "NNPUMultiLabelLoss", "class_prior is required for fixed priors."
            )
        if prior_multiplier <= 0 or not 0 < min_prior < max_prior < 1:
            raise_validation_error(
                "NNPUMultiLabelLoss",
                "prior_multiplier must be positive and prior bounds must satisfy 0 < min < max < 1.",
            )
        self.prior_method = prior_method
        self.class_prior = class_prior
        self.prior_multiplier = float(prior_multiplier)
        self.min_prior = float(min_prior)
        self.max_prior = float(max_prior)
        self.register_buffer("class_priors", torch.empty(0), persistent=False)
        self.register_buffer("active_classes", torch.empty(0, dtype=torch.bool), persistent=False)
        self._config = {
            "head_id": head_id,
            "target_field": target_field,
            "prior_method": prior_method,
            "class_prior": class_prior,
            "prior_multiplier": self.prior_multiplier,
            "min_prior": self.min_prior,
            "max_prior": self.max_prior,
        }

    def on_phase_start(
        self,
        model: torch.nn.Module,
        dataset: MSIBaseDataset,
        transient_cache: Dict[str, Any],
    ) -> None:
        """Estimate or validate class priors using train-only annotations."""
        del model
        cache_key = f"multilabel_training_targets::{self.target_field}"
        cached = transient_cache.get(cache_key)
        if cached is None:
            cached = collect_training_multilabel_targets(dataset, self.target_field)
            transient_cache[cache_key] = cached
        self._configure_training_statistics(*cached)

    def _configure_training_statistics(
        self,
        targets: torch.Tensor,
        mask: torch.Tensor,
    ) -> None:
        valid_count = mask.sum(dim=0)  # (C,)
        positive_count = (targets * mask).sum(dim=0)  # (C,)
        observed = positive_count / valid_count.clamp_min(1)  # (C,)
        if self.prior_method == "fixed":
            priors = torch.as_tensor(self.class_prior, dtype=torch.float32)
            if priors.ndim == 0:
                priors = priors.repeat(targets.shape[1])
            if priors.shape != (targets.shape[1],):
                raise_validation_error(
                    "NNPUMultiLabelLoss",
                    "class_prior must be scalar or contain one value per class.",
                )
        else:
            multiplier = (
                self.prior_multiplier
                if self.prior_method == "scaled_train_observed"
                else 1.0
            )
            priors = observed * multiplier
        self.class_priors = priors.clamp(self.min_prior, self.max_prior)
        self.active_classes = (
            (positive_count > 0)
            & ((valid_count - positive_count) > 0)
            & (valid_count > 0)
        )

    def forward(
        self,
        model_outputs: Dict[str, torch.Tensor],
        batch_data: Tuple[torch.Tensor, ...],
        **kwargs: Any,
    ) -> torch.Tensor:
        """Return the class-mean non-negative PU risk from raw logits."""
        del kwargs
        logits, targets, mask = self.multilabel_batch(model_outputs, batch_data)
        class_count = logits.shape[1]
        if self.class_priors.numel() != class_count:
            if self.prior_method != "fixed":
                self._configure_training_statistics(targets.detach().cpu(), mask.detach().cpu())
            else:
                self._configure_training_statistics(
                    targets.detach().cpu(),
                    mask.detach().cpu(),
                )
        priors = self.class_priors.to(device=logits.device, dtype=logits.dtype)  # (C,)
        labelled_positive = (targets > 0.5) & mask  # (B, C)
        unlabelled = (targets <= 0.5) & mask  # (B, C)
        positive_count = labelled_positive.sum(dim=0)  # (C,)
        unlabelled_count = unlabelled.sum(dim=0)  # (C,)
        positive_loss = F.softplus(-logits)  # (B, C)
        negative_loss = F.softplus(logits)  # (B, C)
        positive_risk = priors * (
            (positive_loss * labelled_positive).sum(dim=0)
            / positive_count.clamp_min(1)
        )  # (C,)
        negative_risk = (
            (negative_loss * unlabelled).sum(dim=0)
            / unlabelled_count.clamp_min(1)
            - priors
            * (
                (negative_loss * labelled_positive).sum(dim=0)
                / positive_count.clamp_min(1)
            )
        )  # (C,)
        active = (
            self.active_classes.to(device=logits.device)
            if self.active_classes.numel() == class_count
            else (positive_count > 0) & (unlabelled_count > 0)
        )
        if not bool(active.any()):
            return logits.sum() * 0.0
        return (positive_risk + negative_risk.clamp_min(0.0))[active].mean()
