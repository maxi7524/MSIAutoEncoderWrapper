"""Train-frequency-balanced BCE for multi-label molecular heads."""

from __future__ import annotations

from typing import Any, Dict, Tuple

import torch
import torch.nn.functional as F

from ...autoencoder_base_criterions import MSIHeadCriterion
from ...criterions_manager import CriterionsManager
from .....models.datasets.base_dataset import MSIBaseDataset
from .....utils.exceptions import raise_validation_error
from .training_targets import collect_training_multilabel_targets


@CriterionsManager.register_criterion(
    "autoencoder", "head", "ClassBalancedMultiLabelBCELoss"
)
class MSIClassBalancedMultiLabelBCELoss(MSIHeadCriterion):
    """Balance positive and negative train contributions independently per class.

    The loss uses ``BCEWithLogits`` and therefore expects raw model logits. For
    class ``c``, ``pos_weight[c] = N_neg[c] / N_pos[c]`` is computed exclusively
    from the training partition and capped by ``max_positive_weight``. Weighted
    values are normalized within each class before a class mean is taken.

    :param head_id: Model head identifier.
    :type head_id: str
    :param target_field: Dataset multi-label target key.
    :type target_field: str
    :param class_balance_method: Currently ``train_inverse_prevalence``.
    :type class_balance_method: str
    :param max_positive_weight: Maximum rare-positive weight.
    :type max_positive_weight: float
    :param reduction: Currently ``balanced_class_mean``.
    :type reduction: str
    :param unknown_label_policy: ``assume_negative`` or ``mask``. The latter
        requires the dataset to provide a per-class availability mask.
    :type unknown_label_policy: str
    """

    def __init__(
        self,
        head_id: str,
        target_field: str,
        class_indices: tuple[int, ...] | list[int] | None = None,
        class_balance_method: str = "train_inverse_prevalence",
        max_positive_weight: float = 20.0,
        reduction: str = "balanced_class_mean",
        unknown_label_policy: str = "assume_negative",
    ) -> None:
        super().__init__(
            head_id=head_id,
            target_field=target_field,
            class_indices=class_indices,
        )
        if class_balance_method != "train_inverse_prevalence":
            raise_validation_error(
                "ClassBalancedMultiLabelBCELoss",
                "class_balance_method must be 'train_inverse_prevalence'.",
            )
        if max_positive_weight < 1.0:
            raise_validation_error(
                "ClassBalancedMultiLabelBCELoss",
                "max_positive_weight must be at least one.",
            )
        if reduction != "balanced_class_mean":
            raise_validation_error(
                "ClassBalancedMultiLabelBCELoss",
                "reduction must be 'balanced_class_mean'.",
            )
        if unknown_label_policy not in {"assume_negative", "mask"}:
            raise_validation_error(
                "ClassBalancedMultiLabelBCELoss",
                "unknown_label_policy must be 'assume_negative' or 'mask'.",
            )
        self.max_positive_weight = float(max_positive_weight)
        self.unknown_label_policy = unknown_label_policy
        self.register_buffer("positive_weights", torch.empty(0), persistent=False)
        self.register_buffer("active_classes", torch.empty(0, dtype=torch.bool), persistent=False)
        self._config = {
            "head_id": head_id,
            "target_field": target_field,
            "class_indices": self.class_indices,
            "class_balance_method": class_balance_method,
            "max_positive_weight": self.max_positive_weight,
            "reduction": reduction,
            "unknown_label_policy": unknown_label_policy,
        }

    def on_phase_start(
        self,
        model: torch.nn.Module,
        dataset: MSIBaseDataset,
        transient_cache: Dict[str, Any],
    ) -> None:
        """Compute immutable class weights from the train partition once."""
        del model
        cache_key = f"multilabel_training_targets::{self.target_field}"
        cached = transient_cache.get(cache_key)
        if cached is None:
            cached = collect_training_multilabel_targets(dataset, self.target_field)
            transient_cache[cache_key] = cached
        targets, mask = cached
        if self.class_indices is not None:
            selection = torch.as_tensor(self.class_indices, dtype=torch.long)
            targets = targets.index_select(1, selection)
            mask = mask.index_select(1, selection)
        if self.unknown_label_policy == "assume_negative":
            mask = torch.ones_like(targets, dtype=torch.bool)
        self._configure_training_statistics(targets, mask)

    def _configure_training_statistics(
        self,
        targets: torch.Tensor,
        mask: torch.Tensor,
    ) -> None:
        positive = (targets * mask).sum(dim=0)  # (C,)
        negative = ((1.0 - targets) * mask).sum(dim=0)  # (C,)
        weights = negative / positive.clamp_min(1.0)  # (C,)
        self.positive_weights = weights.clamp(
            torch.finfo(torch.float32).eps,
            self.max_positive_weight,
        )
        self.active_classes = (positive > 0) & (negative > 0)

    def forward(
        self,
        model_outputs: Dict[str, torch.Tensor],
        batch_data: Tuple[torch.Tensor, ...],
        **kwargs: Any,
    ) -> torch.Tensor:
        """Return train-balanced BCE evaluated directly from logits."""
        del kwargs
        logits, targets, mask = self.multilabel_batch(model_outputs, batch_data)
        if self.unknown_label_policy == "assume_negative":
            mask = torch.ones_like(targets, dtype=torch.bool)
        class_count = logits.shape[1]
        positive_weights = (
            self.positive_weights
            if self.positive_weights.numel() == class_count
            else torch.ones(class_count, device=logits.device)
        ).to(device=logits.device, dtype=logits.dtype)
        active_classes = (
            self.active_classes
            if self.active_classes.numel() == class_count
            else torch.ones(class_count, dtype=torch.bool, device=logits.device)
        ).to(device=logits.device)
        values = F.binary_cross_entropy_with_logits(
            logits,
            targets,
            pos_weight=positive_weights,
            reduction="none",
        )  # (B, C)
        element_weights = torch.where(
            targets > 0.5,
            positive_weights.unsqueeze(0),
            torch.ones_like(values),
        )  # (B, C)
        weighted_mask = element_weights * mask.to(dtype=logits.dtype)  # (B, C)
        denominators = weighted_mask.sum(dim=0)  # (C,)
        class_losses = (values * mask).sum(dim=0) / denominators.clamp_min(1.0)  # (C,)
        valid_classes = active_classes & (denominators > 0)
        if not bool(valid_classes.any()):
            return logits.sum() * 0.0
        return class_losses[valid_classes].mean()
