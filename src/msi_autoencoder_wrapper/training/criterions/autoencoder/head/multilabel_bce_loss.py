"""Binary cross-entropy objective for multi-label molecular heads."""

from __future__ import annotations

import math
from typing import Any, Dict, Tuple

import torch
import torch.nn.functional as F

from ...autoencoder_base_criterions import MSIHeadCriterion
from ...criterions_manager import CriterionsManager
from .....models.datasets.base_dataset import MSIBaseDataset
from .....utils.exceptions import raise_validation_error
from .training_targets import collect_training_multilabel_targets


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
        class_indices: tuple[int, ...] | list[int] | None = None,
        reduction: str = "mean",
    ) -> None:
        super().__init__(
            head_id=head_id,
            target_field=target_field,
            class_indices=class_indices,
        )
        if reduction not in {"mean", "sum"}:
            raise_validation_error(
                "MultiLabelBCELoss", "reduction must be 'mean' or 'sum'."
            )
        self.reduction = reduction
        self._config = {
            "head_id": head_id,
            "target_field": target_field,
            "class_indices": self.class_indices,
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


@CriterionsManager.register_criterion(
    "autoencoder", "head", "PositiveWeightedMultiLabelBCELoss"
)
class MSIPositiveWeightedMultiLabelBCELoss(MSIHeadCriterion):
    """Apply train-derived positive weights while treating every absence as N.

    The positive multiplier has the same square-root P/N definition as
    :class:`SignalMaskedBCELoss`, but this criterion does not construct U or
    mask examples. Consequently every target zero contributes as a negative.

    :param head_id: Model head identifier used in the output mapping.
    :type head_id: str
    :param target_field: Dataset multi-label target key.
    :type target_field: str
    :param class_indices: Optional selected target columns.
    :type class_indices: tuple[int, ...] | list[int] | None
    :param positive_weight_mode: Global or per-class square-root P/N multiplier.
    :type positive_weight_mode: str
    :param max_positive_weight: Inclusive upper bound for a positive multiplier.
    :type max_positive_weight: float
    :param minimum_positive_count: Minimum P count for a stable per-class weight.
    :type minimum_positive_count: int
    """

    _WEIGHT_MODES = {
        "global_sqrt_negative_to_positive",
        "per_class_sqrt_negative_to_positive",
    }

    def __init__(
        self,
        head_id: str,
        target_field: str,
        class_indices: tuple[int, ...] | list[int] | None = None,
        positive_weight_mode: str = "global_sqrt_negative_to_positive",
        max_positive_weight: float = 10.0,
        minimum_positive_count: int = 10,
    ) -> None:
        super().__init__(head_id=head_id, target_field=target_field, class_indices=class_indices)
        if positive_weight_mode not in self._WEIGHT_MODES:
            raise_validation_error(
                "PositiveWeightedMultiLabelBCELoss",
                f"positive_weight_mode must be one of {sorted(self._WEIGHT_MODES)}.",
            )
        if not math.isfinite(max_positive_weight) or max_positive_weight < 1.0:
            raise_validation_error(
                "PositiveWeightedMultiLabelBCELoss",
                "max_positive_weight must be finite and at least one.",
            )
        if (
            isinstance(minimum_positive_count, bool)
            or not isinstance(minimum_positive_count, int)
            or minimum_positive_count < 1
        ):
            raise_validation_error(
                "PositiveWeightedMultiLabelBCELoss",
                "minimum_positive_count must be a positive integer.",
            )
        self.positive_weight_mode = positive_weight_mode
        self.max_positive_weight = float(max_positive_weight)
        self.minimum_positive_count = minimum_positive_count
        self.register_buffer("positive_weights", torch.empty(0), persistent=False)
        self._config = {
            "head_id": head_id,
            "target_field": target_field,
            "class_indices": self.class_indices,
            "positive_weight_mode": positive_weight_mode,
            "max_positive_weight": self.max_positive_weight,
            "minimum_positive_count": minimum_positive_count,
        }

    def on_phase_start(
        self,
        model: torch.nn.Module,
        dataset: MSIBaseDataset,
        transient_cache: Dict[str, Any],
    ) -> None:
        """Compute P/N multipliers from the fixed train partition only."""
        del model
        cache_key = f"positive_weighted_bce_targets::{self.target_field}"
        cached = transient_cache.get(cache_key)
        if cached is None:
            cached = collect_training_multilabel_targets(dataset, self.target_field)
            transient_cache[cache_key] = cached
        targets, _ = cached  # (N_train, C_all), (N_train, C_all)
        if self.class_indices is not None:
            selection = torch.as_tensor(self.class_indices, dtype=torch.long)
            targets = targets.index_select(1, selection)  # (N_train, C)
        positive = targets.sum(dim=0)  # (C,)
        negative = len(targets) - positive  # (C,)
        self.positive_weights = self._weights_from_counts(positive, negative)  # (C,)

    def _weights_from_counts(
        self,
        positive: torch.Tensor,
        negative: torch.Tensor,
    ) -> torch.Tensor:
        """Convert train P/N counts into clipped square-root multipliers."""
        global_weight = torch.sqrt(
            negative.sum().to(torch.float32) / positive.sum().clamp_min(1).to(torch.float32)
        ).clamp(1.0, self.max_positive_weight)  # ()
        if self.positive_weight_mode == "global_sqrt_negative_to_positive":
            return global_weight.expand_as(positive).clone()  # (C,)
        per_class = torch.sqrt(
            negative.to(torch.float32) / positive.clamp_min(1).to(torch.float32)
        ).clamp(1.0, self.max_positive_weight)  # (C,)
        stable = positive >= self.minimum_positive_count  # (C,)
        return torch.where(stable, per_class, global_weight)  # (C,)

    def forward(
        self,
        model_outputs: Dict[str, torch.Tensor],
        batch_data: Tuple[torch.Tensor, ...],
        **kwargs: Any,
    ) -> torch.Tensor:
        """Return unmasked BCE with a positive multiplier for every class."""
        del kwargs
        logits, targets, _ = self.multilabel_batch(model_outputs, batch_data)
        if logits.shape != targets.shape:
            raise_validation_error(
                "PositiveWeightedMultiLabelBCELoss",
                "Logits and multi-label targets must have identical shapes.",
            )
        weights = (
            self.positive_weights.to(dtype=logits.dtype, device=logits.device)
            if self.positive_weights.numel() == logits.shape[1]
            else torch.ones(logits.shape[1], dtype=logits.dtype, device=logits.device)
        )  # (C,)
        values = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")  # (B, C)
        multipliers = torch.where(
            targets > 0.5,
            weights.unsqueeze(0),
            torch.ones((), dtype=logits.dtype, device=logits.device),
        )  # (B, C)
        return (values * multipliers).mean()  # ()
