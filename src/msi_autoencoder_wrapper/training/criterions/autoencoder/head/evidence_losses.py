"""Binary and three-state objectives sharing the dataset's signal evidence."""

from __future__ import annotations

from dataclasses import asdict
import math
from typing import Any

import torch
import torch.nn.functional as F
from torch.utils.data import Subset

from ...autoencoder_base_criterions import MSIHeadCriterion
from ...criterions_manager import CriterionsManager
from .....data.annotation_evidence import IonCatalogue, SignalEvidencePolicy, POSITIVE, NEGATIVE


class EvidenceHeadCriterion(MSIHeadCriterion):
    """Bind ion evidence once per phase and share its semantics across losses.

    :param head_id: Model output head identifier.
    :param target_field: Binary ion target field.
    :param class_indices: Optional target-column selection.
    :param evidence: Parameters of :class:`SignalEvidencePolicy`.
    """

    def __init__(self, head_id: str, target_field: str,
                 class_indices: tuple[int, ...] | list[int] | None = None,
                 evidence: dict[str, Any] | None = None) -> None:
        super().__init__(head_id, target_field, class_indices)
        self.evidence = SignalEvidencePolicy(**(evidence or {}))
        self.catalogue: IonCatalogue | None = None
        self._config = dict(head_id=head_id, target_field=target_field,
                            class_indices=self.class_indices, evidence=asdict(self.evidence))

    def on_phase_start(self, model, dataset, transient_cache) -> None:
        """Bind the catalogue to this dataset and selected target columns."""
        del model, transient_cache
        catalogue = IonCatalogue.from_dataset(dataset, self.target_field)
        if self.class_indices is not None:
            catalogue = IonCatalogue(
                tuple(catalogue.class_names[i] for i in self.class_indices),
                tuple(catalogue.bins[i] for i in self.class_indices), catalogue.feature_count,
            )
        self.catalogue = catalogue

    def evidence_batch(self, model_outputs, batch_data):
        """Return logits, binary targets, availability and P/N/U states."""
        logits, targets, mask = self.head_batch(model_outputs, batch_data)
        targets = targets.to(logits)
        if mask.ndim == 1:
            mask = mask.unsqueeze(1).expand_as(targets)  # (B, C)
        if targets.ndim != 2 or mask.shape != targets.shape:
            raise ValueError("Ion targets and masks must have shape (B, C).")
        if self.catalogue is None:
            raise RuntimeError("Signal evidence must be bound in on_phase_start().")
        states = self.evidence.classify(batch_data[1].to(logits), targets, mask, self.catalogue)
        return logits, targets, mask, states


@CriterionsManager.register_criterion("autoencoder", "head", "SignalMaskedBCELoss")
class SignalMaskedBCELoss(EvidenceHeadCriterion):
    """Optimize BCE on annotated positives and signal-derived negatives only.

    :param positive_weight_mode: ``none``, a global square-root P/N ratio, or
        a classwise square-root P/N ratio with a global fallback.
    :param max_positive_weight: Upper bound for every positive BCE multiplier.
    :param minimum_positive_count: Per-class positive count required before a
        classwise multiplier is considered stable.

    REMARK: P/N counts are measured on the train partition only after applying
    the configured evidence rule. They are not annotation prevalence counts.
    """

    def __init__(
        self,
        head_id: str,
        target_field: str,
        class_indices: tuple[int, ...] | list[int] | None = None,
        evidence: dict[str, Any] | None = None,
        positive_weight_mode: str = "none",
        max_positive_weight: float = 10.0,
        minimum_positive_count: int = 10,
    ) -> None:
        super().__init__(head_id, target_field, class_indices, evidence)
        if positive_weight_mode not in {
            "none",
            "global_sqrt_negative_to_positive",
            "per_class_sqrt_negative_to_positive",
        }:
            raise ValueError("Unsupported P/N positive_weight_mode.")
        if not math.isfinite(max_positive_weight) or max_positive_weight < 1:
            raise ValueError("max_positive_weight must be finite and at least one.")
        if (
            isinstance(minimum_positive_count, bool)
            or not isinstance(minimum_positive_count, int)
            or minimum_positive_count < 1
        ):
            raise ValueError("minimum_positive_count must be a positive integer.")
        self.positive_weight_mode = positive_weight_mode
        self.max_positive_weight = float(max_positive_weight)
        self.minimum_positive_count = minimum_positive_count
        self.register_buffer("positive_weights", torch.empty(0), persistent=False)
        self._config.update(
            positive_weight_mode=positive_weight_mode,
            max_positive_weight=self.max_positive_weight,
            minimum_positive_count=minimum_positive_count,
        )

    def on_phase_start(self, model, dataset, transient_cache) -> None:
        """Bind evidence and calculate train-only P/N positive multipliers."""
        super().on_phase_start(model, dataset, transient_cache)
        if self.positive_weight_mode == "none":
            self.positive_weights = torch.ones(len(self.catalogue.bins))  # (C,)
            return
        cache_key = (
            "signal_masked_bce_weights",
            self.target_field,
            self.class_indices,
            tuple(sorted(asdict(self.evidence).items())),
            self.positive_weight_mode,
            self.max_positive_weight,
            self.minimum_positive_count,
        )
        weights = transient_cache.get(cache_key)
        if weights is None:
            positives, negatives = self._train_evidence_counts(dataset)
            weights = self._positive_weights_from_counts(positives, negatives)
            transient_cache[cache_key] = weights
        self.positive_weights = weights

    def _train_evidence_counts(self, dataset) -> tuple[torch.Tensor, torch.Tensor]:
        """Count P and operational N over the fixed train partition only."""
        partition = dataset.create_partitions().train
        positive_count = torch.zeros(len(self.catalogue.bins), dtype=torch.long)  # (C,)
        negative_count = torch.zeros_like(positive_count)  # (C,)
        owner = partition.dataset if isinstance(partition, Subset) else partition
        indices = partition.indices if isinstance(partition, Subset) else range(len(partition))
        for index in indices:
            sample = owner[int(index)]
            if len(sample) < 4:
                raise ValueError("P/N weighting requires spectra, targets, and target masks.")
            spectrum = sample[1].detach().cpu().reshape(1, -1)  # (1, M)
            target = sample[2][self.target_field].detach().cpu().reshape(1, -1)  # (1, C_all)
            mask = sample[3][self.target_field].detach().cpu()
            if mask.ndim == 0:
                mask = mask.expand_as(target.reshape(-1))
            mask = mask.reshape(1, -1)  # (1, C_all)
            if self.class_indices is not None:
                columns = torch.as_tensor(self.class_indices, dtype=torch.long)
                target = target.index_select(1, columns)  # (1, C)
                mask = mask.index_select(1, columns)  # (1, C)
            states = self.evidence.classify(spectrum, target, mask, self.catalogue)  # (1, C)
            positive_count += (states == POSITIVE).sum(dim=0).to(torch.long)  # (C,)
            negative_count += (states == NEGATIVE).sum(dim=0).to(torch.long)  # (C,)
        return positive_count, negative_count

    def _positive_weights_from_counts(
        self,
        positive_count: torch.Tensor,
        negative_count: torch.Tensor,
    ) -> torch.Tensor:
        """Convert train P/N counts into conservative clipped BCE multipliers."""
        global_positive = positive_count.sum()
        global_negative = negative_count.sum()
        global_weight = torch.sqrt(
            global_negative.to(torch.float32) / global_positive.clamp_min(1).to(torch.float32)
        ).clamp(1, self.max_positive_weight)  # ()
        if self.positive_weight_mode == "global_sqrt_negative_to_positive":
            return global_weight.expand_as(positive_count).clone()  # (C,)
        class_weight = torch.sqrt(
            negative_count.to(torch.float32) / positive_count.clamp_min(1).to(torch.float32)
        ).clamp(1, self.max_positive_weight)  # (C,)
        stable = positive_count >= self.minimum_positive_count  # (C,)
        return torch.where(stable, class_weight, global_weight)  # (C,)

    def forward(self, model_outputs, batch_data, **kwargs):
        """Return mean P/N BCE; uncertain and unavailable entries have no gradient."""
        logits, targets, _, states = self.evidence_batch(model_outputs, batch_data)
        if logits.shape != targets.shape:
            raise ValueError("Binary logits must match the ion targets.")
        selected = (states == POSITIVE) | (states == NEGATIVE)  # (B, C)
        if not bool(selected.any()):
            return logits.sum() * 0.0
        unweighted = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")  # (B, C)
        positive_weights = (
            self.positive_weights.to(logits)
            if self.positive_weights.numel() == logits.shape[1]
            else torch.ones(logits.shape[1], dtype=logits.dtype, device=logits.device)
        )  # (C,)
        weights = torch.where(
            targets > 0.5,
            positive_weights.unsqueeze(0),
            torch.ones((), dtype=logits.dtype, device=logits.device),
        )  # (B, C)
        return (unweighted[selected] * weights[selected]).mean()  # ()


@CriterionsManager.register_criterion("autoencoder", "head", "ThreeStateCrossEntropyLoss")
class ThreeStateCrossEntropyLoss(EvidenceHeadCriterion):
    """Predict N/P/U evidence independently for each ion, in this state order."""

    def forward(self, model_outputs, batch_data, **kwargs):
        """Return categorical evidence loss for available ion targets."""
        logits, targets, mask, states = self.evidence_batch(model_outputs, batch_data)
        if logits.shape != (*targets.shape, 3):
            raise ValueError("Three-state logits must have shape (B, C, 3).")
        if not bool(mask.any()):
            return logits.sum() * 0.0
        return F.cross_entropy(logits[mask], states[mask])
