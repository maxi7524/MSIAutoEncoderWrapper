"""Binary and three-state objectives sharing the dataset's signal evidence."""

from __future__ import annotations

from dataclasses import asdict
from typing import Any

import torch
import torch.nn.functional as F

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
    """Optimize BCE on annotated positives and signal-derived negatives only."""

    def forward(self, model_outputs, batch_data, **kwargs):
        """Return mean P/N BCE; uncertain and unavailable entries have no gradient."""
        logits, targets, _, states = self.evidence_batch(model_outputs, batch_data)
        if logits.shape != targets.shape:
            raise ValueError("Binary logits must match the ion targets.")
        selected = (states == POSITIVE) | (states == NEGATIVE)  # (B, C)
        if not bool(selected.any()):
            return logits.sum() * 0.0
        return F.binary_cross_entropy_with_logits(logits[selected], targets[selected])


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
