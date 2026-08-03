"""Training adapter for the model-independent Masserstein spectrum metric."""

from __future__ import annotations

from typing import Any, Dict, Tuple

import torch

from ...autoencoder_base_criterions import MSIReconstructionCriterion
from ...criterions_manager import CriterionsManager
from .....data import SpectrumBatch
from .....metrics.strategies.masserstein import SpectrumMasserstein
from .....metrics.compatibility import validate_metric_compatibility
from .....models.datasets.base_dataset import MSIBaseDataset


@CriterionsManager.register_criterion(
    "autoencoder",
    "reconstruction",
    "MassersteinLoss",
)
class MSIMassersteinLoss(MSIReconstructionCriterion):
    """Reduce the shared Masserstein metric for autoencoder optimization."""

    def __init__(self, **params: Any) -> None:
        super().__init__()
        self.metric = SpectrumMasserstein(**params)
        self._config = dict(self.metric._config)

    @property
    def reduction(self) -> str:
        """Return the reduction owned by the shared metric."""
        return self.metric.reduction

    @property
    def _mass_axis(self) -> torch.Tensor:
        """Expose the migration buffer for backward-compatible callers."""
        return self.metric._mass_axis

    @_mass_axis.setter
    def _mass_axis(self, value: torch.Tensor) -> None:
        self.metric._mass_axis = value

    def on_phase_start(
        self,
        model: torch.nn.Module,
        dataset: MSIBaseDataset,
        transient_cache: Dict[str, Any],
    ) -> None:
        """Capture the dataset axis once for legacy tuple training batches."""
        del model, transient_cache
        binner = getattr(getattr(dataset, "active_context", None), "binner", None)
        axis_getter = getattr(binner, "GetXAxis", None)
        if callable(axis_getter):
            self.metric._mass_axis = torch.as_tensor(axis_getter(), dtype=torch.float64)

    def forward(
        self,
        model_outputs: Dict[str, torch.Tensor],
        batch_data: Tuple[torch.Tensor, ...],
        **kwargs: Any,
    ) -> torch.Tensor:
        """Evaluate the shared metric using typed or legacy batch inputs."""
        del kwargs
        reconstruction, original = self.reconstruction_pair(model_outputs, batch_data)
        if isinstance(batch_data, SpectrumBatch):
            validate_metric_compatibility(
                self.metric.requirements,
                batch_data.normalization_trace,
            )
        mass_axis = (
            batch_data.space.mass_axis
            if isinstance(batch_data, SpectrumBatch)
            else None
        )
        return self.metric(reconstruction, original, mass_axis=mass_axis)
