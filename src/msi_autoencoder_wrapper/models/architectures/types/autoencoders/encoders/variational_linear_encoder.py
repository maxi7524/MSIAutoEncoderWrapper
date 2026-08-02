"""PyTorch variational encoder for MSI spectra.

The mean/log-variance/sampling contract follows the ConvVAE design used by
AutoMSI (DKFZ, MIT license), adapted to the wrapper's one-dimensional spectral
component architecture.
"""

from __future__ import annotations

from typing import Any, Dict, Sequence

import torch
import torch.nn as nn

from ....architectures_manager import ArchitecturesManager
from .base_encoder import MSIBaseEncoder


@ArchitecturesManager.register_component(
    "autoencoder", "encoder", "VariationalLinearEncoder"
)
class VariationalLinearEncoder(MSIBaseEncoder):
    """Encode spectra as mean, log variance, and reparameterized samples."""

    def __init__(
        self,
        input_dim: int,
        latent_dim: int,
        hidden_dims: Sequence[int] = (256, 128),
    ) -> None:
        super().__init__()
        dimensions = [int(input_dim), *(int(value) for value in hidden_dims)]
        layers = []
        for source, target in zip(dimensions, dimensions[1:]):
            layers.extend((nn.Linear(source, target), nn.BatchNorm1d(target), nn.ReLU()))
        self.backbone = nn.Sequential(*layers)
        last_dim = dimensions[-1]
        self.mean = nn.Linear(last_dim, int(latent_dim))
        self.log_variance = nn.Linear(last_dim, int(latent_dim))
        self._config: Dict[str, Any] = {
            "input_dim": int(input_dim),
            "latent_dim": int(latent_dim),
            "hidden_dims": list(hidden_dims),
        }

    def forward(self, spectra: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Return variational parameters and a reparameterized latent sample."""
        hidden = self.backbone(spectra)
        mean = self.mean(hidden)
        log_variance = self.log_variance(hidden)
        if self.training:
            sample = mean + torch.exp(0.5 * log_variance) * torch.randn_like(mean)
        else:
            sample = mean
        return {
            "latent_mean": mean,
            "latent_log_variance": log_variance,
            "latent_space": sample,
        }
