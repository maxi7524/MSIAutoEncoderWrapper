"""Linear decoder for dense MSI spectra."""

from __future__ import annotations

from typing import Any, Dict, Sequence

import torch
import torch.nn as nn

from ....architectures_manager import ArchitecturesManager
from ......utils.configuration import ConfigurableComponent


@ArchitecturesManager.register_component(
    "autoencoder", "decoder", "LinearSpectrumDecoder"
)
class LinearSpectrumDecoder(nn.Module, ConfigurableComponent):
    """Decode latent vectors into dense nonnegative spectra."""

    def __init__(
        self,
        latent_dim: int,
        output_dim: int,
        hidden_dims: Sequence[int] = (128, 256),
    ) -> None:
        super().__init__()
        dimensions = [int(latent_dim), *(int(value) for value in hidden_dims), int(output_dim)]
        layers = []
        for index, (source, target) in enumerate(zip(dimensions, dimensions[1:])):
            layers.append(nn.Linear(source, target))
            layers.append(nn.Softplus() if index == len(dimensions) - 2 else nn.ReLU())
        self.network = nn.Sequential(*layers)
        self._config: Dict[str, Any] = {
            "latent_dim": int(latent_dim),
            "output_dim": int(output_dim),
            "hidden_dims": list(hidden_dims),
        }

    def forward(self, latent: torch.Tensor) -> torch.Tensor:
        """Return reconstructed spectra."""
        return self.network(latent)
