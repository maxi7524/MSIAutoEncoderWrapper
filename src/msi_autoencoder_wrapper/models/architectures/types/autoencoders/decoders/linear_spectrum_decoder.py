"""Linear decoder for dense MSI spectra."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any, Dict, Sequence

import torch
import torch.nn as nn

from ....architectures_manager import ArchitecturesManager
from ......utils.configuration import ConfigurableComponent
from .output_activation import build_output_activation


@ArchitecturesManager.register_component(
    "autoencoder", "decoder", "LinearSpectrumDecoder"
)
class LinearSpectrumDecoder(nn.Module, ConfigurableComponent):
    """Decode latent vectors into dense nonnegative spectra."""

    def __init__(
        self,
        latent_dim: int,
        output_dim: int,
        output_activation: Mapping[str, Any],
        hidden_dims: Sequence[int] = (128, 256),
    ) -> None:
        """Construct a dense spectrum decoder.

        :param latent_dim: Latent vector width.
        :type latent_dim: int
        :param output_dim: Reconstructed spectrum width.
        :type output_dim: int
        :param output_activation: Final activation configuration. Supported types
            are declared by ``SUPPORTED_OUTPUT_ACTIVATIONS``.
        :type output_activation: Mapping[str, Any]
        :param hidden_dims: Hidden linear layer widths.
        :type hidden_dims: Sequence[int]
        """
        super().__init__()
        dimensions = [int(latent_dim), *(int(value) for value in hidden_dims), int(output_dim)]
        layers = []
        for index, (source, target) in enumerate(zip(dimensions, dimensions[1:])):
            layers.append(nn.Linear(source, target))
            layers.append(
                build_output_activation(output_activation)
                if index == len(dimensions) - 2
                else nn.ReLU()
            )
        self.network = nn.Sequential(*layers)
        self._config: Dict[str, Any] = {
            "latent_dim": int(latent_dim),
            "output_dim": int(output_dim),
            "hidden_dims": list(hidden_dims),
            "output_activation": dict(output_activation),
        }

    def forward(self, latent: torch.Tensor) -> torch.Tensor:
        """Return reconstructed spectra."""
        return self.network(latent)
