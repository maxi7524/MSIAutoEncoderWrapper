"""Deterministic multilayer perceptron encoder for MSI spectra."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import torch
import torch.nn as nn

from ....architectures_manager import ArchitecturesManager
from ......utils.exceptions import raise_validation_error
from .base_encoder import MSIBaseEncoder


@ArchitecturesManager.register_component("autoencoder", "encoder", "MLPEncoder")
class MLPEncoder(MSIBaseEncoder):
    """Encode dense spectra with configurable fully connected hidden layers."""

    def __init__(
        self,
        input_dim: int,
        latent_dim: int,
        hidden_dims: Sequence[int],
        batch_normalization: bool = True,
    ) -> None:
        """Construct the deterministic encoder.

        :param input_dim: Number of input spectrum bins.
        :type input_dim: int
        :param latent_dim: Latent representation width.
        :type latent_dim: int
        :param hidden_dims: Ordered widths of hidden MLP layers.
        :type hidden_dims: Sequence[int]
        :param batch_normalization: Apply BatchNorm before each hidden ReLU.
        :type batch_normalization: bool
        :raises ValidationError: If dimensions are empty, invalid, or not positive.
        """
        super().__init__()
        resolved_hidden_dims = self._validate_dimensions(
            input_dim,
            latent_dim,
            hidden_dims,
        )

        # Hidden feature extraction
        layers: list[nn.Module] = []
        source_dim = int(input_dim)
        for hidden_dim in resolved_hidden_dims:
            layers.append(nn.Linear(source_dim, hidden_dim))
            if batch_normalization:
                layers.append(nn.BatchNorm1d(hidden_dim))
            layers.append(nn.ReLU())
            source_dim = hidden_dim
        self.backbone = nn.Sequential(*layers)

        # Deterministic bottleneck projection
        self.latent_layer = nn.Linear(source_dim, int(latent_dim))
        self._config: dict[str, Any] = {
            "input_dim": int(input_dim),
            "latent_dim": int(latent_dim),
            "hidden_dims": resolved_hidden_dims,
            "batch_normalization": bool(batch_normalization),
        }

    @staticmethod
    def _validate_dimensions(
        input_dim: int,
        latent_dim: int,
        hidden_dims: Sequence[int],
    ) -> list[int]:
        """Validate and return the ordered hidden-layer widths."""
        values = {
            "input_dim": input_dim,
            "latent_dim": latent_dim,
        }
        invalid = [
            name
            for name, value in values.items()
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0
        ]
        if invalid:
            raise_validation_error(
                "MLPEncoder",
                f"{', '.join(invalid)} must be positive integers.",
            )
        if (
            isinstance(hidden_dims, (str, bytes))
            or not isinstance(hidden_dims, Sequence)
            or not hidden_dims
        ):
            raise_validation_error(
                "MLPEncoder",
                "hidden_dims must be a non-empty sequence of positive integers.",
            )
        resolved = list(hidden_dims)
        if any(
            isinstance(value, bool) or not isinstance(value, int) or value <= 0
            for value in resolved
        ):
            raise_validation_error(
                "MLPEncoder",
                "hidden_dims must contain only positive integers.",
            )
        return resolved

    def forward(self, spectra: torch.Tensor) -> torch.Tensor:
        """Return deterministic latent representations.

        :param spectra: Dense input spectra. Shape ``(B, M)``.
        :type spectra: torch.Tensor
        :return: Latent representations. Shape ``(B, D)``.
        :rtype: torch.Tensor
        """
        hidden = self.backbone(spectra)  # (B, H)
        return self.latent_layer(hidden)  # (B, D)
