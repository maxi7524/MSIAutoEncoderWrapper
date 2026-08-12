"""Multilayer perceptron decoder for dense MSI spectra."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import torch
import torch.nn as nn

from ....architectures_manager import ArchitecturesManager
from ......utils.exceptions import raise_validation_error
from .base_decoder import MSIBaseDecoder
from .output_activation import build_output_activation


@ArchitecturesManager.register_component("autoencoder", "decoder", "MLPDecoder")
class MLPDecoder(MSIBaseDecoder):
    """Reconstruct spectra with configurable fully connected hidden layers."""

    def __init__(
        self,
        latent_dim: int,
        output_dim: int,
        hidden_dims: Sequence[int],
        output_activation: Mapping[str, Any],
        batch_normalization: bool = True,
    ) -> None:
        """Construct the spectrum decoder.

        :param latent_dim: Latent representation width.
        :type latent_dim: int
        :param output_dim: Number of reconstructed spectrum bins.
        :type output_dim: int
        :param hidden_dims: Ordered widths of hidden MLP layers.
        :type hidden_dims: Sequence[int]
        :param output_activation: Final nonnegative activation configuration.
        :type output_activation: Mapping[str, Any]
        :param batch_normalization: Apply BatchNorm before each hidden ReLU.
        :type batch_normalization: bool
        :raises ValidationError: If dimensions are empty, invalid, or not positive.
        """
        super().__init__()
        resolved_hidden_dims = self._validate_dimensions(
            latent_dim,
            output_dim,
            hidden_dims,
        )

        # Hidden reconstruction stages
        layers: list[nn.Module] = []
        source_dim = int(latent_dim)
        for hidden_dim in resolved_hidden_dims:
            layers.append(nn.Linear(source_dim, hidden_dim))
            if batch_normalization:
                layers.append(nn.BatchNorm1d(hidden_dim))
            layers.append(nn.ReLU())
            source_dim = hidden_dim

        # Spectrum projection and domain constraint
        layers.extend(
            (
                nn.Linear(source_dim, int(output_dim)),
                build_output_activation(output_activation),
            )
        )
        self.network = nn.Sequential(*layers)
        self._config: dict[str, Any] = {
            "latent_dim": int(latent_dim),
            "output_dim": int(output_dim),
            "hidden_dims": resolved_hidden_dims,
            "batch_normalization": bool(batch_normalization),
            "output_activation": dict(output_activation),
        }

    @staticmethod
    def _validate_dimensions(
        latent_dim: int,
        output_dim: int,
        hidden_dims: Sequence[int],
    ) -> list[int]:
        """Validate and return the ordered hidden-layer widths."""
        values = {
            "latent_dim": latent_dim,
            "output_dim": output_dim,
        }
        invalid = [
            name
            for name, value in values.items()
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0
        ]
        if invalid:
            raise_validation_error(
                "MLPDecoder",
                f"{', '.join(invalid)} must be positive integers.",
            )
        if (
            isinstance(hidden_dims, (str, bytes))
            or not isinstance(hidden_dims, Sequence)
            or not hidden_dims
        ):
            raise_validation_error(
                "MLPDecoder",
                "hidden_dims must be a non-empty sequence of positive integers.",
            )
        resolved = list(hidden_dims)
        if any(
            isinstance(value, bool) or not isinstance(value, int) or value <= 0
            for value in resolved
        ):
            raise_validation_error(
                "MLPDecoder",
                "hidden_dims must contain only positive integers.",
            )
        return resolved

    def forward(self, latent: torch.Tensor) -> torch.Tensor:
        """Return reconstructed spectra.

        :param latent: Latent representations. Shape ``(B, D)``.
        :type latent: torch.Tensor
        :return: Reconstructed spectra. Shape ``(B, M)``.
        :rtype: torch.Tensor
        """
        return self.network(latent)  # (B, M)
