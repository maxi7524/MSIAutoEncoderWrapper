"""Deep-Gambler-style P/N/reserve prediction head."""

from __future__ import annotations

import torch

from ....architectures_manager import ArchitecturesManager
from .linear_classification_head import LinearClassificationHead


@ArchitecturesManager.register_component("autoencoder", "head", "DeepGamblerHead")
class DeepGamblerHead(LinearClassificationHead):
    """Return negative, positive, and reserve logits for every molecular class."""

    def __init__(
        self,
        latent_dim: int,
        output_dim: int,
        hidden_dim: int | None = None,
        dropout: float = 0.0,
    ) -> None:
        super().__init__(latent_dim, output_dim * 3, hidden_dim, dropout)
        self.output_dim = output_dim
        self._config["output_dim"] = output_dim

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """Return N/P/reserve logits with shape ``(B, C, 3)``."""
        return self.network(z).reshape(len(z), self.output_dim, 3)  # (B, C, 3)
