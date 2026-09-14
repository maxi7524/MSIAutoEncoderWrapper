"""Dirichlet-evidence prediction head for binary molecular labels."""

from __future__ import annotations

import torch

from ....architectures_manager import ArchitecturesManager
from .linear_classification_head import LinearClassificationHead


@ArchitecturesManager.register_component("autoencoder", "head", "EvidentialHead")
class EvidentialHead(LinearClassificationHead):
    """Return raw negative/positive evidence logits for each molecular class.

    The criterion transforms raw values to nonnegative evidence with
    ``softplus`` and then to a binary Dirichlet concentration vector. The
    head therefore deliberately does not apply a terminal activation.
    """

    def __init__(
        self,
        latent_dim: int,
        output_dim: int,
        hidden_dim: int | None = None,
        dropout: float = 0.0,
    ) -> None:
        super().__init__(latent_dim, output_dim * 2, hidden_dim, dropout)
        self.output_dim = output_dim
        self._config["output_dim"] = output_dim

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """Return raw N/P evidence logits with shape ``(B, C, 2)``."""
        return self.network(z).reshape(len(z), self.output_dim, 2)  # (B, C, 2)
