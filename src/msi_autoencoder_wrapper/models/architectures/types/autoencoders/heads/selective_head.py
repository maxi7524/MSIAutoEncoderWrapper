"""SelectiveNet-style classifier, selection, and auxiliary head."""

from __future__ import annotations

import torch

from ....architectures_manager import ArchitecturesManager
from .base_head import MSIBaseHead
from .jerm_head import _projection


@ArchitecturesManager.register_component("autoencoder", "head", "SelectiveHead")
class SelectiveHead(MSIBaseHead):
    """Return classifier, acceptance, and auxiliary logits for each class."""

    def __init__(
        self,
        latent_dim: int,
        output_dim: int,
        hidden_dim: int | None = None,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.classifier = _projection(latent_dim, output_dim, hidden_dim, dropout)
        self.selection = _projection(latent_dim, output_dim, hidden_dim, dropout)
        self.auxiliary = _projection(latent_dim, output_dim, hidden_dim, dropout)
        self._config = {
            "latent_dim": latent_dim,
            "output_dim": output_dim,
            "hidden_dim": hidden_dim,
            "dropout": dropout,
        }

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """Return ``(classifier, selection, auxiliary)`` logits, shape ``(B, C, 3)``."""
        return torch.stack(
            (self.classifier(z), self.selection(z), self.auxiliary(z)),
            dim=-1,
        )  # (B, C, 3)
