"""Linear classification head operating on autoencoder latent vectors."""

from __future__ import annotations

import torch
import torch.nn as nn

from ....architectures_manager import ArchitecturesManager
from .base_head import MSIBaseHead


@ArchitecturesManager.register_component("autoencoder", "head", "LinearClassificationHead")
class LinearClassificationHead(MSIBaseHead):
    """Map latent vectors to unnormalized multi-label class logits.

    :param latent_dim: Width of the encoder latent vector.
    :type latent_dim: int
    :param output_dim: Number of molecular classes.
    :type output_dim: int
    :param hidden_dim: Optional hidden-layer width. When omitted, a single
        linear projection is used.
    :type hidden_dim: int | None
    :param dropout: Dropout probability applied before the output projection.
    :type dropout: float
    """

    def __init__(
        self,
        latent_dim: int,
        output_dim: int,
        hidden_dim: int | None = None,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        layers: list[nn.Module] = []
        input_dim = latent_dim
        if hidden_dim is not None:
            layers.extend((nn.Linear(latent_dim, hidden_dim), nn.ReLU()))
            input_dim = hidden_dim
        if dropout > 0:
            layers.append(nn.Dropout(dropout))
        layers.append(nn.Linear(input_dim, output_dim))
        self.network = nn.Sequential(*layers)
        self._config = {
            "latent_dim": latent_dim,
            "output_dim": output_dim,
            "hidden_dim": hidden_dim,
            "dropout": dropout,
        }

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """Return one logit per molecular class."""
        return self.network(z)
