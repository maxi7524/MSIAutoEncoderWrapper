"""Categorical evidence prediction for multilabel ion annotations."""

import torch

from ....architectures_manager import ArchitecturesManager
from .linear_classification_head import LinearClassificationHead


@ArchitecturesManager.register_component("autoencoder", "head", "ThreeStateHead")
class ThreeStateHead(LinearClassificationHead):
    """Return N/P/U logits for each ion.

    :param latent_dim: Encoder output width.
    :param output_dim: Number of ion labels, not the number of state logits.
    :param hidden_dim: Optional hidden layer width.
    :param dropout: Dropout probability.
    """

    def __init__(self, latent_dim: int, output_dim: int,
                 hidden_dim: int | None = None, dropout: float = 0.0) -> None:
        super().__init__(latent_dim, output_dim * 3, hidden_dim, dropout)
        self.output_dim = output_dim
        self._config["output_dim"] = output_dim

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """Return state logits of shape ``(B, C, 3)`` from ``(B, D)`` latents."""
        return self.network(z).reshape(len(z), self.output_dim, 3)  # (B, C, 3)

    @staticmethod
    def presence_probability(logits: torch.Tensor) -> torch.Tensor:
        """Return the positive evidence probability for evaluation, shape ``(B, C)``."""
        return logits.softmax(dim=-1)[..., 1]  # (B, C)
