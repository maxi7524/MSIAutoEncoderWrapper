"""
Concrete linear projection MLP mapping embeddings into high-dimensional contrastive spaces.
"""

from typing import Any
import torch
import torch.nn as nn

from ....architectures_manager import ArchitecturesManager
from .base_projector import MSIBaseProjector

@ArchitecturesManager.register_component("autoencoder", "projector", "LinearProjector")
class LinearProjector(MSIBaseProjector):
    """
    Multi-Layer Perceptron strategy preparing embeddings for contrastive alignment operations.
    """

    def __init__(self, latent_dim: int, projection_dim: int, **kwargs: Any) -> None:
        """
        Constructs structural multilayer feature mappings networks.
        """
        super().__init__()
        self._config = {"latent_dim": latent_dim, "projection_dim": projection_dim}
        
        self.projection_network = nn.Sequential(
            nn.Linear(latent_dim, latent_dim),
            nn.LayerNorm(latent_dim),
            nn.ReLU(),
            nn.Linear(latent_dim, projection_dim)
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """
        Projects latent spaces representations onto contrastive metrics boundaries.
        """
        return self.projection_network(z)