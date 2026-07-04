from typing import Dict, Any
import torch
import torch.nn as nn

# Purely relative imports context mapping
from ..manager import ArchitectureManager
from .base_projector import MSIBaseProjector


@ArchitectureManager.register_projector("LinearProjector")
class LinearProjector(MSIBaseProjector):
    """
    Multi-Layer Perceptron strategy projecting structural embeddings into contrastive fields.

    Projects stable latent-space coordinates into an explicit projection-space optimized for
    InfoNCE metric evaluation steps, stabilizing the feature extractor graph configuration.
    """

    def __init__(self, latent_dim: int, projection_dim: int) -> None:
        """
        Constructs the projection multi-layer perceptron graph structure.

        :param latent_dim: Feature width depth tracking the bottleneck latent-space configuration.
        :type latent_dim: int
        :param projection_dim: Target feature mapping size for the evaluation projection-space layers.
        :type projection_dim: int
        """
        super().__init__()
        
        # Cache initialization layout values snapshot
        self._config = {"latent_dim": latent_dim, "projection_dim": projection_dim}
        
        # Dynamic MLP mapping layers construction block
        self.projection_network = nn.Sequential(
            nn.Linear(latent_dim, latent_dim),
            nn.LayerNorm(latent_dim),
            nn.ReLU(),
            nn.Linear(latent_dim, projection_dim)
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """
        Projects latent representations into contrastive metric calculation layers.

        :param z: Latent coordinate tensors derived from the encoder module. Shape: [Batch, Latent_Dim].
        :type z: torch.Tensor
        :return: High-dimensional projection alignment matrix tensor. Shape: [Batch, Projection_Dim].
        :rtype: torch.Tensor
        """
        return self.projection_network(z)