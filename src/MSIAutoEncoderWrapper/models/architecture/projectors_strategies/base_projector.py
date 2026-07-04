from abc import ABC, abstractmethod
from typing import Any
import torch
import torch.nn as nn


class MSIBaseProjector(nn.Module, ABC):
    """
    Abstract Base Class for contrastive projection heads.
    
    The projector maps stable latent-space features into a dedicated projection-space 
    where contrastive metrics (e.g., InfoNCE) are calculated. This block is typically 
    pruned from the computational graph during inference/transformation steps.
    """

    def __init__(self) -> None:
        """Initializes the base projector module and its internal configuration state."""
        super().__init__()
        self._config: dict[str, Any] = {}

    def GetConfig(self) -> dict[str, Any]:
        """
        Retrieves the initialization parameters required for object reconstruction.

        :return: Parameter dictionary mapping layer and structural properties.
        :rtype: dict
        """
        return self._config

    @abstractmethod
    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """
        Projects latent-space vectors into contrastive metric spaces.

        :param z: Latent embedding coordinates tensor matrix. Expected shape: ``[Batch, Latent_Dim]``.
        :type z: torch.Tensor
        :return: High-dimensional projection alignment matrix tensor. Expected shape: ``[Batch, Projection_Dim]``.
        :rtype: torch.Tensor
        """
        pass