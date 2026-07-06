from abc import ABC, abstractmethod
from typing import Any
import torch
import torch.nn as nn


class MSIBaseDecoder(nn.Module, ABC):
    """
    Abstract Base Class defining the contractual interface for all MSI spectrum decoders.
    
    The decoder executes the inverse operation, expanding lower-dimensional latent-space 
    coordinates back into dense reconstructed intensity profiles matching the grid-x-axis footprint.
    """

    def __init__(self) -> None:
        """Initializes the base decoder module and its internal configuration state."""
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
        Reconstructs full spectral intensity signatures from latent-space embeddings.

        :param z: Latent coordinate tensors derived from an encoder. Expected shape: ``[Batch, Latent_Dim]``.
        :type z: torch.Tensor
        :return: Reconstructed spectrum profiles mapped directly onto the master grid-x-axis. 
                 Expected shape: ``[Batch, Grid_Depth]``.
        :rtype: torch.Tensor
        """
        pass