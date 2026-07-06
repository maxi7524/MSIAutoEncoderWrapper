from abc import ABC, abstractmethod
from typing import Any
import torch
import torch.nn as nn


class MSIBaseEncoder(nn.Module, ABC):
    """
    Abstract Base Class defining the contractual interface for all MSI spectrum encoders.
    
    The encoder is responsible for lossy compression, mapping high-dimensional 
    intensities aligned to the grid-x-axis into a dense bottleneck latent-space.
    """

    def __init__(self) -> None:
        """Initializes the base encoder module and its internal configuration state."""
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
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Compresses input intensities into a lower-dimensional latent-space tensor.

        :param x: Input spectral batch aligned to the master grid-x-axis. 
                  Expected shape: ``[Batch, Grid_Depth]`` or ``[Batch, 1, Grid_Depth]``.
        :type x: torch.Tensor
        :return: Extracted bottleneck latent embeddings. Expected shape: ``[Batch, Latent_Dim]``.
        :rtype: torch.Tensor
        """
        pass