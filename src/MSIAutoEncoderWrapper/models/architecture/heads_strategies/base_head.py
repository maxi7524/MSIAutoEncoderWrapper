from abc import ABC, abstractmethod
from typing import Any
import torch
import torch.nn as nn


class MSIBaseHead(nn.Module, ABC):
    """
    Abstract Base Class for auxiliary multi-task evaluation heads.
    
    Enables parallel downstream tasks (such as segmentation, classification, 
    or regression) to anchor directly onto the core latent-space backbone.
    """

    def __init__(self) -> None:
        """Initializes the auxiliary head module and its internal configuration state."""
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
        Maps latent-space vectors to task-specific objective target fields.

        :param z: Bottleneck latent coordinates matrix tensor. Expected shape: ``[Batch, Latent_Dim]``.
        :type z: torch.Tensor
        :return: Task-specific tensor output (e.g., logits, spatial classes). Shape varies by objective.
        :rtype: torch.Tensor
        """
        pass