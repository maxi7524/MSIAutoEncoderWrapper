from abc import ABC, abstractmethod
from typing import Any
import torch
import torch.nn as nn

from ......utils.configuration import ConfigurableComponent

class MSIBaseDecoder(nn.Module, ConfigurableComponent, ABC):
    """Contractual interface for expanding latent coordinates back into spectral reconstructions inside AE."""
    def __init__(self) -> None:
        super().__init__()
        self._config: dict[str, Any] = {}

    @abstractmethod
    def forward(self, z: torch.Tensor) -> torch.Tensor:
        pass
