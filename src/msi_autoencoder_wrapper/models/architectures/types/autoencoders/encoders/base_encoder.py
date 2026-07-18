from abc import ABC, abstractmethod
from typing import Any
import torch
import torch.nn as nn

from ......utils.configuration import ConfigurableComponent

class MSIBaseEncoder(nn.Module, ConfigurableComponent, ABC):
    """Contractual interface for lossy compression mapping intensities into latent bottlenecks inside AE."""
    def __init__(self) -> None:
        super().__init__()
        self._config: dict[str, Any] = {}

    @abstractmethod
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        pass
