from abc import ABC, abstractmethod
from typing import Any
import torch
import torch.nn as nn

class MSIBaseProjector(nn.Module, ABC):
    """Contractual interface mapping features into high-dimensional layers for contrastive tasks inside AE."""
    def __init__(self) -> None:
        super().__init__()
        self._config: dict[str, Any] = {}

    @abstractmethod
    def forward(self, z: torch.Tensor) -> torch.Tensor:
        pass