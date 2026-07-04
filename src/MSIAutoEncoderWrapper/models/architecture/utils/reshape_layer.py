from typing import List
import torch
import torch.nn as nn


class ReshapeLayer(nn.Module):
    """
    Structural transformation utility node executing explicit tensor dimension modifications.
    """

    def __init__(self, target_shape: List[int]) -> None:
        """
        Configures the target layout projection blueprint parameters.

        :param target_shape: Target structural size footprint dimensions.
        :type target_shape: List[int]
        """
        super().__init__()
        self.target_shape = target_shape

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Transforms flat linear layers back into high-dimensional geometric channel configurations.
        """
        return x.view(x.size(0), *self.target_shape)