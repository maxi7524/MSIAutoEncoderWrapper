from typing import List, Dict, Any
import torch
import torch.nn as nn

# Purely relative imports linking back to interface and registry contexts
from ..manager import ArchitectureManager
from .base_encoder import MSIBaseEncoder


@ArchitectureManager.register_encoder("CNNEncoder")
class CNNEncoder(MSIBaseEncoder):
    """
    1D Convolutional Neural Network strategy for structural grid-x-axis profiling.

    This component reduces the sequence length of highly dimensional binned mass spectrometry 
    intensities using deep splot blocks, projecting feature vectors directly into the stable latent-space.
    """

    def __init__(
        self,
        input_dim: int,
        latent_dim: int,
        channels: List[int],
        kernels: List[int],
        strides: List[int]
    ) -> None:
        """
        Constructs the sequential convolutional embedding layers.

        :param input_dim: Structural feature depth of the master grid-x-axis alignment vector.
        :type input_dim: int
        :param latent_dim: Dimensional capacity targeting the bottleneck latent-space topology.
        :type latent_dim: int
        :param channels: Vector tracking filter depths across cascading layer operations.
        :type channels: List[int]
        :param kernels: Operational receptive field windows mapped across layer convolutions.
        :type kernels: List[int]
        :param strides: Down-sampling factor metrics tracking index strides across the grid-x-axis.
        :type strides: List[int]
        """
        super().__init__()
        
        # Capture configurations snapshot for explicit pipeline recreation steps
        self._config = {
            "input_dim": input_dim,
            "latent_dim": latent_dim,
            "channels": channels,
            "kernels": kernels,
            "strides": strides
        }
        
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.channels = channels
        self.kernels = kernels
        self.strides = strides

        # Structural space breakdown processing loops
        self.spatial_dims = [input_dim]
        current_dim = input_dim
        
        # Calculate sequence degradation across sequential pooling boundaries
        for i in range(len(kernels)):
            current_dim = (current_dim - kernels[i]) // strides[i] + 1
            self.spatial_dims.append(current_dim)
            
        # Core convolution blocks architecture
        self.conv_blocks = nn.ModuleList()
        for i in range(len(kernels)):
            block = nn.Sequential(
                nn.Conv1d(
                    in_channels=channels[i],
                    out_channels=channels[i+1],
                    kernel_size=kernels[i],
                    stride=strides[i]
                ),
                nn.LayerNorm(self.spatial_dims[i+1]),
                nn.ReLU()
            )
            self.conv_blocks.append(block)
            
        # Linear compression bridge execution step
        flattened_dim = channels[-1] * self.spatial_dims[-1]
        self.bottleneck_layer = nn.Sequential(
            nn.Flatten(),
            nn.Linear(flattened_dim, latent_dim),
            nn.LayerNorm(latent_dim)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Compresses grid-x-axis intensity profiles into latent-space vectors.

        :param x: Batch tensor matching grid-x-axis formatting specifications. Shape: [Batch, Grid_Depth].
        :type x: torch.Tensor
        :return: Dense representations mapped inside latent boundaries. Shape: [Batch, Latent_Dim].
        :rtype: torch.Tensor
        """
        # Inject standard single channel dimension if structural shape evaluates flat
        if x.dim() == 2:
            x = x.unsqueeze(1)
            
        # Feed-forward sequence driving calculations through deep splot kernels
        for conv_layer in self.conv_blocks:
            x = conv_layer(x)
            
        return self.bottleneck_layer(x)