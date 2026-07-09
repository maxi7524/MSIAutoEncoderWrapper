"""
Concrete convolutional encoding strategy mapping dense inputs into latent coordinates.
"""

from typing import List, Any
import torch
import torch.nn as nn

from ....architectures_manager import ArchitecturesManager
from .base_encoder import MSIBaseEncoder

# Internal wrapper module logger injection
from .....utils.logger import get_custom_logger
logger = get_custom_logger(__name__)


@ArchitecturesManager.register_component("autoencoder", "encoder", "CNNEncoder")
class CNNEncoder(MSIBaseEncoder):
    """
    Symmetric 1D Convolutional Neural Network processing grid intensity streams into latent contexts.
    """

    def __init__(
        self,
        input_dim: int,
        latent_dim: int,
        channels: List[int],
        kernels: List[int],
        strides: List[int],
        spatial_dims: List[int],
        **kwargs: Any
    ) -> None:
        """
        Constructs sequential 1D convolutional feature extractors.
        """
        super().__init__()
        
        # Save parameter setups snapshots
        self._config = {
            "input_dim": input_dim,
            "latent_dim": latent_dim,
            "channels": channels,
            "kernels": kernels,
            "strides": strides,
            "spatial_dims": spatial_dims
        }
        
        self.spatial_dims = spatial_dims
        self.conv_blocks = nn.ModuleList()
        
        # Neural pipeline architecture compilation loops
        ## Assemble consecutive splot blocks combining Norm layers and non-linear activations
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
            
        # Compile linear compression compression bottleneck layer bridge
        flattened_dim = channels[-1] * self.spatial_dims[-1]
        self.bottleneck_layer = nn.Sequential(
            nn.Flatten(),
            nn.Linear(flattened_dim, latent_dim),
            nn.LayerNorm(latent_dim)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Compresses input mass spectrometry arrays into dense numerical embeddings. Shape: [Batch, Latent_Dim].
        """
        # Tensor dimensionality safety checkpoint
        ## Coerce plain flat matrix dimensions to support explicit single channel 1D maps
        if x.dim() == 2:
            x = x.unsqueeze(1)
            
        # Execute forward evaluation processing loop across network segments
        for conv_layer in self.conv_blocks:
            x = conv_layer(x)
            
        return self.bottleneck_layer(x)