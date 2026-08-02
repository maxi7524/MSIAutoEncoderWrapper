"""
Concrete convolutional reconstruction strategy decoding latent features back into spectral grids.
"""

from collections.abc import Mapping
from typing import List, Any
import torch
import torch.nn as nn

from ....architectures_manager import ArchitecturesManager
from .base_decoder import MSIBaseDecoder
from .output_activation import build_output_activation

# Synchronized logger initialization
from ......utils.logger import get_custom_logger
logger = get_custom_logger(__name__)


class ReshapeLayer(nn.Module):
    """Utility structural layer to transform tensor topologies within nn.Sequential passes."""
    def __init__(self, target_shape: List[int]) -> None:
        super().__init__()
        self.target_shape = target_shape

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x.view(x.size(0), *self.target_shape)


@ArchitecturesManager.register_component("autoencoder", "decoder", "CNNDecoder")
class CNNDecoder(MSIBaseDecoder):
    """
    1D Transposed Convolutional network reproducing full mass signatures from bottleneck states.
    """

    def __init__(
        self,
        latent_dim: int,
        spatial_dims: List[int],
        channels: List[int],
        kernels: List[int],
        strides: List[int],
        output_activation: Mapping[str, Any],
        **kwargs: Any
    ) -> None:
        """Construct symmetric transposed upsampling layer blocks.

        :param latent_dim: Latent vector width.
        :type latent_dim: int
        :param spatial_dims: Spectrum widths before and after each convolution.
        :type spatial_dims: List[int]
        :param channels: Channel counts for the symmetric convolutional stages.
        :type channels: List[int]
        :param kernels: Kernel sizes for the transposed convolutions.
        :type kernels: List[int]
        :param strides: Strides for the transposed convolutions.
        :type strides: List[int]
        :param output_activation: Final activation configuration. Supported types
            are declared by ``SUPPORTED_OUTPUT_ACTIVATIONS``.
        :type output_activation: Mapping[str, Any]
        """
        super().__init__()
        
        self._config = {
            "latent_dim": latent_dim,
            "spatial_dims": spatial_dims,
            "channels": channels,
            "kernels": kernels,
            "strides": strides,
            "output_activation": dict(output_activation),
        }

        self.output_activation = build_output_activation(output_activation)
        
        # Inversion layer dimension layout assignment
        ## Initial expanding layer targets properties setup
        self.initial_expansion = nn.Sequential(
            nn.Linear(latent_dim, spatial_dims[-1] * channels[-1]),
            nn.LayerNorm(spatial_dims[-1] * channels[-1]),
            ReshapeLayer([channels[-1], spatial_dims[-1]])
        )
        
        self.deconv_blocks = nn.ModuleList()
        
        # Loop backwards through configuration profiles to mirror the encoder downsampling graph
        for i in range(len(kernels) - 1, -1, -1):
            in_width = spatial_dims[i+1]
            target_width = spatial_dims[i]
            
            # Mathematical calculation to resolve required output padding
            out_pad = target_width - ((in_width - 1) * strides[i] + kernels[i])
            
            block = nn.Sequential(
                nn.ConvTranspose1d(
                    in_channels=channels[i+1],
                    out_channels=channels[i],
                    kernel_size=kernels[i],
                    stride=strides[i],
                    output_padding=out_pad
                ),
                nn.LayerNorm(target_width) if i > 0 else nn.Identity(),
                nn.ReLU() if i > 0 else self.output_activation
            )
            self.deconv_blocks.append(block)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """
        Expands latent coordinates back to target spectral channels capacity. Shape: [Batch, Grid_Depth].
        """
        x = self.initial_expansion(z)
        
        for deconv_layer in self.deconv_blocks:
            x = deconv_layer(x)
            
        return x.squeeze(1)
