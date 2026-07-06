from typing import List, Dict, Any
import torch
import torch.nn as nn

# Purely relative imports linking down execution graphs
from ..manager import ArchitectureManager
from .base_decoder import MSIBaseDecoder
from ..utils.reshape_layer import ReshapeLayer


@ArchitectureManager.register_decoder("CNNDecoder")
class CNNDecoder(MSIBaseDecoder):
    """
    1D Transposed Convolutional network strategy for symmetric mass spectra recovery.

    This component expands latent coordinates to compute reciprocal spatial widths,
    outputting reconstructed profiles exactly matching the master grid-x-axis structure.
    """

    def __init__(
        self,
        latent_dim: int,
        spatial_dims: List[int],
        channels: List[int],
        kernels: List[int],
        strides: List[int]
    ) -> None:
        """
        Constructs the structural transposed convolutional network pipeline graph.

        :param latent_dim: Quantitative sizing allocating bottleneck latent-space configurations.
        :type latent_dim: int
        :param spatial_dims: Layer array widths trace extracted directly from the corresponding encoder setup.
        :type spatial_dims: List[int]
        :param channels: Block channel configurations tracking reciprocal layer filter allocations.
        :type channels: List[int]
        :param kernels: Window sizes tracking the convolutional operations footprint geometry.
        :type kernels: List[int]
        :param strides: Target strides determining spatial step definitions.
        :type strides: List[int]
        """
        super().__init__()
        
        # Cache configuration parameters snapshot for structural serialization routines
        self._config = {
            "latent_dim": latent_dim,
            "spatial_dims": spatial_dims,
            "channels": channels,
            "kernels": kernels,
            "strides": strides
        }

        # Baseline linear dimension translation block
        ## Project compressed vectors back into equivalent geometric tensors matching the deepest splot layers
        self.initial_expansion = nn.Sequential(
            nn.Linear(latent_dim, spatial_dims[-1] * channels[-1]),
            nn.LayerNorm(spatial_dims[-1] * channels[-1]),
            ReshapeLayer([channels[-1], spatial_dims[-1]])
        )
        
        self.deconv_blocks = nn.ModuleList()
        
        # Build inversion operations sequence bottom-up from deep tracking parameters
        for i in range(len(kernels) - 1, -1, -1):
            in_width = spatial_dims[i+1]
            target_width = spatial_dims[i]
            
            # Mathematical calculation of explicit output padding adjustments to secure clean matrix matching
            out_pad = target_width - ((in_width - 1) * strides[i] + kernels[i])
            
            # Assemble discrete transposed layer pipeline nodes
            block = nn.Sequential(
                nn.ConvTranspose1d(
                    in_channels=channels[i+1],
                    out_channels=channels[i],
                    kernel_size=kernels[i],
                    stride=strides[i],
                    output_padding=out_pad
                ),
                nn.LayerNorm(target_width) if i > 0 else nn.Identity(),
                nn.ReLU() if i > 0 else nn.Identity()
            )
            self.deconv_blocks.append(block)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """
        Decodes compressed bottleneck vectors into full spectral signatures on the grid-x-axis.

        :param z: Latent embedding coordinates tensor matrix. Shape: [Batch, Latent_Dim].
        :type z: torch.Tensor
        :return: Reconstructed profile vector matching baseline grid feature shapes. Shape: [Batch, Grid_Depth].
        :rtype: torch.Tensor
        """
        # 1. Expand flat bottleneck layers back to multi-channel feature shapes
        x = self.initial_expansion(z)
        
        # 2. Drive arrays sequentially through up-sampling transposition paths
        for deconv_layer in self.deconv_blocks:
            x = deconv_layer(x)
            
        # 3. Strip structural extra dimensional padding frames to output flat grid vectors
        return x.squeeze(1)