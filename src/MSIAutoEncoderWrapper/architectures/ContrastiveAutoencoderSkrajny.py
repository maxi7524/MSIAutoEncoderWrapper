from __future__ import annotations

import torch 
import torch.nn as nn
import torch.nn.functional as F
from .base import MSIBaseAutoencoderArchitecture
from ..dataset import MSIPyTorchDataset
#TODO ADD CORRECT LIBRARY 
from ..utils.architecture_utils import estimate_max_peak_width # Will be moved later
import copy 
import numpy as np

# from turtle import forward ??




class ContrastiveAutoencoderSkrajny(MSIBaseAutoencoderArchitecture):
    """
    CNN-based Contrastive Autoencoder implementation.
    """
    def __init__(self, input_dim, latent_dim, channels, kernels, strides):
        """
        Initializes the ContrastiveAutoencoderSkrajny with specific CNN dimensions.

        :param input_dim: Number of input features (m/z bins).
        :type input_dim: int
        :param latent_dim: Dimension of the compressed latent space.
        :type latent_dim: int
        :param channels: List of channel counts for each convolutional layer.
        :type channels: list[int]
        :param kernels: List of kernel sizes for each convolutional layer.
        :type kernels: list[int]
        :param strides: List of strides for each convolutional layer.
        :type strides: list[int]
        """
        super().__init__()
        ## encoder end decoder elements 
        self.encoder = Encoder(
            input_dim=input_dim,
            latent_dim=latent_dim,
            channels=channels,
            kernels=kernels,
            strides=strides)
        self.decoder = Decoder(
            # here we use spatial from encoder
            spatial_dims=self.encoder.spatial_dims,
            latent_dim=latent_dim,
            channels=channels,
            kernels=kernels,
            strides=strides)


    def forward(self, x):
        z = self.encoder(x)
        x_hat = self.decoder(z)
        return z, x_hat

    def encode(self, x):
        return self.encoder(x)

    def decode(self, z):
        return self.decoder(z)

    @staticmethod
    def SetHyperparameters(MSIDataset: MSIPyTorchDataset, latent_dim: int, user_hyperparameters: dict=None, initialize_model: bool = True) -> dict | ContrastiveAutoencoderSkrajny:
        """
        Dynamically suggests hyperparameters based on spectral peak width and input depth.

        The logic estimates the peak envelope width to set the initial kernel size.

        :param MSILoader: Loader used to retrieve X-axis depth and peak statistics.
        :type MSILoader: MSILoader
        :param latent_dim: The target dimension for the latent bottleneck.
        :type latent_dim: int
        :param user_hyperparams: Predefined parameters that override the auto-suggestion.
        :type user_hyperparams: dict, optional
        :param initialize_model: If true return initialized model, if false return dicts with params 
        :type initialize_model: bool, optional
        :returns: Dictionary containing 'input_dim', 'latent_dim', 'channels', 'kernels', and 'strides'.
        :rtype: dict
        :raises ValueError: If peak width estimation fails.
        """

        input_dim = MSIDataset.GetGridXAxisDepth()
        print(f"[Optimization] Setting architecture hyperparameters ...")
        
        # If hyperparameters are provided, return them.  
        if user_hyperparameters is not None:
            params = copy.deepcopy(user_hyperparameters)
            params['input_dim'] = input_dim
            params['latent_dim'] = latent_dim
            return params

        # Initial size
        ## Estimate envelope width
        ### REMARK, starting from 10_000 we obtain same results 
        print(f"[Optimization] Estimating peak envelope width ...")
        auto_kernel_1 = estimate_max_peak_width(MSIDataset, sample_size=10_000)
        print(f"[Optimization] Estimated peak envelope width: {auto_kernel_1} bins")

        # Layer 1: Wide kernel (15) for peak envelope detection (~0.15 Da at 0.01 Da res)
        channels = [1, 2, 4, 16, 32, 64]
        kernels = [auto_kernel_1, 7, 5, 5, 5]
        strides = [2, 3, 3, 3, 3]

    

        architecture_params = {
            'input_dim': input_dim,
            'latent_dim': latent_dim,
            'channels': channels,
            'kernels': kernels,
            'strides': strides
        }

        if initialize_model:
            return ContrastiveAutoencoderSkrajny(**architecture_params)
        return architecture_params

# ---------------------
# Encoder & decoder
# ---------------------

class Encoder(nn.Module):
    """
    Sub-module responsible for compressing MSI spectra into latent space.

    It returns normalized value. 
    """
    def __init__(self, input_dim, latent_dim, channels, kernels, strides):
        super().__init__()
        self.layers = nn.ModuleList()
        # TODO maybe change name
        self.spatial_dims = [input_dim]

        # CNN
        ## input dimension for CNN
        current_dim = input_dim
        ## recurrent construction for CNN
        for i in range(len(kernels)):
            ### channels dim update
            current_dim = (current_dim - kernels[i]) // strides[i] + 1
            self.spatial_dims.append(current_dim)

            ### BLock:  Conv1d -> LayerNorm -> ReLU
            self.layers.append(nn.Sequential(
                nn.Conv1d(
                    in_channels=channels[i], 
                    out_channels=channels[i+1], 
                    kernel_size=kernels[i], 
                    stride=strides[i]
                    ),
                nn.LayerNorm(current_dim),
                nn.ReLU()
            ))
            
        ## projection
        self.flatten = nn.Flatten()
        self.projector = nn.Sequential(
            nn.Linear(current_dim * channels[-1], latent_dim),
            nn.LayerNorm(latent_dim)
        )
        ## latent dim
        self.spatial_dims.append(latent_dim)

    def forward(self, x: torch.Tensor):
        x = x.unsqueeze(1) # x: (Batch, Features) -> (Batch, 1, Features)
        for layer in self.layers:
            x = layer(x)
        x = self.flatten(x)
        z = self.projector(x)
        z_norm = F.normalize(z, p=2, dim=1)
        return z_norm
    
class Decoder(nn.Module):
    """
    Sub-module responsible for reconstructing MSI spectra from latent space.
    """
    def __init__(self, spatial_dims, latent_dim, channels, kernels, strides):
        super().__init__()
        self.layers = nn.ModuleList()

        # CNN
        ## Linear expansion back to spatial representation
        self.initial_expansion = nn.Sequential(
            nn.Linear(latent_dim, spatial_dims[-2] * channels[-1]),
            nn.LayerNorm(spatial_dims[-2] * channels[-1]),
            ReshapeLayer([channels[-1], spatial_dims[-2]])
        ) 
        ## Transposed convolutions
        for i in range(len(kernels) - 1, -1, -1):
            in_dim = spatial_dims[i+1]
            target_dim = spatial_dims[i]
            
            # Calculate output_padding to perfectly match original dimensions
            out_pad = target_dim - ((in_dim - 1) * strides[i] + kernels[i])
            
            block = nn.Sequential(
                nn.ConvTranspose1d(channels[i+1], channels[i], kernels[i], 
                                   stride=strides[i], output_padding=out_pad),
                # We apply Norm and Activation to all but the very last layer
                nn.LayerNorm(target_dim) if i > 0 else nn.Identity(),
                nn.ReLU() if i > 0 else nn.Identity()
            )
            self.layers.append(block)
        
    def forward(self, x):
        x = self.initial_expansion(x)
        for layer in self.layers:
            x = layer(x)
        return x.squeeze(1)

# ---------------------
# Helpers
# ---------------------

class ReshapeLayer(nn.Module):
    """Layer to reshape tensors within nn.Sequential."""
    def __init__(self, vec_shape):
        super().__init__()
        self.vec_shape = vec_shape

    def forward(self, x):
        return x.view([x.shape[0]] + self.vec_shape)
