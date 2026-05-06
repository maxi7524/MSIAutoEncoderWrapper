from __future__ import annotations
from abc import ABC, abstractmethod
import torch.nn as nn
import torch

from ..dataset import IMSPyTorchDataset



class IMSBaseAutoencoderArchitecture(nn.Module, ABC):
    """
    Abstract base class for Ion Mobility Spectrometry (IMS) Autoencoder architectures.

    This class defines a consistent interface for encoding, decoding, and 
    automated hyperparameter suggestion tailored for mass spectrometry data.
    Subclasses must implement the abstract methods to define specific 
    architectural behaviors.
    """

    def __init__(self):
        """
        Initializes the IMSBaseAutoencoder module.
        """
        super().__init__()

    @abstractmethod
    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Performs a full pass through the autoencoder (encoding and decoding).

        :param x: Input spectral tensor.
        :type x: torch.Tensor
        :returns: A tuple containing the latent representation and the reconstructed output.
        :rtype: tuple(torch.Tensor, torch.Tensor)
        """
        pass

    @abstractmethod
    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """
        Compresses input spectra into a lower-dimensional latent space.

        :param x: Input spectral tensor.
        :type x: torch.Tensor
        :returns: Normalized latent representation.
        :rtype: torch.Tensor
        """
        pass

    @abstractmethod
    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """
        Reconstructs the spectra from latent vectors.

        :param z: Latent representation tensor.
        :type z: torch.Tensor
        :returns: Reconstructed spectral tensor.
        :rtype: torch.Tensor
        """
        pass

    @staticmethod
    @abstractmethod
    def SetHyperparameters(IMSDataset: IMSPyTorchDataset, latent_dim: int, user_hyperparameters: dict=None, initialize_model: bool = True) -> dict | IMSBaseAutoencoderArchitecture:
        """
        Analyzes input data to suggest optimal architecture-specific hyperparameters.

        This method is used BEFORE model existence, to initialize it.

        This method should evaluate data characteristics, such as peak width, 
        to determine appropriate kernel sizes, strides, and layer counts.

        :param IMSLoader: Data loader object containing spectral metadata.
        :type IMSLoader: IMSLoader
        :param latent_dim: Desired dimensionality of the bottleneck layer.
        :type latent_dim: int
        :param user_hyperparams: Optional dictionary to override suggested parameters.
        :type user_hyperparams: dict, optional
        :param initialize_model: If true return initialized model, if false return dicts with params 
        :type initialize_model: bool, optional
        :returns: A dictionary of suggested hyperparameters (e.g., channels, kernels, strides).
        :rtype: dict
        """
        pass