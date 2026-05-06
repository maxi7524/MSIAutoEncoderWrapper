from abc import ABC, abstractmethod
import torch.nn as nn
import torch
from ..architectures.base import IMSBaseAutoencoderArchitecture

class IMSABaseAutoEncoderCriterion(nn.Module, ABC):
    """
    Abstract base class for all IMS Criterions.

    This class defines the training strategy and has full access to the model, 
    spectral data, and spatial indices. This allows for complex loss functions 
    that can incorporate spatial context or neighborhood relationships.
    """
    # List of setup functions to be executed before training loop 
    REQUIRED_SETUP = []

    def __init__(self):
        super().__init__()

    @abstractmethod
    def forward(self, 
                batch_idx: int, 
                batch_data: tuple[torch.Tensor, torch.Tensor], 
                model: IMSBaseAutoencoderArchitecture, 
                dataloader: torch.utils.data.DataLoader,
                device: torch.device) -> tuple[torch.Tensor, dict[str, float]]:
        """
        Executes the specific training logic for a given batch.

        :param model: The autoencoder model providing .encode() and .decode().
        :param batch_data: A tuple (spatial_indices, spectra_tensors) returned by the Dataset.
        :param batch_idx: The current iteration index within the epoch.
        :param dataloader: The DataLoader reference, providing access to the underlying Dataset.
        :param device: Torch device parametr for tensor handling.
        
        :returns: A tuple (total_loss, loss_dict) where loss_dict contains components for logging.

        Example: How to extract spatial information:
        -------------------------------------------
        >>> indices, spectra = batch_data
        >>> # Get (x, y, z) coordinates for the first spectrum in the batch:
        >>> first_idx = indices[0].item()
        >>> coords = dataloader.dataset.img.GetSpectrumPosition(first_idx)
        >>> print(f"Processing spectrum at X: {coords[0]}, Y: {coords[1]}")
        """
        pass