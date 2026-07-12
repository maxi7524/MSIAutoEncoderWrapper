"""
Mean Squared Error reconstruction loss strategy for autoencoder spectral profiles.
"""

from typing import Any, Dict, Tuple
import torch
import torch.nn as nn

from ...base_criterion import MSIBaseCriterion
from ...criterions_manager import CriterionsManager
from .....utils.logger import get_custom_logger

# Logger initialization
logger = get_custom_logger(__name__)


@CriterionsManager.register_criterion("autoencoder", "MSELoss")
class MSIMSELoss(MSIBaseCriterion):
    """
    Mean Squared Error (MSE) loss strategy evaluating raw profile reconstruction accuracy.
    """

    def __init__(self, reduction: str = "mean") -> None:
        """
        Initializes the underlying PyTorch MSE loss module block.

        :param reduction: Specifies the reduction scheme to apply to output tensors: 'mean' or 'sum'.
        :type reduction: str
        """
        super().__init__()
        self._config = {"reduction": reduction}
        self.loss_fn = nn.MSELoss(reduction=reduction)

    def forward(
        self,
        model_outputs: Dict[str, torch.Tensor],
        batch_data: Tuple[torch.Tensor, ...],
        **kwargs: Any
    ) -> torch.Tensor:
        """
        Evaluates the mean squared reconstruction error between target arrays.
        """
        # Heading 1 (Reconstruction Discrepancy Evaluation Pass)
        if "reconstruction" not in model_outputs:
            logger.error("Evaluation aborted: 'reconstruction' key missing from model outputs.")
            raise KeyError("MSELoss requires 'reconstruction' field in model outputs.")

        ## Unpack spectra matrix from data stream batch
        _, original_spectra = batch_data
        reconstructed_spectra = model_outputs["reconstruction"]

        ## Compute and return the mathematical reduction error matrix score
        return self.loss_fn(reconstructed_spectra, original_spectra.to(reconstructed_spectra.device))