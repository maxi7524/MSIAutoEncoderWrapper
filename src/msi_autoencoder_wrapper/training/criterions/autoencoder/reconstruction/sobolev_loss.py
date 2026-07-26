"""
Sobolev first-order derivative regularized loss function strategy.
"""

from typing import Any, Dict, Tuple
import torch

from ...autoencoder_base_criterions import MSIReconstructionCriterion
from ...criterions_manager import CriterionsManager
from .....utils.logger import get_custom_logger

# Logger initialization
logger = get_custom_logger(__name__)


@CriterionsManager.register_criterion("autoencoder", "reconstruction", "SobolevLoss")
class MSISobolevLoss(MSIReconstructionCriterion):
    """
    Weighted RMSE Sobolev loss strategy calculating spectral derivative deviations.
    """

    def __init__(self, sobolev_weight: float = 0.5, eps: float = 1e-6) -> None:
        """
        Configures the derivative factor weight coefficients.

        :param sobolev_weight: Multiplier tracking balancing properties for the derivative term.
        :type sobolev_weight: float
        :param eps: Boundary regularization constant avoiding divide-by-zero errors.
        :type eps: float
        """
        super().__init__()
        self._config = {"sobolev_weight": sobolev_weight, "eps": eps}
        self.sobolev_weight = sobolev_weight
        self.eps = eps

    def forward(
        self,
        model_outputs: Dict[str, torch.Tensor],
        batch_data: Tuple[torch.Tensor, ...],
        **kwargs: Any
    ) -> torch.Tensor:
        """
        Computes the weighted mean squared derivative penalty scores.
        """
        # Heading 1 (Sobolev High-Order Gradient Discrepancy Pass)
        reconstructed_spectra, original_spectra = self.reconstruction_pair(
            model_outputs,
            batch_data,
        )

        ## 1. Compute baseline dynamic spatial coordinate weights from average magnitudes
        mean_intensity = torch.mean(original_spectra, dim=1, keepdim=True)
        weights = 1.0 / (mean_intensity + self.eps)

        ## 2. Evaluate standard zero-order RMSE reconstruction deviations
        diff_zero = reconstructed_spectra - original_spectra
        loss_zero = torch.mean(weights * (diff_zero ** 2))

        ## 3. Compute discrete first-order derivative vectors via adjacent bins subtraction
        deriv_orig = original_spectra[:, 1:] - original_spectra[:, :-1]
        deriv_recon = reconstructed_spectra[:, 1:] - reconstructed_spectra[:, :-1]

        ## 4. Evaluate first-order derivative error variations
        diff_first = deriv_recon - deriv_orig
        loss_first = torch.mean(weights * (diff_first ** 2))

        ## 5. Compile final balanced linear loss combination score
        return loss_zero + (self.sobolev_weight * loss_first)
