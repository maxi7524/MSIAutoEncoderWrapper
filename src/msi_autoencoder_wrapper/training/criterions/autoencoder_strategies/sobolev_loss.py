from typing import Any, Dict, Tuple
import torch
import torch.nn as nn

from ..criterions_manager import CriterionsManager
from ..base_criterion import MSIBaseCriterion


@CriterionsManager.register_criterion("SobolevLoss")
class MSISobolevLoss(MSIBaseCriterion):
    """
    Weighted RMSE Sobolev loss strategy calculating spectral derivative deviations.

    This loss evaluates discrepancy vectors both on raw profile channels and 
    discrete first-order derivative variations across adjacent mass-to-charge (m/z) bins,
    dynamically weighting errors relative to average empirical magnitudes.
    """

    def __init__(self, sobolev_weight: float = 0.5, eps: float = 1e-6) -> None:
        """
        Configures the derivative factor weight coefficients.

        :param sobolev_weight: Multiplier tracking balancing properties for the derivative term.
        :type sobolev_weight: float
        :param eps: Boundary regularization constant avoiding divide-by-zero errors during variance steps.
        :type eps: float
        """
        super().__init__()
        self._config = {"sobolev_weight": sobolev_weight, "eps": eps}
        self.sobolev_weight = sobolev_weight
        self.eps = eps

    @property
    def requires_reconstruction(self) -> bool:
        """Requires full spectrum reconstruction arrays to validate derivative profiles."""
        return True

    @property
    def requires_projection(self) -> bool:
        """Does not utilize contrastive representation elements."""
        return False

    def forward(
        self,
        model_outputs: Dict[str, torch.Tensor],
        batch_data: Tuple[torch.Tensor, torch.Tensor],
        **kwargs: Any
    ) -> torch.Tensor:
        """
        Computes the weighted mean squared derivative penalty scores.
        """
        if "reconstruction" not in model_outputs:
            raise KeyError("The 'reconstruction' matrix is missing from forward pass evaluations dictionary maps.")

        _, original_spectra = batch_data
        reconstructed_spectra = model_outputs["reconstruction"]

        # 1. Compute baseline dynamic spatial coordinate weights from average spectrum magnitude footprints
        mean_intensity = torch.mean(original_spectra, dim=1, keepdim=True)
        weights = 1.0 / (mean_intensity + self.eps)

        # 2. Evaluate standard weighted zero-order reconstruction errors (Weighted RMSE)
        diff_zero = reconstructed_spectra - original_spectra
        loss_zero = torch.mean(weights * (diff_zero ** 2))

        # 3. Compute discrete first-order derivative vectors via finite differences across neighboring channels
        deriv_orig = original_spectra[:, 1:] - original_spectra[:, :-1]
        deriv_recon = reconstructed_spectra[:, 1:] - reconstructed_spectra[:, :-1]

        # 4. Evaluate first-order derivative errors (Sobolev regularizer term)
        diff_first = deriv_recon - deriv_orig
        loss_first = torch.mean(weights * (diff_first ** 2))

        # 5. Compile weighted aggregate total score step vectors
        return loss_zero + (self.sobolev_weight * loss_first)