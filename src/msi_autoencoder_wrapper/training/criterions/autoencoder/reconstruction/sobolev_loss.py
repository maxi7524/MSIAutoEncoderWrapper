"""
Sobolev first-order derivative regularized loss function strategy.
"""

from typing import Any, Dict, Tuple
import torch

from ...autoencoder_base_criterions import MSIReconstructionCriterion
from ...criterions_manager import CriterionsManager
from .....utils.logger import get_custom_logger
from .....metrics import sobolev
from .....data import SpectrumBatch

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

        mass_axis = batch_data.space.mass_axis if isinstance(batch_data, SpectrumBatch) else None
        return sobolev(
            reconstructed_spectra,
            original_spectra,
            mass_axis=mass_axis,
            derivative_weight=self.sobolev_weight,
            epsilon=self.eps,
        ).mean()
