"""Criterion implementation registration package."""

from msi_autoencoder_wrapper.utils.module_search import discover_modules

from .autoencoder_base_criterions import (
    MSIContrastiveCriterion,
    MSIHeadCriterion,
    MSIReconstructionCriterion,
    MSIRegularizationCriterion,
)
from .base_criterion import MSIBaseCriterion
from .criterions_manager import CompositeLoss, CriterionsManager

discover_modules(__name__)

from .autoencoder.contrastive.infoNCE_loss import MSIInfoNCELoss
from .autoencoder.head.cross_entropy_loss import MSIMaskedCrossEntropyLoss
from .autoencoder.head.class_balanced_multilabel_bce_loss import (
    MSIClassBalancedMultiLabelBCELoss,
)
from .autoencoder.head.multilabel_bce_loss import MSIMultiLabelBCELoss
from .autoencoder.head.nnpu_multilabel_loss import MSINNPUMultiLabelLoss
from .autoencoder.reconstruction.masserstein_loss import MSIMassersteinLoss
from .autoencoder.reconstruction.mse_loss import MSIMSELoss
from .autoencoder.reconstruction.sobolev_loss import MSISobolevLoss
from .autoencoder.regularization.contractive_loss import MSIContractiveLoss
from .autoencoder.regularization.uniformity_loss import MSIUniformityLoss

__all__ = [
    "CompositeLoss",
    "CriterionsManager",
    "MSIBaseCriterion",
    "MSIContrastiveCriterion",
    "MSIContractiveLoss",
    "MSIHeadCriterion",
    "MSIInfoNCELoss",
    "MSIMaskedCrossEntropyLoss",
    "MSIClassBalancedMultiLabelBCELoss",
    "MSIMassersteinLoss",
    "MSIMSELoss",
    "MSIMultiLabelBCELoss",
    "MSINNPUMultiLabelLoss",
    "MSIReconstructionCriterion",
    "MSIRegularizationCriterion",
    "MSISobolevLoss",
    "MSIUniformityLoss",
]
