"""Criterion implementation registration package."""

from msi_autoencoder_wrapper.utils.module_search import discover_modules

from .autoencoder_base_criterions import (
    MSIContrastiveCriterion,
    MSIHeadCriterion,
    MSIReconstructionCriterion,
)
from .base_criterion import MSIBaseCriterion
from .criterions_manager import CompositeLoss, CriterionsManager

discover_modules(__name__)

from .autoencoder.contrastive.infoNCE_loss import MSIInfoNCELoss
from .autoencoder.head.cross_entropy_loss import MSIMaskedCrossEntropyLoss
from .autoencoder.head.multilabel_bce_loss import MSIMultiLabelBCELoss
from .autoencoder.reconstruction.masserstein_loss import MSIMassersteinLoss
from .autoencoder.reconstruction.mse_loss import MSIMSELoss
from .autoencoder.reconstruction.sobolev_loss import MSISobolevLoss

__all__ = [
    "CompositeLoss",
    "CriterionsManager",
    "MSIBaseCriterion",
    "MSIContrastiveCriterion",
    "MSIHeadCriterion",
    "MSIInfoNCELoss",
    "MSIMaskedCrossEntropyLoss",
    "MSIMassersteinLoss",
    "MSIMSELoss",
    "MSIMultiLabelBCELoss",
    "MSIReconstructionCriterion",
    "MSISobolevLoss",
]
