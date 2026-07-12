"""
Abstract base module defining the mandatory structural contract for all MSI master neural network architectures.
"""

from abc import ABC, abstractmethod
from typing import Dict, Any
import torch
import torch.nn as nn

from ...utils.logger import get_custom_logger

# Logger initialization
logger = get_custom_logger(__name__)


class MSIBaseMasterArchitecture(nn.Module, ABC):
    """
    Abstract Base Class establishing the definitive behavioral and operational contract 
    for all multi-task processing networks inside the architecture ecosystem.
    """

    def __init__(self, resolved_components: Dict[str, nn.Module], **kwargs: Any) -> None:
        """
        Initializes the base master network, anchoring components configurations and metrics parameters.

        :param resolved_components: Mapping configuration pairing subcomponent keys to instantiated nn.Module layers.
        :type resolved_components: Dict[str, nn.Module]
        :param kwargs: Arbitrary backend extension footprints preserved for downstream model frameworks.
        """
        super().__init__()
        
        # State tracking block
        ## Anchor internal parameters storage dictionaries
        self._config: Dict[str, Any] = {}
        
        logger.debug("Instantiating master neural network architecture framework base layer.")

    @abstractmethod
    def forward(self, x: torch.Tensor, **kwargs: Any) -> Dict[str, torch.Tensor]:
        """
        Executes the baseline mathematical execution mapping pass across separate model subcomponents layers.

        :param x: Aligned raw spectrum tensor matrix feed. Shape: [Batch, Features].
        :type x: torch.Tensor
        :return: Standardized evaluation ledger dictionary containing generated outputs tensors.
        :rtype: Dict[str, torch.Tensor]
        """
        pass

    @abstractmethod
    def freeze_backbone(self, freeze: bool = True) -> None:
        """
        Modifies layers gradient requirement parameters status to isolate model weights during training optimization steps.

        :param freeze: Boolean flag activating parameter gradients lock when True, defaults to True.
        :type freeze: bool
        """
        pass