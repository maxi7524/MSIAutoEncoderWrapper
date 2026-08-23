"""
Abstract base module establishing the definitive mathematical contract for all MSI loss functions.
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, Tuple
import torch
import torch.nn as nn

from ...models.datasets.base_dataset import MSIBaseDataset
from ...utils.logger import get_custom_logger
from ...configuration import ConfigurableComponent
from ...utils.exceptions import raise_incompatible_interface_error

# Logger initialization
logger = get_custom_logger(__name__)


class MSIBaseCriterion(nn.Module, ConfigurableComponent, ABC):
    """
    Abstract Base Class defining operational frameworks and lifecycle hooks for MSI cost functions.
    """

    def __init__(self) -> None:
        """
        Initializes the base criterion abstraction layer.
        """
        super().__init__()
        self._config: Dict[str, Any] = {}

    requires_input_grad = False

    def on_phase_start(
        self,
        model: nn.Module,
        dataset: MSIBaseDataset,
        transient_cache: Dict[str, Any],
    ) -> None:
        """Run an optional hook before a training phase starts.

        :param model: Model optimized during the current phase.
        :type model: torch.nn.Module
        :param dataset: Dataset used during the current phase.
        :type dataset: MSIBaseDataset
        :param transient_cache: Shared mutable training cache.
        :type transient_cache: Dict[str, Any]
        """
        del model, dataset, transient_cache

    def on_batch_start(
        self,
        batch_data: Tuple[torch.Tensor, ...],
        transient_cache: Dict[str, Any],
    ) -> Tuple[torch.Tensor, ...]:
        """Run an optional hook before a batch forward pass.

        :param batch_data: Batch returned by the active data loader.
        :type batch_data: Tuple[torch.Tensor, ...]
        :param transient_cache: Shared mutable training cache.
        :type transient_cache: Dict[str, Any]
        :return: Original or transformed batch.
        :rtype: Tuple[torch.Tensor, ...]
        """
        del transient_cache
        return batch_data

    @abstractmethod
    def forward(
        self,
        model_outputs: Dict[str, torch.Tensor],
        batch_data: Tuple[torch.Tensor, ...],
        **kwargs: Any,
    ) -> torch.Tensor:
        """Compute a scalar or unreduced criterion value.

        :param model_outputs: Mapping returned by the model forward pass.
        :type model_outputs: Dict[str, torch.Tensor]
        :param batch_data: Batch returned by the active data loader.
        :type batch_data: Tuple[torch.Tensor, ...]
        :param kwargs: Additional criterion-specific arguments.
        :return: Differentiable criterion tensor.
        :rtype: torch.Tensor
        """
        raise NotImplementedError
