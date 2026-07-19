"""
Abstract base module establishing the definitive mathematical contract for all MSI loss functions.
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, Tuple
import torch
import torch.nn as nn

from ...models.datasets.base_dataset import MSIBaseDataset
from ...utils.logger import get_custom_logger
from ...utils.configuration import ConfigurableComponent
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

    def on_phase_start(self, model: nn.Module, dataset: MSIBaseDataset, transient_cache: Dict[str, Any]) -> None:
        """
        Optional execution lifecycle hook triggered once before the phase epoch sequence loop initiates.

        Useful for heavy pre-computations such as peak-picking or initializing shared statistical arrays.

        :param model: The compiled master architecture model graph currently being optimized.
        :type model: nn.Module
        :param dataset: Bound dataset instance containing raw target spectra profiles.
        :type dataset: MSIBaseDataset
        :param transient_cache: Mutable global scratchpad tracking state variables across the active session.
        :type transient_cache: Dict[str, Any]
        """
        pass


class MSIReconstructionCriterion(MSIBaseCriterion, ABC):
    """Base contract for losses comparing input and reconstructed spectra."""

    criterion_type = "reconstruction"

    @staticmethod
    def reconstruction_pair(
        model_outputs: Dict[str, torch.Tensor],
        batch_data: Tuple[torch.Tensor, ...],
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Return reconstructed and target spectra after interface validation.

        :param model_outputs: Output mapping returned by the model.
        :type model_outputs: Dict[str, torch.Tensor]
        :param batch_data: Dataset batch containing indices and input spectra.
        :type batch_data: Tuple[torch.Tensor, ...]
        :return: Reconstruction followed by the target tensor on the same device.
        :rtype: Tuple[torch.Tensor, torch.Tensor]
        :raises IncompatibleInterfaceError: If required tensors are unavailable.
        """
        if "reconstruction" not in model_outputs:
            raise_incompatible_interface_error(
                context_name="ReconstructionCriterion",
                message="Model outputs must contain a 'reconstruction' tensor.",
            )
        if len(batch_data) < 2 or not isinstance(batch_data[1], torch.Tensor):
            raise_incompatible_interface_error(
                context_name="ReconstructionCriterion",
                message="Batch data must contain an input spectrum tensor at index 1.",
            )
        reconstruction = model_outputs["reconstruction"]
        return reconstruction, batch_data[1].to(reconstruction.device)


class MSIContrastiveCriterion(MSIBaseCriterion, ABC):
    """Base contract for representation losses using augmented spectrum pairs."""

    criterion_type = "contrastive"


class MSIHeadCriterion(MSIBaseCriterion, ABC):
    """Base contract for objectives attached to one named model head."""

    criterion_type = "head"

    def __init__(self, head_name: str) -> None:
        """Initialize the output key used by a downstream objective.

        :param head_name: Name used by the model's ``head_<name>`` output.
        :type head_name: str
        """
        super().__init__()
        self.head_name = head_name

    @property
    def output_key(self) -> str:
        """Return the standardized model-output key for this head."""
        return f"head_{self.head_name}"

    def on_batch_start(self, batch_data: Tuple[torch.Tensor, ...], transient_cache: Dict[str, Any]) -> Tuple[torch.Tensor, ...]:
        """
        Optional execution lifecycle hook triggered for every step batch stream iteration before forward calls.

        Enables on-the-fly computational operations like chemical noise augmentations on the GPU.

        :param batch_data: Tensor tuple returned from the active data loader.
        :type batch_data: Tuple[torch.Tensor, ...]
        :param transient_cache: Mutable global scratchpad tracking state variables across the active session.
        :type transient_cache: Dict[str, Any]
        :return: Transformed or untouched batch variables aligned to data specifications.
        :rtype: Tuple[torch.Tensor, ...]
        """
        return batch_data

    @abstractmethod
    def forward(
        self,
        model_outputs: Dict[str, torch.Tensor],
        batch_data: Tuple[torch.Tensor, ...],
        **kwargs: Any
    ) -> torch.Tensor:
        """
        Computes the target objective mathematical loss matrix score using extracted forward tensors.

        :param model_outputs: Collection mapping outputs generated by the active model forward pass.
        :type model_outputs: Dict[str, torch.Tensor]
        :param batch_data: Variables tuple matching data loader outputs.
        :type batch_data: Tuple[torch.Tensor, ...]
        :return: Singular scalar tensor containing tracking gradients.
        :rtype: torch.Tensor
        """
        pass
