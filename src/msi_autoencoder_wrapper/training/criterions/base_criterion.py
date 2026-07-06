from abc import ABC, abstractmethod
from typing import Any, Dict, Tuple
import torch
import torch.nn as nn

# Purely relative imports pointing to the unified dataset interface
from ...models.datasets.strategies.base_dataset import MSIBaseDataset


class MSIBaseCriterion(nn.Module, ABC):
    """
    Abstract Base Class establishing the mathematical contract for all MSI loss functions.

    This interface decouples empirical loss calculations from explicit training execution loops.
    It exposes properties indicating data requirements (e.g., reconstruction layers or 
    contrastive projections) to allow the execution engine to optimize forward-pass sequences.
    """

    def __init__(self) -> None:
        """Initializes the base criterion module and its isolated configuration state."""
        super().__init__()
        self._config: Dict[str, Any] = {}

    def GetConfig(self) -> Dict[str, Any]:
        """
        Retrieves serialized parameter properties required to replicate the criterion state.

        :return: Parameter configuration map containing structural variables.
        :rtype: dict
        """
        return self._config

    def REQUIRED_SETUP(self, dataset: MSIBaseDataset) -> None:
        """
        Executes global pre-computation steps across the dataset before the training loop starts.

        .. note::
           This method is optional and acts as a baseline hook. Subclasses requiring heavy 
           statistical extractions (e.g., building a dynamic Noise Peak Bank via scipy) 
           must override this method to prepare memory caches.

        :param dataset: Fully initialized concrete subclass of MSIBaseDataset.
        :type dataset: MSIBaseDataset
        """
        pass

    @property
    @abstractmethod
    def requires_reconstruction(self) -> bool:
        """
        Indicates whether this loss function requires full grid-x-axis spectral reconstruction.

        :return: True if the criterion evaluates decoder outputs, False otherwise.
        :rtype: bool
        """
        pass

    @property
    @abstractmethod
    def requires_projection(self) -> bool:
        """
        Indicates whether this loss function requires contrastive space projections.

        :return: True if the criterion evaluates projector head outputs, False otherwise.
        :rtype: bool
        """
        pass

    @abstractmethod
    def forward(
        self,
        model_outputs: Dict[str, torch.Tensor],
        batch_data: Tuple[torch.Tensor, torch.Tensor],
        **kwargs: Any
    ) -> torch.Tensor:
        """
        Computes the target objective loss score using extracted model tensors and raw batch variables.

        :param model_outputs: Sourced dictionary containing evaluated outputs from the model forward pass:
                              - ``"latent_space"``: Core bottleneck tensor.
                              - ``"reconstruction"``: Reconstructed grid-x-axis tensor (optional).
                              - ``"projection"``: Contrastive projection tensor (optional).
        :type model_outputs: dict[str, torch.Tensor]
        :param batch_data: Matched tuple returned by the DataLoader containing (spatial_indices, binned_spectra).
        :type batch_data: tuple(torch.Tensor, torch.Tensor)
        :return: Singular scalar loss tensor tracking graph gradients.
        :rtype: torch.Tensor
        """
        pass