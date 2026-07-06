from abc import ABC, abstractmethod
from typing import Any, Tuple
import torch
from torch.utils.data import Dataset
import numpy as np

# Purely relative imports within the package hierarchy
from ....readers.strategies.base_reader import MSIBaseReader
from ....binners.binners_strategies.base_binner import MSIBaseBinner


class MSIBaseDataset(Dataset, ABC):
    """
    Abstract Base Class establishing the architectural contract for all MSI PyTorch datasets.

    This interface forces all custom sampling strategies (pixel-level, spatial, contrastive)
    to securely wrap a data loader and a spectral binner, exposing unified structural property 
    getters for automated grid-x-axis dimensionality verification.
    """

    def __init__(self, loader: MSIBaseReader, binner: MSIBaseBinner) -> None:
        """
        Initializes the base dataset wrapper via explicit dependency injection.

        :param loader: Concrete implementation of MSIBaseReader driving storage file I/O operations.
        :type loader: MSIBaseReader
        :param binner: Concrete implementation of MSIBaseBinner defining target grid-x-axis mapping properties.
        :type binner: MSIBaseBinner
        """
        super().__init__()
        
        # Injected dependency bindings
        self._loader = loader
        self._binner = binner
        
        # Configuration registry snapshot for pipeline serialization
        self._config: dict[str, Any] = {}

    def GetConfig(self) -> dict[str, Any]:
        """
        Retrieves baseline configuration definitions required for dataset serialization routines.

        :return: Map specifying metadata, parameters, and bound components configurations.
        :rtype: dict
        """
        return self._config

    @abstractmethod
    def __len__(self) -> int:
        """
        Computes total computational elements available for model sampling loops.

        :return: Cumulative sample space density.
        :rtype: int
        """
        pass

    @abstractmethod
    def __getitem__(self, idx: int) -> Tuple[Any, ...]:
        """
        Generates processed structural tensor outputs aligned directly to the master grid-x-axis layout.

        :param idx: Sequence pointer targeting a specific dataset computational sample.
        :type idx: int
        :return: Tuple containing spatial alignment references and processed standard PyTorch tensors.
        :rtype: tuple
        """
        pass

    # Enforced abstract getters to guarantee decoupling across architectures and trainers
    @property
    def loader(self) -> MSIBaseReader:
        """Exposes the internal bound file storage reader interface."""
        return self._loader

    @property
    def binner(self) -> MSIBaseBinner:
        """Exposes the internal bound forward spectral transformation interface."""
        return self._binner

    def GetGridXMin(self) -> float:
        """Retrieves absolute lower boundary limits configured on the target alignment axis."""
        return self._binner.GetXMin()

    def GetGridXMax(self) -> float:
        """Retrieves absolute upper boundary limits configured on the target alignment axis."""
        return self._binner.GetXMax()

    def GetGridXAxis(self) -> np.ndarray:
        """Retrieves the master spectral vector mapping central mass-to-charge (m/z) positions."""
        return self._binner.GetXAxis()

    def GetGridXAxisDepth(self) -> int:
        """Computes input features capacity required to define compatible encoder neural weights layers."""
        return self._binner.GetXAxisDepth()