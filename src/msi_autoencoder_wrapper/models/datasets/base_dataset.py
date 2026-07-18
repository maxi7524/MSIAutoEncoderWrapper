"""
Abstract base dataset architecture mapping out minimalist structural contracts for mass spectrometry samples.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Tuple, Optional, TYPE_CHECKING
from torch.utils.data import Dataset
if TYPE_CHECKING:
    from ...core.mixins.active_context.active_context_mixin import ActiveContextProxy
from ...utils.logger import get_custom_logger

# Logger initialization
logger = get_custom_logger(__name__)


class MSIBaseDataset(Dataset, ABC):
    """
    Abstract Base Class establishing the core contract for all decoupled dataset entities.
    
    Restricts the operational parameters exclusively to execution loops driven by an active context proxy.
    """

    def __init__(self, active_context: Optional[ActiveContextProxy] = None, **kwargs: Any) -> None:
        """
        Initializes the base dataset wrapper via implicit context binding.

        :param active_context: Active execution session proxy tracking live reader and binner drivers.
        :type active_context: ActiveContextProxy
        :param kwargs: Arbitrary parameter footprints preserved for downstream strategy instantiation.
        """
        super().__init__()
        
        # State tracking block
        ## Anchor the unified active session tracking reference hook
        self.active_context = active_context
        self._config: dict[str, Any] = {}

    def GetConfig(self) -> dict[str, Any]:
        """
        Retrieves baseline parameters configured across setup pipelines.

        :return: Map containing foundational serialized configurations.
        :rtype: dict[str, Any]
        """
        return self._config

    @abstractmethod
    def __len__(self) -> int:
        """
        Computes total computational samples capacity mapped inside the underlying data structures.

        :return: Total length mapping dataset tracking limits.
        :rtype: int
        """
        pass

    @abstractmethod
    def __getitem__(self, idx: int) -> Tuple[Any, ...]:
        """
        Generates processed structural outputs aligned directly to the verified grid layout.

        :param idx: Sequence pointer targeting a specific dataset computational sample.
        :type idx: int
        :return: Variables tuple containing sample index tracking keys and intensity tensors.
        :rtype: Tuple[Any, ...]
        """
        pass