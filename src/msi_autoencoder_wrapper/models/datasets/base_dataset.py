"""
Abstract base dataset architecture mapping out minimalist structural contracts for mass spectrometry samples.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Dict, Mapping, Tuple, Optional, TYPE_CHECKING
from torch.utils.data import Dataset
if TYPE_CHECKING:
    from ...core.mixins.active_context.active_context_mixin import ActiveContextProxy
from ...utils.logger import get_custom_logger
from ...utils.configuration import ConfigurableComponent

# Logger initialization
logger = get_custom_logger(__name__)


class MSIBaseDataset(Dataset, ConfigurableComponent, ABC):
    """
    Abstract Base Class establishing the core contract for all decoupled dataset entities.
    
    Restricts the operational parameters exclusively to execution loops driven by an active context proxy.
    """

    def __init__(
        self,
        active_context: Optional[ActiveContextProxy] = None,
        split: Optional[Mapping[str, Any]] = None,
        **kwargs: Any,
    ) -> None:
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
        self._split_config = dict(split) if split is not None else None
        self._partitions: Optional[Any] = None

    def get_sample_id(self, index: int) -> Any:
        """Return the stable identifier used by split manifests."""
        return index

    def create_partitions(self) -> Any:
        """Create and cache configured model-dataset partitions."""
        from .splitting import DatasetPartitions, DatasetSplitter, SplitManifest

        if self._partitions is not None:
            return self._partitions
        if self._split_config is None:
            self._partitions = DatasetPartitions(
                train=self,
                validation=None,
                test=None,
                manifest=SplitManifest(
                    strategy="none",
                    seed=0,
                    assignments={
                        "train": tuple(self.get_sample_id(index) for index in range(len(self))),
                        "validation": (),
                        "test": (),
                    },
                ),
            )
        else:
            self._partitions = DatasetSplitter.split(self, self._split_config)
        return self._partitions

    def get_split_config(self) -> Optional[Dict[str, Any]]:
        """Return an isolated split definition owned by this dataset."""
        return dict(self._split_config) if self._split_config is not None else None

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
