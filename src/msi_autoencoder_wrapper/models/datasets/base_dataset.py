"""Shared indexed dataset contracts with optional virtual source selection."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Dict, Mapping, Optional, TYPE_CHECKING

from torch.utils.data import Dataset

from ...configuration import ConfigurableComponent

if TYPE_CHECKING:
    from ...core.mixins.active_context.active_context_mixin import ActiveContextProxy


class MSIBaseDataset(Dataset, ConfigurableComponent, ABC):
    """Expose one public index space over a complete or selected source dataset."""

    def __init__(
        self,
        active_context: Optional[ActiveContextProxy] = None,
        split: Optional[Mapping[str, Any]] = None,
        subset: Optional[Mapping[str, Any]] = None,
        **_: Any,
    ) -> None:
        """Initialize dataset state without reading source samples.

        :param active_context: Runtime context providing data readers and components.
        :type active_context: ActiveContextProxy | None
        :param split: Optional train, validation, and test split definition.
        :type split: Mapping[str, Any] | None
        :param subset: Optional virtual source-selection definition.
        :type subset: Mapping[str, Any] | None
        """
        super().__init__()
        self.active_context = active_context
        self._config: dict[str, Any] = {}
        self._split_config = dict(split) if split is not None else None
        self._subset_config = dict(subset) if subset is not None else None
        self._selection = None
        self._partitions: Optional[Any] = None

    # Public indexed dataset interface
    ## Every consumer uses public indices; source-index mapping occurs exactly once here.
    def __len__(self) -> int:
        """Return the number of samples visible through the current selection."""
        source_length = self._source_length()
        return source_length if self._selection is None else len(self._selection)

    def __getitem__(self, index: int) -> Any:
        """Load one sample through the current public-to-source index mapping."""
        return self._get_source_item(self._source_index(index))

    def get_sample_id(self, index: int) -> Any:
        """Return the stable source identifier for one public dataset index."""
        return self._get_source_sample_id(self._source_index(index))

    def subset(self, config: Optional[Mapping[str, Any]] = None) -> "MSIBaseDataset":
        """Apply or clear a virtual selection on this dataset instance.

        The source reader, active context, and dataset object remain unchanged.
        Only public indices are mapped to selected source indices.

        :param config: Selection definition, or ``None`` to expose all samples.
        :type config: Mapping[str, Any] | None
        :return: This dataset instance.
        :rtype: MSIBaseDataset
        """
        from .subsetting import DatasetSubsetter, IndexSelection

        self._subset_config = dict(config) if config is not None else None
        self._selection = (
            None
            if config is None
            else IndexSelection(
                DatasetSubsetter.select_indices(
                    source_length=self._source_length(),
                    group_provider=self._source_subset_groups,
                    config=self._subset_config,
                )
            )
        )
        self._partitions = None
        return self

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
                        "train": tuple(
                            self.get_sample_id(index) for index in range(len(self))
                        ),
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

    def get_split_target(self, index: int, **parameters: Any) -> Any:
        """Return one split target through the public-to-source mapping."""
        return self._get_source_split_target(self._source_index(index), **parameters)

    def get_split_mask(self, index: int, **parameters: Any) -> Any:
        """Return one split mask through the public-to-source mapping."""
        return self._get_source_split_mask(self._source_index(index), **parameters)

    def get_split_group(self, index: int, **parameters: Any) -> Any:
        """Return one split group through the public-to-source mapping."""
        return self._get_source_split_group(self._source_index(index), **parameters)

    # Source dataset hooks
    ## Concrete datasets implement these methods only with original source indices.
    @abstractmethod
    def _source_length(self) -> int:
        """Return the complete source length before virtual selection."""

    @abstractmethod
    def _get_source_item(self, source_index: int) -> Any:
        """Load one item by its original source index."""

    def _get_source_sample_id(self, source_index: int) -> Any:
        """Return the default stable identifier for one source sample."""
        return source_index

    def _source_subset_groups(
        self,
        source_indices: range,
        **_: Any,
    ) -> list[Any]:
        """Return default single-stratum metadata for source selection."""
        return ["__all_samples__"] * len(source_indices)

    def _get_source_split_target(self, source_index: int, **_: Any) -> Any:
        """Reject target splitting when a dataset does not expose targets."""
        raise NotImplementedError(
            f"{type(self).__name__} does not expose split targets."
        )

    def _get_source_split_mask(self, source_index: int, **_: Any) -> Any:
        """Reject mask splitting when a dataset does not expose masks."""
        raise NotImplementedError(
            f"{type(self).__name__} does not expose split masks."
        )

    def _get_source_split_group(self, source_index: int, **_: Any) -> Any:
        """Reject grouped splitting when a dataset does not expose groups."""
        raise NotImplementedError(
            f"{type(self).__name__} does not expose split groups."
        )

    def _source_index(self, index: int) -> int:
        """Resolve one public index to an original source index."""
        source_length = self._source_length()
        size = source_length if self._selection is None else len(self._selection)
        if index < 0:
            index += size
        if index < 0 or index >= size:
            raise IndexError(index)
        return index if self._selection is None else self._selection[index]


class RawMSIBaseDataset(MSIBaseDataset, ABC):
    """Add raw-spectrum access while retaining the common index mapping."""

    def get_raw_item(self, index: int) -> Any:
        """Return one raw source sample through the current selection."""
        return self._get_raw_source_item(self._source_index(index))

    def get_raw_batch(self, indices: list[int]) -> Any:
        """Return raw source samples for public indices in their given order."""
        source_indices = [self._source_index(index) for index in indices]
        return self._get_raw_source_batch(source_indices)

    @abstractmethod
    def _get_raw_source_item(self, source_index: int) -> Any:
        """Return one raw sample by original source index."""

    def _get_raw_source_batch(self, source_indices: list[int]) -> Any:
        """Return raw source samples when no native batch reader is available."""
        return [self._get_raw_source_item(index) for index in source_indices]
