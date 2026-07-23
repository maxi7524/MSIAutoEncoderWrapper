"""Base contract for dataset and spectrum annotation readers."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Mapping, Optional

from ..utils.configuration import ConfigurableComponent


class MSIBaseAnnotationReader(ConfigurableComponent, ABC):
    """Expose normalized annotations without imposing model target semantics."""

    def __init__(self, active_context: Optional[Any] = None) -> None:
        self.active_context = active_context
        self._config: Dict[str, Any] = {}

    @abstractmethod
    def get_dataset_metadata(self) -> Dict[str, Any]:
        """Return complete metadata for the selected source dataset."""

    @abstractmethod
    def get_annotations(
        self,
        filters: Optional[Mapping[str, Any]] = None,
    ) -> List[Dict[str, Any]]:
        """Return annotations, applying only explicitly requested filters."""

    def get_spectrum_annotations(
        self,
        spatial_id: int,
        filters: Optional[Mapping[str, Any]] = None,
    ) -> List[Dict[str, Any]]:
        """Return annotations associated with a source spectrum/spatial ID.

        The base implementation selects records whose optional ``spatial_ids``
        collection contains the requested ID. Provider strategies may override
        this method to use ion-image storage more efficiently.
        """
        selected: List[Dict[str, Any]] = []
        for annotation in self.get_annotations(filters):
            spatial_ids = annotation.get("spatial_ids")
            if spatial_ids is not None and spatial_id in spatial_ids:
                selected.append(annotation)
        return selected
