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
        spectrum_id: int,
        filters: Optional[Mapping[str, Any]] = None,
    ) -> List[Dict[str, Any]]:
        """Return annotations associated with a source spectrum ID.

        The base implementation selects records whose optional ``spectrum_ids``
        collection contains the requested ID. Canonical SQLite readers override
        this method with an indexed lookup.
        """
        selected: List[Dict[str, Any]] = []
        for annotation in self.get_annotations(filters):
            spectrum_ids = annotation.get("spectrum_ids")
            if spectrum_ids is not None and spectrum_id in spectrum_ids:
                selected.append(annotation)
        return selected

    def get_spectrum_metadata(self, spectrum_id: int) -> Dict[str, Any]:
        """Return source dataset metadata associated with one spectrum.

        :param spectrum_id: Reader-compatible zero-based spectrum identifier.
        :type spectrum_id: int
        :return: Complete metadata record for the owning source dataset.
        :rtype: Dict[str, Any]
        """
        return self.get_dataset_metadata()
