"""Base contract for external MSI dataset providers."""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional

from ...utils.configuration import ConfigurableComponent


class DatasetSource(ConfigurableComponent, ABC):
    """Define provider-independent dataset discovery and download operations."""

    source_name: str

    @abstractmethod
    def get_available_filters(self) -> Dict[str, Any]:
        """Return the source's supported provider and local filter schema."""

    @abstractmethod
    def filter(self, filters: Mapping[str, Any]) -> List[Dict[str, Any]]:
        """Execute filtering and return complete accepted metadata records."""

    @abstractmethod
    def get_accepted_records(self) -> List[Dict[str, Any]]:
        """Return records accepted by the most recent filtering call."""

    @abstractmethod
    def get_rejected_records(self) -> List[Dict[str, Any]]:
        """Return records rejected by the most recent filtering call."""

    def available_filters(self) -> Dict[str, Any]:
        """Compatibility alias for :meth:`get_available_filters`."""
        return self.get_available_filters()

    def search_datasets(self, filters: Mapping[str, Any]) -> List[Dict[str, Any]]:
        """Compatibility alias for :meth:`filter`."""
        return self.filter(filters)

    def get_search_diagnostics(self) -> List[Dict[str, Any]]:
        """Compatibility alias for :meth:`get_rejected_records`."""
        return self.get_rejected_records()

    @abstractmethod
    def get_dataset_metadata(self, dataset_id: str) -> Dict[str, Any]:
        """Return complete metadata for one external dataset."""

    @abstractmethod
    def get_annotations(
        self,
        dataset_id: str,
        options: Optional[Mapping[str, Any]] = None,
    ) -> List[Dict[str, Any]]:
        """Return available annotations without experiment-specific filtering."""

    @abstractmethod
    def download_dataset(self, dataset_id: str, destination: Path | str) -> Path:
        """Materialize the imzML/ibd pair and return its directory."""
