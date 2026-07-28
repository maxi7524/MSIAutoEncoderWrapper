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
    def search_datasets(self, filters: Mapping[str, Any]) -> List[Dict[str, Any]]:
        """Return complete metadata records matching provider-side filters."""

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
