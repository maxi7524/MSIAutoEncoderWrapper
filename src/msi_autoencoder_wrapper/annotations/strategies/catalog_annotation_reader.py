"""Annotation reader backed by the workspace dataset catalog."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional

from ...workspace.dataset_catalog import DatasetCatalog
from ..annotations_manager import AnnotationReaderManager
from ..base_annotation_reader import MSIBaseAnnotationReader


@AnnotationReaderManager.register_reader("CatalogAnnotationReader")
class CatalogAnnotationReader(MSIBaseAnnotationReader):
    """Read imported source records from ``catalog.sqlite``.

    :param catalog_path: Workspace catalog path.
    :type catalog_path: pathlib.Path | str
    :param source: Source adapter key.
    :type source: str
    :param dataset_id: External dataset identifier.
    :type dataset_id: str
    :param default_filters: Optional read-time filters. Empty by default.
    :type default_filters: Mapping[str, Any] | None
    :param active_context: Optional active context reference.
    :type active_context: Any | None
    """

    def __init__(
        self,
        catalog_path: Path | str,
        source: str,
        dataset_id: str,
        default_filters: Optional[Mapping[str, Any]] = None,
        active_context: Optional[Any] = None,
    ) -> None:
        super().__init__(active_context=active_context)
        self.catalog = DatasetCatalog(catalog_path)
        self.source = source
        self.dataset_id = dataset_id
        self.default_filters = dict(default_filters or {})
        self._config = {
            "catalog_path": str(catalog_path),
            "source": source,
            "dataset_id": dataset_id,
            "default_filters": self.default_filters,
        }

    def get_dataset_metadata(self) -> Dict[str, Any]:
        """Return the complete catalog record for the configured dataset."""
        record = self.catalog.get_dataset(self.source, self.dataset_id)
        return record or {}

    def get_annotations(
        self,
        filters: Optional[Mapping[str, Any]] = None,
    ) -> List[Dict[str, Any]]:
        """Return all annotations unless explicit filters are supplied."""
        effective_filters = {
            **self.default_filters,
            **dict(filters or {}),
        }
        return self.catalog.get_annotations(
            source=self.source,
            dataset_id=self.dataset_id,
            filters=effective_filters,
        )
