"""Annotation reader resolving merged indices back to source datasets."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional

from ...workspace.dataset_catalog import DatasetCatalog
from ..annotations_manager import AnnotationReaderManager
from ..base_annotation_reader import MSIBaseAnnotationReader


@AnnotationReaderManager.register_reader("MergedCatalogAnnotationReader")
class MergedCatalogAnnotationReader(MSIBaseAnnotationReader):
    """Read annotations for an imzML assembled from multiple source datasets.

    :param catalog_path: Workspace SQLite catalog.
    :type catalog_path: pathlib.Path | str
    :param merged_dataset_id: Identifier registered by :class:`ImzMLMerger`.
    :type merged_dataset_id: str
    :param default_filters: Optional read-time annotation filters.
    :type default_filters: Mapping[str, Any] | None
    :param active_context: Optional active context reference.
    :type active_context: Any | None
    """

    def __init__(
        self,
        catalog_path: Path | str,
        merged_dataset_id: str,
        default_filters: Optional[Mapping[str, Any]] = None,
        active_context: Optional[Any] = None,
    ) -> None:
        super().__init__(active_context=active_context)
        self.catalog = DatasetCatalog(catalog_path)
        self.merged_dataset_id = merged_dataset_id
        self.default_filters = dict(default_filters or {})
        self._config = {
            "catalog_path": str(catalog_path),
            "merged_dataset_id": merged_dataset_id,
            "default_filters": self.default_filters,
        }

    def get_dataset_metadata(self) -> Dict[str, Any]:
        """Return metadata records for every contributing source dataset."""
        sources = self.catalog.list_merged_sources(self.merged_dataset_id)
        return {
            "merged_dataset_id": self.merged_dataset_id,
            "sources": [
                self.catalog.get_dataset(item["source"], item["source_dataset_id"])
                for item in sources
            ],
        }

    def get_annotations(
        self,
        filters: Optional[Mapping[str, Any]] = None,
    ) -> List[Dict[str, Any]]:
        """Return annotations for all contributing datasets without deduplication."""
        effective_filters = {**self.default_filters, **dict(filters or {})}
        annotations: List[Dict[str, Any]] = []
        for source in self.catalog.list_merged_sources(self.merged_dataset_id):
            source_annotations = self.catalog.get_annotations(
                source=source["source"],
                dataset_id=source["source_dataset_id"],
                filters=effective_filters,
            )
            annotations.extend(
                {
                    **annotation,
                    "source": source["source"],
                    "source_dataset_id": source["source_dataset_id"],
                }
                for annotation in source_annotations
            )
        return annotations

    def get_spectrum_annotations(
        self,
        spatial_id: int,
        filters: Optional[Mapping[str, Any]] = None,
    ) -> List[Dict[str, Any]]:
        """Resolve a merged spectrum index and return its source annotations.

        Spatial filtering is applied when imported records expose a
        ``spatial_ids`` collection. Dataset-level records without such a field
        remain available because their precise target semantics are intentionally
        deferred to the training dataset strategy.
        """
        source = self.catalog.get_source_index(
            merged_dataset_id=self.merged_dataset_id,
            merged_spectrum_index=spatial_id,
        )
        if source is None:
            return []
        effective_filters = {**self.default_filters, **dict(filters or {})}
        annotations = self.catalog.get_annotations(
            source=source["source"],
            dataset_id=source["source_dataset_id"],
            filters=effective_filters,
        )
        source_spatial_id = int(source["source_spatial_id"])
        return [
            {
                **annotation,
                "source": source["source"],
                "source_dataset_id": source["source_dataset_id"],
                "source_spatial_id": source_spatial_id,
            }
            for annotation in annotations
            if annotation.get("spatial_ids") is None
            or source_spatial_id in annotation["spatial_ids"]
        ]
