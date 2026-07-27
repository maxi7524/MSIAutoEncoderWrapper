"""Canonical annotation reader backed exclusively by the workspace SQLite store."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional

from ..utils.exceptions import raise_validation_error
from ..workspace.dataset_catalog import DatasetCatalog
from .annotations_manager import AnnotationReaderManager
from .base_annotation_reader import MSIBaseAnnotationReader
from .validators import validate_annotation_store


@AnnotationReaderManager.register_reader("SQLiteAnnotationReader")
class SQLiteAnnotationReader(MSIBaseAnnotationReader):
    """Read normalized dataset and spectrum annotations from one SQLite schema.

    :param catalog_path: Path to the canonical workspace SQLite database.
    :type catalog_path: pathlib.Path | str
    :param source: Source key for an unmerged dataset.
    :type source: str | None
    :param dataset_id: Source dataset identifier for an unmerged dataset.
    :type dataset_id: str | None
    :param merged_dataset_id: Identifier of a locally merged dataset.
    :type merged_dataset_id: str | None
    :param default_filters: Optional read-time molecular filters.
    :type default_filters: Mapping[str, Any] | None
    :param active_context: Optional active context reference.
    :type active_context: Any | None
    :raises ValidationError: If neither one source dataset nor one merged
        dataset is selected.

    External database formats are normalized before this reader is used. This
    class therefore contains no METASPACE-specific parsing or provider strategy.
    """

    def __init__(
        self,
        catalog_path: Path | str,
        source: Optional[str] = None,
        dataset_id: Optional[str] = None,
        merged_dataset_id: Optional[str] = None,
        default_filters: Optional[Mapping[str, Any]] = None,
        active_context: Optional[Any] = None,
    ) -> None:
        super().__init__(active_context=active_context)
        source_selected = source is not None or dataset_id is not None
        if source_selected != (source is not None and dataset_id is not None):
            raise_validation_error(
                "SQLiteAnnotationReader",
                "source and dataset_id must be provided together.",
            )
        if (source is not None) == (merged_dataset_id is not None):
            raise_validation_error(
                "SQLiteAnnotationReader",
                "Select exactly one source dataset or one merged dataset.",
            )
        self.catalog = DatasetCatalog(catalog_path)
        validate_annotation_store(self.catalog.path)
        self.source = source
        self.dataset_id = dataset_id
        self.merged_dataset_id = merged_dataset_id
        self.default_filters = dict(default_filters or {})
        self._config = {
            "catalog_path": str(catalog_path),
            "source": source,
            "dataset_id": dataset_id,
            "merged_dataset_id": merged_dataset_id,
            "default_filters": self.default_filters,
        }

    def get_dataset_metadata(self) -> Dict[str, Any]:
        """Return complete source metadata without assigning model classes."""
        if self.merged_dataset_id is None:
            return self.catalog.get_dataset(str(self.source), str(self.dataset_id)) or {}
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
        """Return all normalized molecular annotations by default."""
        effective_filters = {**self.default_filters, **dict(filters or {})}
        if self.merged_dataset_id is None:
            return self.catalog.get_annotations(
                source=str(self.source),
                dataset_id=str(self.dataset_id),
                filters=effective_filters,
            )
        results: List[Dict[str, Any]] = []
        for item in self.catalog.list_merged_sources(self.merged_dataset_id):
            annotations = self.catalog.get_annotations(
                source=item["source"],
                dataset_id=item["source_dataset_id"],
                filters=effective_filters,
            )
            results.extend(
                {
                    **annotation,
                    "source": item["source"],
                    "source_dataset_id": item["source_dataset_id"],
                }
                for annotation in annotations
            )
        return results

    def get_spectrum_annotations(
        self,
        spectrum_id: int,
        filters: Optional[Mapping[str, Any]] = None,
    ) -> List[Dict[str, Any]]:
        """Return every molecular annotation linked to ``spectrum_id``."""
        effective_filters = {**self.default_filters, **dict(filters or {})}
        if self.merged_dataset_id is None:
            return self.catalog.get_spectrum_annotations(
                source=str(self.source),
                dataset_id=str(self.dataset_id),
                spectrum_id=spectrum_id,
                filters=effective_filters,
            )
        source_record = self.catalog.get_source_index(
            merged_dataset_id=self.merged_dataset_id,
            merged_spectrum_index=spectrum_id,
        )
        if source_record is None:
            return []
        source_spectrum_id = int(source_record["source_spectrum_id"])
        annotations = self.catalog.get_spectrum_annotations(
            source=source_record["source"],
            dataset_id=source_record["source_dataset_id"],
            spectrum_id=source_spectrum_id,
            filters=effective_filters,
        )
        return [
            {
                **annotation,
                "source": source_record["source"],
                "source_dataset_id": source_record["source_dataset_id"],
                "source_spectrum_id": source_spectrum_id,
            }
            for annotation in annotations
        ]

    def get_spectrum_metadata(self, spectrum_id: int) -> Dict[str, Any]:
        """Return metadata of the source dataset owning ``spectrum_id``."""
        if self.merged_dataset_id is None:
            return self.get_dataset_metadata()
        source_record = self.catalog.get_source_index(
            merged_dataset_id=self.merged_dataset_id,
            merged_spectrum_index=spectrum_id,
        )
        if source_record is None:
            return {}
        return self.catalog.get_dataset(
            source_record["source"],
            source_record["source_dataset_id"],
        ) or {}
