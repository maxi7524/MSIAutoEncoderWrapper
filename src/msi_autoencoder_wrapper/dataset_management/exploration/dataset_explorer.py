"""Interactive dataset discovery without materializing MSI data."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional

import pandas as pd

from ...utils.logger import get_custom_logger
from ..sources.base import DatasetSource
from ..sources.source_manager import DatasetSourceManager
from ..validators import validate_source_record


logger = get_custom_logger(__name__)


class DatasetExplorer:
    """Explore source filters and export reproducible query configuration.

    :param source: Registered source key or initialized source adapter.
    :type source: str | DatasetSource
    :param source_options: Options used when constructing a registered source.
    :type source_options: Any

    The explorer performs metadata discovery only. Its exported JSON is passed
    directly to the existing ``query --filters`` command; the resulting
    selection is then consumed by ``download``.
    """

    def __init__(self, source: str | DatasetSource, **source_options: Any) -> None:
        if isinstance(source, str):
            DatasetSourceManager.discover_strategies()
            self.source = DatasetSourceManager.get_source(source, **source_options)
        else:
            self.source = source
        self._filters: Dict[str, Any] = {}
        self._records: List[Dict[str, Any]] = []
        self._excluded_ids: set[str] = set()

    @property
    def filters(self) -> Dict[str, Any]:
        """Return a copy of the current query filters."""
        return dict(self._filters)

    def available_filters(self) -> Dict[str, Any]:
        """Return provider filter documentation when exposed by the adapter."""
        provider_method = getattr(self.source, "available_filters", None)
        if callable(provider_method):
            return dict(provider_method())
        return {
            "native_filters": {
                "type": "mapping",
                "description": (
                    f"Native filters accepted by the '{self.source.source_name}' source."
                ),
            }
        }

    def set_filters(
        self,
        filters: Optional[Mapping[str, Any]] = None,
        **updates: Any,
    ) -> Dict[str, Any]:
        """Replace current filters and apply optional keyword updates.

        :param filters: Complete initial filter mapping.
        :type filters: Mapping[str, Any] | None
        :param updates: Individual values overriding the supplied mapping.
        :type updates: Any
        :return: Resolved filter configuration.
        :rtype: Dict[str, Any]
        """
        self._filters = dict(filters or {})
        self._filters.update(updates)
        return self.filters

    def search(
        self,
        filters: Optional[Mapping[str, Any]] = None,
    ) -> pd.DataFrame:
        """Run metadata discovery and return accepted datasets as a table.

        :param filters: Optional replacement filters for this search.
        :type filters: Mapping[str, Any] | None
        :return: Notebook-friendly accepted dataset summary.
        :rtype: pandas.DataFrame
        """
        if filters is not None:
            self.set_filters(filters)
        discovered = self.source.search_datasets(self._filters)
        self._records = []
        for record in discovered:
            validate_source_record(record)
            normalized = dict(record)
            normalized.setdefault("source", self.source.source_name)
            self._records.append(normalized)
        logger.info(
            "Explorer retained %s records from source %s",
            len(self._records),
            self.source.source_name,
        )
        return self.results()

    def results(self, *, include_excluded: bool = False) -> pd.DataFrame:
        """Return accepted records, optionally including manually excluded IDs."""
        rows = [
            _summary_row(record, str(record["dataset_id"]) in self._excluded_ids)
            for record in self._records
            if include_excluded or str(record["dataset_id"]) not in self._excluded_ids
        ]
        return pd.DataFrame(rows, columns=_RESULT_COLUMNS)

    def rejected(self) -> pd.DataFrame:
        """Return provider diagnostics for records rejected during discovery."""
        provider_method = getattr(self.source, "get_search_diagnostics", None)
        diagnostics = list(provider_method()) if callable(provider_method) else []
        return pd.DataFrame(diagnostics)

    def exclude(self, dataset_ids: str | Iterable[str]) -> pd.DataFrame:
        """Exclude one or more reviewed dataset IDs from exported filters.

        :param dataset_ids: One ID or an iterable of IDs visible in results.
        :type dataset_ids: str | Iterable[str]
        :return: Remaining accepted records.
        :rtype: pandas.DataFrame
        :raises ValueError: If an ID is not present in the current results.
        """
        selected = [dataset_ids] if isinstance(dataset_ids, str) else list(dataset_ids)
        known = {str(record["dataset_id"]) for record in self._records}
        unknown = sorted(set(selected) - known)
        if unknown:
            raise ValueError(f"Cannot exclude unknown dataset IDs: {unknown}")
        self._excluded_ids.update(str(value) for value in selected)
        return self.results()

    def include(self, dataset_ids: str | Iterable[str]) -> pd.DataFrame:
        """Remove one or more IDs from the manual exclusion set."""
        selected = [dataset_ids] if isinstance(dataset_ids, str) else list(dataset_ids)
        self._excluded_ids.difference_update(str(value) for value in selected)
        return self.results()

    def export_config(self, path: Path | str) -> Path:
        """Write filters and manual exclusions for the existing query command.

        :param path: Destination JSON path.
        :type path: pathlib.Path | str
        :return: Written path.
        :rtype: pathlib.Path
        """
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        config = dict(self._filters)
        configured_exclusions = {
            str(value) for value in config.get("exclude_dataset_ids", ())
        }
        config["exclude_dataset_ids"] = sorted(
            configured_exclusions | self._excluded_ids
        )
        target.write_text(
            json.dumps(config, ensure_ascii=False, indent=2, default=str) + "\n",
            encoding="utf-8",
        )
        logger.info("Exported dataset query configuration to %s", target)
        return target


_RESULT_COLUMNS = [
    "dataset_id",
    "name",
    "source",
    "project_accession",
    "project_url",
    "organisms",
    "organism_parts",
    "diseases",
    "instruments",
    "total_size_bytes",
    "download_size_bytes",
    "annotation_status",
    "excluded",
]


def _summary_row(record: Mapping[str, Any], excluded: bool) -> Dict[str, Any]:
    metadata = dict(record.get("metadata", {}))
    project = dict(metadata.get("project", {}))
    return {
        "dataset_id": str(record["dataset_id"]),
        "name": str(record.get("name", record["dataset_id"])),
        "source": str(record.get("source", "")),
        "project_accession": metadata.get("project_accession"),
        "project_url": metadata.get("project_url"),
        "organisms": _names(project.get("organisms", metadata.get("organisms"))),
        "organism_parts": _names(
            project.get("organismParts", metadata.get("organism_parts"))
        ),
        "diseases": _names(project.get("diseases", metadata.get("diseases"))),
        "instruments": _names(
            project.get("instruments", metadata.get("instruments"))
        ),
        "total_size_bytes": metadata.get("total_size_bytes"),
        "download_size_bytes": metadata.get("download_size_bytes"),
        "annotation_status": metadata.get("annotation_status"),
        "excluded": excluded,
    }


def _names(value: Any) -> str:
    if value is None:
        return ""
    values = value if isinstance(value, list) else [value]
    return ", ".join(
        str(item.get("name", item.get("value", "")))
        if isinstance(item, Mapping)
        else str(item)
        for item in values
    )
