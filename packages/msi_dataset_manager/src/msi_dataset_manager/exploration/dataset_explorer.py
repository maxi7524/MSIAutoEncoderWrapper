"""Interactive dataset discovery without materializing MSI data."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional

import pandas as pd

from ..utils.logger import get_custom_logger
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
    # TODO - czy dało by się zrobić tak, ze wyświetlamy tutaj dostępen klucze które można wpisać ?? - to samo w innych miejscach

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

    def get_available_filters(self) -> Dict[str, Any]:
        """Return the selected source's complete filter schema."""
        return dict(self.source.get_available_filters())

    def available_filters(self) -> Dict[str, Any]:
        """Compatibility alias for :meth:`get_available_filters`."""
        return self.get_available_filters()

    def get_available_values(self, filter_key: str) -> pd.DataFrame:
        """Return selectable values for one source filter.

        :param filter_key: Key returned by :meth:`get_available_filters`.
        :type filter_key: str
        :return: Values, display labels, and dataset occurrence counts. For
            free-text fields with a declared synonym table (METASPACE's
            ``organism``, ``organism_part``, ``condition``,
            ``growth_conditions``), spelling variants are pre-grouped into one
            row and the ``variants`` column lists the raw spellings folded
            into it; other filters leave ``variants`` empty.
        :rtype: pandas.DataFrame
        :raises ValueError: If the key is unknown or is not enumerable.
        """
        values = self.source.get_available_values(filter_key)
        return pd.DataFrame(values, columns=["value", "label", "count", "variants"])

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
        *,
        summarise: bool = False,
    ) -> pd.DataFrame:
        """Run metadata discovery and return accepted datasets as a table.

        :param filters: Optional replacement filters for this search.
        :type filters: Mapping[str, Any] | None
        :return: Notebook-friendly accepted dataset summary.
        :rtype: pandas.DataFrame
        """
        if filters is not None:
            self.set_filters(filters)
        provider_filters = dict(self._filters)
        configured_exclusions = {
            str(value) for value in provider_filters.get("exclude_dataset_ids", ())
        }
        if self.source.source_name != "metaspace":
            provider_filters.pop("exclude_dataset_ids", None)
        self.source.filter(provider_filters)
        discovered = self.source.get_accepted_records()
        self._records = []
        for record in discovered:
            validate_source_record(record)
            normalized = dict(record)
            normalized.setdefault("source", self.source.source_name)
            if str(normalized["dataset_id"]) not in configured_exclusions:
                self._records.append(normalized)
        logger.info(
            "Explorer retained %s records from source %s",
            len(self._records),
            self.source.source_name,
        )
        return self.results(summarise=summarise)

    def filter(
        self,
        filters: Optional[Mapping[str, Any]] = None,
        *,
        summarise: bool = False,
    ) -> pd.DataFrame:
        """Compatibility-friendly filtering entry point for notebooks."""
        return self.search(filters, summarise=summarise)

    def accepted(
        self,
        *,
        include_excluded: bool = False,
        summarise: bool = False,
    ) -> pd.DataFrame:
        """Return accepted records from the most recent filtering call."""
        return self.results(
            include_excluded=include_excluded,
            summarise=summarise,
        )

    def results(
        self,
        *,
        include_excluded: bool = False,
        summarise: bool = False,
    ) -> pd.DataFrame:
        """Return records and optionally append a provider-aware summary row."""
        selected = [
            record
            for record in self._records
            if include_excluded or str(record["dataset_id"]) not in self._excluded_ids
        ]
        rows = [
            _summary_row(record, str(record["dataset_id"]) in self._excluded_ids)
            for record in selected
        ]
        if summarise and hasattr(self.source, "summarise"):
            rows.append(_aggregate_summary_row(self.source.summarise(selected)))
        return pd.DataFrame(rows, columns=_RESULT_COLUMNS)

    def rejected(self) -> pd.DataFrame:
        """Return provider diagnostics for records rejected during discovery."""
        return pd.DataFrame(self.source.get_rejected_records())

    def count_mz_range_coverage(
        self,
        lower_bounds: Iterable[float],
        upper_bounds: Iterable[float],
    ) -> pd.DataFrame:
        """Count discovered datasets covering every requested m/z interval.

        A dataset covers an interval when its minimum is no greater than the
        requested lower bound and its maximum is no less than the requested
        upper bound. Invalid or missing provider values never match.
        """
        results = self.results()
        minima = pd.to_numeric(results["mz_min"], errors="coerce")
        maxima = pd.to_numeric(results["mz_max"], errors="coerce")
        rows: List[Dict[str, Any]] = []
        for lower_bound in lower_bounds:
            for upper_bound in upper_bounds:
                lower = float(lower_bound)
                upper = float(upper_bound)
                if lower >= upper:
                    continue
                mask = minima.le(lower) & maxima.ge(upper)
                rows.append(
                    {
                        "lower_bound": lower,
                        "upper_bound": upper,
                        "range_width": upper - lower,
                        "dataset_count": int(mask.sum()),
                    }
                )
        return pd.DataFrame(
            rows,
            columns=["lower_bound", "upper_bound", "range_width", "dataset_count"],
        )

    def select_mz_range(self, min_mz: float, max_mz: float) -> pd.DataFrame:
        """Preview discovered datasets covering the complete requested range.

        This helper does not modify the active filters or manual exclusions.
        To apply the range to a query, pass ``mz_min`` and ``mz_max`` to
        :meth:`filter` instead.

        :param min_mz: Requested lower m/z bound.
        :type min_mz: float
        :param max_mz: Requested upper m/z bound.
        :type max_mz: float
        :return: Currently accepted datasets covering the complete range.
        :rtype: pandas.DataFrame
        :raises ValueError: If the range is invalid.
        """
        lower = float(min_mz)
        upper = float(max_mz)
        if lower < 0 or lower >= upper:
            raise ValueError("m/z range requires 0 <= min_mz < max_mz.")
        results = self.results()
        minima = pd.to_numeric(results["mz_min"], errors="coerce")
        maxima = pd.to_numeric(results["mz_max"], errors="coerce")
        matching = minima.le(lower) & maxima.ge(upper)
        return results.loc[matching].reset_index(drop=True)

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

    def export_selection(
        self,
        directory: Path | str,
        *,
        sort_by: str | None = None,
        ascending: bool = True,
    ) -> Dict[str, Path]:
        """Export query configuration and the frozen accepted dataset records."""
        target = Path(directory)
        target.mkdir(parents=True, exist_ok=True)
        filters_path = self.export_config(target / "filter.json")
        selected = [
            record for record in self._records
            if str(record["dataset_id"]) not in self._excluded_ids
        ]
        if sort_by is not None:
            populated = [
                record for record in selected
                if _metadata_value(record, sort_by) is not None
            ]
            missing = [
                record for record in selected
                if _metadata_value(record, sort_by) is None
            ]
            populated.sort(
                key=lambda record: _metadata_value(record, sort_by),
                reverse=not ascending,
            )
            selected = populated + missing
        exported_at = datetime.now(timezone.utc).isoformat()
        selection_path = target / "selection.json"
        selection_path.write_text(
            json.dumps(
                {
                    "schema_version": 2,
                    "source": self.source.source_name,
                    "exported_at": exported_at,
                    "catalog_retrieved_at": getattr(
                        self.source, "catalog_retrieved_at", None
                    ),
                    "filters_path": filters_path.name,
                    "filters": json.loads(filters_path.read_text(encoding="utf-8")),
                    "sort": (
                        {"field": sort_by, "direction": "ascending" if ascending else "descending"}
                        if sort_by is not None else None
                    ),
                    "dataset_ids": [str(record["dataset_id"]) for record in selected],
                    "datasets": selected,
                },
                ensure_ascii=False,
                indent=2,
                default=str,
            ) + "\n",
            encoding="utf-8",
        )
        return {"filters": filters_path, "selection": selection_path}


def _metadata_value(record: Mapping[str, Any], field: str) -> Any:
    """Return one top-level or nested metadata value for selection sorting."""
    return record.get(field, _mapping(record.get("metadata")).get(field))


_RESULT_COLUMNS = [
    "dataset_id",
    "name",
    "source",
    "project_accession",
    "project_url",
    "organisms",
    "organism_parts",
    "condition",
    "growth_conditions",
    "diseases",
    "instruments",
    "ionisation_source",
    "analyzer_type",
    "analyzer_resolving_power",
    "submitter",
    "group",
    "projects",
    "total_size_bytes",
    "download_size_bytes",
    "annotation_status",
    "polarity",
    "processing_status",
    "image_size",
    "pixel_count",
    "mz_tolerance_ppm",
    "analysis_method",
    "mz_min",
    "mz_max",
    "mz_intersection",
    "analysis_methods",
    "measurement_methods",
    "has_optical_image",
    "databases",
    "annotation_count",
    "annotated_pixel_count",
    "unannotated_pixel_count",
    "annotated_pixel_fraction",
    "annotation_fdr",
    "spatial_annotation_count",
    "spatial_annotation_database_count",
    "spatial_stats_status",
    "molecule_count",
    "unique_molecule_count",
    "unique_molecules",
    "excluded",
]


def _summary_row(record: Mapping[str, Any], excluded: bool) -> Dict[str, Any]:
    metadata = dict(record.get("metadata", {}))
    project = dict(metadata.get("project", {}))
    sample_information = _mapping(metadata.get("Sample_Information"))
    analyzer = _mapping(metadata.get("analyzer"))
    ms_analysis = _mapping(metadata.get("MS_Analysis"))
    return {
        "dataset_id": str(record["dataset_id"]),
        "name": str(record.get("name", record["dataset_id"])),
        "source": str(record.get("source", "")),
        "project_accession": metadata.get("project_accession"),
        "project_url": metadata.get("project_url"),
        "organisms": _names(
            project.get(
                "organisms",
                metadata.get(
                    "organisms",
                    metadata.get("organism", sample_information.get("Organism")),
                ),
            )
        ),
        "organism_parts": _names(
            project.get(
                "organismParts",
                metadata.get(
                    "organism_parts",
                    metadata.get(
                        "organism_part",
                        metadata.get(
                            "organismPart",
                            metadata.get(
                                "tissue", sample_information.get("Organism_Part")
                            ),
                        ),
                    ),
                ),
            )
        ),
        "condition": metadata.get(
            "condition", sample_information.get("Condition")
        ),
        "growth_conditions": metadata.get(
            "growth_conditions", sample_information.get("Sample_Growth_Conditions")
        ),
        "diseases": _names(project.get("diseases", metadata.get("diseases"))),
        "instruments": _names(
            project.get(
                "instruments",
                metadata.get(
                    "instruments",
                    analyzer.get(
                        "type",
                        ms_analysis.get("Analyzer"),
                    ),
                ),
            )
        ),
        "ionisation_source": metadata.get(
            "ionisation_source",
            ms_analysis.get("Ionisation_Source"),
        ),
        "analyzer_type": analyzer.get("type", ms_analysis.get("Analyzer")),
        "analyzer_resolving_power": analyzer.get(
            "resolvingPower",
            _mapping(ms_analysis.get("Detector_Resolving_Power")).get(
                "Resolving_Power"
            ),
        ),
        "submitter": _display_identity(metadata.get("submitter")),
        "group": _display_identity(metadata.get("group")),
        "projects": _names(metadata.get("projects")),
        "total_size_bytes": metadata.get("total_size_bytes"),
        "download_size_bytes": metadata.get("download_size_bytes"),
        "annotation_status": metadata.get("annotation_status"),
        "polarity": metadata.get("polarity"),
        "processing_status": metadata.get("status"),
        "image_size": _display_mapping(metadata.get("image_size")),
        "pixel_count": metadata.get("pixel_count"),
        "mz_tolerance_ppm": metadata.get("mz_tolerance_ppm"),
        "analysis_method": metadata.get("analysis_method"),
        "mz_min": metadata.get("mz_min"),
        "mz_max": metadata.get("mz_max"),
        "mz_intersection": None,
        "analysis_methods": None,
        "measurement_methods": None,
        "has_optical_image": metadata.get("has_optical_image"),
        "databases": _database_names(metadata.get("databases")),
        "annotation_count": metadata.get("annotation_count"),
        "annotated_pixel_count": metadata.get("annotated_pixel_count"),
        "unannotated_pixel_count": metadata.get("unannotated_pixel_count"),
        "annotated_pixel_fraction": metadata.get("annotated_pixel_fraction"),
        "annotation_fdr": metadata.get("annotation_fdr"),
        "spatial_annotation_count": metadata.get("spatial_annotation_count"),
        "spatial_annotation_database_count": metadata.get(
            "spatial_annotation_database_count"
        ),
        "spatial_stats_status": metadata.get("spatial_stats_status"),
        "molecule_count": metadata.get("molecule_count"),
        "unique_molecule_count": metadata.get("unique_molecule_count"),
        "unique_molecules": ", ".join(metadata.get("unique_molecules", [])),
        "excluded": excluded,
    }


def _aggregate_summary_row(summary: Mapping[str, Any]) -> Dict[str, Any]:
    """Build the final, visibly labelled aggregate table row."""
    minimum = summary.get("mz_intersection_min")
    maximum = summary.get("mz_intersection_max")
    size_label = "complete" if summary.get("size_complete") else "partial"
    row = {column: None for column in _RESULT_COLUMNS}
    row.update(
        {
            "dataset_id": "SUMMARY",
            "name": f"{summary.get('dataset_count', 0)} datasets ({size_label} size data)",
            "source": "aggregate",
            "total_size_bytes": summary.get("total_size_bytes"),
            "mz_intersection": (
                f"{minimum:g}–{maximum:g}"
                if minimum is not None and maximum is not None
                else None
            ),
            "analysis_methods": ", ".join(summary.get("analysis_methods", [])),
            "measurement_methods": ", ".join(
                summary.get("measurement_methods", [])
            ),
            "excluded": False,
        }
    )
    return row


def _mapping(value: Any) -> Mapping[str, Any]:
    return value if isinstance(value, Mapping) else {}


def _display_mapping(value: Any) -> str:
    if not isinstance(value, Mapping):
        return ""
    return ", ".join(f"{key}={item}" for key, item in value.items())


def _display_identity(value: Any) -> str:
    if not isinstance(value, Mapping):
        return ""
    return str(value.get("name", value.get("id", "")))


def _database_names(value: Any) -> str:
    if not isinstance(value, list):
        return ""
    return ", ".join(
        " ".join(
            str(part)
            for part in (item.get("name"), item.get("version"))
            if part
        )
        if isinstance(item, Mapping)
        else str(item)
        for item in value
    )


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
