"""METASPACE adapter using the optional official Python client."""

from __future__ import annotations

from collections import Counter, defaultdict
from contextlib import contextmanager
from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Set, Tuple

import numpy as np
import requests
from tqdm.auto import tqdm

from ....utils.exceptions import (
    raise_download_limit_error,
    raise_external_service_error,
    raise_project_config_error,
)
from ....utils.logger import get_custom_logger
from .metaspace_authentication import metaspace_client_options
from .metaspace_metadata import attach_api_metadata, summarise_records
from .metaspace_parameters import (
    FREE_TEXT_FILTER_FIELDS,
    FREE_TEXT_VALUE_GROUPS,
    canonical_free_text,
    filter_schema,
    split_filters,
)
from ..base import DatasetSource
from ..source_manager import DatasetSourceManager


logger = get_custom_logger(__name__)


# Catalogue cache
_CACHE_SCHEMA_VERSION = 1
_AVAILABLE_DATASETS_CACHE_FILE = "available-datasets.json"


@DatasetSourceManager.register_source("metaspace")
class MetaspaceDatasetSource(DatasetSource):
    """Discover and download METASPACE datasets through ``SMInstance``.

    :param client: Optional initialized ``SMInstance``. Primarily useful for
        dependency injection and offline tests.
    :type client: Any | None
    :param client_options: Options forwarded to ``SMInstance`` when a client is
        not supplied, for example ``api_key`` or ``config_path``.
    :type client_options: Mapping[str, Any] | None
    :param cache_dir: Optional directory, resolved relative to the current
        working directory, for the reusable METASPACE catalogue file. ``None``
        keeps the catalogue only in memory and does not write to disk.
    :type cache_dir: pathlib.Path | str | None
    :param refresh_cache: Ignore an existing catalogue file, retrieve current
        metadata from METASPACE, and replace the file when ``cache_dir`` is set.
    :type refresh_cache: bool
    :param load_catalog: Load discovery metadata during construction. Disable
        this for download-only workflows whose dataset IDs are already frozen.
    :type load_catalog: bool

    The external package is imported lazily, so the base library remains usable
    without METASPACE credentials or its optional client dependency.
    """

    source_name = "metaspace"

    def __init__(
        self,
        client: Optional[Any] = None,
        client_options: Optional[Mapping[str, Any]] = None,
        cache_dir: Optional[Path | str] = None,
        refresh_cache: bool = False,
        load_catalog: bool = True,
    ) -> None:
        self._client_options = dict(client_options or {})
        self._client = client
        self._accepted_records: List[Dict[str, Any]] = []
        self._rejected_records: List[Dict[str, Any]] = []
        self._cache_dir = Path(cache_dir) if cache_dir is not None else None
        self._refresh_cache = refresh_cache
        self.catalog_retrieved_at: Optional[str] = None
        self._config = {
            "client_options": {
                key: "***" if key in {"api_key", "password"} else value
                for key, value in self._client_options.items()
            },
            "cache_dir": str(self._cache_dir) if self._cache_dir else None,
            "refresh_cache": refresh_cache,
            "load_catalog": load_catalog,
        }
        self._available_values_cache = (
            self._load_available_datasets() if load_catalog else []
        )

    @staticmethod
    def get_available_filters() -> Dict[str, Any]:
        """Return notebook-friendly documentation for METASPACE filters."""
        return filter_schema()

    def get_available_values(self, filter_key: str) -> List[Dict[str, Any]]:
        """Return values currently present in accessible METASPACE datasets.

        :param filter_key: Enumerable key returned by
            :meth:`get_available_filters`.
        :type filter_key: str
        :return: Choice records sorted by frequency and label.
        :rtype: List[Dict[str, Any]]
        :raises ValueError: If the filter is unknown or represents free text or
            a quantitative constraint rather than a finite value collection.

        The first call retrieves metadata for accessible finished datasets and
        caches it on this source instance. No annotations, ion images, imzML,
        or ibd files are downloaded.
        """
        schema = self.get_available_filters()
        if filter_key not in schema:
            raise ValueError(f"Unknown METASPACE filter '{filter_key}'.")
        if filter_key == "has_optical_image":
            return super().get_available_values(filter_key)
        enumerable = {
            "dataset_ids",
            "submitter_id",
            "group_id",
            "project_id",
            "polarity",
            "organism",
            "organism_part",
            "condition",
            "growth_conditions",
            "analyzer_type",
            "ionisation_source",
            "maldi_matrix",
            "status",
        }
        if filter_key not in enumerable:
            raise ValueError(
                f"METASPACE filter '{filter_key}' is free text or quantitative "
                "and does not expose a finite value list."
            )
        return _available_values(self._available_values_cache, filter_key)

    def get_accepted_records(self) -> List[Dict[str, Any]]:
        """Return records accepted by the most recent filtering call."""
        return [dict(record) for record in self._accepted_records]

    def get_rejected_records(self) -> List[Dict[str, Any]]:
        """Return records rejected by local quantitative filters."""
        return [dict(record) for record in self._rejected_records]

    @property
    def client(self) -> Any:
        """Return the injected client or lazily construct ``SMInstance``."""
        if self._client is None:
            try:
                from metaspace import SMInstance
            except ImportError:
                try:
                    from metaspace.sm_annotation_utils import SMInstance
                except ImportError:
                    raise_project_config_error(
                        context_name="MetaspaceSource",
                        message=(
                            "The optional METASPACE Python client is not installed. "
                            "Install the official client or inject an SMInstance."
                        ),
                    )
            self._client = SMInstance(**metaspace_client_options(self._client_options))
        return self._client

    def filter(self, filters: Mapping[str, Any]) -> List[Dict[str, Any]]:
        """Filter METASPACE datasets and attach review-oriented statistics."""
        native_filters, local_filters = split_filters(filters)
        native_filters, cache_rejected = _restrict_to_cached_free_text_matches(
            native_filters,
            local_filters,
            self._available_values_cache,
        )
        molecule_statistics_enabled = bool(
            local_filters.get("include_molecule_stats")
        ) or any(
            local_filters.get(key) is not None
            for key in ("min_molecule_count", "min_unique_molecule_count")
        )
        overall = tqdm(
            total=3,
            desc="METASPACE discovery",
            unit="stage",
            position=0,
            dynamic_ncols=True,
        )
        current = tqdm(
            total=1,
            desc="Current operation",
            unit="operation",
            position=1,
            leave=False,
            dynamic_ncols=True,
        )
        try:
            with _current_operation(overall, current, "Query dataset catalogue"):
                datasets = (
                    []
                    if native_filters.get("idMask") == []
                    else self.client.datasets(**native_filters)
                )
            records = [self._dataset_record(dataset) for dataset in datasets]
            spatial_statistics_enabled = bool(
                local_filters.get("include_spatial_annotation_stats")
            )
            with _current_operation(overall, current, "Retrieve metadata and m/z ranges"):
                attach_api_metadata(self.client, records)
            records, early_rejected = _apply_early_filters(records, local_filters)
            records, free_text_rejected = _apply_free_text_filters(records, local_filters)
            fdr = float(local_filters.get("annotation_fdr", 0.1))
            with _current_operation(overall, current, "Retrieve annotation counts"):
                _attach_annotation_counts(self.client, records, fdr)
            if spatial_statistics_enabled:
                overall.total += len(records)
                overall.refresh()
                image_count = sum(
                    int(record["metadata"].get("annotation_count") or 0)
                    for record in records
                )
                overall.set_postfix(datasets=len(records), ion_images=image_count)
                logger.info(
                    "METASPACE spatial plan contains %s datasets and %s annotation images",
                    len(records),
                    image_count,
                )
                for record in records:
                    dataset_id = str(record["dataset_id"])
                    dataset_image_count = int(
                        record["metadata"].get("annotation_count") or 0
                    )
                    with _current_operation(
                        overall,
                        current,
                        f"Spatial annotations: {dataset_id}",
                        total=max(dataset_image_count, 1),
                        unit="image",
                    ):
                        _attach_spatial_annotation_statistics(
                            self.client, [record], fdr, progress=current
                        )
            if molecule_statistics_enabled:
                overall.total += 1
                overall.refresh()
                batch_count = max(1, (len(records) + 99) // 100)
                with _current_operation(
                    overall,
                    current,
                    "Retrieve molecular identities",
                    total=batch_count,
                    unit="batch",
                ):
                    _attach_molecule_statistics(
                        self.client, records, fdr, progress=current
                    )
            accepted, quantitative_rejected = _apply_local_filters(records, local_filters)
        finally:
            current.close()
            overall.close()
        self._accepted_records = accepted
        rejected_by_id = {
            str(record["dataset_id"]): record
            for record in (
                cache_rejected + early_rejected + free_text_rejected + quantitative_rejected
            )
        }
        self._rejected_records = list(rejected_by_id.values())
        logger.info(
            "METASPACE discovery accepted %s datasets and rejected %s datasets",
            len(self._accepted_records),
            len(self._rejected_records),
        )
        return self.get_accepted_records()

    def _load_available_datasets(self) -> List[Dict[str, Any]]:
        """Load the METASPACE catalogue into memory, optionally using disk."""
        path = (
            self._cache_dir / _AVAILABLE_DATASETS_CACHE_FILE
            if self._cache_dir is not None
            else None
        )
        if path is not None and path.is_file() and not self._refresh_cache:
            try:
                payload = json.loads(path.read_text(encoding="utf-8"))
                if payload.get("schema_version") == _CACHE_SCHEMA_VERSION:
                    records = payload.get("records")
                    if isinstance(records, list):
                        self.catalog_retrieved_at = datetime.fromtimestamp(
                            path.stat().st_mtime,
                            timezone.utc,
                        ).isoformat()
                        logger.info(
                            "Loaded %s METASPACE catalogue records from %s",
                            len(records),
                            path,
                        )
                        return [dict(record) for record in records]
            except (OSError, TypeError, ValueError, json.JSONDecodeError) as error:
                logger.warning("Ignoring invalid METASPACE catalogue %s: %s", path, error)

        datasets = self.client.datasets(status="FINISHED")
        self.catalog_retrieved_at = datetime.now(timezone.utc).isoformat()
        records = [self._dataset_record(dataset) for dataset in datasets]
        logger.info("Loaded %s METASPACE catalogue records from API", len(records))
        if path is not None:
            path.parent.mkdir(parents=True, exist_ok=True)
            temporary = path.with_suffix(".json.tmp")
            temporary.write_text(
                json.dumps(
                    {
                        "schema_version": _CACHE_SCHEMA_VERSION,
                        "records": records,
                    },
                    ensure_ascii=False,
                    indent=2,
                    default=str,
                )
                + "\n",
                encoding="utf-8",
            )
            temporary.replace(path)
            logger.info("Saved METASPACE catalogue to %s", path)
        return records

    def get_dataset_metadata(self, dataset_id: str) -> Dict[str, Any]:
        """Return source metadata and stable summary fields for one dataset."""
        record = self._dataset_record(self.client.dataset(id=dataset_id))
        attach_api_metadata(self.client, [record])
        return record

    @staticmethod
    def summarise(records: List[Mapping[str, Any]]) -> Dict[str, Any]:
        """Return aggregate storage, m/z intersection and analysis methods."""
        return summarise_records(records)

    def get_annotations(
        self,
        dataset_id: str,
        options: Optional[Mapping[str, Any]] = None,
    ) -> List[Dict[str, Any]]:
        """Retrieve all annotations exposed by requested database/FDR options.

        ``options`` may contain ``databases``, ``annotation_fdr``, and
        ``include_spatial``. When databases are omitted, every database listed
        on the dataset is queried. ``annotation_fdr`` is the same threshold
        used by discovery statistics and spatial annotation retrieval.
        """
        options = dict(options or {})
        dataset = self.client.dataset(id=dataset_id)
        databases = options.get("databases") or [
            (database.name, database.version)
            for database in getattr(dataset, "database_details", [])
        ]
        annotation_fdr = float(options.get("annotation_fdr", 0.1))
        include_spatial = bool(options.get("include_spatial", True))
        records: List[Dict[str, Any]] = []
        for database in databases:
            results = dataset.results(database=database, fdr=annotation_fdr)
            database_name, database_version = _database_parts(database)
            database_records = _records_from_table(results)
            spatial_images = _spatial_images_by_molecule(
                dataset,
                database,
                annotation_fdr,
                enabled=include_spatial,
            )
            missing_spatial_annotations: List[Tuple[str, str]] = []
            for row in database_records:
                formula = row.get("formula", row.get("sumFormula"))
                key = (str(formula or ""), str(row.get("adduct") or ""))
                if include_spatial and key not in spatial_images:
                    missing_spatial_annotations.append(key)
                records.append(
                    {
                        **row,
                        "database_name": database_name,
                        "database_version": database_version,
                        **(
                            {"ion_image": spatial_images[key].tolist()}
                            if key in spatial_images
                            else {}
                        ),
                    }
                )
            if missing_spatial_annotations:
                preview = ", ".join(
                    f"{formula}{adduct}"
                    for formula, adduct in missing_spatial_annotations[:5]
                )
                raise_external_service_error(
                    context_name="METASPACE",
                    message=(
                        f"Dataset '{dataset_id}' returned "
                        f"{len(missing_spatial_annotations)} molecular annotations "
                        f"without matching ion images for database "
                        f"'{database_name} {database_version}' at annotation_fdr="
                        f"{annotation_fdr}. Missing examples: {preview}. Spatial "
                        "pixel annotations cannot be constructed completely."
                    ),
                )
        logger.info(
            "Retrieved %s METASPACE annotations for dataset %s",
            len(records),
            dataset_id,
        )
        return records

    def download_dataset(self, dataset_id: str, destination: Path | str) -> Path:
        """Download an imzML/ibd pair with the official METASPACE client."""
        target = Path(destination)
        target.mkdir(parents=True, exist_ok=True)
        expected = target / f"{dataset_id}.imzML"
        expected_ibd = expected.with_suffix(".ibd")

        # Dataset download
        ## Local cache
        if (
            expected.is_file()
            and expected.stat().st_size > 0
            and expected_ibd.is_file()
            and expected_ibd.stat().st_size > 0
        ):
            logger.info(
                "METASPACE dataset %s already contains a complete local pair at %s",
                dataset_id,
                target,
            )
            return target

        ## Remote response validation
        dataset = self.client.dataset(id=dataset_id)
        if hasattr(dataset, "download_links"):
            download = dataset.download_links()
            files = list((download or {}).get("files", []))
            if not files:
                message = (download or {}).get(
                    "message", "METASPACE returned no downloadable files."
                )
                raise_external_service_error(
                    context_name="METASPACE",
                    message=f"Cannot download dataset '{dataset_id}': {message}",
                )
            _validate_download_files(files, dataset_id)

        ## Official client transfer
        ### Delegate signed-link handling, parallel transfer, and file naming to
        ### SMDataset instead of maintaining a second downloader in this adapter.
        try:
            dataset.download_to_dir(target, base_name=dataset_id)
        except (OSError, ValueError, requests.RequestException) as error:
            raise_external_service_error(
                context_name="METASPACE",
                message=f"Download failed for dataset '{dataset_id}': {error}",
            )

        ## Output verification
        if not expected.is_file() or not expected.with_suffix(".ibd").is_file():
            raise_project_config_error(
                context_name="MetaspaceSource",
                message=(
                    f"METASPACE download for '{dataset_id}' did not create the "
                    "expected imzML/ibd pair."
                ),
            )
        return target

    @staticmethod
    def _dataset_record(dataset: Any) -> Dict[str, Any]:
        metadata = _object_mapping(getattr(dataset, "metadata", {}))
        provider = _object_mapping(getattr(dataset, "_info", {}))
        sample_information = _object_mapping(metadata.get("Sample_Information"))
        database_details = [
            {
                "name": getattr(database, "name", None),
                "version": getattr(database, "version", None),
                "id": getattr(database, "id", None),
            }
            for database in getattr(dataset, "database_details", [])
        ]
        return {
            "dataset_id": str(getattr(dataset, "id")),
            "name": str(getattr(dataset, "name", getattr(dataset, "id"))),
            "metadata": {
                **metadata,
                "project_url": f"https://metaspace2020.eu/dataset/{dataset.id}",
                "provider_metadata": provider,
                "submitter": provider.get("submitter"),
                "group": provider.get("group"),
                "projects": provider.get("projects", []),
                "upload_datetime": provider.get("uploadDT"),
                "polarity": getattr(dataset, "polarity", None),
                "status": getattr(dataset, "status", None),
                "image_size": _object_mapping(getattr(dataset, "image_size", {})),
                "pixel_count": _pixel_count(provider.get("acquisitionGeometry")),
                "condition": provider.get("condition")
                or metadata.get("condition")
                or sample_information.get("Condition"),
                "growth_conditions": provider.get("growthConditions")
                or sample_information.get("Sample_Growth_Conditions"),
                "organism": provider.get("organism")
                or metadata.get("organism")
                or sample_information.get("Organism"),
                "organism_part": provider.get("organismPart")
                or sample_information.get("Organism_Part"),
                "ionisation_source": provider.get("ionisationSource"),
                "analyzer": provider.get("analyzer"),
                "maldi_matrix": provider.get("maldiMatrix"),
                "has_optical_image": provider.get("opticalImage"),
                "databases": database_details,
            },
        }


def _object_mapping(value: Any) -> Dict[str, Any]:
    if isinstance(value, Mapping):
        return dict(value)
    if hasattr(value, "_asdict"):
        return dict(value._asdict())
    if hasattr(value, "__dict__"):
        return {
            key: item
            for key, item in vars(value).items()
            if not key.startswith("_")
        }
    return {"value": value} if value is not None else {}


def _restrict_to_cached_free_text_matches(
    native_filters: Mapping[str, Any],
    local_filters: Mapping[str, Any],
    cached_records: List[Dict[str, Any]],
) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
    """Use the cached catalogue to avoid retrieving known local mismatches."""
    if not cached_records or not any(
        local_filters.get(key) is not None for key in _FREE_TEXT_FILTER_FIELDS
    ):
        return dict(native_filters), []
    scoped_records = _cached_native_scope(cached_records, native_filters)
    matches, rejected = _apply_free_text_filters(scoped_records, local_filters)
    matched_ids = {str(record["dataset_id"]) for record in matches}
    requested_ids = native_filters.get("idMask")
    if requested_ids is not None:
        values = (
            requested_ids.split("|")
            if isinstance(requested_ids, str)
            else requested_ids
        )
        matched_ids.intersection_update(str(value) for value in values)
    restricted = dict(native_filters)
    restricted["idMask"] = sorted(matched_ids)
    logger.info(
        "METASPACE cache reduced discovery to %s candidate datasets",
        len(matched_ids),
    )
    return restricted, rejected


def _cached_native_scope(
    records: List[Dict[str, Any]],
    native_filters: Mapping[str, Any],
) -> List[Dict[str, Any]]:
    """Apply exact cached fields before producing local rejection diagnostics."""
    requested_ids = native_filters.get("idMask")
    id_values = None
    if requested_ids is not None:
        values = (
            requested_ids.split("|")
            if isinstance(requested_ids, str)
            else requested_ids
        )
        id_values = {str(value) for value in values}
    field_filters = {
        "polarity": native_filters.get("polarity"),
        "status": native_filters.get("status"),
        "ionisation_source": native_filters.get("ionisation_source"),
        "maldi_matrix": native_filters.get("maldi_matrix"),
    }
    scoped: List[Dict[str, Any]] = []
    for record in records:
        if id_values is not None and str(record["dataset_id"]) not in id_values:
            continue
        metadata = record["metadata"]
        if any(
            expected is not None
            and str(metadata.get(field, "")).casefold()
            != str(expected).casefold()
            for field, expected in field_filters.items()
        ):
            continue
        analyzer_type = native_filters.get("analyzer_type")
        if analyzer_type is not None and str(
            _object_mapping(metadata.get("analyzer")).get("type", "")
        ).casefold() != str(analyzer_type).casefold():
            continue
        scoped.append(record)
    return scoped


@contextmanager
def _current_operation(
    overall: Any,
    current: Any,
    description: str,
    *,
    total: int = 1,
    unit: str = "operation",
) -> Iterable[None]:
    """Reset one reusable current-operation bar below overall progress."""
    current.reset(total=total)
    current.set_description(description)
    current.unit = unit
    current.refresh()
    try:
        yield
    finally:
        if current.n < current.total:
            current.update(current.total - current.n)
        overall.update(1)


@contextmanager
def _hide_client_progress() -> Iterable[None]:
    """Suppress the official client's nested bar beneath wrapper progress."""
    try:
        from metaspace import sm_annotation_utils
    except ImportError:
        yield
        return
    original = sm_annotation_utils.tqdm
    sm_annotation_utils.tqdm = lambda iterable, *args, **kwargs: iterable
    try:
        yield
    finally:
        sm_annotation_utils.tqdm = original


def _available_values(
    records: List[Dict[str, Any]],
    filter_key: str,
) -> List[Dict[str, Any]]:
    """Aggregate one METASPACE field into notebook-friendly choices."""
    values: List[Tuple[Any, str]] = []
    for record in records:
        metadata = record["metadata"]
        if filter_key == "dataset_ids":
            values.append((record["dataset_id"], record["name"]))
        elif filter_key in {"submitter_id", "group_id"}:
            field = "submitter" if filter_key == "submitter_id" else "group"
            identity = _object_mapping(metadata.get(field))
            if identity.get("id"):
                values.append(
                    (identity["id"], str(identity.get("name", identity["id"])))
                )
        elif filter_key == "project_id":
            for project in metadata.get("projects", []):
                identity = _object_mapping(project)
                if identity.get("id"):
                    values.append(
                        (
                            identity["id"],
                            str(identity.get("name", identity["id"])),
                        )
                    )
        else:
            field = {
                "analyzer_type": "analyzer",
            }.get(filter_key, filter_key)
            value = metadata.get(field)
            if filter_key == "analyzer_type":
                value = _object_mapping(value).get("type")
            if value not in {None, ""}:
                values.append((value, str(value)))
    counts = Counter(values)
    if filter_key in _FREE_TEXT_FILTER_FIELDS:
        return _group_free_text_values(counts)
    return [
        {"value": value, "label": label, "count": count}
        for (value, label), count in sorted(
            counts.items(),
            key=lambda item: (-item[1], item[0][1].casefold()),
        )
    ]


def _group_free_text_values(
    counts: "Counter[Tuple[Any, str]]",
) -> List[Dict[str, Any]]:
    """Collapse spelling/formatting variants of a free-text field into one row.

    :param counts: Occurrence count per raw ``(value, label)`` pair, as
        collected in :func:`_available_values`.
    :type counts: Counter[Tuple[Any, str]]
    :return: One row per group with a summed ``count`` and a ``variants``
        string listing every raw spelling folded into it.
    :rtype: List[Dict[str, Any]]

    Grouping is keyed by :func:`_canonical_free_text` -- the same declared
    alias table used by ``filter()`` (:func:`_free_text_matches`) -- so
    this view always agrees with what a filter query will actually match.
    Only spelling variants listed in ``FREE_TEXT_VALUE_GROUPS`` are merged;
    anything not reviewed there stays on its own row.
    """
    groups: Dict[str, List[Tuple[Any, str, int]]] = defaultdict(list)
    for (value, label), count in counts.items():
        groups[_canonical_free_text(label)].append((value, label, count))

    rows: List[Dict[str, Any]] = []
    for canonical, members in groups.items():
        members.sort(key=lambda member: -member[2])
        canonical_value, canonical_label, _ = members[0]
        if canonical in FREE_TEXT_VALUE_GROUPS:
            canonical_value = canonical_label = canonical
        rows.append(
            {
                "value": canonical_value,
                "label": canonical_label,
                "count": sum(count for _, _, count in members),
                "variants": ", ".join(
                    f"{label} ({count})" for _, label, count in members
                ),
            }
        )
    rows.sort(key=lambda row: (-row["count"], row["label"].casefold()))
    return rows


_FREE_TEXT_FILTER_FIELDS = FREE_TEXT_FILTER_FIELDS


def _attach_annotation_counts(
    client: Any,
    records: List[Dict[str, Any]],
    fdr: float,
) -> None:
    """Attach aggregate annotation counts with one paginated GraphQL query."""
    graph = getattr(client, "_gqclient", None)
    if graph is None or not hasattr(graph, "listQuery") or not records:
        return
    query = """
        query datasetStatistics(
            $filter: DatasetFilter, $fdrLevels: [Int!]!,
            $offset: Int, $limit: Int
        ) {
          allDatasets(filter: $filter, offset: $offset, limit: $limit) {
            id
            opticalImage
            annotationCounts(inpFdrLvls: $fdrLevels) {
              databaseId dbName dbVersion counts { level n } isTargeted total
            }
          }
        }
    """
    summaries: List[Dict[str, Any]] = []
    for batch in _batches(records, 100):
        dataset_ids = "|".join(str(record["dataset_id"]) for record in batch)
        summaries.extend(
            graph.listQuery(
                "allDatasets",
                query,
                {"filter": {"ids": dataset_ids}, "fdrLevels": [round(fdr * 100)]},
            )
        )
    by_id = {str(summary["id"]): summary for summary in summaries}
    for record in records:
        metadata = record["metadata"]
        summary = by_id.get(str(record["dataset_id"]), {})
        database_counts = list(summary.get("annotationCounts") or [])
        for count in database_counts:
            values = list(count.get("counts") or [])
            count["count_at_fdr"] = int(values[0].get("n", 0)) if values else 0
        metadata["annotation_fdr"] = fdr
        metadata["annotation_counts_by_database"] = database_counts
        metadata["annotation_count"] = sum(
            int(count["count_at_fdr"]) for count in database_counts
        )
        metadata["has_optical_image"] = bool(summary.get("opticalImage"))


def _attach_molecule_statistics(
    client: Any,
    records: List[Dict[str, Any]],
    fdr: float,
    progress: Optional[Any] = None,
) -> None:
    """Attach per-dataset molecule counts without downloading ion images."""
    graph = getattr(client, "_gqclient", None)
    if graph is None or not hasattr(graph, "getAnnotations") or not records:
        return
    annotations: List[Dict[str, Any]] = []
    for batch in _batches(records, 100):
        dataset_ids = "|".join(str(record["dataset_id"]) for record in batch)
        annotations.extend(
            graph.getAnnotations(
                annotationFilter={"fdrLevel": fdr},
                datasetFilter={"ids": dataset_ids},
            )
        )
        if progress is not None:
            progress.update(1)
    identities: Dict[str, Set[Tuple[str, str]]] = defaultdict(set)
    for annotation in annotations:
        dataset = _object_mapping(annotation.get("dataset"))
        dataset_id = str(dataset.get("id", ""))
        formula = str(annotation.get("sumFormula", ""))
        adduct = str(annotation.get("adduct", ""))
        if dataset_id and formula:
            identities[dataset_id].add((formula, adduct))
    occurrence = Counter(
        identity for values in identities.values() for identity in values
    )
    for record in records:
        values = identities[str(record["dataset_id"])]
        unique = sorted(identity for identity in values if occurrence[identity] == 1)
        record["metadata"].update(
            {
                "molecule_count": len(values),
                "unique_molecule_count": len(unique),
                "unique_molecules": [f"{formula}{adduct}" for formula, adduct in unique],
            }
        )


def _attach_spatial_annotation_statistics(
    client: Any,
    records: List[Dict[str, Any]],
    fdr: float,
    progress: Optional[Any] = None,
) -> None:
    """Attach unique annotated-pixel counts from METASPACE ion images.

    :param client: Initialized official METASPACE client.
    :type client: Any
    :param records: Dataset records to enrich in place.
    :type records: List[Dict[str, Any]]
    :param fdr: Maximum FDR used to select source annotations.
    :type fdr: float
    :raises ValueError: If ion images within a dataset have inconsistent
        shapes or their union exceeds the reported acquired-pixel count.

    Intensity magnitudes are not interpreted. A spatial position is counted
    once when at least one annotation selected by ``fdr`` has non-zero signal.
    """
    for record in records:
        dataset_id = str(record["dataset_id"])
        dataset = client.dataset(id=dataset_id)
        annotated_mask: Optional[np.ndarray] = None
        spatial_annotation_count = 0
        database_count = 0

        for database in getattr(dataset, "database_details", []):
            with _hide_client_progress():
                annotation_images = dataset.all_annotation_images(
                    database=(database.name, database.version),
                    fdr=fdr,
                    only_first_isotope=True,
                    scale_intensity=False,
                )
            database_has_images = False
            for images_for_annotation in annotation_images:
                if progress is not None:
                    progress.update(1)
                if (
                    len(images_for_annotation) == 0
                    or images_for_annotation[0] is None
                ):
                    continue
                image = np.asarray(images_for_annotation[0])
                signal_mask = np.isfinite(image) & (image != 0)
                if annotated_mask is None:
                    annotated_mask = signal_mask.copy()
                elif annotated_mask.shape != signal_mask.shape:
                    raise ValueError(
                        "METASPACE ion-image shape mismatch for dataset "
                        f"'{dataset_id}': {annotated_mask.shape} != {signal_mask.shape}."
                    )
                else:
                    annotated_mask |= signal_mask
                spatial_annotation_count += 1
                database_has_images = True
            database_count += int(database_has_images)

        annotated_pixel_count = (
            int(np.count_nonzero(annotated_mask))
            if annotated_mask is not None
            else 0
        )
        metadata = record["metadata"]
        acquired_pixel_count = metadata.get("pixel_count")
        metadata.update(
            {
                "annotated_pixel_count": annotated_pixel_count,
                "annotation_fdr": fdr,
                "spatial_annotation_count": spatial_annotation_count,
                "spatial_annotation_database_count": database_count,
            }
        )
        if acquired_pixel_count is None:
            metadata.update(
                {
                    "unannotated_pixel_count": None,
                    "annotated_pixel_fraction": None,
                    "spatial_stats_status": "missing_acquired_pixel_count",
                }
            )
            continue
        acquired_pixel_count = int(acquired_pixel_count)
        if annotated_pixel_count > acquired_pixel_count:
            raise ValueError(
                "METASPACE annotated-pixel count exceeds acquisition geometry "
                f"for dataset '{dataset_id}': {annotated_pixel_count} > "
                f"{acquired_pixel_count}."
            )
        metadata.update(
            {
                "unannotated_pixel_count": (
                    acquired_pixel_count - annotated_pixel_count
                ),
                "annotated_pixel_fraction": (
                    annotated_pixel_count / acquired_pixel_count
                    if acquired_pixel_count > 0
                    else None
                ),
                "spatial_stats_status": "complete",
            }
        )


def _canonical_free_text(value: Any) -> str:
    """Return the canonical group key for one free-text metadata value.

    The explicit groups live in :mod:`metaspace_parameters`. Unreviewed values
    remain separate instead of being merged heuristically.
    """
    return canonical_free_text(value)


def _free_text_matches(field_value: Any, query_value: str) -> bool:
    """Check one submitter free-text field against one query term.

    Both sides are reduced to their canonical group via
    :func:`_canonical_free_text` before comparison, so a filter query
    always agrees with the grouping shown by ``get_available_values`` --
    both are driven by the same declared alias table.
    """
    canonical_query = _canonical_free_text(query_value)
    if not canonical_query:
        return True
    return _canonical_free_text(field_value) == canonical_query


def _free_text_filter_matches(field_value: Any, query: Any) -> bool:
    """Check a free-text field against a query string or an iterable of OR'd queries."""
    queries = [query] if isinstance(query, str) else list(query)
    return any(_free_text_matches(field_value, str(term)) for term in queries)


def _apply_free_text_filters(
    records: List[Dict[str, Any]],
    filters: Mapping[str, Any],
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """Apply tolerant local matching for unstructured METASPACE metadata fields.

    :param records: Dataset records fetched from METASPACE, before annotation
        or molecule statistics are attached.
    :type records: List[Dict[str, Any]]
    :param filters: Local filter mapping; only keys in
        :data:`_FREE_TEXT_FILTER_FIELDS` are consulted.
    :type filters: Mapping[str, Any]
    :return: Accepted records and rejection diagnostics, in the same shape as
        :func:`_apply_local_filters`.
    :rtype: Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]

    Run before the annotation/molecule/spatial enrichment steps so that
    expensive per-dataset METASPACE queries are not spent on datasets that
    are rejected here anyway.
    """
    accepted: List[Dict[str, Any]] = []
    rejected: List[Dict[str, Any]] = []
    for record in records:
        metadata = record["metadata"]
        reason = None
        for filter_key, field in _FREE_TEXT_FILTER_FIELDS.items():
            query = filters.get(filter_key)
            if query is None:
                continue
            if not _free_text_filter_matches(metadata.get(field), query):
                reason = (
                    f"{field} does not match {query!r}; observed "
                    f"{metadata.get(field)!r}"
                )
                break
        if reason is None:
            accepted.append(record)
        else:
            rejected.append(
                {
                    "dataset_id": record["dataset_id"],
                    "name": record["name"],
                    "reason": reason,
                    "project_url": metadata.get("project_url"),
                }
            )
    if rejected:
        logger.debug(
            "METASPACE free-text filters rejected %s datasets before enrichment",
            len(rejected),
        )
    return accepted, rejected


def _apply_early_filters(
    records: List[Dict[str, Any]],
    filters: Mapping[str, Any],
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """Apply exclusions and m/z coverage before annotation API operations."""
    excluded_ids = {str(value) for value in filters.get("exclude_dataset_ids", ())}
    mz_range = filters.get("mz_range")
    if mz_range is not None:
        if not isinstance(mz_range, Mapping):
            raise ValueError("mz_range must be an object with min and max values")
        lower = float(mz_range["min"])
        upper = float(mz_range["max"])
        if lower < 0 or lower >= upper:
            raise ValueError("m/z range requires 0 <= min < max")
        if mz_range.get("mode", "covers") != "covers":
            raise ValueError("mz_range mode must be 'covers'")
    accepted: List[Dict[str, Any]] = []
    rejected: List[Dict[str, Any]] = []
    for record in records:
        dataset_id = str(record["dataset_id"])
        reason = None
        if dataset_id in excluded_ids:
            reason = "dataset_id is listed in exclude_dataset_ids"
        elif mz_range is not None:
            metadata = record["metadata"]
            dataset_min = metadata.get("mz_min")
            dataset_max = metadata.get("mz_max")
            if (
                dataset_min is None
                or dataset_max is None
                or float(dataset_min) > lower
                or float(dataset_max) < upper
            ):
                reason = (
                    f"m/z range [{dataset_min}, {dataset_max}] does not cover "
                    f"[{lower}, {upper}]"
                )
        if reason is None:
            accepted.append(record)
        else:
            rejected.append(
                {"dataset_id": dataset_id, "name": record["name"], "reason": reason}
            )
    return accepted, rejected


def _apply_local_filters(
    records: List[Dict[str, Any]],
    filters: Mapping[str, Any],
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """Apply quantitative constraints and retain explicit rejection reasons."""
    constraints = (
        ("min_annotation_count", "annotation_count", "at least", lambda a, b: a >= b),
        ("max_annotation_count", "annotation_count", "at most", lambda a, b: a <= b),
        ("min_molecule_count", "molecule_count", "at least", lambda a, b: a >= b),
        (
            "min_unique_molecule_count",
            "unique_molecule_count",
            "at least",
            lambda a, b: a >= b,
        ),
    )
    accepted: List[Dict[str, Any]] = []
    rejected: List[Dict[str, Any]] = []
    for record in records:
        reason = None
        metadata = record["metadata"]
        expected_optical = filters.get("has_optical_image")
        if (
            expected_optical is not None
            and metadata.get("has_optical_image") != bool(expected_optical)
        ):
            reason = (
                "has_optical_image must be "
                f"{bool(expected_optical)}; observed {metadata.get('has_optical_image')}"
            )
        for option, field, comparison, predicate in constraints:
            if reason is not None:
                break
            limit = filters.get(option)
            if limit is None:
                continue
            value = metadata.get(field)
            if value is None or not predicate(int(value), int(limit)):
                reason = f"{field} must be {comparison} {limit}; observed {value}"
                break
        if reason is None:
            accepted.append(record)
        else:
            rejected.append(
                {
                    "dataset_id": record["dataset_id"],
                    "name": record["name"],
                    "reason": reason,
                    "project_url": metadata.get("project_url"),
                }
            )
    return accepted, rejected


def _batches(values: List[Dict[str, Any]], size: int) -> Iterable[List[Dict[str, Any]]]:
    """Yield bounded record batches for METASPACE ID filters."""
    for start in range(0, len(values), size):
        yield values[start : start + size]


def _pixel_count(acquisition_geometry: Any) -> Optional[int]:
    """Read the number of acquired pixels from METASPACE geometry JSON."""
    if not acquisition_geometry:
        return None
    try:
        import json

        geometry = (
            json.loads(acquisition_geometry)
            if isinstance(acquisition_geometry, str)
            else acquisition_geometry
        )
        return int(geometry.get("pixel_count"))
    except (TypeError, ValueError, AttributeError):
        return None


def _records_from_table(value: Any) -> List[Dict[str, Any]]:
    if hasattr(value, "to_dict"):
        try:
            table = value.reset_index() if hasattr(value, "reset_index") else value
            return [dict(record) for record in table.to_dict(orient="records")]
        except TypeError:
            pass
    if isinstance(value, list):
        return [
            dict(record) if isinstance(record, Mapping) else {"value": record}
            for record in value
        ]
    return []


def _database_parts(database: Any) -> tuple[Optional[str], Optional[str]]:
    if isinstance(database, (tuple, list)) and len(database) >= 2:
        return str(database[0]), str(database[1])
    return str(database), None


# Dataset download response validation
def _validate_download_files(
    files: List[Mapping[str, Any]],
    dataset_id: str,
) -> None:
    """Validate a complete METASPACE imzML download response before transfer.

    :param files: File records returned by ``SMDataset.download_links()``.
    :type files: List[Mapping[str, Any]]
    :param dataset_id: Dataset being materialized.
    :type dataset_id: str
    :raises DownloadLimitError: If METASPACE returns its quota sentinel file.
    :raises ExternalServiceError: If the response is not a complete imzML/ibd
        pair.
    """
    filenames = [str(file_record.get("filename", "")) for file_record in files]
    if any(filename.casefold() == "download_limit_reached.txt" for filename in filenames):
        raise_download_limit_error(
            "METASPACE",
            (
                f"METASPACE refused dataset '{dataset_id}' because the account or "
                "service download quota has been reached. Retry after the quota "
                "resets or contact METASPACE support. No data files were downloaded."
            ),
        )
    suffixes = {Path(filename).suffix.lower() for filename in filenames}
    unsupported = [
        filename
        for filename in filenames
        if Path(filename).suffix.lower() not in {".imzml", ".ibd"}
    ]
    if unsupported:
        raise_external_service_error(
            "METASPACE",
            (
                f"Dataset '{dataset_id}' returned unsupported download files: "
                f"{', '.join(unsupported)}."
            ),
        )
    if not {".imzml", ".ibd"}.issubset(suffixes):
        raise_external_service_error(
            "METASPACE",
            f"Dataset '{dataset_id}' did not return a complete imzML/ibd pair.",
        )


def _spatial_images_by_molecule(
    dataset: Any,
    database: Any,
    fdr: float,
    *,
    enabled: bool,
) -> Dict[tuple[str, str], np.ndarray]:
    """Return first-isotope ion images keyed by formula and adduct.

    :param dataset: Official METASPACE dataset object.
    :type dataset: Any
    :param database: METASPACE molecular database selector.
    :type database: Any
    :param fdr: Retrieval FDR threshold.
    :type fdr: float
    :param enabled: Whether spatial annotation retrieval is requested.
    :type enabled: bool
    :return: Ion images indexed by canonical molecule identity.
    :rtype: Dict[tuple[str, str], numpy.ndarray]

    Older or injected clients may expose tabular annotations without the ion
    image API. In that case dataset-level annotations are still preserved.
    """
    if not enabled or not hasattr(dataset, "all_annotation_images"):
        return {}
    images = dataset.all_annotation_images(
        database=database,
        fdr=fdr,
        only_first_isotope=True,
        scale_intensity=True,
    )
    result: Dict[tuple[str, str], np.ndarray] = {}
    for images_for_annotation in images:
        if len(images_for_annotation) == 0 or images_for_annotation[0] is None:
            continue
        key = (
            str(getattr(images_for_annotation, "formula", "") or ""),
            str(getattr(images_for_annotation, "adduct", "") or ""),
        )
        result[key] = np.asarray(images_for_annotation[0])
    return result
