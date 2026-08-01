"""METASPACE adapter using the optional official Python client."""

from __future__ import annotations

from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Set, Tuple

import numpy as np
import requests

from ....utils.exceptions import (
    raise_download_limit_error,
    raise_external_service_error,
    raise_project_config_error,
)
from ....utils.logger import get_custom_logger
from .metaspace_authentication import metaspace_client_options
from ..base import DatasetSource
from ..source_manager import DatasetSourceManager


logger = get_custom_logger(__name__)


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
    ) -> None:
        self._client_options = dict(client_options or {})
        self._client = client
        self._accepted_records: List[Dict[str, Any]] = []
        self._rejected_records: List[Dict[str, Any]] = []
        self._cache_dir = Path(cache_dir) if cache_dir is not None else None
        self._refresh_cache = refresh_cache
        self._config = {
            "client_options": {
                key: "***" if key in {"api_key", "password"} else value
                for key, value in self._client_options.items()
            },
            "cache_dir": str(self._cache_dir) if self._cache_dir else None,
            "refresh_cache": refresh_cache,
        }
        self._available_values_cache = self._load_available_datasets()

    @staticmethod
    def get_available_filters() -> Dict[str, Any]:
        """Return notebook-friendly documentation for METASPACE filters."""
        return {
            "name": {"type": "string", "api_field": "nameMask"},
            "dataset_ids": {"type": "string | list[string]", "api_field": "idMask"},
            "submitter_id": {"type": "string", "api_field": "submitter"},
            "group_id": {"type": "string", "api_field": "group"},
            "project_id": {"type": "string", "api_field": "project"},
            "molecule": {
                "type": "string",
                "api_field": "hasAnnotationMatching.compoundQuery",
            },
            "polarity": {
                "type": "Positive | Negative",
                "api_field": "polarity",
                "choices": ["Positive", "Negative"],
            },
            "organism": {"type": "string", "api_field": "organism"},
            "organism_part": {"type": "string", "api_field": "organismPart"},
            "condition": {"type": "string", "api_field": "condition"},
            "growth_conditions": {"type": "string", "api_field": "growthConditions"},
            "analyzer_type": {"type": "string", "api_field": "analyzerType"},
            "ionisation_source": {"type": "string", "api_field": "ionisationSource"},
            "maldi_matrix": {"type": "string", "api_field": "maldiMatrix"},
            "has_optical_image": {
                "type": "boolean",
                "local": True,
                "choices": [True, False],
                "description": "Applied to the optical-image field returned by GraphQL.",
            },
            "status": {"type": "string", "default": "FINISHED"},
            "annotation_fdr": {"type": "float", "default": 0.1},
            "min_annotation_count": {"type": "integer | null", "local": True},
            "max_annotation_count": {"type": "integer | null", "local": True},
            "include_molecule_stats": {"type": "boolean", "default": False},
            "include_spatial_annotation_stats": {
                "type": "boolean",
                "default": False,
                "local": True,
            },
            "min_molecule_count": {"type": "integer | null", "local": True},
            "min_unique_molecule_count": {"type": "integer | null", "local": True},
            "exclude_dataset_ids": {
                "type": "list[string]",
                "description": "Local exclusions applied after provider discovery.",
            },
        }

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
        native_filters, local_filters = _split_filters(filters)
        datasets = self.client.datasets(**native_filters)
        records = [self._dataset_record(dataset) for dataset in datasets]
        fdr = float(local_filters.get("annotation_fdr", 0.1))
        _attach_annotation_counts(self.client, records, fdr)
        if bool(local_filters.get("include_spatial_annotation_stats")):
            _attach_spatial_annotation_statistics(self.client, records, fdr)
        if bool(local_filters.get("include_molecule_stats")) or any(
            local_filters.get(key) is not None
            for key in ("min_molecule_count", "min_unique_molecule_count")
        ):
            _attach_molecule_statistics(self.client, records, fdr)
        self._accepted_records, self._rejected_records = _apply_local_filters(
            records, local_filters
        )
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
                        logger.info(
                            "Loaded %s METASPACE catalogue records from %s",
                            len(records),
                            path,
                        )
                        return [dict(record) for record in records]
            except (OSError, TypeError, ValueError, json.JSONDecodeError) as error:
                logger.warning("Ignoring invalid METASPACE catalogue %s: %s", path, error)

        datasets = self.client.datasets(status="FINISHED")
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
        return self._dataset_record(self.client.dataset(id=dataset_id))

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
        """Download a dataset's source imzML/ibd pair into a local directory."""
        target = Path(destination)
        target.mkdir(parents=True, exist_ok=True)
        expected = target / f"{dataset_id}.imzML"
        expected_ibd = expected.with_suffix(".ibd")
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
            try:
                with ThreadPoolExecutor(max_workers=min(2, len(files))) as executor:
                    list(
                        executor.map(
                            lambda file_record: _download_file(
                                file_record, target, dataset_id
                            ),
                            files,
                        )
                    )
            except (requests.RequestException, ValueError) as error:
                raise_external_service_error(
                    context_name="METASPACE",
                    message=f"Download failed for dataset '{dataset_id}': {error}",
                )
        else:
            dataset.download_to_dir(target, base_name=dataset_id)
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
    return [
        {"value": value, "label": label, "count": count}
        for (value, label), count in sorted(
            counts.items(),
            key=lambda item: (-item[1], item[0][1].casefold()),
        )
    ]


_METASPACE_FILTER_ALIASES = {
    "name": "nameMask",
    "dataset_ids": "idMask",
    "organism_part": "organismPart",
    "growth_conditions": "growthConditions",
}
_LOCAL_FILTERS = {
    "annotation_fdr",
    "include_molecule_stats",
    "include_spatial_annotation_stats",
    "min_annotation_count",
    "max_annotation_count",
    "min_molecule_count",
    "min_unique_molecule_count",
    "has_optical_image",
}


def _split_filters(filters: Mapping[str, Any]) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """Separate METASPACE GraphQL filters from local aggregate constraints."""
    options = dict(filters)
    local = {
        key: options.pop(key)
        for key in list(options)
        if key in _LOCAL_FILTERS
    }
    fdr = float(local.get("annotation_fdr", 0.1))
    molecule = options.pop("molecule", None)
    native = {
        _METASPACE_FILTER_ALIASES.get(key, key): value
        for key, value in options.items()
        if value is not None
    }
    native.setdefault("status", "FINISHED")
    if molecule:
        native["hasAnnotationMatching"] = {
            "compoundQuery": str(molecule),
            "fdrLevel": fdr,
        }
    local.setdefault("annotation_fdr", fdr)
    return native, local


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
            annotation_images = dataset.all_annotation_images(
                database=(database.name, database.version),
                fdr=fdr,
                only_first_isotope=True,
                scale_intensity=False,
            )
            database_has_images = False
            for images_for_annotation in annotation_images:
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


def _download_file(
    file_record: Mapping[str, Any],
    destination: Path,
    dataset_id: str,
) -> Path:
    """Download one signed METASPACE file safely and atomically.

    :param file_record: METASPACE file record containing ``filename`` and
        signed ``link`` fields.
    :type file_record: Mapping[str, Any]
    :param destination: Dataset output directory.
    :type destination: pathlib.Path
    :param dataset_id: Stable dataset ID used as the local file stem.
    :type dataset_id: str
    :return: Completed local path.
    :rtype: pathlib.Path
    :raises ValueError: If the record is malformed or has an unsupported file
        extension.
    :raises requests.RequestException: If the HTTP request fails.
    """
    filename = str(file_record.get("filename", ""))
    link = file_record.get("link")
    suffix = Path(filename).suffix
    if suffix.lower() not in {".imzml", ".ibd"}:
        raise ValueError(f"Unsupported METASPACE download file: '{filename}'.")
    if not isinstance(link, str) or not link:
        raise ValueError(f"METASPACE returned no link for '{filename}'.")
    canonical_suffix = ".imzML" if suffix.lower() == ".imzml" else ".ibd"
    output = destination / f"{dataset_id}{canonical_suffix}"
    partial = output.with_name(f"{output.name}.part")
    if output.is_file() and output.stat().st_size > 0:
        logger.info("METASPACE file already exists: %s", output)
        return output

    logger.info("Downloading METASPACE file %s", filename)
    partial.unlink(missing_ok=True)
    try:
        with requests.get(link, stream=True, timeout=(15, 120)) as response:
            response.raise_for_status()
            with partial.open("wb") as stream:
                for chunk in response.iter_content(chunk_size=1024 * 1024):
                    if chunk:
                        stream.write(chunk)
        if partial.stat().st_size == 0:
            raise ValueError(f"METASPACE returned an empty file for '{filename}'.")
        partial.replace(output)
    except Exception:
        partial.unlink(missing_ok=True)
        raise
    logger.info("Downloaded METASPACE file to %s", output)
    return output


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
