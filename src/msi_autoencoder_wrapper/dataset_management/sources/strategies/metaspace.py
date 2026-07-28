"""METASPACE adapter using the optional official Python client."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional

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


@DatasetSourceManager.register_source("metaspace")
class MetaspaceDatasetSource(DatasetSource):
    """Discover and download METASPACE datasets through ``SMInstance``.

    :param client: Optional initialized ``SMInstance``. Primarily useful for
        dependency injection and offline tests.
    :type client: Any | None
    :param client_options: Options forwarded to ``SMInstance`` when a client is
        not supplied, for example ``api_key`` or ``config_path``.
    :type client_options: Mapping[str, Any] | None

    The external package is imported lazily, so the base library remains usable
    without METASPACE credentials or its optional client dependency.
    """

    source_name = "metaspace"

    def __init__(
        self,
        client: Optional[Any] = None,
        client_options: Optional[Mapping[str, Any]] = None,
    ) -> None:
        self._client_options = dict(client_options or {})
        self._client = client
        self._config = {
            "client_options": {
                key: "***" if key in {"api_key", "password"} else value
                for key, value in self._client_options.items()
            }
        }

    @staticmethod
    def available_filters() -> Dict[str, Any]:
        """Return notebook-friendly documentation for METASPACE filters."""
        return {
            "nameMask": {"type": "string", "description": "Dataset name search."},
            "idMask": {"type": "string | list[string]"},
            "submitter_id": {"type": "string"},
            "group_id": {"type": "string"},
            "project_id": {"type": "string"},
            "polarity": {"type": "Positive | Negative"},
            "ionisation_source": {"type": "string"},
            "analyzer_type": {"type": "string"},
            "maldi_matrix": {"type": "string"},
            "organism": {"type": "string"},
            "exclude_dataset_ids": {
                "type": "list[string]",
                "description": "Local exclusions applied after provider discovery.",
            },
        }

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

    def search_datasets(self, filters: Mapping[str, Any]) -> List[Dict[str, Any]]:
        """Return METASPACE datasets matching native dataset filters."""
        datasets = self.client.datasets(**dict(filters))
        records = [self._dataset_record(dataset) for dataset in datasets]
        logger.info("METASPACE discovery returned %s datasets", len(records))
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

        ``options`` may contain ``databases`` and ``fdr``. When databases are
        omitted, every database listed on the dataset is queried. The default
        FDR of ``0.5`` requests the broadest standard METASPACE result level;
        lower experimental thresholds are applied later by annotation readers.
        """
        options = dict(options or {})
        dataset = self.client.dataset(id=dataset_id)
        databases = options.get("databases") or [
            (database.name, database.version)
            for database in getattr(dataset, "database_details", [])
        ]
        fdr = float(options.get("fdr", 0.5))
        records: List[Dict[str, Any]] = []
        for database in databases:
            results = dataset.results(database=database, fdr=fdr)
            database_name, database_version = _database_parts(database)
            database_records = _records_from_table(results)
            spatial_images = _spatial_images_by_molecule(
                dataset,
                database,
                fdr,
                enabled=bool(options.get("include_spatial", True)),
            )
            for row in database_records:
                formula = row.get("formula", row.get("sumFormula"))
                key = (str(formula or ""), str(row.get("adduct") or ""))
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
        expected = target / f"{dataset_id}.imzML"
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
                "polarity": getattr(dataset, "polarity", None),
                "status": getattr(dataset, "status", None),
                "image_size": _object_mapping(getattr(dataset, "image_size", {})),
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
