"""METASPACE adapter using the optional official Python client."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional

import numpy as np

from ...utils.exceptions import raise_project_config_error
from ...utils.logger import get_custom_logger
from ..base_source import DatasetSource
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
        self._config = {"client_options": self._client_options}

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
            self._client = SMInstance(**self._client_options)
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
