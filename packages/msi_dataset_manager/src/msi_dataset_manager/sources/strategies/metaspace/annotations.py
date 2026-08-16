"""METASPACE annotation-schema adaptation and spatial export."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Tuple

import numpy as np

from ...base import (
    ANNOTATION_EXPORT_SCHEMA_VERSION,
    SourceAnnotationExport,
)
from ....imzml import PyImzMLReader
from ....utils.exceptions import raise_external_service_error
from ....utils.logger import get_custom_logger
from .csv import read_metaspace_annotation_export, write_annotation_csv_pair


logger = get_custom_logger(__name__)

# --------------------------------------------------
# Section: Wrapper, export function
# --------------------------------------------------

def materialize_metaspace_annotations(
    *,
    client: Any,
    dataset_id: str,
    dataset_name: str,
    directory: Path,
    imzml_path: Path,
    options: Optional[Mapping[str, Any]] = None,
) -> int:
    """Retrieve, normalize, and write one METASPACE annotation export.

    METASPACE exposes tabular result records but no formal result-schema
    version. ``provider_record_layout`` therefore records the recognized field
    layout; unsupported future layouts fail during canonical CSV validation
    rather than silently producing incomplete exports.
    """
    # Provider retrieval
    ## Database identity comes from the dataset object, then each result table
    ## is queried independently at the selected FDR threshold.
    from .source import _database_parts, _records_from_table, _spatial_images_by_molecule

    options = dict(options or {})
    dataset = client.dataset(id=dataset_id)
    databases = options.get("databases") or [
        (database.name, database.version)
        for database in getattr(dataset, "database_details", [])
    ]
    annotation_fdr = float(options.get("annotation_fdr", 0.1))
    include_spatial = bool(options.get("include_spatial", True))
    records: List[tuple[Dict[str, Any], np.ndarray | None]] = []
    for database in databases:
        results = dataset.results(database=database, fdr=annotation_fdr)
        database_name, database_version = _database_parts(database)
        database_id = _database_id(dataset, database_name, database_version)
        spatial_images = _spatial_images_by_molecule(
            dataset,
            database,
            annotation_fdr,
            enabled=include_spatial,
        )
        missing_spatial_annotations: List[Tuple[str, str]] = []
        for position, row in enumerate(_records_from_table(results)):
            formula = row.get("formula", row.get("sumFormula"))
            adduct = row.get("adduct")
            key = (str(formula or ""), str(adduct or ""))
            if include_spatial and key not in spatial_images:
                missing_spatial_annotations.append(key)
            records.append(
                (
                    _normalize_metaspace_record(
                    dataset_id=dataset_id,
                    row=row,
                    position=position,
                    database_name=database_name,
                    database_version=database_version,
                    database_id=database_id,
                    ),
                    spatial_images.get(key),
                )
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
    write_annotation_csv_pair(
        directory=directory,
        dataset_id=dataset_id,
        dataset_name=dataset_name,
        annotations=records,
        reader=PyImzMLReader(imzml_path),
    )
    # Obsolete artifact removal
    ## Previous filenames and schemas are intentionally unsupported. They are
    ## removed only after the current pair has been published successfully.
    for obsolete_path in (
        directory / "metaspace_annotations.csv",
        directory / f"{dataset_id}_pixel_intensities.csv",
    ):
        obsolete_path.unlink(missing_ok=True)
    logger.info("Exported %s METASPACE annotations for dataset %s", len(records), dataset_id)
    return len(records)


def read_metaspace_annotations(
    *,
    dataset_id: str,
    directory: Path,
    imzml_path: Path,
) -> SourceAnnotationExport:
    """Read one current-version local METASPACE annotation export."""
    return read_metaspace_annotation_export(
        dataset_id=dataset_id,
        directory=directory,
        imzml_path=imzml_path,
    )

# --------------------------------------------------
# Section: Wrapper, export function
# --------------------------------------------------

def _normalize_metaspace_record(
    *,
    dataset_id: str,
    row: Mapping[str, Any],
    position: int,
    database_name: Optional[str],
    database_version: Optional[str],
    database_id: Optional[str],
) -> Dict[str, Any]:
    """Normalize one recognized METASPACE result row without losing it."""
    source_record = dict(row)
    formula = source_record.get("formula", source_record.get("sumFormula"))
    return {
        **source_record,
        "schema_version": ANNOTATION_EXPORT_SCHEMA_VERSION,
        "source": "metaspace",
        "dataset_id": dataset_id,
        "source_annotation_id": str(
            source_record.get("id")
            or source_record.get("annotationId")
            or source_record.get("annotation_id")
            or position
        ),
        "provider_record_layout": _metaspace_record_layout(source_record),
        "database_name": database_name,
        "database_version": database_version,
        "database_id": database_id,
        "formula": formula,
        "adduct": source_record.get("adduct"),
        "mz": _first_value(source_record, "mz", "m/z", "mz_value"),
        "fdr": _first_value(source_record, "fdr", "FDR"),
        "molecule_names": _first_value(
            source_record,
            "molecule_names",
            "moleculeNames",
        ),
        "molecule_ids": _first_value(
            source_record,
            "molecule_ids",
            "moleculeIds",
        ),
        "source_record": source_record,
    }


def _metaspace_record_layout(record: Mapping[str, Any]) -> str:
    """Identify a supported METASPACE table layout from its field names."""
    if "formula" in record and "mz" in record:
        return "metaspace-results-formula-mz-v1"
    if "sumFormula" in record and "mz" in record:
        return "metaspace-results-sum-formula-mz-v1"
    if "formula" in record and "m/z" in record:
        return "metaspace-results-formula-mass-slash-v1"
    return "metaspace-results-unrecognized-v1"


def _database_id(dataset: Any, name: Optional[str], version: Optional[str]) -> Optional[str]:
    """Return the provider database identifier matching name and version."""
    for database in getattr(dataset, "database_details", []):
        if (
            str(getattr(database, "name", "")) == str(name or "")
            and str(getattr(database, "version", "")) == str(version or "")
        ):
            value = getattr(database, "id", None)
            return str(value) if value is not None else None
    return None


def _first_value(record: Mapping[str, Any], *names: str) -> Any:
    """Return the first available provider value."""
    for name in names:
        if record.get(name) is not None:
            return record[name]
    return None
