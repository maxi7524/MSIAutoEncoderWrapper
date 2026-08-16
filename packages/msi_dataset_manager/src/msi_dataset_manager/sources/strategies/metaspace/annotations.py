"""METASPACE annotation-schema adaptation and spatial export."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional

import numpy as np
from tqdm.auto import tqdm

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
    ## One GraphQL annotation record contains both the table fields and the URL
    ## of the first-isotope spatial matrix. This is METASPACE's actual export
    ## model: the service does not expose a prebuilt pixel-intensity CSV.
    from .source import _database_parts

    options = dict(options or {})
    dataset = client.dataset(id=dataset_id)
    databases = options.get("databases") or [
        (database.name, database.version)
        for database in getattr(dataset, "database_details", [])
    ]
    annotation_fdr = float(options.get("annotation_fdr", 0.1))
    records: List[tuple[Dict[str, Any], np.ndarray]] = []
    progress = tqdm(
        total=0,
        desc=f"METASPACE records: {dataset_id}",
        unit="annotation",
        leave=False,
        dynamic_ncols=True,
    )
    try:
        for database in databases:
            database_name, database_version = _database_parts(database)
            database_id = _database_id(dataset, database_name, database_version)
            provider_records = _get_annotation_records(
                dataset=dataset,
                dataset_id=dataset_id,
                database_id=database_id,
                fdr=annotation_fdr,
            )
            normalized = [
                _normalize_metaspace_record(
                    dataset_id=dataset_id,
                    row=row,
                    database_name=database_name,
                    database_version=database_version,
                    database_id=database_id,
                )
                for row in provider_records
            ]
            progress.total += len(provider_records)
            progress.refresh()
            # Spatial-intensity retrieval
            ## A complete annotation export always includes actual pixel values.
            ## There is intentionally no metadata-only or synthetic-value fallback.
            spatial_images: List[Optional[np.ndarray]] = [None] * len(provider_records)
            with ThreadPoolExecutor() as executor:
                futures = {
                    executor.submit(_download_spatial_intensity_matrix, dataset, row): position
                    for position, row in enumerate(provider_records)
                }
                for future in as_completed(futures):
                    position = futures[future]
                    row = provider_records[position]
                    spatial_images[position] = future.result()
                    progress.set_postfix(
                        formula=str(row.get("sumFormula") or ""),
                        adduct=str(row.get("adduct") or ""),
                        mz=row.get("mz", ""),
                    )
                    progress.update(1)
            records.extend(zip(normalized, spatial_images))
    finally:
        progress.close()
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
    logger.info(
        "Exported %s METASPACE annotations for dataset %s",
        len(records),
        dataset_id,
    )
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


# METASPACE source-schema normalization
def _normalize_metaspace_record(
    *,
    dataset_id: str,
    row: Mapping[str, Any],
    database_name: Optional[str],
    database_version: Optional[str],
    database_id: Optional[str],
) -> Dict[str, Any]:
    """Normalize one recognized METASPACE result row without losing it."""
    source_record = dict(row)
    formula = source_record.get("sumFormula", source_record.get("formula"))
    ion = str(source_record.get("ion") or "")
    if not ion:
        raise_external_service_error(
            "METASPACE",
            "Annotation response does not contain the stable ion identifier required "
            "to join its pixel-intensity matrix.",
        )
    return {
        **source_record,
        "schema_version": ANNOTATION_EXPORT_SCHEMA_VERSION,
        "source": "metaspace",
        "dataset_id": dataset_id,
        "source_annotation_id": f"{database_id or 'unknown-database'}:{ion}",
        "provider_record_layout": _metaspace_record_layout(source_record),
        "database_name": database_name,
        "database_version": database_version,
        "database_id": database_id,
        "formula": formula,
        "adduct": source_record.get("adduct"),
        "mz": _first_value(source_record, "mz", "m/z", "mz_value"),
        "fdr": _first_value(source_record, "fdrLevel", "fdr", "FDR"),
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
    if {"sumFormula", "ion", "mz", "fdrLevel", "isotopeImages"}.issubset(record):
        return "metaspace-graphql-annotation-v1"
    return "metaspace-results-unrecognized-v1"


def _get_annotation_records(
    *,
    dataset: Any,
    dataset_id: str,
    database_id: Optional[str],
    fdr: float,
) -> List[Dict[str, Any]]:
    """Retrieve full METASPACE annotation records in one provider request."""
    graph_client = getattr(dataset, "_gqclient", None)
    if graph_client is None or not hasattr(graph_client, "getAnnotations"):
        raise_external_service_error(
            "METASPACE",
            "The installed METASPACE client does not expose full annotation records. "
            "Update the client before materializing annotations.",
        )
    annotation_filter: Dict[str, Any] = {
        "fdrLevel": fdr,
        "hasChemMod": False,
        "hasNeutralLoss": False,
    }
    if database_id is not None:
        try:
            annotation_filter["databaseId"] = int(database_id)
        except (TypeError, ValueError):
            raise_external_service_error(
                "METASPACE",
                f"Dataset '{dataset_id}' returned an invalid database identifier: "
                f"{database_id!r}.",
            )
    return [
        dict(row)
        for row in graph_client.getAnnotations(
            annotationFilter=annotation_filter,
            datasetFilter={"ids": dataset_id},
        )
    ]


def _download_spatial_intensity_matrix(
    dataset: Any,
    record: Mapping[str, Any],
) -> np.ndarray:
    """Download the first-isotope matrix referenced by one annotation record."""
    images = record.get("isotopeImages") or []
    if not images:
        raise_external_service_error(
            "METASPACE",
            "Annotation response has no first-isotope image, so its pixel intensities "
            "cannot be exported without data loss.",
        )
    image_group = dataset.isotope_images(
        str(record.get("sumFormula") or ""),
        str(record.get("adduct") or ""),
        only_first_isotope=True,
        scale_intensity=True,
        # REMARK: METASPACE's CSV export explicitly states that hotspot removal
        # has been applied. Preserve that provider-level preprocessing rather
        # than silently exporting a different intensity representation.
        hotspot_clipping=True,
        neutral_loss=str(record.get("neutralLoss") or ""),
        chem_mod=str(record.get("chemMod") or ""),
        image_metadata=list(images[:1]),
    )
    if not image_group or image_group[0] is None:
        raise_external_service_error(
            "METASPACE",
            "METASPACE returned an empty first-isotope matrix for an annotation.",
        )
    return np.asarray(image_group[0])


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
