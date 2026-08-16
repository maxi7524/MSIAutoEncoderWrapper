"""Read paired canonical molecular and spectrum-intensity CSV exports."""

from __future__ import annotations

import csv
import json
import math
from pathlib import Path
from typing import Any, Dict, List, Mapping, Sequence

import numpy as np

from ....imzml import PyImzMLReader
from ....utils.exceptions import raise_validation_error
from ...base import ANNOTATION_EXPORT_SCHEMA_VERSION, SourceAnnotationExport
from ...base.validation import validate_annotation_record

# Annotation artifact paths
def annotation_csv_paths(directory: Path, dataset_id: str) -> tuple[Path, Path]:
    """Return the only supported METASPACE annotation export paths."""
    del dataset_id
    return directory / "annotations.csv", directory / "pixel_intensities.csv"


def has_complete_annotation_csv(directory: Path, dataset_id: str) -> bool:
    """Return whether both reader-compatible annotation CSV files are non-empty."""
    return all(
        path.is_file() and path.stat().st_size > 0
        for path in annotation_csv_paths(directory, dataset_id)
    )

def write_annotation_csv_pair(
    *,
    directory: Path,
    dataset_id: str,
    dataset_name: str,
    annotations: Sequence[tuple[Mapping[str, Any], np.ndarray]],
    reader: PyImzMLReader,
) -> tuple[Path, Path]:
    """Write one canonical, reader-compatible annotation CSV pair.

    Source adapters must supply records in the source annotation export
    contract. This function serializes that contract and has no source-specific
    field aliases or database adaptation rules.
    """
    annotations_path, intensities_path = annotation_csv_paths(directory, dataset_id)
    annotations_tmp = annotations_path.with_suffix(".csv.tmp")
    intensities_tmp = intensities_path.with_suffix(".csv.tmp")
    coordinates = [
        f"x{x - 1}_y{y - 1}"
        for spectrum_id in range(reader.GetNumberOfSpectra())
        for x, y, _ in [reader.GetSpectrumPosition(spectrum_id)]
    ]
    rows = []
    intensity_rows = []
    annotation_ids: set[str] = set()
    for position, (annotation, ion_image) in enumerate(annotations):
        formula = annotation.get("formula")
        mz = annotation.get("mz")
        if not formula:
            raise_validation_error(
                "AnnotationCSV", f"Annotation {position} does not contain formula."
            )
        if mz is None:
            raise_validation_error(
                "AnnotationCSV", f"Annotation {position} does not contain m/z."
            )
        annotation_id = str(annotation.get("source_annotation_id") or "")
        if not annotation_id:
            raise_validation_error(
                "AnnotationCSV",
                f"Annotation {position} does not contain a source annotation identifier.",
            )
        if annotation_id in annotation_ids:
            raise_validation_error(
                "AnnotationCSV",
                f"Duplicate source annotation identifier: {annotation_id!r}.",
            )
        annotation_ids.add(annotation_id)
        rows.append(
            {
                "schema_version": annotation.get("schema_version", ""),
                "source": annotation.get("source", ""),
                "source_annotation_id": annotation_id,
                "provider_record_layout": annotation.get("provider_record_layout", ""),
                "group": annotation.get("group", ""),
                "datasetName": dataset_name,
                "datasetId": dataset_id,
                "formula": formula,
                "adduct": annotation.get("adduct", ""),
                "mz": mz,
                "fdr": annotation.get("fdr", ""),
                "database_name": annotation.get("database_name", ""),
                "database_version": annotation.get("database_version", ""),
                "database_id": annotation.get("database_id", ""),
                "source_record_json": json.dumps(
                    annotation.get("source_record", {}),
                    ensure_ascii=False,
                    default=str,
                    sort_keys=True,
                ),
            }
        )
        intensity_row: dict[str, Any] = {
            "source_annotation_id": annotation_id,
            "mol_formula": formula,
            "adduct": annotation.get("adduct", ""),
            "mz": mz,
            "moleculeNames": annotation.get(
                "molecule_names", annotation.get("moleculeNames", "")
            ),
            "moleculeIds": annotation.get(
                "molecule_ids", annotation.get("moleculeIds", "")
            ),
        }
        # Pixel-matrix serialization
        ## The source matrix is the only authority for per-pixel intensity.
        ## Do not substitute presence flags or values from another schema.
        for spectrum_id, coordinate in enumerate(coordinates):
            x, y, _ = reader.GetSpectrumPosition(spectrum_id)
            row_index = y - 1
            column_index = x - 1
            intensity_row[coordinate] = (
                float(ion_image[row_index, column_index])
                if 0 <= row_index < ion_image.shape[0]
                and 0 <= column_index < ion_image.shape[1]
                else ""
            )
        intensity_rows.append(intensity_row)
    try:
        _write_csv(
            annotations_tmp,
            rows,
            [
                "schema_version", "source", "source_annotation_id",
                "provider_record_layout", "group", "datasetName", "datasetId",
                "formula", "adduct", "mz", "fdr", "database_name",
                "database_version", "database_id", "source_record_json",
            ],
        )
        _write_csv(
            intensities_tmp,
            intensity_rows,
            [
                "source_annotation_id",
                "mol_formula",
                "adduct",
                "mz",
                "moleculeNames",
                "moleculeIds",
                *coordinates,
            ],
        )
        annotations_tmp.replace(annotations_path)
        intensities_tmp.replace(intensities_path)
    except Exception:
        annotations_tmp.unlink(missing_ok=True)
        intensities_tmp.unlink(missing_ok=True)
        raise
    return annotations_path, intensities_path


def _read_csv(path: Path) -> List[Dict[str, str]]:
    if not path.is_file():
        raise_validation_error("CanonicalCSV", f"CSV file does not exist: {path}")
    with path.open("r", encoding="utf-8", newline="") as stream:
        return list(csv.DictReader(stream))


def read_metaspace_annotation_export(
    *,
    dataset_id: str,
    directory: Path,
    imzml_path: Path,
) -> SourceAnnotationExport:
    """Read exactly the current METASPACE source annotation schema."""
    image = Path(imzml_path)
    annotations_path, pixel_intensities_path = annotation_csv_paths(
        directory,
        dataset_id,
    )
    rows = _read_csv(annotations_path)
    intensity_rows = _read_csv(pixel_intensities_path)
    # Lossless annotation-to-intensity matching
    ## Formula, adduct, and m/z are descriptive fields, not a stable join key.
    intensities: dict[str, Dict[str, str]] = {}
    for intensity_row in intensity_rows:
        annotation_id = intensity_row.get("source_annotation_id") or ""
        if not annotation_id or annotation_id in intensities:
            raise_validation_error(
                "CanonicalCSV",
                "Pixel-intensity rows must have unique source_annotation_id values.",
            )
        intensities[annotation_id] = intensity_row
    reader = PyImzMLReader(image)
    coordinate_to_spectrum = {
        f"x{x - 1}_y{y - 1}": spectrum_id
        for spectrum_id in range(reader.GetNumberOfSpectra())
        for x, y, _ in [reader.GetSpectrumPosition(spectrum_id)]
    }
    records: List[Dict[str, Any]] = []
    for position, row in enumerate(rows):
        if row.get("schema_version") != str(ANNOTATION_EXPORT_SCHEMA_VERSION):
            raise_validation_error(
                "MetaspaceAnnotationCSV",
                "Unsupported annotation schema version. Remove the old annotation "
                "CSV files and materialize them again.",
            )
        if row.get("source") != "metaspace" or row.get("datasetId") != dataset_id:
            raise_validation_error(
                "MetaspaceAnnotationCSV",
                "Annotation CSV source or dataset identifier does not match its directory.",
            )
        raw_mz = row.get("mz")
        annotation_id = row.get("source_annotation_id") or ""
        intensity_row = intensities.pop(annotation_id, None)
        if intensity_row is None:
            raise_validation_error(
                "CanonicalCSV",
                "No pixel-intensity row matches source_annotation_id="
                f"{annotation_id!r}.",
            )
        spectrum_values: Dict[int, float] = {}
        for column, spectrum_id in coordinate_to_spectrum.items():
            raw_value = intensity_row.get(column, "")
            if raw_value == "":
                continue
            value = float(raw_value)
            if not math.isfinite(value) or value < 0:
                raise_validation_error("CanonicalCSV", f"Invalid intensity {raw_value}")
            if value > 0:
                spectrum_values[spectrum_id] = value
        record: Dict[str, Any] = {
            **row,
            "annotation_id": row["source_annotation_id"],
            "formula": row.get("formula") or intensity_row.get("mol_formula"),
            "adduct": row.get("adduct") or intensity_row.get("adduct"),
            "mz": float(raw_mz),
            "fdr": float(row["fdr"]) if row.get("fdr") else None,
            "spectrum_ids": list(spectrum_values),
            "spectrum_values": spectrum_values,
        }
        source_record_json = row.get("source_record_json")
        if source_record_json:
            try:
                record["source_record"] = json.loads(source_record_json)
            except json.JSONDecodeError:
                raise_validation_error(
                    "CanonicalCSV",
                    "Annotation source_record_json is not valid JSON.",
                )
        validate_annotation_record(record)
        records.append(record)
    first = rows[0] if rows else {}
    return SourceAnnotationExport(
        source="metaspace",
        dataset_id=dataset_id,
        schema_version=ANNOTATION_EXPORT_SCHEMA_VERSION,
        metadata={
            "name": first.get("datasetName") or image.stem,
            "image_path": str(image),
        },
        records=records,
    )
def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]], fields: list[str]) -> None:
    """Write one UTF-8 CSV file with stable columns."""
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
