"""Persist provider annotations as paired, reader-compatible CSV files."""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from ...readers.strategies.pyimzml_reader import PyImzMLReader
from ...utils.exceptions import raise_validation_error


def annotation_csv_paths(directory: Path, dataset_id: str) -> tuple[Path, Path]:
    """Return canonical annotation and pixel-intensity CSV paths."""
    return (
        directory / "metaspace_annotations.csv",
        directory / f"{dataset_id}_pixel_intensities.csv",
    )


def has_complete_annotation_csv(directory: Path, dataset_id: str) -> bool:
    """Return whether both canonical annotation CSV files are non-empty."""
    return all(
        path.is_file() and path.stat().st_size > 0
        for path in annotation_csv_paths(directory, dataset_id)
    )


def write_annotation_csv_pair(
    *,
    directory: Path,
    dataset_id: str,
    dataset_name: str,
    annotations: Sequence[Mapping[str, Any]],
    reader: PyImzMLReader,
) -> tuple[Path, Path]:
    """Atomically write provider annotations and per-pixel intensities."""
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
    for position, annotation in enumerate(annotations):
        formula = str(annotation.get("formula") or annotation.get("sumFormula") or "")
        adduct = str(annotation.get("adduct") or "")
        mz = annotation.get("mz")
        if mz is None:
            raise_validation_error(
                "AnnotationCSV", f"Annotation {position} does not contain m/z."
            )
        ion_image_value = annotation.get("ion_image")
        ion_image = np.asarray(ion_image_value) if ion_image_value is not None else None
        rows.append(
            {
                "group": annotation.get("group", ""),
                "datasetName": dataset_name,
                "datasetId": dataset_id,
                "formula": formula,
                "adduct": adduct,
                "mz": mz,
                "fdr": annotation.get("fdr", ""),
                "database_name": annotation.get("database_name", ""),
                "database_version": annotation.get("database_version", ""),
            }
        )
        intensity_row: dict[str, Any] = {
            "mol_formula": formula,
            "adduct": adduct,
            "mz": mz,
            "moleculeNames": annotation.get("moleculeNames", ""),
            "moleculeIds": annotation.get("moleculeIds", ""),
        }
        spectrum_values = {
            int(key): float(value)
            for key, value in dict(annotation.get("spectrum_values") or {}).items()
        }
        spectrum_ids = {int(value) for value in annotation.get("spectrum_ids") or ()}
        for spectrum_id, coordinate in enumerate(coordinates):
            if ion_image is None:
                intensity_row[coordinate] = spectrum_values.get(
                    spectrum_id,
                    1.0 if spectrum_id in spectrum_ids else "",
                )
                continue
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
        _write_csv(annotations_tmp, rows, list(rows[0]) if rows else [
            "group", "datasetName", "datasetId", "formula", "adduct", "mz", "fdr",
            "database_name", "database_version",
        ])
        _write_csv(
            intensities_tmp,
            intensity_rows,
            ["mol_formula", "adduct", "mz", "moleculeNames", "moleculeIds", *coordinates],
        )
        annotations_tmp.replace(annotations_path)
        intensities_tmp.replace(intensities_path)
    except Exception:
        annotations_tmp.unlink(missing_ok=True)
        intensities_tmp.unlink(missing_ok=True)
        raise
    return annotations_path, intensities_path


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]], fields: list[str]) -> None:
    """Write one UTF-8 CSV file with stable columns."""
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
