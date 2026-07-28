"""Import manually downloaded METASPACE files into the canonical catalog."""

from __future__ import annotations

import csv
import math
from decimal import Decimal
from pathlib import Path
from typing import Any, Dict, Mapping

from ...readers.strategies.pyimzml_reader import PyImzMLReader
from ...utils.exceptions import raise_validation_error
from ...utils.logger import get_custom_logger
from ..catalog.sqlite_catalog import DatasetCatalog


logger = get_custom_logger(__name__)


def import_local_dataset(
    *,
    catalog: DatasetCatalog,
    source: str,
    dataset_id: str,
    name: str,
    imzml_path: Path | str,
    annotations_path: Path | str,
    pixel_intensities_path: Path | str,
    metadata: Mapping[str, Any] | None = None,
) -> Dict[str, int]:
    """Import one local imzML pair and its METASPACE CSV exports.

    :param catalog: Destination canonical SQLite catalog.
    :type catalog: DatasetCatalog
    :param source: Source strategy name.
    :type source: str
    :param dataset_id: Stable source dataset identifier.
    :type dataset_id: str
    :param name: Human-readable dataset name.
    :type name: str
    :param imzml_path: Existing imzML file with a sibling ``.ibd`` file.
    :type imzml_path: pathlib.Path | str
    :param annotations_path: METASPACE annotation table CSV.
    :type annotations_path: pathlib.Path | str
    :param pixel_intensities_path: METASPACE pixel-intensity CSV.
    :type pixel_intensities_path: pathlib.Path | str
    :param metadata: Complete known source metadata.
    :type metadata: Mapping[str, Any] | None
    :return: Imported spectrum, annotation, and spatial-link counts.
    :rtype: Dict[str, int]
    :raises ValidationError: If files or annotation rows cannot be matched.
    """
    imzml = Path(imzml_path)
    if not imzml.is_file() or not imzml.with_suffix(".ibd").is_file():
        raise_validation_error("LocalDataset", f"Incomplete imzML pair: '{imzml}'.")
    annotation_rows = _read_csv(Path(annotations_path))
    intensity_rows = _read_csv(Path(pixel_intensities_path))
    intensity_by_mz = {Decimal(row["mz"]): row for row in intensity_rows}
    if len(intensity_by_mz) != len(intensity_rows):
        raise_validation_error("LocalDataset", "Pixel-intensity m/z values are not unique.")

    reader = PyImzMLReader(imzml)
    coordinate_to_spectrum = {
        f"x{x - 1}_y{y - 1}": spectrum_id
        for spectrum_id in range(reader.GetNumberOfSpectra())
        for x, y, _ in [reader.GetSpectrumPosition(spectrum_id)]
    }
    records = []
    spatial_links = 0
    for position, row in enumerate(annotation_rows):
        intensity_row = intensity_by_mz.get(Decimal(row["mz"]))
        if intensity_row is None:
            raise_validation_error(
                "LocalDataset", f"No pixel-intensity row matches annotation m/z {row['mz']}."
            )
        spectrum_ids = []
        spectrum_values: Dict[int, float] = {}
        for column, spectrum_id in coordinate_to_spectrum.items():
            raw_value = intensity_row.get(column, "")
            if not raw_value:
                continue
            value = float(raw_value)
            if math.isfinite(value) and value > 0:
                spectrum_ids.append(spectrum_id)
                spectrum_values[spectrum_id] = value
        record = dict(row)
        record.update(
            {
                "annotation_id": str(position),
                "formula": row.get("formula") or intensity_row.get("mol_formula"),
                "adduct": row.get("adduct") or intensity_row.get("adduct"),
                "mz": float(row["mz"]),
                "fdr": float(row["fdr"]),
                "spectrum_ids": spectrum_ids,
                "spectrum_values": spectrum_values,
            }
        )
        records.append(record)
        spatial_links += len(spectrum_ids)

    catalog.upsert_dataset(
        source=source,
        dataset_id=dataset_id,
        name=name,
        metadata=dict(metadata or {}),
        local_path=imzml.parent,
        status="materialized",
    )
    catalog.replace_annotations(source=source, dataset_id=dataset_id, annotations=records)
    result = {
        "spectra": reader.GetNumberOfSpectra(),
        "annotations": len(records),
        "spatial_links": spatial_links,
    }
    logger.info("Imported local dataset %s: %s", dataset_id, result)
    return result


def _read_csv(path: Path) -> list[Dict[str, str]]:
    if not path.is_file():
        raise_validation_error("LocalDataset", f"CSV file does not exist: '{path}'.")
    with path.open("r", encoding="utf-8", newline="") as stream:
        return list(csv.DictReader(line for line in stream if not line.startswith("#")))
