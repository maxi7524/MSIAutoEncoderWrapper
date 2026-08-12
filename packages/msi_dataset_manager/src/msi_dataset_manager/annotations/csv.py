"""Read paired canonical molecular and spectrum-intensity CSV exports."""

from __future__ import annotations

import csv
import math
from collections import defaultdict, deque
from decimal import Decimal
from pathlib import Path
from typing import Any, Dict, List

from ..imzml import PyImzMLReader
from ..utils.exceptions import raise_validation_error
from .validation import validate_annotation_record


def _read_csv(path: Path) -> List[Dict[str, str]]:
    if not path.is_file():
        raise_validation_error("CanonicalCSV", f"CSV file does not exist: {path}")
    with path.open("r", encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(line for line in stream if not line.startswith("#")))


def read_canonical_csv_annotations(
    image_path: Path | str,
    annotations_path: Path | str,
    pixel_intensities_path: Path | str,
) -> tuple[Dict[str, Any], List[Dict[str, Any]]]:
    """Normalize paired CSV files to provider-independent annotation records."""
    image = Path(image_path)
    rows = _read_csv(Path(annotations_path))
    intensity_rows = _read_csv(Path(pixel_intensities_path))
    # Annotation-to-intensity matching
    ## Isobaric ions may share m/z; formula and adduct disambiguate their images.
    intensities: dict[tuple[str, str, Decimal], deque[Dict[str, str]]] = defaultdict(deque)
    for intensity_row in intensity_rows:
        intensities[_annotation_key(intensity_row, formula_field="mol_formula")].append(
            intensity_row
        )
    reader = PyImzMLReader(image)
    coordinate_to_spectrum = {
        f"x{x - 1}_y{y - 1}": spectrum_id
        for spectrum_id in range(reader.GetNumberOfSpectra())
        for x, y, _ in [reader.GetSpectrumPosition(spectrum_id)]
    }
    records: List[Dict[str, Any]] = []
    for position, row in enumerate(rows):
        raw_mz = row.get("mz")
        key = _annotation_key(row, formula_field="formula")
        matching_rows = intensities.get(key)
        if not matching_rows:
            raise_validation_error(
                "CanonicalCSV",
                "No intensity row matches annotation "
                f"formula={key[0]!r}, adduct={key[1]!r}, m/z={raw_mz}.",
            )
        intensity_row = matching_rows.popleft()
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
            "annotation_id": str(position),
            "formula": row.get("formula") or intensity_row.get("mol_formula"),
            "adduct": row.get("adduct") or intensity_row.get("adduct"),
            "mz": float(raw_mz),
            "fdr": float(row["fdr"]) if row.get("fdr") else None,
            "spectrum_ids": list(spectrum_values),
            "spectrum_values": spectrum_values,
        }
        validate_annotation_record(record)
        records.append(record)
    first = rows[0] if rows else {}
    return {
        "source": first.get("source", "metaspace"),
        "dataset_id": first.get("datasetId"),
        "name": first.get("datasetName") or image.stem,
        "image_path": str(image),
    }, records


def _annotation_key(
    row: Dict[str, str], *, formula_field: str
) -> tuple[str, str, Decimal]:
    """Return the canonical ion identity used to pair the two CSV exports."""
    raw_mz = row.get("mz")
    if not raw_mz:
        raise_validation_error("CanonicalCSV", "An annotation row has no m/z value")
    return (
        str(row.get(formula_field) or "").strip(),
        str(row.get("adduct") or "").strip(),
        Decimal(raw_mz),
    )
