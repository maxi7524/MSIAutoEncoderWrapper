"""Annotation reader for paired METASPACE CSV exports stored beside an image."""

from __future__ import annotations

import csv
import math
from collections import defaultdict, deque
from decimal import Decimal
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional

from ...readers.strategies.pyimzml_reader import PyImzMLReader
from ...utils.exceptions import raise_validation_error
from ..annotations_manager import AnnotationReaderManager
from ..base_annotation_reader import MSIBaseAnnotationReader


def _read_csv(path: Path) -> List[Dict[str, str]]:
    """Read a METASPACE CSV while ignoring its leading comment records."""
    if not path.is_file():
        raise_validation_error(
            "MetaspaceCSVAnnotationReader", f"CSV file does not exist: '{path}'."
        )
    with path.open("r", encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(line for line in stream if not line.startswith("#")))


def read_metaspace_csv_annotations(
    image_path: Path | str,
    annotations_path: Path | str,
    pixel_intensities_path: Path | str,
) -> tuple[Dict[str, Any], List[Dict[str, Any]]]:
    """Normalize paired METASPACE exports against one imzML spectrum index."""
    image = Path(image_path)
    if not image.is_file() or not image.with_suffix(".ibd").is_file():
        raise_validation_error(
            "MetaspaceCSVAnnotationReader",
            f"Incomplete imzML pair for annotation mapping: '{image}'.",
        )

    annotation_rows = _read_csv(Path(annotations_path))
    intensity_rows = _read_csv(Path(pixel_intensities_path))
    if not annotation_rows:
        raise_validation_error(
            "MetaspaceCSVAnnotationReader", "The annotation CSV contains no records."
        )
    # Annotation-to-intensity matching
    ## METASPACE can export different formula/adduct ions at an identical m/z.
    intensity_by_ion: dict[
        tuple[str, str, Decimal], deque[Dict[str, str]]
    ] = defaultdict(deque)
    for intensity_row in intensity_rows:
        key = (
            str(intensity_row.get("mol_formula") or "").strip(),
            str(intensity_row.get("adduct") or "").strip(),
            Decimal(intensity_row["mz"]),
        )
        intensity_by_ion[key].append(intensity_row)

    image_reader = PyImzMLReader(image)
    coordinate_to_spectrum = {
        f"x{x - 1}_y{y - 1}": spectrum_id
        for spectrum_id in range(image_reader.GetNumberOfSpectra())
        for x, y, _ in [image_reader.GetSpectrumPosition(spectrum_id)]
    }
    records: List[Dict[str, Any]] = []
    for position, row in enumerate(annotation_rows):
        raw_mz = row.get("mz")
        if not raw_mz:
            raise_validation_error(
                "MetaspaceCSVAnnotationReader",
                f"Annotation row {position} does not contain an m/z value.",
            )
        ion_key = (
            str(row.get("formula") or "").strip(),
            str(row.get("adduct") or "").strip(),
            Decimal(raw_mz),
        )
        matching_rows = intensity_by_ion.get(ion_key)
        if not matching_rows:
            raise_validation_error(
                "MetaspaceCSVAnnotationReader",
                "No pixel-intensity row matches annotation "
                f"formula={ion_key[0]!r}, adduct={ion_key[1]!r}, m/z={raw_mz}.",
            )
        intensity_row = matching_rows.popleft()
        spectrum_ids: List[int] = []
        spectrum_values: Dict[int, float] = {}
        for column, spectrum_id in coordinate_to_spectrum.items():
            raw_value = intensity_row.get(column, "")
            if not raw_value:
                continue
            value = float(raw_value)
            if not math.isfinite(value) or value < 0:
                raise_validation_error(
                    "MetaspaceCSVAnnotationReader",
                    (
                        f"Invalid pixel intensity {raw_value!r} for annotation m/z "
                        f"{raw_mz} at coordinate '{column}'. Intensities must be "
                        "finite and non-negative."
                    ),
                )
            if value > 0:
                spectrum_ids.append(spectrum_id)
                spectrum_values[spectrum_id] = value
        record: Dict[str, Any] = dict(row)
        record.update(
            {
                "annotation_id": str(position),
                "formula": row.get("formula") or intensity_row.get("mol_formula"),
                "adduct": row.get("adduct") or intensity_row.get("adduct"),
                "mz": float(raw_mz),
                "fdr": float(row["fdr"]) if row.get("fdr") else None,
                "spectrum_ids": spectrum_ids,
                "spectrum_values": spectrum_values,
            }
        )
        records.append(record)

    first = annotation_rows[0]
    metadata = {
        "source": "metaspace",
        "dataset_id": first.get("datasetId"),
        "name": first.get("datasetName") or image.stem,
        "group": first.get("group"),
        "image_path": str(image),
    }
    return metadata, records


@AnnotationReaderManager.register_reader("MetaspaceCSVAnnotationReader")
class MetaspaceCSVAnnotationReader(MSIBaseAnnotationReader):
    """Read METASPACE annotation and per-pixel intensity CSV exports."""

    def __init__(
        self,
        image_path: Path | str,
        annotations_path: Path | str,
        pixel_intensities_path: Path | str,
        default_filters: Optional[Mapping[str, Any]] = None,
        active_context: Optional[Any] = None,
    ) -> None:
        super().__init__(active_context=active_context)
        self.image_path = Path(image_path)
        self.annotations_path = Path(annotations_path)
        self.pixel_intensities_path = Path(pixel_intensities_path)
        self.default_filters = dict(default_filters or {})
        self._metadata, self._annotations = read_metaspace_csv_annotations(
            self.image_path,
            self.annotations_path,
            self.pixel_intensities_path,
        )
        self._config = {
            "image_path": str(self.image_path),
            "annotations_path": str(self.annotations_path),
            "pixel_intensities_path": str(self.pixel_intensities_path),
            "default_filters": self.default_filters,
        }

    def get_dataset_metadata(self) -> Dict[str, Any]:
        """Return metadata retained in the METASPACE annotation export."""
        return dict(self._metadata)

    def get_annotations(
        self,
        filters: Optional[Mapping[str, Any]] = None,
    ) -> List[Dict[str, Any]]:
        """Return annotations after applying supported read-time filters."""
        effective = {**self.default_filters, **dict(filters or {})}
        selected = list(self._annotations)
        max_fdr = effective.pop("max_fdr", None)
        if max_fdr is not None:
            selected = [
                item
                for item in selected
                if item.get("fdr") is not None and float(item["fdr"]) <= float(max_fdr)
            ]
        for key, value in effective.items():
            selected = [item for item in selected if item.get(key) == value]
        return [dict(item) for item in selected]
