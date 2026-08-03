"""Normalize provider ion images to sparse spectrum-ID annotations."""

from __future__ import annotations

from typing import Any, Dict, List, Mapping, Sequence

import numpy as np

from ...annotations.validators import validate_annotation_record
from ...readers.strategies.pyimzml_reader import PyImzMLReader


def normalize_spectrum_annotations(
    annotations: Sequence[Mapping[str, Any]],
    reader: PyImzMLReader,
) -> List[Dict[str, Any]]:
    """Map provider ion-image coordinates to canonical spectrum IDs."""
    normalized: List[Dict[str, Any]] = []
    for annotation in annotations:
        record = dict(annotation)
        validate_annotation_record(record)
        ion_image_value = record.pop("ion_image", None)
        if ion_image_value is None:
            normalized.append(record)
            continue
        ion_image = np.asarray(ion_image_value)
        spectrum_ids: List[int] = []
        spectrum_values: Dict[int, float] = {}
        for spectrum_id in range(reader.GetNumberOfSpectra()):
            x_position, y_position, _ = reader.GetSpectrumPosition(spectrum_id)
            row = y_position - 1
            column = x_position - 1
            if row < 0 or column < 0 or row >= ion_image.shape[0] or column >= ion_image.shape[1]:
                continue
            intensity = float(ion_image[row, column])
            if np.isfinite(intensity) and intensity > 0:
                spectrum_ids.append(spectrum_id)
                spectrum_values[spectrum_id] = intensity
        record["spectrum_ids"] = spectrum_ids
        record["spectrum_values"] = spectrum_values
        normalized.append(record)
    return normalized
