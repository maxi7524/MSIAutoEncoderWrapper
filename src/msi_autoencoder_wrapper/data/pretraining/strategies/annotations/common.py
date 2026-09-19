"""Shared population access and component construction for annotation strategies."""

from __future__ import annotations

import numpy as np

from ...annotation_population import AnnotationPeakRecord, AnnotationPopulation
from ...sampling import SyntheticComponent, SyntheticSamplingContext


def annotation_population(context: SyntheticSamplingContext) -> AnnotationPopulation:
    """Return the required train-only annotation population.

    :param context: Synthetic sampling context for the active dataset item.
    :type context: SyntheticSamplingContext
    :return: Pixel-aware annotation records.
    :rtype: AnnotationPopulation
    """
    if context.annotation_population is None:
        raise ValueError(
            "Pixel-aware annotation strategies require an annotation peak source."
        )
    return context.annotation_population


def component_from_record(record: AnnotationPeakRecord) -> SyntheticComponent:
    """Convert one complete pixel/bin annotation record into a component."""
    return SyntheticComponent(
        center=record.bin_index,
        label_indices=record.label_indices,
    )


def random_record_for_label(
    rng: np.random.Generator,
    population: AnnotationPopulation,
    label_index: int,
) -> AnnotationPeakRecord:
    """Select one train record containing the requested positive class."""
    records = population.records_for_label(label_index)
    if not records:
        raise ValueError(f"No train annotation record is available for class {label_index}.")
    return records[int(rng.integers(len(records)))]


def peak_count(
    rng: np.random.Generator,
    context: SyntheticSamplingContext,
    minimum: int,
    maximum: int | None,
) -> int:
    """Sample one inclusive component count with shared validation."""
    maximum = context.default_max_peaks if maximum is None else maximum
    if any(isinstance(value, bool) or not isinstance(value, int) for value in (minimum, maximum)):
        raise ValueError("min_peaks and max_peaks must be integers.")
    if minimum < 1 or maximum < minimum:
        raise ValueError("Peak bounds must satisfy 1 <= min_peaks <= max_peaks.")
    return int(rng.integers(minimum, maximum + 1))
