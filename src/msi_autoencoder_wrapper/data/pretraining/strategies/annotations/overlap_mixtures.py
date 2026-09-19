"""Mixtures enriched for genuine multi-label annotation intersections."""

from __future__ import annotations

from ...sampling import (
    SyntheticSampleDefinition,
    SyntheticSamplingContext,
    SyntheticSamplingStrategy,
    register_sampling_strategy,
)
from .common import annotation_population, component_from_record, peak_count


@register_sampling_strategy("annotation_overlap_mixture")
class AnnotationOverlapMixtureSamplingStrategy(SyntheticSamplingStrategy):
    """Generate mixtures from train pixel/bin records with multiple labels."""

    def __init__(self, min_peaks: int = 15, max_peaks: int = 30) -> None:
        """Configure the inclusive number of overlapping components."""
        self.min_peaks = min_peaks
        self.max_peaks = max_peaks

    def sample(self, rng, context: SyntheticSamplingContext, *, label_targets):
        """Return one mixture enriched for observed annotation intersections."""
        population = annotation_population(context)
        records = population.overlap_records
        if not records:
            raise ValueError("annotation_overlap_mixture requires train records with multiple labels.")
        count = peak_count(rng, context, self.min_peaks, self.max_peaks)
        selected = tuple(records[int(rng.integers(len(records)))] for _ in range(count))
        return SyntheticSampleDefinition(
            components=tuple(component_from_record(record) for record in selected),
            label_targets=label_targets,
            metadata={
                "generator": "annotation_overlap_mixture",
                "source_records": tuple(
                    (record.spectrum_id, record.bin_index, record.label_indices)
                    for record in selected
                ),
            },
        )
