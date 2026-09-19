"""Marginally balanced mixtures from train annotation records."""

from __future__ import annotations

from ...sampling import (
    SyntheticSampleDefinition,
    SyntheticSamplingContext,
    SyntheticSamplingStrategy,
    register_sampling_strategy,
)
from .common import annotation_population, component_from_record, peak_count, random_record_for_label


@register_sampling_strategy("annotation_uniform_mixture")
class AnnotationUniformMixtureSamplingStrategy(SyntheticSamplingStrategy):
    """Compose mixtures from a deterministic round-robin sequence of classes.

    The requested class stream cycles deterministically through molecular
    columns before records are expanded to their complete source-local label
    sets. Metadata records both requested and resulting labels so an
    epoch-level quota allocator can audit extra positives from genuine overlap.
    """

    def __init__(self, min_peaks: int = 15, max_peaks: int = 30) -> None:
        """Configure the inclusive number of source pixel/bin components."""
        self.min_peaks = min_peaks
        self.max_peaks = max_peaks

    def sample(self, rng, context: SyntheticSamplingContext, *, label_targets):
        """Return one class-balanced, complete-label mixture."""
        population = annotation_population(context)
        labels = population.positive_labels
        if not labels:
            raise ValueError("annotation_uniform_mixture requires positive train labels.")
        component_count = peak_count(rng, context, self.min_peaks, self.max_peaks)
        requested = context.requested_labels or tuple(
            labels[(context.entry_index + position) % len(labels)]
            for position in range(component_count)
        )
        if len(requested) != component_count:
            raise ValueError("Annotation class schedule and sampled mixture size disagree.")
        records = tuple(random_record_for_label(rng, population, label) for label in requested)
        components = tuple(component_from_record(record) for record in records)
        return SyntheticSampleDefinition(
            components=components,
            label_targets=label_targets,
            metadata={
                "generator": "annotation_uniform_mixture",
                "requested_labels": requested,
                "source_records": tuple(
                    (record.spectrum_id, record.bin_index, record.label_indices)
                    for record in records
                ),
            },
        )
