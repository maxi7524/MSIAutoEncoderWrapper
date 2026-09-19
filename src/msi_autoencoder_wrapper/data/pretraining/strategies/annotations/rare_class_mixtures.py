"""Mixtures enriched for rare train-split molecular classes."""

from __future__ import annotations

from ...sampling import (
    SyntheticSampleDefinition,
    SyntheticSamplingContext,
    SyntheticSamplingStrategy,
    register_sampling_strategy,
)
from .common import annotation_population, component_from_record, peak_count, random_record_for_label


@register_sampling_strategy("annotation_rare_class_mixture")
class AnnotationRareClassMixtureSamplingStrategy(SyntheticSamplingStrategy):
    """Generate mixtures by cycling through the least frequent train classes."""

    def __init__(
        self,
        min_peaks: int = 15,
        max_peaks: int = 30,
        rare_fraction: float = 0.25,
    ) -> None:
        """Configure mixture size and the selected lower-frequency class share."""
        self.min_peaks = min_peaks
        self.max_peaks = max_peaks
        if not 0.0 < rare_fraction <= 1.0:
            raise ValueError("rare_fraction must belong to (0, 1].")
        self.rare_fraction = float(rare_fraction)

    def sample(self, rng, context: SyntheticSamplingContext, *, label_targets):
        """Return one mixture whose requested classes belong to the rare pool."""
        population = annotation_population(context)
        rare_labels = population.rare_labels(self.rare_fraction)
        count = peak_count(rng, context, self.min_peaks, self.max_peaks)
        requested = context.requested_labels or tuple(
            rare_labels[(context.entry_index + position) % len(rare_labels)]
            for position in range(count)
        )
        if len(requested) != count:
            raise ValueError("Rare-class schedule and sampled mixture size disagree.")
        records = tuple(random_record_for_label(rng, population, label) for label in requested)
        return SyntheticSampleDefinition(
            components=tuple(component_from_record(record) for record in records),
            label_targets=label_targets,
            metadata={
                "generator": "annotation_rare_class_mixture",
                "rare_fraction": self.rare_fraction,
                "requested_labels": requested,
            },
        )
