"""Full-axis single-bin coverage from train pixel annotations."""

from __future__ import annotations

import numpy as np

from ...sampling import (
    SyntheticComponent,
    SyntheticSampleDefinition,
    SyntheticSamplingContext,
    SyntheticSamplingStrategy,
    register_sampling_strategy,
)
from .common import annotation_population


@register_sampling_strategy("annotation_axis_coverage")
class AnnotationAxisCoverageSamplingStrategy(SyntheticSamplingStrategy):
    """Cycle through every axis bin while sampling one deterministic train pixel.

    The enclosing ``sampling_plan`` count must be divisible by the active axis
    width. Consequently every bin appears exactly the same number of times in
    the strategy population; labels come from the selected train pixel/bin.
    """

    def sample(self, rng, context: SyntheticSamplingContext, *, label_targets):
        """Return one complete positive or negative pixel/bin component."""
        population = annotation_population(context)
        if context.entry_count % population.feature_count:
            raise ValueError(
                "annotation_axis_coverage count must be divisible by the active bin count."
            )
        bin_index = context.entry_index % population.feature_count
        spectrum_id = population.spectrum_ids[int(rng.integers(len(population.spectrum_ids)))]
        labels = population.labels_for(spectrum_id, bin_index)
        return SyntheticSampleDefinition(
            components=(SyntheticComponent(center=bin_index, label_indices=labels),),
            label_targets=label_targets,
            metadata={
                "generator": "annotation_axis_coverage",
                "source_spectrum_id": spectrum_id,
                "axis_bin": bin_index,
                "axis_repeats": context.entry_count // population.feature_count,
                "positive_labels": labels,
            },
        )
