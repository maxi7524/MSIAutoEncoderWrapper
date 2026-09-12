"""Peak sampling strategies using labelled coordinates from a peak source."""

from __future__ import annotations

import numpy as np

from ..sampling import (
    SyntheticComponent,
    SyntheticSampleDefinition,
    SyntheticSamplingContext,
    SyntheticSamplingStrategy,
    register_sampling_strategy,
)


def _peak_count(
    rng: np.random.Generator,
    context: SyntheticSamplingContext,
    minimum: int,
    maximum: int | None,
) -> int:
    """Sample an inclusive peak count from shared or strategy-local bounds."""
    maximum = context.default_max_peaks if maximum is None else maximum
    if any(isinstance(value, bool) or not isinstance(value, int) for value in (minimum, maximum)):
        raise ValueError("min_peaks and max_peaks must be integers.")
    if minimum < 1 or maximum < minimum:
        raise ValueError("Peak bounds must satisfy 1 <= min_peaks <= max_peaks.")
    return int(rng.integers(minimum, maximum + 1))


def _annotated_components(
    rng: np.random.Generator,
    context: SyntheticSamplingContext,
    count: int,
) -> list[SyntheticComponent]:
    """Sample distinct labelled identities and one permitted coordinate each."""
    if not context.eligible_labels:
        raise ValueError("Annotated synthesis requires eligible labelled peak coordinates.")
    labels = rng.choice(context.eligible_labels, size=min(count, len(context.eligible_labels)), replace=False)
    return [
        SyntheticComponent(int(rng.choice(context.source.bins[int(label)])), int(label))
        for label in labels
    ]


def _background_components(
    rng: np.random.Generator,
    context: SyntheticSamplingContext,
    count: int,
) -> list[SyntheticComponent]:
    """Sample unlabelled coordinates outside protected labelled peak windows."""
    if count < 1:
        return []
    if not len(context.background_bins):
        raise ValueError("No unannotated bins remain for the requested synthetic strategy.")
    centers = rng.choice(context.background_bins, size=min(count, len(context.background_bins)), replace=False)
    return [SyntheticComponent(int(center), None) for center in centers]


@register_sampling_strategy("single_random")
class SingleRandomSamplingStrategy(SyntheticSamplingStrategy):
    """Generate one unlabelled peak outside all labelled peak windows."""

    def sample(self, rng, context, *, label_targets):
        """Return one background component and the requested target policy."""
        return SyntheticSampleDefinition(tuple(_background_components(rng, context, 1)), label_targets)


@register_sampling_strategy("single_annotated")
class SingleAnnotatedSamplingStrategy(SyntheticSamplingStrategy):
    """Generate one component at a labelled coordinate."""

    def sample(self, rng, context, *, label_targets):
        """Return one labelled component and the requested target policy."""
        return SyntheticSampleDefinition(tuple(_annotated_components(rng, context, 1)), label_targets)


@register_sampling_strategy("random")
class RandomSamplingStrategy(SyntheticSamplingStrategy):
    """Generate an unlabelled mixture with ``min_peaks`` and ``max_peaks`` bounds."""

    def __init__(self, min_peaks: int = 1, max_peaks: int | None = None) -> None:
        """Configure inclusive bounds for the unlabelled component count.

        :param min_peaks: Minimum number of generated components.
        :type min_peaks: int
        :param max_peaks: Maximum number of generated components.  When omitted,
            the phase-level ``SyntheticSpectrumConfig.max_peaks`` is used.
        :type max_peaks: int | None
        """
        self.min_peaks = min_peaks
        self.max_peaks = max_peaks

    def sample(self, rng, context, *, label_targets):
        """Return a background mixture and the requested target policy."""
        return SyntheticSampleDefinition(
            tuple(_background_components(rng, context, _peak_count(rng, context, self.min_peaks, self.max_peaks))),
            label_targets,
        )


@register_sampling_strategy("annotated")
class AnnotatedSamplingStrategy(SyntheticSamplingStrategy):
    """Generate a labelled mixture with ``min_peaks`` and ``max_peaks`` bounds."""

    def __init__(self, min_peaks: int = 1, max_peaks: int | None = None) -> None:
        """Configure inclusive bounds for the labelled component count.

        :param min_peaks: Minimum number of generated components.
        :type min_peaks: int
        :param max_peaks: Maximum number of generated components.  When omitted,
            the phase-level ``SyntheticSpectrumConfig.max_peaks`` is used.
        :type max_peaks: int | None
        """
        self.min_peaks = min_peaks
        self.max_peaks = max_peaks

    def sample(self, rng, context, *, label_targets):
        """Return a labelled mixture and the requested target policy."""
        return SyntheticSampleDefinition(
            tuple(_annotated_components(rng, context, _peak_count(rng, context, self.min_peaks, self.max_peaks))),
            label_targets,
        )


@register_sampling_strategy("mixed")
class MixedSamplingStrategy(SyntheticSamplingStrategy):
    """Generate a mixture of labelled and unlabelled peak components."""

    def __init__(self, min_peaks: int = 1, max_peaks: int | None = None) -> None:
        """Configure inclusive bounds for the total mixed component count.

        :param min_peaks: Minimum number of generated components.
        :type min_peaks: int
        :param max_peaks: Maximum number of generated components.  When omitted,
            the phase-level ``SyntheticSpectrumConfig.max_peaks`` is used.
        :type max_peaks: int | None
        """
        self.min_peaks = min_peaks
        self.max_peaks = max_peaks

    def sample(self, rng, context, *, label_targets):
        """Return a mixed component set and the requested target policy."""
        count = _peak_count(rng, context, self.min_peaks, self.max_peaks)
        annotated_count = int(rng.integers(count + 1))
        components = _annotated_components(rng, context, annotated_count)
        components.extend(_background_components(rng, context, count - len(components)))
        return SyntheticSampleDefinition(tuple(components), label_targets)
