"""Candidate-molecule sampling strategies for synthetic pretraining."""

from __future__ import annotations

from typing import Any

import numpy as np

from ..sampling import (
    SyntheticComponent,
    SyntheticSampleDefinition,
    SyntheticSamplingContext,
    SyntheticSamplingStrategy,
    register_sampling_strategy,
)


def _candidate_count(
    rng: np.random.Generator,
    context: SyntheticSamplingContext,
    minimum: int,
    maximum: int | None,
) -> int:
    """Sample a valid inclusive number of distinct candidate components."""
    maximum = context.default_max_peaks if maximum is None else maximum
    if any(
        isinstance(value, bool) or not isinstance(value, int)
        for value in (minimum, maximum)
    ):
        raise ValueError("min_molecules and max_molecules must be integers.")
    if minimum < 1 or maximum < minimum:
        raise ValueError(
            "Molecule bounds must satisfy 1 <= min_molecules <= max_molecules."
        )
    if not context.eligible_labels:
        raise ValueError("Candidate synthesis requires eligible candidate labels.")
    return min(
        int(rng.integers(minimum, maximum + 1)),
        len(context.eligible_labels),
    )


def _candidate_labels(
    rng: np.random.Generator,
    context: SyntheticSamplingContext,
    count: int,
    *,
    class_balanced: bool,
) -> np.ndarray:
    """Sample distinct labels, optionally balancing populated chemical classes."""
    labels = np.asarray(context.eligible_labels, dtype=np.int64)
    if not class_balanced:
        return np.asarray(rng.choice(labels, size=count, replace=False), dtype=np.int64)
    by_class: dict[str, list[int]] = {}
    unclassified: list[int] = []
    for label in labels:
        metadata = context.source.get_label_metadata(int(label))
        classes = tuple(metadata.get("chemical_classes", ()))
        if not classes:
            unclassified.append(int(label))
            continue
        for class_name in classes:
            by_class.setdefault(str(class_name), []).append(int(label))
    if not by_class:
        return np.asarray(rng.choice(labels, size=count, replace=False), dtype=np.int64)
    selected: list[int] = []
    class_names = list(by_class)
    for class_name in rng.permutation(class_names):
        candidates = [label for label in by_class[class_name] if label not in selected]
        if candidates:
            selected.append(int(rng.choice(candidates)))
        if len(selected) == count:
            return np.asarray(selected, dtype=np.int64)
    remaining = [int(label) for label in labels if int(label) not in selected]
    if remaining and len(selected) < count:
        selected.extend(
            int(label)
            for label in rng.choice(
                np.asarray(remaining, dtype=np.int64),
                size=count - len(selected),
                replace=False,
            )
        )
    return np.asarray(selected, dtype=np.int64)


def _candidate_definition(
    rng: np.random.Generator,
    context: SyntheticSamplingContext,
    labels: np.ndarray,
    weights: np.ndarray,
    *,
    label_targets: bool,
    generator: str,
    metadata: dict[str, Any],
) -> SyntheticSampleDefinition:
    """Build weighted components and record their database provenance."""
    components = tuple(
        SyntheticComponent(
            center=int(rng.choice(context.source.bins[int(label)])),
            label_index=int(label),
            intensity_weight=float(weight),
        )
        for label, weight in zip(labels, weights, strict=True)
    )
    component_metadata = tuple(
        {
            **dict(context.source.get_label_metadata(int(label))),
            "weight": float(weight),
        }
        for label, weight in zip(labels, weights, strict=True)
    )
    return SyntheticSampleDefinition(
        components=components,
        label_targets=label_targets,
        metadata={
            "generator": generator,
            "components": component_metadata,
            **metadata,
        },
    )


@register_sampling_strategy("candidate_single")
class CandidateSingleSamplingStrategy(SyntheticSamplingStrategy):
    """Generate one nonempty spectrum from a single database candidate ion."""

    def sample(self, rng, context, *, label_targets):
        """Return one candidate ion with unit total synthetic intensity."""
        label = _candidate_labels(rng, context, 1, class_balanced=False)
        return _candidate_definition(
            rng,
            context,
            label,
            np.ones(1, dtype=np.float64),
            label_targets=label_targets,
            generator="candidate_single",
            metadata={"mixing_mode": "single"},
        )


@register_sampling_strategy("candidate_permuted")
class CandidatePermutedSamplingStrategy(SyntheticSamplingStrategy):
    """Generate weighted candidate variants from occupancy and intensity draws.

    ``permutations`` is interpreted by the sampling-plan expansion as the
    number of independent variants per configured base count. Every variant
    draws a Dirichlet occupancy vector and independent log-normal component
    intensities before TIC normalization.
    """

    def __init__(
        self,
        min_molecules: int = 2,
        max_molecules: int | None = None,
        occupancy_alpha: float = 1.0,
        intensity_log_sigma: float = 0.5,
        permutations: int = 1,
        class_balanced: bool = False,
    ) -> None:
        """Configure molecule-count and two-distribution weight sampling.

        :param min_molecules: Minimum number of sampled candidate ions.
        :type min_molecules: int
        :param max_molecules: Maximum number of candidate ions.
        :type max_molecules: int | None
        :param occupancy_alpha: Symmetric Dirichlet concentration for relative
            candidate occupancy.
        :type occupancy_alpha: float
        :param intensity_log_sigma: Log-normal spread for component intensity.
        :type intensity_log_sigma: float
        :param permutations: Number of generated variants per sampling-plan
            base count.
        :type permutations: int
        :param class_balanced: Prefer distinct chemical classes when possible.
        :type class_balanced: bool
        """
        if occupancy_alpha <= 0 or intensity_log_sigma < 0:
            raise ValueError("Candidate weight-distribution parameters are invalid.")
        if isinstance(permutations, bool) or not isinstance(permutations, int) or permutations < 1:
            raise ValueError("permutations must be a positive integer.")
        self.min_molecules = min_molecules
        self.max_molecules = max_molecules
        self.occupancy_alpha = float(occupancy_alpha)
        self.intensity_log_sigma = float(intensity_log_sigma)
        self.permutations = permutations
        self.class_balanced = class_balanced

    def sample(self, rng, context, *, label_targets):
        """Return one independently weighted candidate-mixture permutation."""
        count = _candidate_count(rng, context, self.min_molecules, self.max_molecules)
        labels = _candidate_labels(
            rng,
            context,
            count,
            class_balanced=self.class_balanced,
        )
        occupancy = rng.dirichlet(np.full(count, self.occupancy_alpha))
        intensities = rng.lognormal(mean=0.0, sigma=self.intensity_log_sigma, size=count)
        weights = occupancy * intensities
        weights /= weights.sum()
        return _candidate_definition(
            rng,
            context,
            labels,
            weights,
            label_targets=label_targets,
            generator="candidate_permuted",
            metadata={
                "mixing_mode": "weighted_sum",
                "permutations": self.permutations,
                "occupancy_alpha": self.occupancy_alpha,
                "intensity_log_sigma": self.intensity_log_sigma,
                "occupancy": tuple(float(value) for value in occupancy),
                "raw_intensities": tuple(float(value) for value in intensities),
            },
        )


@register_sampling_strategy("candidate_convolved")
class CandidateConvolvedSamplingStrategy(CandidatePermutedSamplingStrategy):
    """Generate a class-aware, non-biological weighted sum of candidates."""

    def __init__(
        self,
        min_molecules: int = 1,
        max_molecules: int | None = None,
        occupancy_alpha: float = 1.0,
        intensity_log_sigma: float = 0.5,
        class_balanced: bool = True,
    ) -> None:
        """Configure a bounded weighted candidate mixture.

        The generated spectrum is the sum of component peak profiles with
        sampled weights; no biological co-occurrence assertion is made.
        """
        super().__init__(
            min_molecules=min_molecules,
            max_molecules=max_molecules,
            occupancy_alpha=occupancy_alpha,
            intensity_log_sigma=intensity_log_sigma,
            permutations=1,
            class_balanced=class_balanced,
        )

    def sample(self, rng, context, *, label_targets):
        """Return one class-aware weighted-sum candidate spectrum."""
        definition = super().sample(rng, context, label_targets=label_targets)
        return SyntheticSampleDefinition(
            components=definition.components,
            label_targets=definition.label_targets,
            metadata={
                **definition.metadata,
                "generator": "candidate_convolved",
                "mixing_mode": "convolution_weighted_sum",
            },
        )
