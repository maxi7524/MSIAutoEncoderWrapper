"""Deterministic proportional sampling for image-associated multi-label pixels."""

from __future__ import annotations

from collections import Counter, defaultdict
import math
import random
from typing import Hashable, Mapping, Sequence

from ...utils.exceptions import raise_validation_error


_SPLIT_NAMES = ("train", "validation", "test")


def select_proportional_multilabel_indices(
    groups: Sequence[Hashable],
    positive_labels: Sequence[frozenset[int]],
    *,
    fraction: float,
    seed: int,
    minimum_positive_count: int,
) -> tuple[int, ...]:
    """Select image-proportional pixels while retaining rare positive classes.

    :param groups: Source-image group for every candidate pixel.
    :type groups: Sequence[Hashable]
    :param positive_labels: Positive class-index sets for every candidate pixel.
    :type positive_labels: Sequence[frozenset[int]]
    :param fraction: Fraction of candidate pixels to retain.
    :type fraction: float
    :param seed: Deterministic sampling seed.
    :type seed: int
    :param minimum_positive_count: Required selected support for every positive
        class.
    :type minimum_positive_count: int
    :return: Sorted selected positions in the candidate sequence.
    :rtype: tuple[int, ...]
    :raises ValidationError: If the requested fraction cannot retain every
        positive class with the required support.
    """
    _validate_population(groups, positive_labels)
    if not 0.0 < fraction <= 1.0:
        raise_validation_error("MultilabelSubset", "fraction must belong to (0, 1].")
    if minimum_positive_count < 1:
        raise_validation_error(
            "MultilabelSubset", "minimum_positive_count must be positive."
        )

    target_size = min(len(groups), max(1, math.floor(len(groups) * fraction)))
    group_indices = _group_indices(groups)
    capacities = _proportional_capacities(
        {group: len(indices) for group, indices in group_indices.items()},
        target_size,
    )
    label_positions = _label_positions(positive_labels)
    # Exact per-class sampling targets cannot in general coexist with exact
    # per-image quotas: labels can be concentrated in the same small image.
    # Enforce the scientifically required coverage here, then let the
    # quota-preserving random completion retain class prevalence in expectation.
    required_counts = {
        label: min(len(indices), minimum_positive_count)
        for label, indices in label_positions.items()
    }
    generator = random.Random(seed)
    selected = _select_required_positives(
        groups=groups,
        positive_labels=positive_labels,
        label_positions=label_positions,
        desired_counts=required_counts,
        capacities=capacities,
        generator=generator,
    )

    # Complete every image quota with an unbiased deterministic sample. The
    # forced rare-class positions above are the minimum deviation from the
    # source image and positive-label distributions needed for coverage.
    for group, indices in group_indices.items():
        remaining = capacities[group] - sum(index in selected for index in indices)
        available = [index for index in indices if index not in selected]
        if remaining < 0 or remaining > len(available):
            raise_validation_error(
                "MultilabelSubset",
                "Image quotas are incompatible with required positive-class coverage.",
            )
        selected.update(generator.sample(available, remaining))

    _validate_selected_coverage(selected, label_positions, minimum_positive_count)
    return tuple(sorted(selected))


def split_proportional_multilabel_indices(
    groups: Sequence[Hashable],
    positive_labels: Sequence[frozenset[int]],
    *,
    fractions: Mapping[str, float],
    seed: int,
    minimum_positive_per_split: int,
) -> dict[str, list[int]]:
    """Split pixels with per-image capacities and positive-class coverage.

    Every source image receives its requested train/validation/test allocation
    independently. Rare positive labels are assigned first, then remaining
    samples fill the still-open image capacities while minimizing per-class
    deviation from the requested split fractions.

    :param groups: Source-image group for every selected pixel.
    :type groups: Sequence[Hashable]
    :param positive_labels: Positive class-index sets for every selected pixel.
    :type positive_labels: Sequence[frozenset[int]]
    :param fractions: Fractions keyed by train, validation, and test.
    :type fractions: Mapping[str, float]
    :param seed: Deterministic allocation seed.
    :type seed: int
    :param minimum_positive_per_split: Required positive samples for each class
        in every non-empty split.
    :type minimum_positive_per_split: int
    :return: Public sample positions per split.
    :rtype: dict[str, list[int]]
    :raises ValidationError: If a class cannot be represented in every split.
    """
    _validate_population(groups, positive_labels)
    if minimum_positive_per_split < 1:
        raise_validation_error(
            "MultilabelSplit", "minimum_positive_per_split must be positive."
        )
    active_splits = tuple(name for name in _SPLIT_NAMES if fractions[name] > 0)
    group_indices = _group_indices(groups)
    group_capacities = {
        group: _fraction_capacities(len(indices), fractions)
        for group, indices in group_indices.items()
    }
    label_positions = _label_positions(positive_labels)
    required_count = minimum_positive_per_split * len(active_splits)
    unavailable = {
        label: len(indices)
        for label, indices in label_positions.items()
        if len(indices) < required_count
    }
    if unavailable:
        raise_validation_error(
            "MultilabelSplit",
            "Positive classes cannot cover every split: "
            + _format_label_counts(unavailable),
        )

    generator = random.Random(seed)
    assignments = {name: [] for name in _SPLIT_NAMES}
    assigned: set[int] = set()
    class_counts = {name: Counter() for name in _SPLIT_NAMES}
    remaining = {group: dict(capacities) for group, capacities in group_capacities.items()}

    # Allocate every pixel under exact per-image quotas before enforcing
    # coverage. This keeps class prevalence close to the requested fractions
    # and avoids prematurely consuming a multi-label pixel for one class.
    target_counts = {
        split_name: {
            label: len(indices) * fractions[split_name]
            for label, indices in label_positions.items()
        }
        for split_name in _SPLIT_NAMES
    }
    unassigned = [index for index in range(len(groups)) if index not in assigned]
    generator.shuffle(unassigned)
    unassigned.sort(key=lambda index: len(positive_labels[index]), reverse=True)
    for index in unassigned:
        group = groups[index]
        candidates = [
            split_name
            for split_name in active_splits
            if remaining[group][split_name] > 0
        ]
        if not candidates:
            raise_validation_error(
                "MultilabelSplit", "Image split capacities were exhausted prematurely."
            )
        split_name = max(
            candidates,
            key=lambda name: _assignment_score(
                positive_labels[index],
                name,
                group,
                target_counts,
                class_counts,
                remaining,
            ),
        )
        _assign(
            index,
            split_name,
            groups,
            positive_labels,
            assignments,
            assigned,
            remaining,
            class_counts,
        )

    # Repair missing support through within-image swaps. A swap preserves every
    # image quota exactly while moving a required positive to its missing split.
    _repair_split_coverage(
        assignments=assignments,
        groups=groups,
        positive_labels=positive_labels,
        label_positions=label_positions,
        active_splits=active_splits,
        class_counts=class_counts,
        minimum_positive_per_split=minimum_positive_per_split,
        generator=generator,
    )
    _validate_split_coverage(class_counts, label_positions, active_splits, minimum_positive_per_split)
    for values in assignments.values():
        values.sort()
    return assignments


def _validate_population(groups: Sequence[Hashable], positive_labels: Sequence[frozenset[int]]) -> None:
    """Validate aligned non-empty candidate metadata."""
    if not groups or len(groups) != len(positive_labels):
        raise_validation_error(
            "MultilabelSampling", "groups and positive_labels must be aligned non-empty sequences."
        )


def _group_indices(groups: Sequence[Hashable]) -> dict[Hashable, list[int]]:
    """Return source positions grouped by their image identifier."""
    values: dict[Hashable, list[int]] = defaultdict(list)
    for index, group in enumerate(groups):
        values[group].append(index)
    return dict(values)


def _label_positions(positive_labels: Sequence[frozenset[int]]) -> dict[int, list[int]]:
    """Invert sparse positive-label sets into class-to-position lists."""
    values: dict[int, list[int]] = defaultdict(list)
    for index, labels in enumerate(positive_labels):
        for label in labels:
            values[int(label)].append(index)
    return dict(values)


def _proportional_capacities(group_sizes: Mapping[Hashable, int], target_size: int) -> dict[Hashable, int]:
    """Allocate a fixed total sample count proportionally by largest remainder."""
    source_size = sum(group_sizes.values())
    exact = {group: target_size * size / source_size for group, size in group_sizes.items()}
    capacities = {group: min(size, math.floor(exact[group])) for group, size in group_sizes.items()}
    remainder = target_size - sum(capacities.values())
    for group in sorted(group_sizes, key=lambda value: exact[value] - capacities[value], reverse=True):
        if remainder == 0:
            break
        if capacities[group] < group_sizes[group]:
            capacities[group] += 1
            remainder -= 1
    return capacities


def _fraction_capacities(size: int, fractions: Mapping[str, float]) -> dict[str, int]:
    """Allocate one image's pixels by largest remainder across splits."""
    exact = {name: size * fractions[name] for name in _SPLIT_NAMES}
    capacities = {name: math.floor(exact[name]) for name in _SPLIT_NAMES}
    remainder = size - sum(capacities.values())
    for name in sorted(_SPLIT_NAMES, key=lambda value: exact[value] - capacities[value], reverse=True):
        if remainder == 0:
            break
        capacities[name] += 1
        remainder -= 1
    return capacities


def _select_required_positives(
    *,
    groups: Sequence[Hashable],
    positive_labels: Sequence[frozenset[int]],
    label_positions: Mapping[int, Sequence[int]],
    desired_counts: Mapping[int, int],
    capacities: Mapping[Hashable, int],
    generator: random.Random,
) -> set[int]:
    """Reserve selected positions until every class reaches its target support."""
    selected: set[int] = set()
    selected_counts: Counter[int] = Counter()
    remaining = dict(capacities)
    for label, indices in sorted(label_positions.items(), key=lambda item: len(item[1])):
        while selected_counts[label] < desired_counts[label]:
            candidates = [
                index
                for index in indices
                if index not in selected and remaining[groups[index]] > 0
            ]
            if not candidates:
                raise_validation_error(
                    "MultilabelSubset",
                    f"Class {label} cannot satisfy required support under image quotas.",
                )
            generator.shuffle(candidates)
            candidate = max(
                candidates,
                key=lambda index: sum(
                    selected_counts[other] < desired_counts.get(other, 0)
                    for other in positive_labels[index]
                ),
            )
            selected.add(candidate)
            remaining[groups[candidate]] -= 1
            selected_counts.update(positive_labels[candidate])
    return selected


def _assignment_score(
    labels: frozenset[int],
    split_name: str,
    group: Hashable,
    target_counts: Mapping[str, Mapping[int, float]],
    class_counts: Mapping[str, Counter[int]],
    remaining: Mapping[Hashable, Mapping[str, int]],
) -> float:
    """Score one feasible split by remaining label deficit and image capacity."""
    label_score = sum(
        max(0.0, target_counts[split_name][label] - class_counts[split_name][label])
        / max(target_counts[split_name][label], 1.0)
        for label in labels
    )
    return label_score + remaining[group][split_name] * 1.0e-6


def _assign(
    index: int,
    split_name: str,
    groups: Sequence[Hashable],
    positive_labels: Sequence[frozenset[int]],
    assignments: Mapping[str, list[int]],
    assigned: set[int],
    remaining: Mapping[Hashable, dict[str, int]],
    class_counts: Mapping[str, Counter[int]],
) -> None:
    """Assign one source position and update sparse allocation state."""
    assignments[split_name].append(index)
    assigned.add(index)
    remaining[groups[index]][split_name] -= 1
    class_counts[split_name].update(positive_labels[index])


def _repair_split_coverage(
    *,
    assignments: Mapping[str, list[int]],
    groups: Sequence[Hashable],
    positive_labels: Sequence[frozenset[int]],
    label_positions: Mapping[int, Sequence[int]],
    active_splits: Sequence[str],
    class_counts: Mapping[str, Counter[int]],
    minimum_positive_per_split: int,
    generator: random.Random,
) -> None:
    """Repair sparse coverage using quota-preserving swaps within each image."""
    split_by_index = {
        index: split_name
        for split_name, indices in assignments.items()
        for index in indices
    }
    indices_by_split_and_group = {
        split_name: _group_indices(indices_groups)
        for split_name, indices_groups in (
            (
                split_name,
                [groups[index] for index in indices],
            )
            for split_name, indices in assignments.items()
        )
    }
    # Convert local positions from _group_indices back to public sample indices.
    for split_name, grouped_positions in indices_by_split_and_group.items():
        split_indices = assignments[split_name]
        indices_by_split_and_group[split_name] = {
            group: [split_indices[position] for position in positions]
            for group, positions in grouped_positions.items()
        }

    protected_labels: set[int] = set()
    for label, indices in sorted(label_positions.items(), key=lambda item: len(item[1])):
        for split_name in active_splits:
            while class_counts[split_name][label] < minimum_positive_per_split:
                swap = _find_coverage_swap(
                    label=label,
                    split_name=split_name,
                    label_indices=indices,
                    split_by_index=split_by_index,
                    indices_by_split_and_group=indices_by_split_and_group,
                    groups=groups,
                    positive_labels=positive_labels,
                    class_counts=class_counts,
                    protected_labels=protected_labels | {label},
                    minimum_positive_per_split=minimum_positive_per_split,
                    generator=generator,
                )
                if swap is None:
                    raise_validation_error(
                        "MultilabelSplit",
                        f"Class {label} cannot be allocated to every split under image quotas.",
                    )
                source_index, target_index, _ = swap
                source_split = split_by_index[source_index]
                _swap_split_assignments(
                    source_index=source_index,
                    target_index=target_index,
                    source_split=source_split,
                    target_split=split_name,
                    assignments=assignments,
                    split_by_index=split_by_index,
                    indices_by_split_and_group=indices_by_split_and_group,
                    groups=groups,
                    positive_labels=positive_labels,
                    class_counts=class_counts,
                )
        protected_labels.add(label)


def _find_coverage_swap(
    *,
    label: int,
    split_name: str,
    label_indices: Sequence[int],
    split_by_index: Mapping[int, str],
    indices_by_split_and_group: Mapping[str, Mapping[Hashable, Sequence[int]]],
    groups: Sequence[Hashable],
    positive_labels: Sequence[frozenset[int]],
    class_counts: Mapping[str, Counter[int]],
    protected_labels: set[int],
    minimum_positive_per_split: int,
    generator: random.Random,
) -> tuple[int, int, bool] | None:
    """Find one coverage-preserving swap for a missing label and split.

    Same-image swaps are attempted first. When no such swap exists, a
    cross-image exchange preserves global split sizes and restores the required
    class coverage at the smallest possible image-level deviation.
    """
    sources = [index for index in label_indices if split_by_index[index] != split_name]
    generator.shuffle(sources)
    same_image = _best_coverage_swap(
        sources=sources,
        target_groups={index: (groups[index],) for index in sources},
        split_name=split_name,
        split_by_index=split_by_index,
        indices_by_split_and_group=indices_by_split_and_group,
        groups=groups,
        positive_labels=positive_labels,
        class_counts=class_counts,
        protected_labels=protected_labels,
        minimum_positive_per_split=minimum_positive_per_split,
        generator=generator,
    )
    if same_image is not None:
        return (*same_image, True)

    all_groups = tuple(indices_by_split_and_group[split_name])
    cross_image = _best_coverage_swap(
        sources=sources,
        target_groups={index: all_groups for index in sources},
        split_name=split_name,
        split_by_index=split_by_index,
        indices_by_split_and_group=indices_by_split_and_group,
        groups=groups,
        positive_labels=positive_labels,
        class_counts=class_counts,
        protected_labels=protected_labels,
        minimum_positive_per_split=minimum_positive_per_split,
        generator=generator,
    )
    return None if cross_image is None else (*cross_image, False)


def _best_coverage_swap(
    *,
    sources: Sequence[int],
    target_groups: Mapping[int, Sequence[Hashable]],
    split_name: str,
    split_by_index: Mapping[int, str],
    indices_by_split_and_group: Mapping[str, Mapping[Hashable, Sequence[int]]],
    groups: Sequence[Hashable],
    positive_labels: Sequence[frozenset[int]],
    class_counts: Mapping[str, Counter[int]],
    protected_labels: set[int],
    minimum_positive_per_split: int,
    generator: random.Random,
) -> tuple[int, int] | None:
    """Return the highest-value feasible swap from the requested target groups."""
    best: tuple[int, int] | None = None
    best_score = -math.inf
    for source_index in sources:
        source_split = split_by_index[source_index]
        for group in target_groups[source_index]:
            targets = list(indices_by_split_and_group[split_name][group])
            generator.shuffle(targets)
            for target_index in targets:
                if not _swap_preserves_coverage(
                    source_index=source_index,
                    target_index=target_index,
                    source_split=source_split,
                    target_split=split_name,
                    positive_labels=positive_labels,
                    class_counts=class_counts,
                    protected_labels=protected_labels,
                    minimum_positive_per_split=minimum_positive_per_split,
                ):
                    continue
                score = sum(
                    class_counts[split_name][candidate_label] < minimum_positive_per_split
                    for candidate_label in positive_labels[source_index]
                ) - len(positive_labels[target_index])
                if score > best_score:
                    best = (source_index, target_index)
                    best_score = score
    return best


def _swap_preserves_coverage(
    *,
    source_index: int,
    target_index: int,
    source_split: str,
    target_split: str,
    positive_labels: Sequence[frozenset[int]],
    class_counts: Mapping[str, Counter[int]],
    protected_labels: set[int],
    minimum_positive_per_split: int,
) -> bool:
    """Return whether a swap preserves already-established class coverage."""
    source_labels = positive_labels[source_index]
    target_labels = positive_labels[target_index]
    for label in protected_labels:
        source_delta = int(label in target_labels) - int(label in source_labels)
        target_delta = int(label in source_labels) - int(label in target_labels)
        if class_counts[source_split][label] + source_delta < minimum_positive_per_split:
            return False
        if class_counts[target_split][label] + target_delta < minimum_positive_per_split:
            return False
    return True


def _swap_split_assignments(
    *,
    source_index: int,
    target_index: int,
    source_split: str,
    target_split: str,
    assignments: Mapping[str, list[int]],
    split_by_index: dict[int, str],
    indices_by_split_and_group: Mapping[str, Mapping[Hashable, list[int]]],
    groups: Sequence[Hashable],
    positive_labels: Sequence[frozenset[int]],
    class_counts: Mapping[str, Counter[int]],
) -> None:
    """Swap two same-image indices and update all split bookkeeping."""
    assignments[source_split].remove(source_index)
    assignments[source_split].append(target_index)
    assignments[target_split].remove(target_index)
    assignments[target_split].append(source_index)
    split_by_index[source_index] = target_split
    split_by_index[target_index] = source_split

    source_group = groups[source_index]
    target_group = groups[target_index]
    indices_by_split_and_group[source_split][source_group].remove(source_index)
    indices_by_split_and_group[source_split][target_group].append(target_index)
    indices_by_split_and_group[target_split][target_group].remove(target_index)
    indices_by_split_and_group[target_split][source_group].append(source_index)

    class_counts[source_split].subtract(positive_labels[source_index])
    class_counts[source_split].update(positive_labels[target_index])
    class_counts[target_split].subtract(positive_labels[target_index])
    class_counts[target_split].update(positive_labels[source_index])


def _validate_selected_coverage(
    selected: set[int],
    label_positions: Mapping[int, Sequence[int]],
    minimum_positive_count: int,
) -> None:
    """Ensure subset support for every original positive class."""
    unavailable = {
        label: sum(index in selected for index in indices)
        for label, indices in label_positions.items()
        if sum(index in selected for index in indices) < minimum_positive_count
    }
    if unavailable:
        raise_validation_error(
            "MultilabelSubset",
            "Selected subset lacks required positive support: " + _format_label_counts(unavailable),
        )


def _validate_split_coverage(
    class_counts: Mapping[str, Counter[int]],
    label_positions: Mapping[int, Sequence[int]],
    active_splits: Sequence[str],
    minimum_positive_per_split: int,
) -> None:
    """Ensure every positive class occurs in every active split."""
    missing = {
        label: {
            split_name: class_counts[split_name][label]
            for split_name in active_splits
            if class_counts[split_name][label] < minimum_positive_per_split
        }
        for label in label_positions
    }
    missing = {label: counts for label, counts in missing.items() if counts}
    if missing:
        formatted = ", ".join(
            f"{label}: {dict(counts)}" for label, counts in sorted(missing.items())
        )
        raise_validation_error(
            "MultilabelSplit", "Positive-class coverage failed: " + formatted
        )


def _format_label_counts(values: Mapping[int, int]) -> str:
    """Render compact deterministic class support diagnostics."""
    return ", ".join(f"{label}={count}" for label, count in sorted(values.items()))
