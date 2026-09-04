"""Deterministic source-index selection for model datasets."""

from __future__ import annotations

import math
import random
from collections import defaultdict
from typing import Any, Callable, Mapping, Sequence


class IndexSelection:
    """Map public dataset indices to selected original source indices."""

    def __init__(self, source_indices: Sequence[int]) -> None:
        self._source_indices = tuple(int(index) for index in source_indices)

    def __len__(self) -> int:
        return len(self._source_indices)

    def __getitem__(self, public_index: int) -> int:
        return self._source_indices[public_index]

    @property
    def source_indices(self) -> tuple[int, ...]:
        """Return the immutable selected source-index mapping."""
        return self._source_indices


class DatasetSubsetter:
    """Select reproducible random or metadata-stratified source indices."""

    @classmethod
    def select_indices(
        cls,
        *,
        source_length: int,
        group_provider: Callable[..., Sequence[Any]],
        multilabel_provider: (
            Callable[..., tuple[Sequence[Any], Sequence[frozenset[int]]]] | None
        ) = None,
        config: Mapping[str, Any],
    ) -> tuple[int, ...]:
        """Return selected original indices without modifying a dataset.

        :param source_length: Number of available source samples.
        :type source_length: int
        :param group_provider: Bulk source-metadata provider used only for
            stratified selection.
        :type group_provider: Callable[..., Sequence[Any]]
        :param multilabel_provider: Bulk provider returning image groups and
            sparse positive labels for ``proportional_multilabel`` selection.
        :type multilabel_provider: Callable[..., tuple[Sequence[Any], Sequence[frozenset[int]]]] | None
        :param config: ``fraction``, ``seed``, ``method``, and optional
            strategy parameters.
        :type config: Mapping[str, Any]
        :return: Sorted original source indices.
        :rtype: tuple[int, ...]
        """
        if source_length < 1:
            raise ValueError("A subset requires at least one source sample.")
        fraction = float(config.get("fraction", 1.0))
        if not 0.0 < fraction <= 1.0:
            raise ValueError("subset.fraction must be in the interval (0, 1].")
        target_size = min(source_length, max(1, math.floor(source_length * fraction)))
        if target_size == source_length:
            return tuple(range(source_length))
        seed = int(config.get("seed", 0))
        method = str(config.get("method", "random"))
        if method == "random":
            return cls._random_indices(source_length, target_size, seed)
        if method == "proportional_multilabel":
            if multilabel_provider is None:
                raise ValueError(
                    "proportional_multilabel selection requires sparse positive labels."
                )
            parameters = dict(config.get("parameters", {}))
            minimum_positive_count = int(parameters.pop("minimum_positive_count", 1))
            groups, positive_labels = multilabel_provider(
                range(source_length), **parameters
            )
            from .multilabel_sampling import select_proportional_multilabel_indices

            return select_proportional_multilabel_indices(
                list(groups),
                [frozenset(labels) for labels in positive_labels],
                fraction=fraction,
                seed=seed,
                minimum_positive_count=minimum_positive_count,
            )
        if method != "stratified_random":
            raise ValueError(
                "Unsupported subset method %r; use 'random', "
                "'stratified_random', or 'proportional_multilabel'." % method
            )
        source_indices = range(source_length)
        groups = list(group_provider(source_indices, **dict(config.get("parameters", {}))))
        if len(groups) != source_length:
            raise ValueError("The subset group provider returned an invalid group count.")
        return cls._stratified_indices(groups, target_size, seed)

    @staticmethod
    def _random_indices(source_length: int, target_size: int, seed: int) -> tuple[int, ...]:
        generator = random.Random(seed)
        selected = generator.sample(range(source_length), target_size)
        selected.sort()
        return tuple(selected)

    @staticmethod
    def _stratified_indices(
        groups: Sequence[Any],
        target_size: int,
        seed: int,
    ) -> tuple[int, ...]:
        strata: dict[Any, list[int]] = defaultdict(list)
        for source_index, group in enumerate(groups):
            strata[_hashable(group)].append(source_index)
        source_length = len(groups)
        exact = {
            key: target_size * len(indices) / source_length
            for key, indices in strata.items()
        }
        quotas = {
            key: min(len(strata[key]), math.floor(value))
            for key, value in exact.items()
        }
        remainder = target_size - sum(quotas.values())
        for key in sorted(strata, key=lambda item: exact[item] - quotas[item], reverse=True):
            if remainder == 0:
                break
            if quotas[key] < len(strata[key]):
                quotas[key] += 1
                remainder -= 1
        generator = random.Random(seed)
        selected = [
            source_index
            for key, indices in strata.items()
            for source_index in generator.sample(indices, quotas[key])
        ]
        selected.sort()
        return tuple(selected)


def _hashable(value: Any) -> Any:
    """Convert nested metadata values into stable strata keys."""
    if isinstance(value, Mapping):
        return tuple(sorted((key, _hashable(item)) for key, item in value.items()))
    if isinstance(value, (list, tuple, set)):
        return tuple(_hashable(item) for item in value)
    return value
