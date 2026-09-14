"""Deterministic batch sampling from P/U/N_sim target masks."""

from __future__ import annotations

import math
from collections.abc import Callable, Iterator, Mapping, Sequence
from typing import Any

import torch
from torch.utils.data import Sampler, Subset

from .supervision_masks import simulated_negative_mask_key


class SupervisionMaskBatchSampler(Sampler[list[int]]):
    """Build batches with configured P/U/N_sim sample proportions.

    A source spectrum can be in more than one pool because molecular
    supervision is per target column. The sampler only selects source indices;
    it never changes target values or reinterprets annotation absence.
    """

    _POOL_NAMES = frozenset({"positive", "simulated_negative", "unlabelled"})

    def __init__(
        self,
        targets: torch.Tensor,
        availability: torch.Tensor,
        simulated_negative: torch.Tensor,
        *,
        batch_size: int,
        proportions: Mapping[str, float],
        steps_per_epoch: int | None = None,
        replacement: bool = True,
        seed: int = 0,
        supervision_provider: Callable[
            [], tuple[torch.Tensor, torch.Tensor, torch.Tensor]
        ] | None = None,
    ) -> None:
        """Initialize deterministic pools from target and auxiliary masks."""
        if isinstance(batch_size, bool) or not isinstance(batch_size, int) or batch_size < 1:
            raise ValueError("batch_size must be a positive integer.")
        if steps_per_epoch is not None and (
            isinstance(steps_per_epoch, bool)
            or not isinstance(steps_per_epoch, int)
            or steps_per_epoch < 1
        ):
            raise ValueError("steps_per_epoch must be a positive integer or null.")
        if isinstance(seed, bool) or not isinstance(seed, int):
            raise ValueError("seed must be an integer.")
        unknown = set(proportions) - self._POOL_NAMES
        if unknown:
            raise ValueError(f"Unsupported supervision-mask proportions: {sorted(unknown)}.")
        weights = {name: float(value) for name, value in proportions.items() if float(value) > 0}
        if not weights or any(not math.isfinite(value) for value in weights.values()):
            raise ValueError("At least one finite positive supervision proportion is required.")
        if len(weights) > batch_size:
            raise ValueError("batch_size must accommodate every requested supervision pool.")
        self.batch_size = batch_size
        self.replacement = bool(replacement)
        self.seed = seed
        self.epoch = 0
        self.counts = _proportional_counts(batch_size, weights)
        self.supervision_provider = supervision_provider
        self._set_pools(targets, availability, simulated_negative)
        self.steps_per_epoch = (
            steps_per_epoch if steps_per_epoch is not None else math.ceil(targets.shape[0] / batch_size)
        )

    def set_epoch(self, epoch: int) -> None:
        """Select the deterministic draw sequence for one epoch.

        :param epoch: Nonnegative epoch index.
        :type epoch: int
        """
        if epoch < 0:
            raise ValueError("epoch must be nonnegative.")
        self.epoch = int(epoch)
        if self.supervision_provider is not None:
            self._set_pools(*self.supervision_provider())

    def __iter__(self) -> Iterator[list[int]]:
        """Yield deterministic batches drawn from the declared mask pools."""
        generator = torch.Generator()
        generator.manual_seed(self.seed + self.epoch)
        for _ in range(self.steps_per_epoch):
            batch: list[int] = []
            for name, count in self.counts.items():
                pool = self.pools[name]
                if self.replacement:
                    selected = torch.randint(len(pool), (count,), generator=generator).tolist()
                else:
                    if count > len(pool):
                        raise ValueError(
                            f"Supervision pool '{name}' has {len(pool)} samples, below requested {count}."
                        )
                    selected = torch.randperm(len(pool), generator=generator)[:count].tolist()
                batch.extend(pool[index] for index in selected)
            permutation = torch.randperm(len(batch), generator=generator).tolist()
            yield [batch[index] for index in permutation]

    def __len__(self) -> int:
        """Return the configured number of batches in one epoch."""
        return self.steps_per_epoch

    def _set_pools(
        self,
        targets: torch.Tensor,
        availability: torch.Tensor,
        simulated_negative: torch.Tensor,
    ) -> None:
        """Refresh pools after a generator has changed its supervised labels."""
        self.pools = _supervision_pools(
            targets,
            availability,
            simulated_negative,
            self.counts,
        )


def collect_supervision_masks(
    dataset: Any,
    target_field: str,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return target, availability, and ``N_sim`` matrices for one data view.

    :param dataset: Repository dataset or a ``Subset`` partition.
    :type dataset: Any
    :param target_field: Molecular target field.
    :type target_field: str
    :return: Target, availability, and simulated-negative tensors, ``(N, C)``.
    :rtype: tuple[torch.Tensor, torch.Tensor, torch.Tensor]
    """
    owner = dataset.dataset if isinstance(dataset, Subset) else dataset
    indices: Sequence[int] = (
        [int(index) for index in dataset.indices]
        if isinstance(dataset, Subset)
        else list(range(len(dataset)))
    )
    getter = getattr(owner, "get_target_batch", None)
    if callable(getter):
        target_batch = getter(indices)
        values = target_batch.values[target_field].to(dtype=torch.float32, device="cpu")
        availability = target_batch.masks[target_field].to(dtype=torch.bool, device="cpu")
        simulated_negative = target_batch.masks.get(simulated_negative_mask_key(target_field))
        simulated_negative = (
            torch.zeros_like(availability, dtype=torch.bool)
            if simulated_negative is None
            else simulated_negative.to(dtype=torch.bool, device="cpu")
        )
        return _broadcast_masks(values, availability, simulated_negative)
    values, availability, simulated_negative = [], [], []
    for index in range(len(dataset)):
        sample = dataset[index]
        values.append(sample[2][target_field])
        availability.append(sample[3][target_field])
        simulated_negative.append(
            sample[3].get(
                simulated_negative_mask_key(target_field),
                torch.zeros_like(sample[3][target_field], dtype=torch.bool),
            )
        )
    if not values:
        raise ValueError("Cannot build a supervision sampler from an empty dataset.")
    return _broadcast_masks(torch.stack(values), torch.stack(availability), torch.stack(simulated_negative))


def _broadcast_masks(
    targets: torch.Tensor,
    availability: torch.Tensor,
    simulated_negative: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Validate and broadcast supervision matrices to ``(N, C)``."""
    if targets.ndim != 2 or targets.shape[0] < 1:
        raise ValueError("Targets must have shape (N, C) with N >= 1.")
    resolved = []
    for mask in (availability, simulated_negative):
        if mask.shape == targets.shape[:1]:
            mask = mask.unsqueeze(1).expand_as(targets)  # (N, C)
        if mask.shape != targets.shape:
            raise ValueError("Supervision masks must have shape (N,) or (N, C).")
        resolved.append(mask.bool())
    return targets, resolved[0], resolved[1]


def _supervision_pools(
    targets: torch.Tensor,
    availability: torch.Tensor,
    simulated_negative: torch.Tensor,
    requested: Mapping[str, int],
) -> dict[str, list[int]]:
    """Collect source-index pools from mutually exclusive P/U/N_sim masks."""
    targets, availability, simulated_negative = _broadcast_masks(
        targets, availability, simulated_negative
    )
    positives = availability & (targets > 0.5)  # (N, C)
    simulated_negative = simulated_negative & availability & ~positives  # (N, C)
    unlabelled = availability & ~positives & ~simulated_negative  # (N, C)
    membership = {
        "positive": positives,
        "simulated_negative": simulated_negative,
        "unlabelled": unlabelled,
    }
    pools = {
        name: entries.any(dim=1).nonzero(as_tuple=True)[0].tolist()
        for name, entries in membership.items()
        if name in requested
    }
    empty = [name for name, pool in pools.items() if not pool]
    if empty:
        raise ValueError("Requested supervision pools are empty: " + ", ".join(sorted(empty)))
    return pools


def _proportional_counts(batch_size: int, proportions: Mapping[str, float]) -> dict[str, int]:
    """Allocate integer batch counts while retaining every requested pool."""
    total = sum(proportions.values())
    exact = {name: batch_size * value / total for name, value in proportions.items()}
    counts = {name: max(1, math.floor(value)) for name, value in exact.items()}
    while sum(counts.values()) > batch_size:
        candidate = max(counts, key=lambda name: (counts[name], exact[name] - counts[name]))
        if counts[candidate] == 1:
            raise ValueError("batch_size cannot retain every requested supervision pool.")
        counts[candidate] -= 1
    remaining = batch_size - sum(counts.values())
    for name in sorted(proportions, key=lambda key: exact[key] - counts[key], reverse=True):
        if remaining == 0:
            break
        counts[name] += 1
        remaining -= 1
    return counts
