"""Deterministic splitting of ready model datasets."""

from __future__ import annotations

import hashlib
import json
import random
from collections import defaultdict
from typing import Any, Dict, Hashable, Iterable, List, Mapping, Sequence

from torch.utils.data import Dataset, Subset

from ....utils.exceptions import raise_validation_error
from .config import SplitConfig
from .partitions import DatasetPartitions, SplitManifest


class DatasetSplitter:
    """Create train, validation, and test views without owning source preparation."""

    @classmethod
    def split(cls, dataset: Dataset, config: SplitConfig | Mapping[str, Any]) -> DatasetPartitions:
        """Split ``dataset`` according to one validated strategy."""
        resolved = config if isinstance(config, SplitConfig) else SplitConfig.from_config(config)
        if resolved.strategy == "predefined":
            indices = cls._predefined_indices(dataset, resolved)
        elif resolved.strategy in {"target_stratified", "mask_stratified"}:
            indices = cls._stratified_indices(dataset, resolved)
        else:
            groups = cls._groups(dataset, resolved)
            indices = cls._assign_groups(groups, len(dataset), resolved)
        subsets = {
            name: Subset(dataset, values) if values else None
            for name, values in indices.items()
        }
        if subsets["train"] is None:
            raise_validation_error("DatasetSplit", "The train partition is empty.")
        sample_assignments = {
            name: tuple(cls._sample_id(dataset, index) for index in values)
            for name, values in indices.items()
        }
        manifest = SplitManifest(
            strategy=resolved.strategy,
            seed=resolved.seed,
            assignments=sample_assignments,
            dataset_fingerprint=cls._fingerprint(dataset),
        )
        return DatasetPartitions(
            train=subsets["train"],
            validation=subsets["validation"],
            test=subsets["test"],
            manifest=manifest,
        )

    @classmethod
    def _groups(cls, dataset: Dataset, config: SplitConfig) -> List[List[int]]:
        if config.strategy == "random":
            return [[index] for index in range(len(dataset))]
        selector_name = "get_split_group"
        selector = getattr(dataset, selector_name, None)
        if not callable(selector):
            raise_validation_error(
                "DatasetSplit",
                f"Dataset '{type(dataset).__name__}' does not implement {selector_name}().",
            )
        grouped: Dict[Hashable, List[int]] = defaultdict(list)
        missing_policy = str(config.parameters.get("missing", "exclude"))
        for index in range(len(dataset)):
            value = selector(index, **dict(config.parameters))
            if value is None and missing_policy == "exclude":
                continue
            key: Hashable = "__missing__" if value is None else cls._hashable(value)
            grouped[key].append(index)
        return list(grouped.values())

    @classmethod
    def _stratified_indices(
        cls, dataset: Dataset, config: SplitConfig
    ) -> Dict[str, List[int]]:
        selector_name = (
            "get_split_target"
            if config.strategy == "target_stratified"
            else "get_split_mask"
        )
        selector = getattr(dataset, selector_name, None)
        if not callable(selector):
            raise_validation_error(
                "DatasetSplit",
                f"Dataset '{type(dataset).__name__}' does not implement {selector_name}().",
            )
        strata: Dict[Hashable, List[int]] = defaultdict(list)
        missing_policy = str(config.parameters.get("missing", "exclude"))
        for index in range(len(dataset)):
            value = selector(index, **dict(config.parameters))
            if value is None and missing_policy == "exclude":
                continue
            key: Hashable = "__missing__" if value is None else cls._hashable(value)
            strata[key].append(index)
        result: Dict[str, List[int]] = {"train": [], "validation": [], "test": []}
        for offset, key in enumerate(sorted(strata, key=str)):
            values = list(strata[key])
            random.Random(config.seed + offset).shuffle(values)
            lengths = cls._fraction_lengths(len(values), config)
            cursor = 0
            for name in ("train", "validation", "test"):
                next_cursor = cursor + lengths[name]
                result[name].extend(values[cursor:next_cursor])
                cursor = next_cursor
        for values in result.values():
            values.sort()
        return result

    @staticmethod
    def _assign_groups(
        groups: Sequence[Sequence[int]], total: int, config: SplitConfig
    ) -> Dict[str, List[int]]:
        del total
        shuffled = [list(group) for group in groups]
        random.Random(config.seed).shuffle(shuffled)
        # Group allocation
        ## Place the largest indivisible groups first. The seeded shuffle remains
        ## the deterministic tie-breaker for equal-size groups.
        shuffled.sort(key=len, reverse=True)
        included_total = sum(len(group) for group in shuffled)
        targets = {
            name: config.fractions[name] * included_total
            for name in ("train", "validation", "test")
        }
        result: Dict[str, List[int]] = {"train": [], "validation": [], "test": []}
        for group in shuffled:
            active = [
                name
                for name in ("test", "validation", "train")
                if config.fractions[name] > 0
            ]
            feasible = [
                name
                for name in active
                if len(result[name]) + len(group) <= targets[name]
            ]
            candidates = feasible or active
            destination = max(
                candidates,
                key=lambda name: (
                    targets[name] - len(result[name])
                ) / targets[name],
            )
            result[destination].extend(group)
        for values in result.values():
            values.sort()
        return result

    @staticmethod
    def _fraction_lengths(total: int, config: SplitConfig) -> Dict[str, int]:
        names = ("train", "validation", "test")
        exact = {name: config.fractions[name] * total for name in names}
        lengths = {name: int(exact[name]) for name in names}
        remaining = total - sum(lengths.values())
        order = sorted(names, key=lambda name: exact[name] - lengths[name], reverse=True)
        for name in order[:remaining]:
            lengths[name] += 1
        return lengths

    @classmethod
    def _predefined_indices(
        cls, dataset: Dataset, config: SplitConfig
    ) -> Dict[str, List[int]]:
        lookup = {cls._canonical_id(cls._sample_id(dataset, index)): index for index in range(len(dataset))}
        result: Dict[str, List[int]] = {"train": [], "validation": [], "test": []}
        seen: set[int] = set()
        for name in result:
            for sample_id in (config.assignments or {}).get(name, []):
                index = lookup.get(cls._canonical_id(sample_id))
                if index is None:
                    raise_validation_error(
                        "DatasetSplit", f"Unknown predefined sample ID '{sample_id}'."
                    )
                if index in seen:
                    raise_validation_error(
                        "DatasetSplit", f"Sample ID '{sample_id}' is assigned more than once."
                    )
                seen.add(index)
                result[name].append(index)
        return result

    @staticmethod
    def _sample_id(dataset: Dataset, index: int) -> Any:
        getter = getattr(dataset, "get_sample_id", None)
        return getter(index) if callable(getter) else index

    @classmethod
    def _fingerprint(cls, dataset: Dataset) -> str:
        payload = [cls._sample_id(dataset, index) for index in range(len(dataset))]
        encoded = json.dumps(payload, sort_keys=True, default=str).encode("utf-8")
        return hashlib.sha256(encoded).hexdigest()

    @staticmethod
    def _canonical_id(value: Any) -> str:
        return json.dumps(value, sort_keys=True, default=str)

    @staticmethod
    def _hashable(value: Any) -> Hashable:
        if isinstance(value, Hashable):
            return value
        return json.dumps(value, sort_keys=True, default=str)
