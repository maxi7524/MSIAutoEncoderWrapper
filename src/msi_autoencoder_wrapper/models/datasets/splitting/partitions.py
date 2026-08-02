"""Dataset partition values passed from datasets to trainers."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Iterator, Mapping, Optional

from torch.utils.data import Dataset


@dataclass(frozen=True)
class SplitManifest:
    """Record stable sample assignments produced by a splitter."""

    strategy: str
    seed: int
    assignments: Mapping[str, tuple[Any, ...]]
    dataset_fingerprint: Optional[str] = None

    def get_config(self) -> Dict[str, Any]:
        """Return a JSON-compatible assignment manifest."""
        return {
            "strategy": self.strategy,
            "seed": self.seed,
            "dataset_fingerprint": self.dataset_fingerprint,
            "assignments": {
                name: list(values) for name, values in self.assignments.items()
            },
        }


@dataclass(frozen=True)
class DatasetPartitions:
    """Contain disjoint dataset views and their reproducibility manifest."""

    train: Dataset
    validation: Optional[Dataset]
    test: Optional[Dataset]
    manifest: SplitManifest

    def items(self) -> Iterator[tuple[str, Dataset]]:
        """Iterate over non-empty partitions in execution order."""
        yield "train", self.train
        if self.validation is not None:
            yield "validation", self.validation
        if self.test is not None:
            yield "test", self.test

