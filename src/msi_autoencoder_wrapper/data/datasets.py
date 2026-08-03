"""Dataset views that select a representation without mutating source datasets."""

from __future__ import annotations

from typing import Any

from torch.utils.data import Dataset


class RawDatasetView(Dataset):
    """Expose ``get_raw_item`` while preserving the source dataset lifecycle."""

    def __init__(self, dataset: Any) -> None:
        source = getattr(dataset, "dataset", None)
        if not callable(getattr(dataset, "get_raw_item", None)) and not (
            source is not None
            and callable(getattr(source, "get_raw_item", None))
            and hasattr(dataset, "indices")
        ):
            raise TypeError("RawDatasetView requires a dataset with get_raw_item().")
        self.dataset = dataset

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, index: int) -> Any:
        getter = getattr(self.dataset, "get_raw_item", None)
        if callable(getter):
            return getter(index)
        source_index = self.dataset.indices[index]
        return self.dataset.dataset.get_raw_item(source_index)

    def __getitems__(self, indices: list[int]) -> Any:
        """Fetch a complete batch through the source dataset when supported."""
        getter = getattr(self.dataset, "get_raw_batch", None)
        if callable(getter):
            return getter(indices)
        source = getattr(self.dataset, "dataset", None)
        source_getter = getattr(source, "get_raw_batch", None)
        if callable(source_getter) and hasattr(self.dataset, "indices"):
            source_indices = [self.dataset.indices[index] for index in indices]
            return source_getter(source_indices)
        return [self[index] for index in indices]
