"""Dataset views that select a representation without mutating source datasets."""

from __future__ import annotations

from typing import Any

from torch.utils.data import Dataset


class RawDatasetView(Dataset):
    """Expose ``get_raw_item`` while preserving the source dataset lifecycle."""

    def __init__(self, dataset: Any) -> None:
        if not callable(getattr(dataset, "get_raw_item", None)):
            raise TypeError("RawDatasetView requires a dataset with get_raw_item().")
        self.dataset = dataset

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, index: int) -> Any:
        return self.dataset.get_raw_item(index)
