"""Device-aware transformations from packed raw spectra to model batches."""

from __future__ import annotations

from dataclasses import replace
from typing import Any

import torch

from .batches import RawSpectrumBatch, SpectrumBatch


class BatchPreprocessor:
    """Run binning and normalization on a selected CPU or CUDA device."""

    def __init__(
        self,
        dataset: Any,
        device: torch.device | str,
        compute_device: torch.device | str,
    ) -> None:
        self.dataset = dataset
        self.device = torch.device(device)
        self.compute_device = torch.device(compute_device)
        self.binner = dataset.active_context.binner

    def __call__(self, raw_batch: RawSpectrumBatch) -> SpectrumBatch:
        """Transfer once, bin, normalize, and forward to the compute device."""
        non_blocking = self.device.type == "cuda"
        prepared = raw_batch.to(self.device, non_blocking=non_blocking)
        dense = self.binner.transform_batch(prepared)
        spectra = self.dataset.normalize_batch(dense.spectra)
        normalized = SpectrumBatch(
            sample_ids=dense.sample_ids,
            spectra=spectra,
            space=replace(
                dense.space,
                normalization=self.dataset.normalization,
            ),
            targets=dense.targets,
            views=dense.views,
            metadata=dense.metadata,
        )
        if self.compute_device == self.device:
            return normalized
        if self.device.type == "cpu" and self.compute_device.type == "cuda":
            normalized = normalized.pin_memory()
        return normalized.to(
            self.compute_device,
            non_blocking=self.compute_device.type == "cuda",
        )
