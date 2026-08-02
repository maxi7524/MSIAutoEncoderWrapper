"""Device-aware transformations from packed raw spectra to model batches."""

from __future__ import annotations

from dataclasses import replace
from typing import Any

import torch

from .batches import RawSpectrumBatch, SpectrumBatch
from ..normalization import NormalizationPipeline


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
        normalization = getattr(self.dataset.active_context, "normalization", None)
        raw_trace = None
        if normalization is not None and normalization.stage == "raw":
            raw_values, raw_trace = normalization.transform(
                prepared.intensities,
                sample_indices=prepared.sample_indices,
                batch_size=prepared.batch_size,
            )
            prepared = replace(
                prepared,
                intensities=raw_values,
                normalization_trace=raw_trace,
            )
        dense = self.binner.transform_batch(prepared)
        trace = raw_trace
        if normalization is not None and normalization.stage == "binned":
            spectra, trace = normalization.transform(dense.spectra)
            normalization_name = trace.pipeline_name
        elif normalization is not None:
            spectra = dense.spectra
            normalization_name = raw_trace.pipeline_name if raw_trace else "none"
        else:
            legacy_name = getattr(self.dataset, "normalization", "none")
            if legacy_name == "none":
                spectra = dense.spectra
                normalization_name = "none"
            else:
                legacy = NormalizationPipeline.from_config(
                    {
                        "stage": "binned",
                        "steps": {
                            legacy_name: {
                                "type": legacy_name,
                                "epsilon": getattr(
                                    self.dataset,
                                    "normalization_epsilon",
                                    1e-12,
                                ),
                            }
                        },
                    }
                )
                spectra, trace = legacy.transform(dense.spectra)
                normalization_name = trace.pipeline_name
        normalized = SpectrumBatch(
            sample_ids=dense.sample_ids,
            spectra=spectra,
            space=replace(
                dense.space,
                normalization=normalization_name,
            ),
            targets=dense.targets,
            views=dense.views,
            metadata=dense.metadata,
            normalization_trace=trace,
        )
        if self.compute_device == self.device:
            return normalized
        if self.device.type == "cpu" and self.compute_device.type == "cuda":
            normalized = normalized.pin_memory()
        return normalized.to(
            self.compute_device,
            non_blocking=self.compute_device.type == "cuda",
        )
