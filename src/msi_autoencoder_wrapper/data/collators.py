"""DataLoader collators for packed raw MSI samples."""

from __future__ import annotations

from collections.abc import Sequence

import torch

from .batches import RawSpectrumBatch, SharedAxisRawBatch
from .samples import RawSpectrumSample
from .targets import TargetBatch, TargetSchema


class RawSpectrumCollator:
    """Pack variable-length raw spectra without padding their point axes."""

    def __init__(self, schemas: dict[str, TargetSchema] | None = None) -> None:
        self.schemas = dict(schemas or {})

    def __call__(
        self,
        samples: Sequence[RawSpectrumSample] | RawSpectrumBatch | SharedAxisRawBatch,
    ) -> RawSpectrumBatch | SharedAxisRawBatch:
        """Collate raw samples into flat values, offsets, and sample indices."""
        if isinstance(samples, (RawSpectrumBatch, SharedAxisRawBatch)):
            return samples
        if not samples:
            raise ValueError("RawSpectrumCollator requires at least one sample.")
        lengths = torch.tensor(
            [sample.mass_values.numel() for sample in samples], dtype=torch.long
        )
        offsets = torch.cat(
            [torch.zeros(1, dtype=torch.long), torch.cumsum(lengths, dim=0)]
        )
        values = {
            name: torch.stack([sample.targets.values[name] for sample in samples])
            for name in samples[0].targets.values
        }
        masks = {
            name: torch.stack([sample.targets.masks[name] for sample in samples])
            for name in samples[0].targets.masks
        }
        return RawSpectrumBatch(
            sample_ids=torch.tensor([sample.sample_id for sample in samples], dtype=torch.long),
            mass_values=torch.cat([sample.mass_values for sample in samples]),
            intensities=torch.cat([sample.intensities for sample in samples]),
            offsets=offsets,
            sample_indices=torch.repeat_interleave(
                torch.arange(len(samples), dtype=torch.long), lengths
            ),
            targets=TargetBatch(values=values, masks=masks, schemas=self.schemas),
        )
