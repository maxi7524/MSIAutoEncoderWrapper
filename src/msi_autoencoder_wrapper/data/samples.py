"""Per-sample records produced before DataLoader collation."""

from __future__ import annotations

from dataclasses import dataclass

import torch

from .spaces import SpectrumSpace
from .targets import TargetSample


@dataclass(frozen=True)
class RawSpectrumSample:
    """Store one variable-length spectrum read from source storage."""

    sample_id: int
    mass_values: torch.Tensor
    intensities: torch.Tensor
    targets: TargetSample


@dataclass(frozen=True)
class SpectrumSample:
    """Store one dense spectrum on a shared binned axis."""

    sample_id: int
    spectrum: torch.Tensor
    space: SpectrumSpace
    targets: TargetSample
