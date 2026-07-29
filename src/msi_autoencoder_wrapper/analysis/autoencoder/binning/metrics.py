"""Pure aggregation helpers for binning diagnostics."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Dict

import numpy as np


def summarize_forward(
    finite_fractions: Sequence[float],
    tic_ratios: Sequence[float],
) -> Dict[str, float]:
    """Aggregate forward-binner diagnostics."""
    return {
        "spectrum_count": float(len(finite_fractions)),
        "finite_fraction": float(np.mean(finite_fractions)),
        "mean_tic_ratio": float(np.nanmean(tic_ratios)),
        "min_tic_ratio": float(np.nanmin(tic_ratios)),
        "max_tic_ratio": float(np.nanmax(tic_ratios)),
    }


def summarize_inverse(
    mse_values: Sequence[float],
    mae_values: Sequence[float],
    tic_ratios: Sequence[float],
) -> Dict[str, float]:
    """Aggregate inverse/forward round-trip diagnostics."""
    return {
        "spectrum_count": float(len(mse_values)),
        "mean_mse": float(np.mean(mse_values)),
        "mean_mae": float(np.mean(mae_values)),
        "mean_tic_ratio": float(np.nanmean(tic_ratios)),
    }
