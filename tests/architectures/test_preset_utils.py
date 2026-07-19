"""Tests for data-driven architecture preset utilities."""

from __future__ import annotations

import pytest

from msi_autoencoder_wrapper.models.architectures.utils.presets_utils import (
    estimate_max_peak_width,
)
from msi_autoencoder_wrapper.utils.exceptions import ValidationError


def test_peak_width_estimator_returns_reproducible_odd_kernel(
    mock_active_context,
) -> None:
    """The restored peak-envelope heuristic returns a valid symmetric kernel."""
    first = estimate_max_peak_width(
        mock_active_context,
        sample_size=6,
        random_seed=7,
    )
    second = estimate_max_peak_width(
        mock_active_context,
        sample_size=6,
        random_seed=7,
    )

    assert first == second
    assert first >= 3
    assert first % 2 == 1


def test_peak_width_estimator_validates_sample_size(mock_active_context) -> None:
    """Invalid sampling settings use the global validation error."""
    with pytest.raises(ValidationError, match="sample_size"):
        estimate_max_peak_width(mock_active_context, sample_size=0)
