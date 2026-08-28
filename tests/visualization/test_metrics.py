"""Tests for model-independent metric distribution plots."""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")

import numpy as np
import pytest
from matplotlib.collections import PathCollection

from msi_autoencoder_wrapper.visualization.metrics import plot_violin_with_points


class TestPlotViolinWithPoints:
    def test_scatters_exactly_one_point_per_finite_value(self) -> None:
        values = np.array([1.0, 2.0, 3.0, np.nan, 4.0])

        figure, ax = plot_violin_with_points(values, position=0.0, color="C0")

        scatter_collections = [child for child in ax.collections if isinstance(child, PathCollection)]
        assert len(scatter_collections) == 1
        assert scatter_collections[0].get_offsets().shape[0] == 4  # NaN dropped

    def test_points_are_jittered_around_the_given_position(self) -> None:
        values = np.array([5.0, 5.0, 5.0])

        _, ax = plot_violin_with_points(values, position=2.0, color="C0", jitter=0.1)

        scatter_collections = [child for child in ax.collections if isinstance(child, PathCollection)]
        offsets = scatter_collections[0].get_offsets()
        x_values = offsets[:, 0]
        y_values = offsets[:, 1]
        assert np.all(np.abs(x_values - 2.0) <= 0.1)
        assert np.allclose(y_values, 5.0)

    def test_empty_values_after_dropping_non_finite_draws_nothing(self) -> None:
        values = np.array([np.nan, np.inf, -np.inf])

        figure, ax = plot_violin_with_points(values, position=0.0, color="C0")

        assert len(ax.collections) == 0

    def test_reuses_caller_provided_axis(self) -> None:
        import matplotlib.pyplot as plt

        figure, ax = plt.subplots()

        returned_figure, returned_ax = plot_violin_with_points(
            np.array([1.0, 2.0]), position=0.0, ax=ax, color="C0"
        )

        assert returned_ax is ax
        assert returned_figure is figure
