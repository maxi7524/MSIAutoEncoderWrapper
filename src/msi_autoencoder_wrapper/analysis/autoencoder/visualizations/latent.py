"""Latent-space visualization helpers."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes
from matplotlib.figure import Figure


def plot_projection(
    projection: np.ndarray,
    labels: Optional[Sequence[object]] = None,
    ax: Optional[Axes] = None,
) -> tuple[Figure, Axes]:
    """Plot a two-dimensional latent projection.

    :param projection: Projected points with at least two columns.
    :type projection: numpy.ndarray
    :param labels: Optional numeric color values.
    :type labels: Sequence[object] | None
    :param ax: Optional existing axes.
    :type ax: matplotlib.axes.Axes | None
    :return: Figure and axes.
    :rtype: tuple[matplotlib.figure.Figure, matplotlib.axes.Axes]
    """
    if ax is None:
        figure, ax = plt.subplots()
    else:
        figure = ax.figure
    scatter = ax.scatter(projection[:, 0], projection[:, 1], c=labels, s=8)
    ax.set(xlabel="Component 1", ylabel="Component 2", title="Latent projection")
    if labels is not None:
        figure.colorbar(scatter, ax=ax)
    return figure, ax
