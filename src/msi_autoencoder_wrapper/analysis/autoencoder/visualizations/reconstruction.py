"""Reconstruction and spatial visualization helpers."""

from __future__ import annotations

from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes
from matplotlib.figure import Figure

from ..results import PreparedAnalysis


def plot_metric_distribution(
    prepared: PreparedAnalysis,
    metric: str = "mse",
    bins: int = 50,
    ax: Optional[Axes] = None,
) -> tuple[Figure, Axes]:
    """Plot a cached per-spectrum metric histogram.

    :param prepared: Prepared result cache.
    :type prepared: PreparedAnalysis
    :param metric: Metric name.
    :type metric: str
    :param bins: Histogram bin count.
    :type bins: int
    :param ax: Optional existing axes.
    :type ax: matplotlib.axes.Axes | None
    :return: Figure and axes.
    :rtype: tuple[matplotlib.figure.Figure, matplotlib.axes.Axes]
    """
    if ax is None:
        figure, ax = plt.subplots()
    else:
        figure = ax.figure
    ax.hist(prepared.metric_array(metric), bins=bins)
    ax.set(xlabel=metric, ylabel="Spectrum count", title=f"{metric} distribution")
    return figure, ax


def plot_spatial_image(
    values: np.ndarray,
    title: str,
    z_index: int = 0,
    ax: Optional[Axes] = None,
    colorbar: bool = True,
) -> tuple[Figure, Axes]:
    """Plot one z-plane from a spatial result array.

    :param values: Spatial array in ``(z, y, x)`` order.
    :type values: numpy.ndarray
    :param title: Plot title.
    :type title: str
    :param z_index: Selected z-plane.
    :type z_index: int
    :param ax: Optional existing axes.
    :type ax: matplotlib.axes.Axes | None
    :param colorbar: Add a color scale when true.
    :type colorbar: bool
    :return: Figure and axes.
    :rtype: tuple[matplotlib.figure.Figure, matplotlib.axes.Axes]
    """
    if ax is None:
        figure, ax = plt.subplots()
    else:
        figure = ax.figure
    image = ax.imshow(np.asarray(values)[z_index], origin="lower")
    ax.set_title(title)
    if colorbar:
        figure.colorbar(image, ax=ax)
    return figure, ax


def plot_spectrum_comparison(
    mass_axis: np.ndarray,
    original: np.ndarray,
    reconstruction: np.ndarray,
    ax: Optional[Axes] = None,
) -> tuple[Figure, Axes]:
    """Plot original, reconstructed, and residual spectra.

    :param mass_axis: Shared binner m/z axis.
    :type mass_axis: numpy.ndarray
    :param original: Original binned spectrum.
    :type original: numpy.ndarray
    :param reconstruction: Reconstructed binned spectrum.
    :type reconstruction: numpy.ndarray
    :param ax: Optional existing axes.
    :type ax: matplotlib.axes.Axes | None
    :return: Figure and axes.
    :rtype: tuple[matplotlib.figure.Figure, matplotlib.axes.Axes]
    """
    if ax is None:
        figure, ax = plt.subplots()
    else:
        figure = ax.figure
    ax.plot(mass_axis, original, label="input")
    ax.plot(mass_axis, reconstruction, label="reconstruction")
    ax.plot(mass_axis, original - reconstruction, label="residual", alpha=0.7)
    ax.set(xlabel="m/z", ylabel="Intensity")
    ax.legend()
    return figure, ax
