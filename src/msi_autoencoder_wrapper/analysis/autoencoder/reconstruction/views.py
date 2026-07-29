"""Atomic reconstruction visualizations over already prepared values."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes
from matplotlib.figure import Figure

from ....visualization import VisualizationTheme, resolve_theme
from ....visualization.spatial import plot_image_grid


def plot_metric_distributions(
    values_by_model: Mapping[str, np.ndarray],
    metric: str,
    bins: int = 60,
    ax: Optional[Axes] = None,
    theme: VisualizationTheme | str | None = None,
) -> tuple[Figure, Axes]:
    """Overlay per-spectrum metric distributions for any model count.

    :param values_by_model: Metric arrays keyed by model name.
    :type values_by_model: Mapping[str, numpy.ndarray]
    :param metric: Displayed metric name.
    :type metric: str
    :param bins: Shared histogram bin count.
    :type bins: int
    :param ax: Optional existing axes.
    :type ax: matplotlib.axes.Axes | None
    :param theme: Global graphical strategy.
    :type theme: VisualizationTheme | str | None
    :return: Figure and axes.
    :rtype: tuple[matplotlib.figure.Figure, matplotlib.axes.Axes]
    """
    resolved = resolve_theme(theme)
    if ax is None:
        figure, ax = plt.subplots(
            figsize=resolved.figure_size,
            dpi=resolved.figure_dpi,
        )
    else:
        figure = ax.figure
    finite_values = [
        np.asarray(values)[np.isfinite(values)] for values in values_by_model.values()
    ]
    combined = np.concatenate(finite_values) if finite_values else np.asarray([])
    edges = np.histogram_bin_edges(combined, bins=bins) if combined.size else bins
    for index, (model_name, values) in enumerate(values_by_model.items()):
        ax.hist(
            values,
            bins=edges,
            density=True,
            histtype="stepfilled",
            alpha=resolved.overlapping_signal_alpha,
            color=resolved.color_for_model(model_name, index),
            label=model_name,
        )
    ax.set(
        xlabel=metric,
        ylabel="Density",
        title=f"{metric} distribution",
    )
    ax.grid(resolved.grid_visible, alpha=resolved.grid_alpha)
    ax.legend(
        loc=resolved.legend_location,
        frameon=resolved.legend_frame,
        ncols=resolved.legend_columns,
    )
    return figure, ax


def plot_error_images(
    images_by_model: Mapping[str, np.ndarray],
    metric: str,
    theme: VisualizationTheme | str | None = None,
) -> tuple[Figure, np.ndarray]:
    """Display spatial metric images using a shared scale.

    :param images_by_model: Spatial arrays keyed by model name.
    :type images_by_model: Mapping[str, numpy.ndarray]
    :param metric: Metric name used in panel titles.
    :type metric: str
    :param theme: Global graphical strategy.
    :type theme: VisualizationTheme | str | None
    :return: Figure and flattened axes.
    :rtype: tuple[matplotlib.figure.Figure, numpy.ndarray]
    """
    resolved = resolve_theme(theme)
    named = {
        f"{model_name}: {metric}": values
        for model_name, values in images_by_model.items()
    }
    return plot_image_grid(
        named,
        shared_scale=True,
        cmap=resolved.error_colormap,
        theme=resolved,
    )


def plot_feature_profiles(
    mass_axis: np.ndarray,
    profiles_by_model: Mapping[str, Mapping[str, np.ndarray]],
    metric: str,
    top_n: int = 10,
    ax: Optional[Axes] = None,
    theme: VisualizationTheme | str | None = None,
) -> tuple[Figure, Axes, Mapping[str, np.ndarray]]:
    """Plot mean feature error, quantile bands, and worst m/z labels.

    :param mass_axis: Shared binner axis.
    :type mass_axis: numpy.ndarray
    :param profiles_by_model: Distribution profiles keyed by model.
    :type profiles_by_model: Mapping[str, Mapping[str, numpy.ndarray]]
    :param metric: Feature metric name.
    :type metric: str
    :param top_n: Number of worst mean-error features marked per model.
    :type top_n: int
    :param ax: Optional existing axes.
    :type ax: matplotlib.axes.Axes | None
    :param theme: Global graphical strategy.
    :type theme: VisualizationTheme | str | None
    :return: Figure, axes, and selected feature indices by model.
    :rtype: tuple[matplotlib.figure.Figure, matplotlib.axes.Axes, Mapping[str, numpy.ndarray]]
    """
    resolved = resolve_theme(theme)
    if ax is None:
        figure, ax = plt.subplots(
            figsize=resolved.figure_size,
            dpi=resolved.figure_dpi,
        )
    else:
        figure = ax.figure
    selected: dict[str, np.ndarray] = {}
    for index, (model_name, profile) in enumerate(profiles_by_model.items()):
        color = resolved.color_for_model(model_name, index)
        mean = np.asarray(profile["mean"])
        ax.plot(
            mass_axis,
            mean,
            color=color,
            linewidth=resolved.reconstruction_line_width,
            label=model_name,
        )
        ax.fill_between(
            mass_axis,
            profile["lower"],
            profile["upper"],
            color=color,
            alpha=resolved.uncertainty_alpha,
        )
        count = min(top_n, len(mean))
        indices = np.argsort(mean)[-count:][::-1]
        selected[model_name] = indices
        ax.scatter(
            mass_axis[indices],
            mean[indices],
            color=color,
            s=24,
            zorder=resolved.annotation_zorder,
        )
        for feature_index in indices:
            ax.annotate(
                f"{mass_axis[feature_index]:.3f}",
                (mass_axis[feature_index], mean[feature_index]),
                fontsize=resolved.tick_font_size,
                color=color,
                rotation=45,
            )
    ax.set(
        xlabel="m/z",
        ylabel=metric,
        title=f"Global {metric} profile with distribution interval",
    )
    ax.grid(resolved.grid_visible, alpha=resolved.grid_alpha)
    ax.legend(loc=resolved.legend_location, frameon=resolved.legend_frame)
    return figure, ax, selected
