"""Atomic reconstruction visualizations over already prepared values."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes
from matplotlib.figure import Figure

from ....visualization import VisualizationTheme, resolve_theme
from ....visualization.spatial import plot_spatial_image


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
    annotation_mask: Optional[np.ndarray] = None,
    theme: VisualizationTheme | str | None = None,
) -> tuple[Figure, np.ndarray]:
    """Display spatial metric images using a shared scale.

    :param images_by_model: Spatial arrays keyed by model name.
    :type images_by_model: Mapping[str, numpy.ndarray]
    :param metric: Metric name used in panel titles.
    :type metric: str
    :param annotation_mask: Optional spatial annotation-availability mask.
    :type annotation_mask: numpy.ndarray | None
    :param theme: Global graphical strategy.
    :type theme: VisualizationTheme | str | None
    :return: Figure and flattened axes.
    :rtype: tuple[matplotlib.figure.Figure, numpy.ndarray]
    """
    resolved = resolve_theme(theme)
    error_values = [
        np.asarray(values)[np.isfinite(values)] for values in images_by_model.values()
    ]
    finite = np.concatenate(error_values) if error_values else np.asarray([])
    error_range = (
        (float(np.min(finite)), float(np.max(finite))) if finite.size else None
    )
    rows = len(images_by_model) + int(annotation_mask is not None)
    figure, created = plt.subplots(
        rows,
        1,
        figsize=(7, 5 * rows),
        dpi=resolved.figure_dpi,
        squeeze=False,
        sharex=True,
        sharey=True,
    )
    axes = created.ravel()
    row = 0
    if annotation_mask is not None:
        plot_spatial_image(
            annotation_mask,
            title="annotation availability",
            ax=axes[row],
            cmap="Greys",
            value_range=(0.0, 1.0),
            theme=resolved,
        )
        row += 1
    for model_name, values in images_by_model.items():
        plot_spatial_image(
            values,
            title=f"{model_name}: {metric}",
            ax=axes[row],
            cmap=resolved.error_colormap,
            value_range=error_range,
            theme=resolved,
        )
        row += 1
    figure.tight_layout()
    return figure, axes


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
    model_count = len(profiles_by_model)
    if ax is None:
        figure, created = plt.subplots(
            1,
            model_count,
            figsize=(7 * model_count, 6),
            dpi=resolved.figure_dpi,
            squeeze=False,
            sharex=True,
            sharey=True,
        )
        axes = created.ravel()
    else:
        figure = ax.figure
        axes = np.asarray([ax])
    selected: dict[str, np.ndarray] = {}
    for index, (model_name, profile) in enumerate(profiles_by_model.items()):
        current_axis = axes[min(index, len(axes) - 1)]
        color = resolved.color_for_model(model_name, index)
        mean = np.asarray(profile["mean"])
        current_axis.plot(
            mass_axis,
            mean,
            color=color,
            linewidth=resolved.reconstruction_line_width,
            label=model_name,
        )
        current_axis.plot(
            mass_axis,
            profile["median"],
            color=color,
            linewidth=resolved.reference_line_width,
            linestyle=resolved.baseline_line_style,
            label="median",
        )
        current_axis.fill_between(
            mass_axis,
            profile["lower"],
            profile["upper"],
            color=color,
            alpha=resolved.uncertainty_alpha,
        )
        count = min(top_n, len(mean))
        indices = np.argsort(mean)[-count:][::-1]
        selected[model_name] = indices
        current_axis.scatter(
            mass_axis[indices],
            mean[indices],
            color=color,
            s=24,
            zorder=resolved.annotation_zorder,
        )
        for feature_index in indices:
            current_axis.annotate(
                f"{mass_axis[feature_index]:.3f}",
                (mass_axis[feature_index], mean[feature_index]),
                fontsize=resolved.tick_font_size,
                color=color,
                rotation=45,
            )
        current_axis.set(
            xlabel="m/z",
            ylabel=metric,
            title=f"{model_name}: global {metric} distribution",
        )
        current_axis.grid(resolved.grid_visible, alpha=resolved.grid_alpha)
        current_axis.legend(
            loc=resolved.legend_location,
            frameon=resolved.legend_frame,
        )
    return figure, axes, selected


def plot_ion_image_comparison(
    input_image: np.ndarray,
    annotation_mask: np.ndarray,
    reconstructions: Mapping[str, np.ndarray],
    residuals: Mapping[str, np.ndarray],
    mz: float,
    theme: VisualizationTheme | str | None = None,
) -> tuple[Figure, np.ndarray]:
    """Plot shared input/labels beside model reconstruction and residual rows."""
    resolved = resolve_theme(theme)
    model_names = list(reconstructions)
    rows = max(2, len(model_names))
    figure, axes = plt.subplots(
        rows,
        3,
        figsize=(16, 5 * rows),
        dpi=resolved.figure_dpi,
        squeeze=False,
        sharex=True,
        sharey=True,
    )

    # Shared reference column
    ## Input and labels are invariant across models and therefore rendered once.
    intensity_arrays = [input_image, *reconstructions.values()]
    finite_intensity = np.concatenate(
        [np.asarray(values)[np.isfinite(values)] for values in intensity_arrays]
    )
    intensity_range = (
        float(np.min(finite_intensity)),
        float(np.max(finite_intensity)),
    )
    plot_spatial_image(
        input_image,
        title=f"input: {mz:.5f} m/z",
        ax=axes[0, 0],
        value_range=intensity_range,
        theme=resolved,
    )
    plot_spatial_image(
        annotation_mask,
        title="annotation availability",
        ax=axes[1, 0],
        cmap="Greys",
        value_range=(0.0, 1.0),
        theme=resolved,
    )

    # Model-dependent columns
    ## Residual limits are shared and symmetric so color intensity is directly
    ## comparable between every autoencoder row.
    finite_residual = np.concatenate(
        [np.asarray(values)[np.isfinite(values)] for values in residuals.values()]
    )
    residual_limit = float(np.max(np.abs(finite_residual), initial=0.0))
    residual_range = (-residual_limit, residual_limit)
    for row, model_name in enumerate(model_names):
        plot_spatial_image(
            reconstructions[model_name],
            title=f"{model_name}: reconstruction",
            ax=axes[row, 1],
            value_range=intensity_range,
            theme=resolved,
        )
        plot_spatial_image(
            residuals[model_name],
            title=f"{model_name}: residual",
            ax=axes[row, 2],
            cmap=resolved.residual_colormap,
            value_range=residual_range,
            theme=resolved,
        )
    for row in range(len(model_names), rows):
        axes[row, 1].set_visible(False)
        axes[row, 2].set_visible(False)
    for row in range(2, rows):
        axes[row, 0].set_visible(False)
    figure.tight_layout()
    return figure, axes
