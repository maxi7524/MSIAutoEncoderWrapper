"""Model-independent rendering of MSI spatial arrays."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes
from matplotlib.figure import Figure

from ..theme import VisualizationTheme, resolve_theme


def plot_spatial_image(
    values: np.ndarray,
    title: str = "",
    z_index: int = 0,
    ax: Optional[Axes] = None,
    colorbar: bool = True,
    cmap: Optional[str] = None,
    value_range: Optional[tuple[float, float]] = None,
    theme: VisualizationTheme | str | None = None,
) -> tuple[Figure, Axes]:
    """Render one z-plane from a spatial array.

    :param values: Spatial array in ``(z, y, x)`` or ``(y, x)`` order.
    :type values: numpy.ndarray
    :param title: Axes title.
    :type title: str
    :param z_index: Plane selected from a three-dimensional array.
    :type z_index: int
    :param ax: Optional existing axes.
    :type ax: matplotlib.axes.Axes | None
    :param colorbar: Add a color scale when true.
    :type colorbar: bool
    :param cmap: Optional local colormap override.
    :type cmap: str | None
    :param value_range: Optional shared ``(minimum, maximum)`` scale.
    :type value_range: tuple[float, float] | None
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
    array = np.asarray(values)
    plane = array[z_index] if array.ndim == 3 else array
    limits = value_range or (None, None)
    image = ax.imshow(
        plane,
        origin=resolved.image_origin,
        interpolation=resolved.image_interpolation,
        cmap=cmap or resolved.image_colormap,
        vmin=limits[0],
        vmax=limits[1],
    )
    _style_axes(ax, title, resolved)
    if colorbar:
        figure.colorbar(image, ax=ax)
    return figure, ax


def plot_mask_overlay(
    values: np.ndarray,
    mask: np.ndarray,
    title: str = "",
    z_index: int = 0,
    ax: Optional[Axes] = None,
    mask_color: Optional[str] = None,
    theme: VisualizationTheme | str | None = None,
) -> tuple[Figure, Axes]:
    """Overlay a transparent binary mask on a numeric spatial image.

    :param values: Base spatial values.
    :type values: numpy.ndarray
    :param mask: Boolean spatial mask aligned with ``values``.
    :type mask: numpy.ndarray
    :param title: Axes title.
    :type title: str
    :param z_index: Selected z-plane.
    :type z_index: int
    :param ax: Optional existing axes.
    :type ax: matplotlib.axes.Axes | None
    :param mask_color: Optional semantic mask color override.
    :type mask_color: str | None
    :param theme: Global graphical strategy.
    :type theme: VisualizationTheme | str | None
    :return: Figure and axes.
    :rtype: tuple[matplotlib.figure.Figure, matplotlib.axes.Axes]
    """
    from matplotlib.colors import ListedColormap

    resolved = resolve_theme(theme)
    figure, ax = plot_spatial_image(
        values,
        title=title,
        z_index=z_index,
        ax=ax,
        theme=resolved,
    )
    mask_array = np.asarray(mask)
    mask_plane = mask_array[z_index] if mask_array.ndim == 3 else mask_array
    overlay = np.ma.masked_where(~mask_plane.astype(bool), mask_plane)
    ax.imshow(
        overlay,
        origin=resolved.image_origin,
        interpolation=resolved.image_interpolation,
        cmap=ListedColormap([mask_color or resolved.ground_truth_color]),
        alpha=resolved.mask_alpha,
        zorder=resolved.annotation_zorder,
    )
    return figure, ax


def plot_image_grid(
    images: Mapping[str, np.ndarray],
    columns: Optional[int] = None,
    shared_scale: Optional[bool] = None,
    cmap: Optional[str] = None,
    theme: VisualizationTheme | str | None = None,
) -> tuple[Figure, np.ndarray]:
    """Render named spatial arrays in a consistent grid.

    :param images: Mapping from panel titles to spatial arrays.
    :type images: Mapping[str, numpy.ndarray]
    :param columns: Optional number of subplot columns.
    :type columns: int | None
    :param shared_scale: Share finite extrema across panels.
    :type shared_scale: bool | None
    :param cmap: Optional colormap override.
    :type cmap: str | None
    :param theme: Global graphical strategy.
    :type theme: VisualizationTheme | str | None
    :return: Figure and flattened axes array.
    :rtype: tuple[matplotlib.figure.Figure, numpy.ndarray]
    """
    resolved = resolve_theme(theme)
    items = list(images.items())
    column_count = columns or min(3, max(1, len(items)))
    row_count = int(np.ceil(len(items) / column_count))
    figure, axes = plt.subplots(
        row_count,
        column_count,
        figsize=(5 * column_count, 4.5 * row_count),
        dpi=resolved.figure_dpi,
        squeeze=False,
    )
    flattened = axes.ravel()
    use_shared = resolved.shared_image_scale if shared_scale is None else shared_scale
    value_range = None
    if use_shared and items:
        finite = np.concatenate(
            [np.asarray(values)[np.isfinite(values)] for _, values in items]
        )
        if finite.size:
            value_range = (float(np.min(finite)), float(np.max(finite)))
    for axis, (title, values) in zip(flattened, items):
        plot_spatial_image(
            values,
            title=title,
            ax=axis,
            cmap=cmap,
            value_range=value_range,
            theme=resolved,
        )
    for axis in flattened[len(items) :]:
        axis.set_visible(False)
    figure.tight_layout()
    return figure, flattened


def _style_axes(ax: Axes, title: str, theme: VisualizationTheme) -> None:
    """Apply consistent labels, background, and grid semantics."""
    ax.set_title(
        title,
        fontsize=theme.title_font_size,
        color=theme.text_color,
        loc=theme.title_location,
    )
    ax.set_facecolor(theme.panel_color)
    ax.tick_params(labelsize=theme.tick_font_size, colors=theme.text_color)
