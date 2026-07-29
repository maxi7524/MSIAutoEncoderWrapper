"""Atomic latent-space visualizations."""

from __future__ import annotations

from collections.abc import Mapping

import matplotlib.pyplot as plt
import numpy as np

from ....visualization import VisualizationTheme, resolve_theme


def plot_projection_grid(
    projections: Mapping[str, np.ndarray],
    labels: np.ndarray | None,
    title: str,
    theme: VisualizationTheme | str | None,
):
    """Plot per-model projections without equating their coordinate systems."""
    resolved = resolve_theme(theme)
    figure, axes = plt.subplots(
        1,
        len(projections),
        figsize=(7 * len(projections), 6),
        dpi=resolved.figure_dpi,
        squeeze=False,
    )
    for index, (model_name, projection) in enumerate(projections.items()):
        axis = axes[0, index]
        scatter = axis.scatter(
            projection[:, 0],
            projection[:, 1],
            c=labels,
            color=None if labels is not None else resolved.color_for_model(model_name, index),
            cmap=resolved.image_colormap if labels is not None else None,
            alpha=resolved.secondary_alpha,
            s=10,
        )
        axis.set(
            xlabel="Component 1",
            ylabel="Component 2",
            title=f"{model_name}: {title}",
        )
        if labels is not None:
            figure.colorbar(scatter, ax=axis)
    figure.tight_layout()
    return figure, axes.ravel()
