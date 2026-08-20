"""Atomic head-output spatial visualizations."""

from __future__ import annotations

from collections.abc import Mapping

import matplotlib.pyplot as plt
import numpy as np

from ....visualization import VisualizationTheme, resolve_theme
from ....visualization.spatial import plot_spatial_image


def plot_class_maps(
    model_maps: Mapping[str, Mapping[str, np.ndarray]],
    class_label: str,
    theme: VisualizationTheme | str | None,
    metrics_by_model: Mapping[str, Mapping[str, float]] | None = None,
):
    """Plot ground truth, probability, and signed correctness per model."""
    resolved = resolve_theme(theme)
    models = list(model_maps)
    figure, axes = plt.subplots(
        len(models),
        3,
        figsize=(15, 4.5 * len(models)),
        dpi=resolved.figure_dpi,
        squeeze=False,
    )
    for row, model_name in enumerate(models):
        maps = model_maps[model_name]
        plot_spatial_image(
            maps["ground_truth"],
            title=f"{model_name}: ground truth",
            ax=axes[row, 0],
            cmap="Greys",
            value_range=(0.0, 1.0),
            theme=resolved,
        )
        metrics = (metrics_by_model or {}).get(model_name)
        if metrics:
            summary = " | ".join(
                f"{name}={value:.3f}"
                for name, value in metrics.items()
                if name not in {"class_index", "positive_samples"}
            )
            axes[row, 2].set_xlabel(summary, fontsize=resolved.tick_font_size)
        plot_spatial_image(
            maps["probability"],
            title=f"{model_name}: probability",
            ax=axes[row, 1],
            cmap=resolved.probability_colormap,
            value_range=(0.0, 1.0),
            theme=resolved,
        )
        plot_spatial_image(
            maps["signed_assignment"],
            title=f"{model_name}: green=correct, red=incorrect confidence",
            ax=axes[row, 2],
            cmap=resolved.correctness_colormap,
            value_range=(-1.0, 1.0),
            theme=resolved,
        )
    figure.suptitle(class_label)
    figure.tight_layout()
    return figure, axes
