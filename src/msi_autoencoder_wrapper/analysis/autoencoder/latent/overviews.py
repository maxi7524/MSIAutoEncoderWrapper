"""Composed latent-space visualization panels."""

from __future__ import annotations

from collections.abc import Mapping, Sequence

import matplotlib.pyplot as plt
import numpy as np

from ....visualization import VisualizationTheme, resolve_theme
from ....visualization.spatial import plot_spatial_image


def plot_component_grid(
    images_by_model: Mapping[str, Mapping[int, np.ndarray]],
    component_indices: Sequence[int],
    component_name: str,
    theme: VisualizationTheme | str | None,
    explained_variance: Mapping[str, Mapping[int, float]] | None = None,
):
    """Plot model-by-component spatial image matrix."""
    resolved = resolve_theme(theme)
    models = list(images_by_model)
    components = list(component_indices)
    figure, axes = plt.subplots(
        len(models),
        len(components),
        figsize=(4.5 * len(components), 4.2 * len(models)),
        dpi=resolved.figure_dpi,
        squeeze=False,
    )
    for row, model_name in enumerate(models):
        for column, component in enumerate(components):
            variance = (
                explained_variance.get(model_name, {}).get(component)
                if explained_variance
                else None
            )
            suffix = f" ({variance:.1%})" if variance is not None else ""
            plot_spatial_image(
                images_by_model[model_name][component],
                title=f"{model_name}: {component_name} {component}{suffix}",
                ax=axes[row, column],
                theme=resolved,
            )
    figure.tight_layout()
    return figure, axes
