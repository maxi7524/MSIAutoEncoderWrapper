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
            color=(
                None
                if labels is not None
                else resolved.color_for_model(model_name, index)
            ),
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


def plot_multilabel_projection(
    projections: Mapping[str, np.ndarray],
    targets: np.ndarray,
    target_mask: np.ndarray,
    class_labels: Mapping[int, str],
    probabilities: Mapping[str, np.ndarray] | None,
    mode: str,
    title: str,
    theme: VisualizationTheme | str | None,
):
    """Plot all multi-label classes as overlays or class-specific panels."""
    resolved = resolve_theme(theme)
    if mode not in {"overlay", "panels"}:
        raise ValueError("mode must be 'overlay' or 'panels'.")
    models = list(projections)
    classes = list(class_labels)
    columns = 1 if mode == "overlay" else len(classes)
    figure, axes = plt.subplots(
        len(models),
        columns,
        figsize=(6 * columns, 5 * len(models)),
        dpi=resolved.figure_dpi,
        squeeze=False,
    )
    truth = np.asarray(targets, dtype=bool)
    available = np.asarray(target_mask, dtype=bool).reshape(-1)
    for row, model_name in enumerate(models):
        projection = projections[model_name]
        if mode == "overlay":
            axis = axes[row, 0]
            axis.scatter(
                projection[:, 0],
                projection[:, 1],
                color=resolved.input_color,
                alpha=resolved.uncertainty_alpha,
                s=8,
            )
            for class_index in classes:
                selected = truth[:, class_index] & available
                if not np.any(selected):
                    continue
                strength = (
                    probabilities[model_name][selected, class_index]
                    if probabilities is not None
                    else np.ones(np.sum(selected))
                )
                rgba = np.tile(
                    plt.matplotlib.colors.to_rgba(
                        resolved.color_for_class(class_index)
                    ),
                    (np.sum(selected), 1),
                )
                rgba[:, 3] = np.clip(
                    strength * resolved.primary_alpha,
                    resolved.uncertainty_alpha,
                    resolved.primary_alpha,
                )
                axis.scatter(
                    projection[selected, 0],
                    projection[selected, 1],
                    color=rgba,
                    s=12,
                    label=class_labels[class_index],
                )
            axis.legend(fontsize=resolved.tick_font_size, ncols=2)
            axis.set_title(f"{model_name}: {title}, all molecules")
        else:
            for column, class_index in enumerate(classes):
                axis = axes[row, column]
                axis.scatter(
                    projection[~available, 0],
                    projection[~available, 1],
                    color=resolved.input_color,
                    alpha=resolved.uncertainty_alpha,
                    s=8,
                )
                if probabilities is None:
                    signed = np.where(truth[:, class_index], 1.0, -1.0)
                else:
                    probability = probabilities[model_name][:, class_index]
                    predicted = probability >= 0.5
                    correct = predicted == truth[:, class_index]
                    confidence = np.where(predicted, probability, 1.0 - probability)
                    signed = np.where(correct, confidence, -confidence)
                axis.scatter(
                    projection[available, 0],
                    projection[available, 1],
                    c=signed[available],
                    cmap=resolved.correctness_colormap,
                    vmin=-1.0,
                    vmax=1.0,
                    s=10,
                    alpha=resolved.secondary_alpha,
                )
                axis.set_title(
                    f"{model_name}: {class_labels[class_index]}\n"
                    "green=correct, red=incorrect"
                )
        for axis in axes[row]:
            axis.set(xlabel="Component 1", ylabel="Component 2")
            axis.grid(resolved.grid_visible, alpha=resolved.grid_alpha)
    figure.tight_layout()
    return figure, axes
