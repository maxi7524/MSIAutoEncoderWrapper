"""Anatomical reference previews with explicit coordinate space and vocabulary."""

from __future__ import annotations

from collections.abc import Mapping

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import BoundaryNorm, ListedColormap
from matplotlib.patches import Patch

from ..theme import VisualizationTheme, resolve_theme


def plot_reference_comparison(image: np.ndarray, labels: np.ndarray, class_names: Mapping[str, str],
                              colors: Mapping[str, str], *, title: str = "manual_annotation",
                              theme: VisualizationTheme | str | None = None):
    """Render the original reference beside a lossless-indexed mask preview.

    :param image: Source image RGB/RGBA array, shape ``(H, W, C)``.
    :param labels: Disjoint integer mask, shape ``(H, W)``, -1 unknown.
    :param class_names: Class vocabulary with string integer keys.
    :param colors: Colors keyed identically to class_names.
    :param title: Coordinate-space and annotation-status context.
    :param theme: Shared graphical theme.
    :return: Matplotlib figure and two axes.
    :raises ValueError: If shapes or class IDs are inconsistent.
    """
    resolved = resolve_theme(theme)
    if labels.ndim != 2 or image.shape[:2] != labels.shape or set(class_names) != set(colors):
        raise ValueError("Reference image, labels and vocabulary are inconsistent.")
    ids = sorted(map(int, class_names))
    if not set(np.unique(labels)).issubset({-1, *ids}):
        raise ValueError("Undefined reference class.")
    # Use compact display codes, retaining canonical numeric IDs in saved data
    display = np.zeros(labels.shape, dtype=np.int16)  # (H, W)
    for code, label in enumerate(ids, 1):
        display[labels == label] = code
    palette = ["#303030", *[colors[str(i)] for i in ids]]
    cmap = ListedColormap(palette)
    figure, axes = plt.subplots(1, 2, figsize=(15, 6), dpi=resolved.figure_dpi)
    axes[0].imshow(image, origin="upper")
    axes[0].set_title("Source expert illustration")
    axes[1].imshow(display, origin="upper", cmap=cmap, norm=BoundaryNorm(np.arange(len(palette) + 1) - .5, len(palette)), interpolation="nearest")
    axes[1].set_title("Digitized mask — H&E image pixels; provisional")
    for ax in axes:
        ax.set_xlabel("Image X (pixels)")
        ax.set_ylabel("Image Y (pixels)")
        ax.grid(False)
    patches = [Patch(color=palette[0], label="-1: unknown / unannotated")]
    patches.extend(Patch(color=colors[str(i)], label=f"{i}: {class_names[str(i)]}") for i in ids)
    figure.legend(handles=patches, loc="lower center", ncol=4, frameon=False)
    figure.suptitle(title)
    figure.tight_layout(rect=(0, .12, 1, .95))
    return figure, axes


def plot_partition_comparison(reference: np.ndarray, aligned: np.ndarray, *, title: str,
                             theme: VisualizationTheme | str | None = None):
    """Render known tissue, matched predictions and errors on the same raster.

    :param reference: Reference raster, -1 unknown.
    :param aligned: Matched prediction raster, -2 unmatched/noise.
    :param title: Dataset/pipeline/seed context.
    :param theme: Shared plotting theme.
    :return: Figure and axes.
    """
    resolved = resolve_theme(theme)
    figure, axes = plt.subplots(1, 3, figsize=(13, 4), dpi=resolved.figure_dpi)
    available = reference >= 0  # (H, W)
    ids = sorted(map(int, np.unique(reference[available])))
    palette = ListedColormap(["#777777", *[resolved.color_for_class(i) for i in range(len(ids))]])
    norm = BoundaryNorm(np.arange(len(ids) + 2) - .5, len(ids) + 1)
    displays = []
    for values in [reference, aligned]:
        display = np.zeros(reference.shape, dtype=np.int32)  # (H, W)
        for code, label in enumerate(ids, 1):
            display[values == label] = code
        displays.append(np.ma.masked_where(~available, display))
    for ax, values, name in zip(axes[:2], displays, ["Reference tissue", "Matched clusters"], strict=True):
        ax.imshow(values, origin="upper", interpolation="nearest", cmap=palette, norm=norm)
        ax.set_title(name)
    error = np.ma.masked_where(~available, (reference != aligned).astype(float))  # (H, W)
    axes[2].imshow(error, origin="upper", interpolation="nearest", cmap="cividis", vmin=0, vmax=1)
    axes[2].set_title("Errors (yellow = incorrect)")
    for ax in axes:
        ax.set_xlabel("Source X (pixels)")
        ax.set_ylabel("Source Y (pixels)")
        ax.grid(False)
    figure.suptitle(title)
    figure.tight_layout()
    return figure, axes


def plot_source_annotation(xy: np.ndarray, labels: np.ndarray, class_names: Mapping[str, str], *, title: str,
                           theme: VisualizationTheme | str | None = None):
    """Show native source coordinates and diagnostic annotation availability.

    :param xy: Original coordinate positions, shape ``(N, 2)``.
    :param labels: Original numeric labels, -1 unavailable, shape ``(N,)``.
    :param class_names: Numeric vocabulary.
    :param title: Dataset/image/annotation context.
    :param theme: Shared visual settings.
    :return: Matplotlib figure and axes.
    """
    resolved = resolve_theme(theme)
    figure, ax = plt.subplots(figsize=resolved.figure_size, dpi=resolved.figure_dpi)
    for label in [-1, *sorted(map(int, class_names))]:
        selected = labels == label  # (N,)
        if not selected.any():
            continue
        color = "#888888" if label == -1 else resolved.color_for_class(label)
        name = "unknown" if label == -1 else class_names[str(label)]
        ax.scatter(xy[selected, 0], xy[selected, 1], color=color, marker="s", s=8,
                   label=f"{label}: {name} ({int(selected.sum())} pixels)")
    ax.set_aspect("equal")
    ax.set(xlabel="Source X (native units)", ylabel="Source Y (native units)", title=title)
    ax.grid(False)
    ax.legend()
    figure.tight_layout()
    return figure, ax


def plot_source_signal_and_mask(xy: np.ndarray, signal: np.ndarray, labels: np.ndarray,
                               class_names: Mapping[str, str], *, title: str, signal_label: str,
                               theme: VisualizationTheme | str | None = None):
    """Show an observed MSI signal beside available source annotations.

    :param xy: Original source positions, shape ``(N, 2)``.
    :param signal: Observed feature intensity, shape ``(N,)``.
    :param labels: Source annotation IDs, -1 unavailable, shape ``(N,)``.
    :param class_names: Available numeric vocabulary; empty if no MSI mask exists.
    :param title: Dataset/acquisition and annotation context.
    :param signal_label: Source feature identity and intensity representation.
    :param theme: Shared visual theme.
    :return: Figure and two axes.
    :raises ValueError: For inconsistent or nonfinite data.
    """
    resolved = resolve_theme(theme)
    if xy.shape != (len(signal), 2) or labels.shape != signal.shape or not np.isfinite(signal).all():
        raise ValueError("Source signal and pixel labels are inconsistent.")
    figure, axes = plt.subplots(1, 2, figsize=(13, 5), dpi=resolved.figure_dpi)
    points = axes[0].scatter(xy[:, 0], xy[:, 1], c=signal, cmap=resolved.image_colormap, marker="s", s=14)
    figure.colorbar(points, ax=axes[0], label=signal_label)
    axes[0].set_title("Observed MSI signal")
    for label in [-1, *sorted(map(int, class_names))]:
        selected = labels == label  # (N,)
        if not selected.any():
            continue
        name = "unknown / no annotation" if label == -1 else class_names[str(label)]
        axes[1].scatter(xy[selected, 0], xy[selected, 1], s=14, marker="s",
                        color="#888888" if label == -1 else resolved.color_for_class(label),
                        label=f"{label}: {name} ({int(selected.sum())})")
    axes[1].set_title("Available source mask" if class_names else "Numeric MSI annotation unavailable")
    axes[1].legend(loc="upper center", bbox_to_anchor=(0.5, -0.16), ncol=3)
    figure.subplots_adjust(bottom=0.23)
    for ax in axes:
        ax.set_aspect("equal")
        ax.set(xlabel="Source X (native units)", ylabel="Source Y (native units)")
        ax.grid(False)
    figure.suptitle(title)
    figure.tight_layout()
    return figure, axes


def plot_clustering_methods(reference: np.ndarray, predictions: Mapping[str, np.ndarray],
                            summaries: Mapping[str, Mapping[str, float]], *, title: str,
                            errors: bool = False, theme: VisualizationTheme | str | None = None):
    """Compare all matched method outputs using identical class colours and axes.

    :param reference: Known integer raster, -1 unavailable.
    :param predictions: Score-only matched rasters by method.
    :param summaries: Accuracy/Dice/noise scores by method.
    :param title: Layout, input seed, noise and representation identity.
    :param errors: Display binary error panels instead of predicted regions.
    :param theme: Shared visual configuration.
    :return: Figure and axes.
    :raises ValueError: For prediction shape mismatches.
    """
    resolved = resolve_theme(theme)
    available = reference >= 0  # (H, W)
    ids = sorted(map(int, np.unique(reference[available])))
    palette = ListedColormap(["#777777", *[resolved.color_for_class(i) for i in range(len(ids))]])
    norm = BoundaryNorm(np.arange(len(ids) + 2) - .5, len(ids) + 1)
    figure, axes = plt.subplots(1, len(predictions) + 1, figsize=(4 * (len(predictions) + 1), 4.8),
                                dpi=resolved.figure_dpi, squeeze=False)
    items = [("Reference", reference), *predictions.items()]
    for ax, (name, labels) in zip(axes[0], items, strict=True):
        if labels.shape != reference.shape:
            raise ValueError("All partition maps must share the reference shape.")
        if errors and name != "Reference":
            values = np.ma.masked_where(~available, (labels != reference).astype(float))  # (H, W)
            ax.imshow(values, origin="upper", interpolation="nearest", cmap="cividis", vmin=0, vmax=1)
        else:
            display = np.zeros(reference.shape, dtype=np.int16)  # (H, W)
            for code, label in enumerate(ids, 1):
                display[labels == label] = code
            ax.imshow(np.ma.masked_where(~available, display), origin="upper", cmap=palette, norm=norm,
                      interpolation="nearest")
        if name == "Reference":
            heading = f"Reference: {len(ids)} regions"
        else:
            summary = summaries[name]
            heading = f"{name}\naccuracy={summary['matched_accuracy']:.3f}; Dice={summary['macro_dice']:.3f}\nnoise={summary['noise_fraction']:.3f}"
        ax.set(title=heading, xlabel="Source X (pixels)", ylabel="Source Y (pixels)")
        ax.grid(False)
    legend = [Patch(color=resolved.color_for_class(i), label=f"Region {label}") for i, label in enumerate(ids)]
    legend += [Patch(color="#777777", label="Unmatched / noise")]
    if errors:
        legend += [Patch(color="#fde725", label="Incorrect assignment")]
    figure.legend(handles=legend, loc="lower center", ncol=min(7, len(legend)), frameon=False)
    figure.suptitle(title)
    figure.tight_layout(rect=(0, .09, 1, .93))
    return figure, axes


def plot_spectral_region_profiles(mass_axis: np.ndarray, spectra: np.ndarray, labels: np.ndarray, *, title: str,
                                 reconstructions: np.ndarray | None = None,
                                 theme: VisualizationTheme | str | None = None):
    """Show all region means with 10–90% spectral variation envelopes.

    :param mass_axis: Physical source m/z axis, shape ``(M,)``.
    :param spectra: Identical input observations, shape ``(N, M)``.
    :param labels: Ground-truth region labels used only for descriptive grouping.
    :param title: Layout, noise, input seed and representation context.
    :param reconstructions: Optional decoder outputs, same shape as spectra.
    :param theme: Shared theme.
    :return: Figure and axes.
    """
    resolved = resolve_theme(theme)
    classes = np.unique(labels)
    figure, axes = plt.subplots(len(classes), 1, figsize=(13, 2.2 * len(classes)), sharex=True, squeeze=False,
                                dpi=resolved.figure_dpi)
    for row, label in enumerate(classes):
        selected = spectra[labels == label]  # (N_c, M)
        mean = selected.mean(axis=0)  # (M,)
        low, high = np.quantile(selected, [.1, .9], axis=0)  # (M,), (M,)
        ax = axes[row, 0]
        ax.fill_between(mass_axis, low, high, color=resolved.color_for_class(row), alpha=resolved.uncertainty_alpha,
                        label="Input 10–90% interval")
        ax.plot(mass_axis, mean, color=resolved.color_for_class(row), label="Mean input")
        if reconstructions is not None:
            ax.plot(mass_axis, reconstructions[labels == label].mean(axis=0), color=resolved.residual_color,
                    linestyle="--", label="Mean decoder output")
        ax.set(ylabel="TIC intensity", title=f"Region {int(label)}; n={len(selected)}")
        ax.legend(loc="upper right")
    axes[-1, 0].set_xlabel("m/z")
    figure.suptitle(title)
    figure.tight_layout()
    return figure, axes
