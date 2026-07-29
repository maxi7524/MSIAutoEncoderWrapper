"""Composed reconstruction panels built from atomic visualization tools."""

from __future__ import annotations

from collections.abc import Mapping, Sequence

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.figure import Figure

from ....visualization import VisualizationTheme, resolve_theme
from ....visualization.spectra import plot_spectrum_comparison


def plot_selected_spectra(
    mass_axis: np.ndarray,
    spectrum_ids: Sequence[int],
    input_spectra: Mapping[int, np.ndarray],
    reconstructions: Mapping[str, Mapping[int, np.ndarray]],
    metric_values: Mapping[int, float],
    selection_name: str,
    clip_reconstruction: bool,
    theme: VisualizationTheme | str | None,
) -> tuple[Figure, np.ndarray]:
    """Create aligned signal/residual panels for selected spectra.

    :param mass_axis: Shared binner m/z axis.
    :type mass_axis: numpy.ndarray
    :param spectrum_ids: Selected stable spectrum identifiers.
    :type spectrum_ids: Sequence[int]
    :param input_spectra: Input arrays keyed by spectrum identifier.
    :type input_spectra: Mapping[int, numpy.ndarray]
    :param reconstructions: Model and spectrum keyed reconstructions.
    :type reconstructions: Mapping[str, Mapping[int, numpy.ndarray]]
    :param metric_values: Selection metric keyed by spectrum identifier.
    :type metric_values: Mapping[int, float]
    :param selection_name: Best, median, or worst selection label.
    :type selection_name: str
    :param clip_reconstruction: Clip displayed reconstruction values to zero.
    :type clip_reconstruction: bool
    :param theme: Global graphical strategy.
    :type theme: VisualizationTheme | str | None
    :return: Figure and subplot axes.
    :rtype: tuple[matplotlib.figure.Figure, numpy.ndarray]
    """
    resolved = resolve_theme(theme)
    figure, axes = plt.subplots(
        2,
        len(spectrum_ids),
        figsize=(6 * len(spectrum_ids), 8),
        dpi=resolved.figure_dpi,
        sharex="col",
        squeeze=False,
        gridspec_kw={"height_ratios": (2.0, 1.0)},
    )
    for column, spectrum_id in enumerate(spectrum_ids):
        model_values = {
            model_name: values[int(spectrum_id)]
            for model_name, values in reconstructions.items()
        }
        plot_spectrum_comparison(
            mass_axis,
            input_spectra[int(spectrum_id)],
            model_values,
            axes=(axes[0, column], axes[1, column]),
            clip_reconstruction=clip_reconstruction,
            theme=resolved,
        )
        axes[0, column].set_title(
            f"{selection_name}: spectrum {spectrum_id}\n"
            f"score={metric_values[int(spectrum_id)]:.3e}",
            loc=resolved.title_location,
        )
    figure.tight_layout()
    return figure, axes
