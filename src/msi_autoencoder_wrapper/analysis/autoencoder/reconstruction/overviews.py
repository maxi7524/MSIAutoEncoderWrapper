"""Composed reconstruction panels built from atomic visualization tools."""

from __future__ import annotations

from collections.abc import Mapping, Sequence

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.figure import Figure

from ....visualization import VisualizationTheme, resolve_theme


def plot_selected_spectra(
    mass_axis: np.ndarray,
    spectrum_ids: Sequence[int],
    input_spectra: Mapping[int, np.ndarray],
    reconstructions: Mapping[str, Mapping[int, np.ndarray]],
    metric_values: Mapping[int, float],
    selection_name: str,
    clip_reconstruction: bool,
    theme: VisualizationTheme | str | None,
    labels: Mapping[int, str] | None = None,
) -> tuple[Figure, np.ndarray]:
    """Create one mirrored reconstruction panel per model and spectrum.

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
    model_names = list(reconstructions)
    figure, axes = plt.subplots(
        2 * len(model_names),
        len(spectrum_ids),
        figsize=(6 * len(spectrum_ids), 6 * len(model_names)),
        dpi=resolved.figure_dpi,
        sharex="col",
        squeeze=False,
        gridspec_kw={"height_ratios": tuple((2.0, 1.0) * len(model_names))},
    )
    for model_index, model_name in enumerate(model_names):
        color = resolved.color_for_model(model_name, model_index)
        signal_row = 2 * model_index
        residual_row = signal_row + 1
        for column, spectrum_id in enumerate(spectrum_ids):
            original = np.asarray(input_spectra[int(spectrum_id)])
            reconstruction = np.asarray(reconstructions[model_name][int(spectrum_id)])
            displayed = (
                np.clip(reconstruction, 0.0, None)
                if clip_reconstruction
                else reconstruction
            )
            residual = original - reconstruction

            # Mirrored spectrum comparison
            ## Negation is an intentional display operation: both physical
            ## spectra remain non-negative while upper/lower placement removes
            ## the need for competing colors inside one model panel.
            signal_axis = axes[signal_row, column]
            residual_axis = axes[residual_row, column]
            signal_axis.plot(
                mass_axis,
                original,
                color=color,
                alpha=resolved.overlapping_signal_alpha,
                linewidth=resolved.input_line_width,
                label="input (+)",
            )
            signal_axis.plot(
                mass_axis,
                -displayed,
                color=color,
                alpha=resolved.overlapping_signal_alpha,
                linewidth=resolved.reconstruction_line_width,
                label="mirrored reconstruction (-)",
            )
            residual_axis.plot(
                mass_axis,
                residual,
                color=color,
                alpha=resolved.residual_alpha,
                linewidth=resolved.residual_line_width,
                label="input - reconstruction",
            )
            residual_limit = float(np.max(np.abs(residual), initial=0.0))
            if residual_limit > 0.0:
                residual_axis.set_ylim(-residual_limit, residual_limit)
            for axis in (signal_axis, residual_axis):
                axis.axhline(
                    0.0,
                    color=resolved.baseline_color,
                    linewidth=resolved.reference_line_width,
                )
                axis.grid(resolved.grid_visible, alpha=resolved.grid_alpha)
                axis.legend(fontsize=resolved.tick_font_size)
            signal_axis.set_ylabel(f"{model_name}\nintensity")
            residual_axis.set(xlabel="m/z", ylabel="residual")
            label_text = f"\n{labels[int(spectrum_id)]}" if labels else ""
            signal_axis.set_title(
                f"{selection_name}: spectrum {spectrum_id}\n"
                f"score={metric_values[int(spectrum_id)]:.3e}{label_text}",
                loc=resolved.title_location,
            )
    figure.tight_layout()
    return figure, axes
