"""Model-independent spectrum comparison rendering."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes
from matplotlib.figure import Figure

from ..theme import VisualizationTheme, resolve_theme


def plot_sparse_spectrum_match(
    reference_mz: np.ndarray,
    reference_intensity: np.ndarray,
    candidate_mz: np.ndarray,
    candidate_intensity: np.ndarray,
    match: object,
    axes: Optional[tuple[Axes, Axes]] = None,
    candidate_label: str = "candidate",
    theme: VisualizationTheme | str | None = None,
) -> tuple[Figure, tuple[Axes, Axes]]:
    """Plot native sparse spectra and matched intensity residuals.

    The lower panel is defined only on matched reference coordinates. Positional
    displacement is shown by connectors in the upper panel and is never turned
    into two artificial opposite intensity impulses.
    """
    resolved = resolve_theme(theme)
    if axes is None:
        figure, created = plt.subplots(2, 1, figsize=resolved.figure_size, dpi=resolved.figure_dpi, sharex=True, gridspec_kw={"height_ratios": (2.0, 1.0)})
        signal_axis, residual_axis = created
    else:
        signal_axis, residual_axis = axes; figure = signal_axis.figure
    reference_mz = np.asarray(reference_mz); reference_intensity = np.asarray(reference_intensity)
    candidate_mz = np.asarray(candidate_mz); candidate_intensity = np.asarray(candidate_intensity)
    signal_axis.vlines(reference_mz, 0.0, reference_intensity, color=resolved.input_color, linewidth=resolved.input_line_width, alpha=resolved.primary_alpha, label="reference (+)")
    color = resolved.color_for_model(candidate_label)
    signal_axis.vlines(candidate_mz, 0.0, -candidate_intensity, color=color, linewidth=resolved.reconstruction_line_width, alpha=resolved.secondary_alpha, label=f"{candidate_label} (-)")
    for reference_index, candidate_index in zip(match.matched_reference_indices, match.matched_candidate_indices):
        signal_axis.plot([reference_mz[reference_index], candidate_mz[candidate_index]], [reference_intensity[reference_index], -candidate_intensity[candidate_index]], color=resolved.baseline_color, alpha=0.12, linewidth=resolved.reference_line_width)
    residual = match.matched_reference_intensity - match.matched_candidate_intensity
    residual_axis.vlines(reference_mz[match.matched_reference_indices], 0.0, residual, color=resolved.residual_color, linewidth=resolved.residual_line_width, label="matched intensity residual")
    residual_axis.scatter(reference_mz[match.unmatched_reference_indices], reference_intensity[match.unmatched_reference_indices], marker="x", color=resolved.false_negative_color, s=resolved.marker_size ** 2, linewidths=resolved.marker_edge_width, alpha=resolved.marker_alpha, label="unmatched reference")
    residual_axis.scatter(candidate_mz[match.unmatched_candidate_indices], -candidate_intensity[match.unmatched_candidate_indices], marker="x", color=resolved.false_positive_color, s=resolved.marker_size ** 2, linewidths=resolved.marker_edge_width, alpha=resolved.marker_alpha, label="unmatched candidate")
    for axis in (signal_axis, residual_axis):
        axis.axhline(0.0, color=resolved.baseline_color, linewidth=resolved.reference_line_width, linestyle=resolved.baseline_line_style); axis.grid(resolved.grid_visible, axis=resolved.grid_axis, color=resolved.grid_color, alpha=resolved.grid_alpha, linewidth=resolved.grid_line_width); axis.legend(loc=resolved.legend_location, ncols=resolved.legend_columns, frameon=resolved.legend_frame, fontsize=resolved.legend_font_size)
    signal_axis.set_ylabel("Intensity"); residual_axis.set(xlabel="m/z", ylabel="Matched Δ intensity")
    return figure, (signal_axis, residual_axis)


def plot_sparse_spectrum_multi_match(
    reference_mz: np.ndarray,
    reference_intensity: np.ndarray,
    candidates: Mapping[str, tuple[np.ndarray, np.ndarray]],
    matches: Mapping[str, object],
    axes: Optional[tuple[Axes, Axes]] = None,
    theme: VisualizationTheme | str | None = None,
) -> tuple[Figure, tuple[Axes, Axes]]:
    """Plot one reference spectrum against several candidates on one mirrored plot.

    Same reference (+) axis as :func:`plot_sparse_spectrum_match`, but every candidate
    in ``candidates`` is mirrored (-) and colored by ``theme.color_for_model(label)`` on
    the *same* signal axis, and every candidate's matched-intensity residual is drawn in
    its own color on the *same* residual axis — so several methods' reconstructions of
    the same spectrum are directly comparable at a glance: same reference, same axes,
    only the candidate trace differs by color.

    Deliberately simpler than :func:`plot_sparse_spectrum_match` for the multi-candidate
    case: no per-pair connector lines and no unmatched-point markers, since N candidates
    worth of those would make the plot unreadable. Use
    :func:`plot_sparse_spectrum_match` instead when you need that point-by-point detail
    for a *single* candidate (there, "unmatched reference" marks reference peaks with no
    candidate within tolerance — i.e. lost signal; "unmatched candidate" marks candidate
    points with no reference peak within tolerance — i.e. spurious/extra signal).

    :param candidates: ``{label: (mz, intensity)}`` — one entry per method/parameter
        setting being compared, all against the same ``reference_mz``/``reference_intensity``.
    :param matches: ``{label: SpectralPointMatch}`` — the match for that candidate
        against the same reference (from ``metrics.match_spectral_points``), same keys
        as ``candidates``.
    """
    resolved = resolve_theme(theme)
    if axes is None:
        figure, created = plt.subplots(2, 1, figsize=resolved.figure_size, dpi=resolved.figure_dpi, sharex=True, gridspec_kw={"height_ratios": (2.0, 1.0)})
        signal_axis, residual_axis = created
    else:
        signal_axis, residual_axis = axes; figure = signal_axis.figure
    reference_mz = np.asarray(reference_mz); reference_intensity = np.asarray(reference_intensity)
    signal_axis.vlines(reference_mz, 0.0, reference_intensity, color=resolved.input_color, linewidth=resolved.input_line_width, alpha=resolved.primary_alpha, label="reference (+)")
    for index, (label, (candidate_mz, candidate_intensity)) in enumerate(candidates.items()):
        candidate_mz = np.asarray(candidate_mz); candidate_intensity = np.asarray(candidate_intensity)
        match = matches[label]
        color = resolved.color_for_model(label, index)
        signal_axis.vlines(candidate_mz, 0.0, -candidate_intensity, color=color, linewidth=resolved.reconstruction_line_width, alpha=resolved.overlapping_signal_alpha, label=f"{label} (-)")
        residual = match.matched_reference_intensity - match.matched_candidate_intensity
        residual_axis.vlines(reference_mz[match.matched_reference_indices], 0.0, residual, color=color, linewidth=resolved.residual_line_width, alpha=resolved.residual_alpha, label=f"{label} residual")
    for axis in (signal_axis, residual_axis):
        axis.axhline(0.0, color=resolved.baseline_color, linewidth=resolved.reference_line_width, linestyle=resolved.baseline_line_style); axis.grid(resolved.grid_visible, axis=resolved.grid_axis, color=resolved.grid_color, alpha=resolved.grid_alpha, linewidth=resolved.grid_line_width); axis.legend(loc=resolved.legend_location, ncols=resolved.legend_columns, frameon=resolved.legend_frame, fontsize=resolved.legend_font_size)
    signal_axis.set_ylabel("Intensity"); residual_axis.set(xlabel="m/z", ylabel="Matched Δ intensity")
    return figure, (signal_axis, residual_axis)


def plot_spectrum_comparison(
    mass_axis: np.ndarray,
    original: np.ndarray,
    reconstructions: Mapping[str, np.ndarray] | np.ndarray,
    axes: Optional[tuple[Axes, Axes]] = None,
    clip_reconstruction: bool = True,
    mirrored: bool = False,
    theme: VisualizationTheme | str | None = None,
) -> tuple[Figure, tuple[Axes, Axes]]:
    """Plot input/reconstructions above signed residuals on aligned axes.

    Negative decoder outputs can be clipped for physical display while signed
    residuals remain unmodified, preserving over- and under-estimation.

    :param mass_axis: Shared binner m/z axis.
    :type mass_axis: numpy.ndarray
    :param original: Input spectrum.
    :type original: numpy.ndarray
    :param reconstructions: Named model reconstructions or one unnamed array.
    :type reconstructions: Mapping[str, numpy.ndarray] | numpy.ndarray
    :param axes: Optional aligned signal and residual axes.
    :type axes: tuple[matplotlib.axes.Axes, matplotlib.axes.Axes] | None
    :param clip_reconstruction: Clip displayed reconstructions to zero.
    :type clip_reconstruction: bool
    :param theme: Global graphical strategy.
    :type theme: VisualizationTheme | str | None
    :return: Figure and ``(signal_axis, residual_axis)``.
    :rtype: tuple[matplotlib.figure.Figure, tuple[matplotlib.axes.Axes, matplotlib.axes.Axes]]
    """
    resolved = resolve_theme(theme)
    named = (
        {"reconstruction": np.asarray(reconstructions)}
        if isinstance(reconstructions, np.ndarray)
        else dict(reconstructions)
    )
    if axes is None:
        figure, created = plt.subplots(
            2,
            1,
            figsize=resolved.figure_size,
            dpi=resolved.figure_dpi,
            sharex=True,
            gridspec_kw={"height_ratios": (2.0, 1.0)},
        )
        signal_axis, residual_axis = created
    else:
        signal_axis, residual_axis = axes
        figure = signal_axis.figure
    signal_axis.plot(
        mass_axis,
        original,
        color=resolved.input_color,
        alpha=resolved.overlapping_signal_alpha,
        linewidth=resolved.input_line_width,
        label="input",
        zorder=resolved.input_zorder,
    )
    residual_limit = 0.0
    for index, (model_name, raw_reconstruction) in enumerate(named.items()):
        color = resolved.color_for_model(model_name, index)
        reconstruction = np.asarray(raw_reconstruction)
        displayed = np.clip(reconstruction, 0.0, None) if clip_reconstruction else reconstruction
        residual = np.asarray(original) - reconstruction
        residual_limit = max(residual_limit, float(np.max(np.abs(residual), initial=0.0)))
        signal_axis.plot(
            mass_axis,
            -displayed if mirrored else displayed,
            color=color,
            alpha=resolved.secondary_alpha,
            linewidth=resolved.reconstruction_line_width,
            label=f"{model_name} (-)" if mirrored else model_name,
            zorder=resolved.reconstruction_zorder,
        )
        residual_axis.plot(
            mass_axis,
            residual,
            color=color,
            alpha=resolved.residual_alpha,
            linewidth=resolved.residual_line_width,
            label=f"{model_name} residual",
            zorder=resolved.residual_zorder,
        )
    residual_axis.axhline(
        0.0,
        color=resolved.baseline_color,
        linewidth=resolved.reference_line_width,
        linestyle=resolved.baseline_line_style,
    )
    if residual_limit > 0:
        residual_axis.set_ylim(-residual_limit, residual_limit)
    signal_axis.set_ylabel("Intensity", fontsize=resolved.label_font_size)
    residual_axis.set(xlabel="m/z", ylabel="Input - reconstruction")
    signal_axis.legend(
        loc=resolved.legend_location,
        frameon=resolved.legend_frame,
        ncols=resolved.legend_columns,
        fontsize=resolved.legend_font_size,
    )
    residual_axis.legend(
        loc=resolved.legend_location,
        frameon=resolved.legend_frame,
        ncols=resolved.legend_columns,
        fontsize=resolved.legend_font_size,
    )
    for axis in (signal_axis, residual_axis):
        axis.set_facecolor(resolved.panel_color)
        axis.grid(resolved.grid_visible, axis=resolved.grid_axis, color=resolved.grid_color, alpha=resolved.grid_alpha, linewidth=resolved.grid_line_width)
        axis.tick_params(labelsize=resolved.tick_font_size)
    return figure, (signal_axis, residual_axis)
