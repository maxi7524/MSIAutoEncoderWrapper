"""m/z-localization profile: where along the mass axis does error concentrate?

Pools matched-pair localization error (Da and ppm) and the unmatched-reference
fraction across every sampled spectrum, bucketed by m/z position into fixed-width
windows — a coarser, purely-for-aggregation/visualization binning than the analysis's
own Δm, never used as a basis for matching itself (matching stays position/tolerance
based throughout, per ``methodology.md`` §1). See ``methodology.md`` §6 step 6.
"""

from __future__ import annotations

from typing import Any, Mapping, Optional, Sequence

import matplotlib.pyplot as plt
import numpy as np

from ....metrics import match_spectral_points
from ....utils.logger import get_custom_logger
from ....visualization import VisualizationTheme, resolve_theme
from .binner_forward_analysis import crop_spectrum
from .inverse_binner_analysis import inverse_binner_factory
from .precompute import BinningPrecompute

logger = get_custom_logger(__name__)


def localization_profile(
    precompute: BinningPrecompute,
    method_grid_point: dict[str, Any],
    comparison: str,
    delta_m: float,
    mz_bin_width: float = 50.0,
    tolerance: float = 0.01,
    tolerance_unit: str = "Da",
    x_min: Optional[float] = None,
    x_max: Optional[float] = None,
) -> list[dict[str, Any]]:
    """Pool every sampled spectrum's matched-pair localization error, bucketed by
    reference m/z position into ``mz_bin_width``-wide windows.

    :param method_grid_point: The ``{"label", "method", "params"}`` dict for the
        inverse-binner configuration to profile (same shape as
        ``inverse_binner_analysis.inverse_sweep_records``'s ``method_grid`` entries).
    :param comparison: ``"inverse_binned"`` (reference = B(X)) or ``"inverse_original"``
        (reference = raw X).
    :return: One row per m/z bin: ``mz_bin_start``, ``mz_bin_end``, ``mz_bin_mid``,
        ``reference_point_count`` (all reference points in the bin, matched or not),
        ``matched_count``, ``unmatched_fraction`` (unmatched reference points / all
        reference points in the bin — "how much signal near this m/z gets lost"),
        ``median_da``/``q95_da``/``median_ppm``/``q95_ppm`` (over *matched* points only).
        Bins with zero reference points are omitted.
    """
    label, method, params = method_grid_point["label"], method_grid_point["method"], dict(method_grid_point.get("params", {}))
    forward_binner, inverse_binner, inverse_cache = precompute.inverse(delta_m, inverse_binner_factory(method, **params), x_min, x_max, cache_key=label)
    _, forward_cache = precompute.forward(delta_m, x_min, x_max)
    grid_mz = np.asarray(forward_binner.GetXAxis(), dtype=np.float64)
    logger.info("Building localization profile for label=%s, comparison=%s, bin width=%s.", label, comparison, mz_bin_width)

    all_reference_mz: list[np.ndarray] = []
    matched_reference_mz: list[np.ndarray] = []
    matched_da: list[np.ndarray] = []
    matched_ppm: list[np.ndarray] = []
    for spectrum_id in precompute.spectrum_ids:
        raw_mz, raw_y = crop_spectrum(*precompute.raw(spectrum_id), forward_binner.x_min, forward_binner.x_max)
        grid_y = forward_cache[int(spectrum_id)]
        result = inverse_cache[int(spectrum_id)]
        reference_mz, reference_y = (grid_mz, grid_y) if comparison == "inverse_binned" else (raw_mz, raw_y)
        match = match_spectral_points(reference_mz, reference_y, result.mz, result.intensity, tolerance, tolerance_unit, "one_to_one")
        all_reference_mz.append(reference_mz)
        if match.matched_reference_indices.size:
            matched_reference_mz.append(reference_mz[match.matched_reference_indices])
            matched_da.append(np.abs(match.mz_errors_da))
            matched_ppm.append(np.abs(match.mz_errors_ppm))

    pooled_all_mz = np.concatenate(all_reference_mz) if all_reference_mz else np.asarray([])
    pooled_matched_mz = np.concatenate(matched_reference_mz) if matched_reference_mz else np.asarray([])
    pooled_da = np.concatenate(matched_da) if matched_da else np.asarray([])
    pooled_ppm = np.concatenate(matched_ppm) if matched_ppm else np.asarray([])

    lo, hi = forward_binner.x_min, forward_binner.x_max
    edges = np.arange(np.floor(lo / mz_bin_width) * mz_bin_width, hi + mz_bin_width, mz_bin_width)
    records: list[dict[str, Any]] = []
    for start, end in zip(edges[:-1], edges[1:]):
        all_mask = (pooled_all_mz >= start) & (pooled_all_mz < end)
        reference_count = int(np.sum(all_mask))
        if reference_count == 0:
            continue
        matched_mask = (pooled_matched_mz >= start) & (pooled_matched_mz < end)
        matched_count = int(np.sum(matched_mask))
        bin_da = pooled_da[matched_mask]
        bin_ppm = pooled_ppm[matched_mask]
        records.append({
            "mz_bin_start": float(start), "mz_bin_end": float(end), "mz_bin_mid": float((start + end) / 2),
            "reference_point_count": reference_count, "matched_count": matched_count,
            "unmatched_fraction": 1.0 - matched_count / reference_count,
            "median_da": float(np.median(bin_da)) if bin_da.size else np.nan,
            "q95_da": float(np.quantile(bin_da, 0.95)) if bin_da.size else np.nan,
            "median_ppm": float(np.median(bin_ppm)) if bin_ppm.size else np.nan,
            "q95_ppm": float(np.quantile(bin_ppm, 0.95)) if bin_ppm.size else np.nan,
        })
    return records


def localization_profiles(
    precompute: BinningPrecompute,
    method_grid: Sequence[dict[str, Any]],
    comparison: str,
    delta_m: float,
    mz_bin_width: float = 50.0,
    tolerance: float = 0.01,
    tolerance_unit: str = "Da",
    x_min: Optional[float] = None,
    x_max: Optional[float] = None,
) -> dict[str, list[dict[str, Any]]]:
    """:func:`localization_profile` for every entry in ``method_grid`` at once, so
    several methods'/parameters' error-vs-m/z and unmatched-vs-m/z behavior can be
    compared on one plot instead of only ever looking at one config
    (``CHOSEN_INVERSE_CONFIG``) in isolation.

    :return: ``{label: profile}``, same per-label schema as :func:`localization_profile`.
    """
    return {
        point["label"]: localization_profile(precompute, point, comparison, delta_m, mz_bin_width, tolerance, tolerance_unit, x_min, x_max)
        for point in method_grid
    }


def plot_localization_profile(
    profile: list[dict[str, Any]],
    unit: str = "da",
    statistic: str = "median",
    ax=None,
    label: Optional[str] = None,
    color: Optional[str] = None,
    theme: VisualizationTheme | str | None = None,
):
    """Plot error-vs-m/z-position (``median_{unit}``/``q95_{unit}``) from
    :func:`localization_profile`, one line — call twice with different ``statistic`` on
    the same ``ax`` to overlay median and q95, or use :func:`plot_localization_profiles`
    to overlay several configs (one color per config) automatically.

    :param label: Legend label; defaults to ``"{statistic} |error| ({unit})"``. Set this
        explicitly (together with ``color``) when overlaying more than one profile on
        the same ``ax`` so the legend distinguishes them.
    :param color: Line color; defaults to matplotlib's automatic cycling.
    """
    resolved = resolve_theme(theme)
    if ax is None:
        figure, ax = plt.subplots(figsize=resolved.figure_size, dpi=resolved.figure_dpi)
    else:
        figure = ax.figure
    xs = [record["mz_bin_mid"] for record in profile]
    ys = [record[f"{statistic}_{unit}"] for record in profile]
    ax.plot(xs, ys, marker="o", color=color, alpha=resolved.primary_alpha, label=label or f"{statistic} |error| ({unit})")
    ax.set(xlabel="m/z", ylabel=f"localization error ({unit})")
    ax.grid(resolved.grid_visible, alpha=resolved.grid_alpha)
    ax.legend(fontsize=resolved.tick_font_size, loc=resolved.legend_location, frameon=resolved.legend_frame)
    figure.tight_layout()
    return figure, ax


def plot_localization_profiles(
    profiles: Mapping[str, list[dict[str, Any]]],
    unit: str = "da",
    statistic: str = "median",
    ax=None,
    theme: VisualizationTheme | str | None = None,
):
    """:func:`plot_localization_profile` for every ``{label: profile}`` entry overlaid
    on one plot, one color per label (``theme.color_for_model`` — same color convention,
    and the same color for a given label, as every other plot in this package)."""
    resolved = resolve_theme(theme)
    figure = None
    for index, (label, profile) in enumerate(profiles.items()):
        figure, ax = plot_localization_profile(profile, unit, statistic, ax, label=f"{label} ({statistic})", color=resolved.color_for_model(label, index), theme=resolved)
    return figure, ax


def plot_unmatched_fraction_profile(
    profile: list[dict[str, Any]],
    ax=None,
    label: Optional[str] = None,
    color: Optional[str] = None,
    theme: VisualizationTheme | str | None = None,
):
    """Plot the fraction of reference points left unmatched, vs m/z position.

    :param label: Legend label; defaults to ``"unmatched reference fraction"``.
    :param color: Line color; defaults to ``theme.residual_color``.
    """
    resolved = resolve_theme(theme)
    if ax is None:
        figure, ax = plt.subplots(figsize=resolved.figure_size, dpi=resolved.figure_dpi)
    else:
        figure = ax.figure
    xs = [record["mz_bin_mid"] for record in profile]
    ys = [record["unmatched_fraction"] for record in profile]
    ax.plot(xs, ys, marker="o", color=color or resolved.residual_color, alpha=resolved.primary_alpha, label=label or "unmatched reference fraction")
    ax.set(xlabel="m/z", ylabel="unmatched fraction")
    ax.set_ylim(0.0, 1.0)
    ax.grid(resolved.grid_visible, alpha=resolved.grid_alpha)
    ax.legend(fontsize=resolved.tick_font_size, loc=resolved.legend_location, frameon=resolved.legend_frame)
    figure.tight_layout()
    return figure, ax


def plot_unmatched_fraction_profiles(
    profiles: Mapping[str, list[dict[str, Any]]],
    ax=None,
    theme: VisualizationTheme | str | None = None,
):
    """:func:`plot_unmatched_fraction_profile` for every ``{label: profile}`` entry
    overlaid on one plot, one color per label (same convention as
    :func:`plot_localization_profiles`)."""
    resolved = resolve_theme(theme)
    figure = None
    for index, (label, profile) in enumerate(profiles.items()):
        figure, ax = plot_unmatched_fraction_profile(profile, ax, label=label, color=resolved.color_for_model(label, index), theme=resolved)
    return figure, ax
