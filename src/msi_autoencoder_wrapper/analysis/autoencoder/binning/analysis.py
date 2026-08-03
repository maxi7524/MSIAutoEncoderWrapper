"""Coordinate-aware forward and inverse binning analysis."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any, Optional

import matplotlib.pyplot as plt
import numpy as np

from ....binners.binners_strategies.linear_binner import LinearBinning
from ....metrics import match_spectral_points, spectral_point_metrics
from ....utils.exceptions import raise_validation_error
from ....visualization.metrics import plot_metric_distribution, plot_metric_tradeoff
from ....visualization.spectra import plot_sparse_spectrum_match
from ..reconstruction.metrics import reconstruction_metrics, summarize
from .metrics import normalize_intensity, summaries

BINNING_COMPARISONS = ("binned_original", "inverse_binned", "inverse_original")


class BinningAnalysis:
    """Orchestrate matching, metric records, sweeps, and views for MSI binning."""

    def __init__(self, owner: Any) -> None:
        self.owner = owner
        self._cache: dict[tuple[Any, ...], list[dict[str, Any]]] = {}

    def _selected(self, spectrum_ids: Optional[Sequence[int]]) -> np.ndarray:
        runtime = self.owner.models[self.owner.default_model_name]
        return self.owner.selected_ids(spectrum_ids, runtime.dataset)

    def representations(self, spectrum_id: int, binner: Any | None = None, inverse_binner: Any | None = None) -> dict[str, tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]]:
        """Return the three unprojected comparison pairs for one spectrum."""
        forward_binner = binner or self.owner.binner
        inverse = inverse_binner or getattr(self.owner.context, "inverse_binner", None)
        if inverse is None: raise_validation_error("BinningAnalysis", "The active context has no inverse binner.")
        raw_x, raw_y = (np.asarray(values, dtype=float) for values in self.owner.reader.GetSpectrum(int(spectrum_id)))
        grid_x = np.asarray(forward_binner.GetXAxis(), dtype=float); forward_y = np.asarray(forward_binner(xs=raw_x, ys=raw_y), dtype=float)
        inverse_x, inverse_y = (np.asarray(values, dtype=float) for values in inverse(forward_y))
        return {"binned_original": (raw_x, raw_y, grid_x, forward_y), "inverse_binned": (grid_x, forward_y, inverse_x, inverse_y), "inverse_original": (raw_x, raw_y, inverse_x, inverse_y)}

    def forward_representation(self, spectrum_id: int, binner: Any | None = None) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Return ``X`` and ``B(X)`` without resolving or invoking an inverse binner."""
        forward_binner = binner or self.owner.binner
        raw_x, raw_y = (np.asarray(values, dtype=float) for values in self.owner.reader.GetSpectrum(int(spectrum_id)))
        grid_x = np.asarray(forward_binner.GetXAxis(), dtype=float)
        forward_y = np.asarray(forward_binner(xs=raw_x, ys=raw_y), dtype=float)
        return raw_x, raw_y, grid_x, forward_y

    def records(self, spectrum_ids: Optional[Sequence[int]] = None, tolerance: float = 0.01, tolerance_unit: str = "Da", matching_strategy: str = "local_mass", normalizations: Sequence[str] = ("raw", "tic", "max"), binner: Any | None = None, inverse_binner: Any | None = None) -> list[dict[str, Any]]:
        """Return long-form per-spectrum records for every comparison and normalization."""
        selected = self._selected(spectrum_ids)
        key = (tuple(selected), tolerance, tolerance_unit, matching_strategy, tuple(normalizations), id(binner or self.owner.binner), id(inverse_binner or getattr(self.owner.context, "inverse_binner", None)))
        if key in self._cache: return self._cache[key]
        output: list[dict[str, Any]] = []
        for spectrum_id in selected:
            raw_x, raw_y = self.owner.reader.GetSpectrum(int(spectrum_id)); raw_x = np.asarray(raw_x); raw_y = np.asarray(raw_y)
            properties = {"tic": float(np.sum(raw_y)), "nonzero_point_count": int(np.count_nonzero(raw_y > 0)), "maximum_intensity": float(np.max(raw_y, initial=0.0)), "dominant_peak_fraction": float(np.max(raw_y, initial=0.0) / np.sum(raw_y)) if np.sum(raw_y) else 0.0, "mz_span": float(np.ptp(raw_x)) if len(raw_x) else 0.0}
            for comparison, (reference_x, reference_y, candidate_x, candidate_y) in self.representations(int(spectrum_id), binner, inverse_binner).items():
                diagnostics = getattr(inverse_binner or getattr(self.owner.context, "inverse_binner", None), "last_diagnostics", {}) if comparison != "binned_original" else {}
                for normalization in normalizations:
                    normalized_reference = normalize_intensity(reference_y, normalization); normalized_candidate = normalize_intensity(candidate_y, normalization)
                    match = match_spectral_points(reference_x, normalized_reference, candidate_x, normalized_candidate, tolerance, tolerance_unit, matching_strategy)
                    values = spectral_point_metrics(reference_x, normalized_reference, candidate_x, normalized_candidate, match)
                    # TIC is always evaluated in source intensity space.
                    values["tic_relative_error"] = abs(float(np.sum(reference_y)) - float(np.sum(candidate_y))) / (abs(float(np.sum(reference_y))) + np.finfo(float).eps)
                    forward_component = binner or self.owner.binner
                    inverse_component = inverse_binner or getattr(self.owner.context, "inverse_binner", None)
                    parameters = {"tolerance": float(tolerance), "tolerance_unit": tolerance_unit, "matching_strategy": matching_strategy, "forward_binner": type(forward_component).__name__, "forward_config": repr(forward_component.get_config()), "inverse_binner": type(inverse_component).__name__ if comparison != "binned_original" else None, "inverse_config": repr(inverse_component.get_config()) if comparison != "binned_original" else None}
                    for metric, value in values.items(): output.append({"spectrum_id": int(spectrum_id), "comparison": comparison, "normalization": normalization, "metric": metric, "value": float(value), **parameters, **properties, **diagnostics})
        self._cache[key] = output
        return output

    def metric_summaries(self, *args: Any, **kwargs: Any) -> list[dict[str, Any]]:
        """Summarize every metric distribution from :meth:`records`."""
        if "include_masserstein" in kwargs:
            results = self.spectrum_metrics(*args, **kwargs)
            return [{"comparison": comparison, "metric": metric, **summarize(np.asarray([record[metric] for record in spectra.values()]))} for comparison, spectra in results.items() for metric in next(iter(spectra.values())).keys()]
        return summaries(self.records(*args, **kwargs))

    def forward_report(self, spectrum_ids: Optional[Sequence[int]] = None, **kwargs: Any) -> Mapping[str, float]:
        """Compatibility report restricted to forward-binner comparisons."""
        finite, ratios = [], []
        for spectrum_id in self._selected(spectrum_ids):
            raw_x, raw_y = self.owner.reader.GetSpectrum(int(spectrum_id)); values = np.asarray(self.owner.binner(xs=raw_x, ys=raw_y), dtype=float); total = float(np.sum(raw_y))
            finite.append(float(np.mean(np.isfinite(values)))); ratios.append(float(np.sum(values)) / total if total else np.nan)
        return {"spectrum_count": float(len(finite)), "finite_fraction": float(np.mean(finite)), "mean_tic_ratio": float(np.nanmean(ratios)), "min_tic_ratio": float(np.nanmin(ratios)), "max_tic_ratio": float(np.nanmax(ratios))}

    def inverse_report(self, spectrum_ids: Optional[Sequence[int]] = None, **kwargs: Any) -> Mapping[str, float]:
        """Compatibility report restricted to inverse-binner comparisons."""
        mse_values, mae_values, ratios = [], [], []
        inverse = getattr(self.owner.context, "inverse_binner", None)
        for spectrum_id in self._selected(spectrum_ids):
            _, _, _, forward = self.representations(int(spectrum_id))["binned_original"]
            inverse_x, inverse_y = inverse(forward); round_trip = np.asarray(self.owner.binner(xs=inverse_x, ys=inverse_y), dtype=float); residual = forward - round_trip; total = float(np.sum(forward))
            mse_values.append(float(np.mean(residual ** 2))); mae_values.append(float(np.mean(np.abs(residual)))); ratios.append(float(np.sum(round_trip)) / total if total else np.nan)
        return {"spectrum_count": float(len(mse_values)), "mean_mse": float(np.mean(mse_values)), "mean_mae": float(np.mean(mae_values)), "mean_tic_ratio": float(np.nanmean(ratios))}

    def overview(self, spectrum_ids: Optional[Sequence[int]] = None, **kwargs: Any) -> Mapping[str, Any]:
        """Return separate forward and inverse summaries."""
        return {"forward": self.forward_report(spectrum_ids, **kwargs), "inverse": self.inverse_report(spectrum_ids, **kwargs)}

    def plot_metric_distributions(self, metric: str, normalization: str = "raw", spectrum_ids: Optional[Sequence[int]] = None, kind: str = "histogram", bins: int = 40, **kwargs: Any):
        """Render one panel per mathematically distinct comparison."""
        records = self.records(spectrum_ids, **kwargs); figure, axes = plt.subplots(1, 3, figsize=(18, 5), dpi=self.owner.theme.figure_dpi, squeeze=False)
        for index, comparison in enumerate(BINNING_COMPARISONS):
            values = [record["value"] for record in records if record["comparison"] == comparison and record["normalization"] == normalization and record["metric"] == metric]
            plot_metric_distribution(np.asarray(values), metric, axes[0, index], bins, kind, comparison, self.owner.theme); axes[0, index].set_title(comparison)
        figure.tight_layout(); return figure, axes.ravel()

    def plot_spectrum(self, spectrum_id: int, comparison: str = "inverse_original", inverse_binner: Any | None = None, tolerance: float = 0.01, tolerance_unit: str = "Da", matching_strategy: str = "local_mass"):
        """Plot native sparse coordinates and a matching-aware intensity residual."""
        if comparison not in BINNING_COMPARISONS: raise_validation_error("BinningAnalysis", f"Unknown comparison '{comparison}'.")
        rx, ry, cx, cy = self.representations(spectrum_id, inverse_binner=inverse_binner)[comparison]
        match = match_spectral_points(rx, ry, cx, cy, tolerance, tolerance_unit, matching_strategy)
        return plot_sparse_spectrum_match(rx, ry, cx, cy, match, candidate_label=comparison, theme=self.owner.theme)

    def spectrum_metrics(self, spectrum_ids: Optional[Sequence[int]] = None, include_masserstein: bool = False, **_: Any) -> dict[str, dict[int, dict[str, float]]]:
        """Compatibility dense diagnostics; use :meth:`records` for quality analysis."""
        aliases = {"binned_original": "forward_original", "inverse_binned": "inverse_forward", "inverse_original": "inverse_original"}; output = {name: {} for name in aliases.values()}
        for spectrum_id in self._selected(spectrum_ids):
            for comparison, (rx, ry, cx, cy) in self.representations(int(spectrum_id)).items():
                axis = np.unique(np.concatenate((rx, cx))); reference = np.interp(axis, rx, ry, left=0, right=0); candidate = np.interp(axis, cx, cy, left=0, right=0) if cx.size else np.zeros_like(axis)
                values = reconstruction_metrics(reference[None, :], candidate[None, :])
                output[aliases[comparison]][int(spectrum_id)] = {metric: float(values[metric][0]) for metric in ("mse", "mae", "cosine_similarity", "spectral_angle", "tic_error")}
        return output

    def feature_distribution(self, comparison: str, spectrum_ids: Optional[Sequence[int]] = None, metric: str = "mse", **_: Any) -> dict[str, np.ndarray]:
        """Compatibility report-grid profile; core metrics remain projection-free."""
        comparison = {"forward_original": "binned_original", "inverse_forward": "inverse_binned"}.get(comparison, comparison); axis = np.asarray(self.owner.binner.GetXAxis()); contributions = []
        for spectrum_id in self._selected(spectrum_ids):
            rx, ry, cx, cy = self.representations(int(spectrum_id))[comparison]; residual = np.interp(axis, rx, ry, left=0, right=0) - (np.interp(axis, cx, cy, left=0, right=0) if cx.size else 0)
            contributions.append(residual ** 2 if metric == "mse" else np.abs(residual))
        values = np.asarray(contributions)
        return {"mean": np.mean(values, axis=0), "median": np.median(values, axis=0), "lower": np.quantile(values, .05, axis=0), "upper": np.quantile(values, .95, axis=0)}

    def forward_sweep(self, bin_steps: Sequence[float], spectrum_ids: Optional[Sequence[int]] = None, baseline_step: float | None = None, **kwargs: Any) -> list[dict[str, Any]]:
        """Evaluate forward-binner resolution trade-offs through the same record pipeline."""
        if not bin_steps or any(step <= 0 for step in bin_steps): raise_validation_error("BinningAnalysis", "bin_steps must contain positive values.")
        baseline = baseline_step or min(bin_steps); output: list[dict[str, Any]] = []
        tolerance = float(kwargs.pop("tolerance", 0.01)); tolerance_unit = kwargs.pop("tolerance_unit", "Da"); matching_strategy = kwargs.pop("matching_strategy", "local_mass"); normalizations = kwargs.pop("normalizations", ("raw", "tic", "max"))
        if kwargs: raise_validation_error("BinningAnalysis", f"Unknown forward sweep parameters: {sorted(kwargs)}.")
        for step in bin_steps:
            binner = LinearBinning(step, self.owner.binner.GetXMin(), self.owner.binner.GetXMax())
            records: list[dict[str, Any]] = []
            for spectrum_id in self._selected(spectrum_ids):
                rx, ry, cx, cy = self.forward_representation(int(spectrum_id), binner)
                for normalization in normalizations:
                    normalized_reference = normalize_intensity(ry, normalization); normalized_candidate = normalize_intensity(cy, normalization)
                    match = match_spectral_points(rx, normalized_reference, cx, normalized_candidate, tolerance, tolerance_unit, matching_strategy)
                    values = spectral_point_metrics(rx, normalized_reference, cx, normalized_candidate, match)
                    values["tic_relative_error"] = abs(float(np.sum(ry)) - float(np.sum(cy))) / (abs(float(np.sum(ry))) + np.finfo(float).eps)
                    for metric, value in values.items(): records.append({"spectrum_id": int(spectrum_id), "comparison": "binned_original", "normalization": normalization, "metric": metric, "value": float(value), "tolerance": tolerance, "tolerance_unit": tolerance_unit, "matching_strategy": matching_strategy, "forward_binner": type(binner).__name__, "forward_config": repr(binner.get_config()), "inverse_binner": None, "inverse_config": None})
            dimension = binner.GetXAxisDepth()
            for record in summaries(records): output.append({**record, "bin_step": float(step), "feature_dimension": dimension, "dimension_reduction": 1.0 - baseline / float(step)})
        return output

    def plot_ranked_spectra(self, metrics: Sequence[str], comparison: str, normalization: str, spectrum_ids: Optional[Sequence[int]] = None, inverse_binner: Any | None = None, tolerance: float = 0.01, tolerance_unit: str = "Da", matching_strategy: str = "local_mass"):
        """Plot best, median, and worst spectra independently for every metric."""
        records = self.records(spectrum_ids, tolerance, tolerance_unit, matching_strategy, (normalization,), inverse_binner=inverse_binner)
        figure, axes = plt.subplots(2 * len(metrics), 3, figsize=(18, 7 * len(metrics)), dpi=self.owner.theme.figure_dpi, squeeze=False)
        maximize = {"cosine_similarity", "peak_precision", "peak_recall", "matched_intensity_fraction", "size_reduction"}
        for metric_index, metric in enumerate(metrics):
            selected = [record for record in records if record["comparison"] == comparison and record["metric"] == metric and np.isfinite(record["value"])]
            ordered = sorted(selected, key=lambda record: record["value"], reverse=metric in maximize)
            choices = (ordered[0], ordered[len(ordered) // 2], ordered[-1])
            for column, (label, record) in enumerate(zip(("best", "median", "worst"), choices)):
                rx, ry, cx, cy = self.representations(record["spectrum_id"], inverse_binner=inverse_binner)[comparison]
                match = match_spectral_points(rx, ry, cx, cy, tolerance, tolerance_unit, matching_strategy)
                plot_sparse_spectrum_match(rx, ry, cx, cy, match, axes=(axes[2 * metric_index, column], axes[2 * metric_index + 1, column]), candidate_label=comparison, theme=self.owner.theme)
                axes[2 * metric_index, column].set_title(f"{metric}: {label} | spectrum {record['spectrum_id']} | {record['value']:.4g}")
        figure.tight_layout(); return figure, axes

    def plot_tradeoff(self, records: Sequence[Mapping[str, Any]], metric: str, x: str = "bin_step", normalization: str = "raw"):
        """Plot one selected sweep metric without combining metric semantics."""
        selected = [record for record in records if record["metric"] == metric and record["normalization"] == normalization]
        return plot_metric_tradeoff([record[x] for record in selected], [record["median"] for record in selected], x, metric, theme=self.owner.theme)
