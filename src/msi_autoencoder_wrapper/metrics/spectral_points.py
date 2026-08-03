"""Coordinate-aware matching and metrics for sparse spectra on distinct m/z axes."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
from scipy.stats import wasserstein_distance

from ..utils.exceptions import raise_validation_error

ToleranceUnit = Literal["Da", "ppm"]
MatchingStrategy = Literal["nearest", "one_to_one", "local_mass"]


@dataclass(frozen=True)
class SpectralPointMatch:
    """Store coordinate matches, unmatched points, and aligned local intensities."""

    matched_reference_indices: np.ndarray
    matched_candidate_indices: np.ndarray
    candidate_groups: tuple[np.ndarray, ...]
    mz_errors_da: np.ndarray
    mz_errors_ppm: np.ndarray
    unmatched_reference_indices: np.ndarray
    unmatched_candidate_indices: np.ndarray
    matched_reference_intensity: np.ndarray
    matched_candidate_intensity: np.ndarray


def _inputs(axis: np.ndarray, intensity: np.ndarray, name: str) -> tuple[np.ndarray, np.ndarray]:
    x, y = np.asarray(axis, dtype=np.float64), np.asarray(intensity, dtype=np.float64)
    if x.ndim != 1 or y.ndim != 1 or x.size != y.size or np.any(np.diff(x) < 0):
        raise_validation_error("SpectralPointMetric", f"{name} axis and intensity must be equal, one-dimensional, and sorted.")
    valid = np.isfinite(x) & np.isfinite(y)
    return x[valid], y[valid]


def match_spectral_points(reference_mz: np.ndarray, reference_intensity: np.ndarray, candidate_mz: np.ndarray, candidate_intensity: np.ndarray, tolerance: float, tolerance_unit: ToleranceUnit = "Da", matching_strategy: MatchingStrategy = "one_to_one") -> SpectralPointMatch:
    """Match two sorted sparse spectra without projecting them onto a shared grid."""
    if tolerance < 0 or tolerance_unit not in {"Da", "ppm"} or matching_strategy not in {"nearest", "one_to_one", "local_mass"}:
        raise_validation_error("SpectralPointMetric", "Invalid tolerance, tolerance unit, or matching strategy.")
    rx, ry = _inputs(reference_mz, reference_intensity, "Reference")
    cx, cy = _inputs(candidate_mz, candidate_intensity, "Candidate")
    radii = np.full(rx.size, tolerance) if tolerance_unit == "Da" else rx * tolerance * 1e-6
    pairs: list[tuple[int, int]] = []; groups: list[np.ndarray] = []
    if matching_strategy == "nearest":
        for index in range(rx.size):
            left = int(np.searchsorted(cx, rx[index] - radii[index], side="left")); right = int(np.searchsorted(cx, rx[index] + radii[index], side="right"))
            candidates = np.arange(left, right, dtype=int)
            if candidates.size:
                chosen = int(candidates[np.argmin(np.abs(cx[candidates] - rx[index]))]); pairs.append((index, chosen)); groups.append(np.asarray([chosen]))
    elif matching_strategy == "local_mass":
        for index in range(rx.size):
            left = int(np.searchsorted(cx, rx[index] - radii[index], side="left")); right = int(np.searchsorted(cx, rx[index] + radii[index], side="right"))
            candidates = np.arange(left, right, dtype=int)
            if candidates.size:
                chosen = int(candidates[np.argmin(np.abs(cx[candidates] - rx[index]))]); pairs.append((index, chosen)); groups.append(candidates)
    elif rx.size and cx.size:
        # Ordered greedy matching is linear-memory and deterministic. The next
        # admissible candidate is used once; local ties prefer smaller error.
        candidate_start = 0
        for index in range(rx.size):
            left = max(candidate_start, int(np.searchsorted(cx, rx[index] - radii[index], side="left")))
            right = int(np.searchsorted(cx, rx[index] + radii[index], side="right"))
            if left < right:
                candidates = np.arange(left, right, dtype=int); chosen = int(candidates[np.argmin(np.abs(cx[candidates] - rx[index]))])
                pairs.append((index, chosen)); groups.append(np.asarray([chosen])); candidate_start = chosen + 1
    ref = np.asarray([pair[0] for pair in pairs], dtype=int); cand = np.asarray([pair[1] for pair in pairs], dtype=int)
    da = cx[cand] - rx[ref] if ref.size else np.asarray([], dtype=float)
    matched_candidate = np.asarray([np.sum(cy[group]) for group in groups], dtype=float)
    used = np.unique(np.concatenate(groups)) if groups else np.asarray([], dtype=int)
    return SpectralPointMatch(ref, cand, tuple(groups), da, np.divide(da, rx[ref], out=np.zeros_like(da), where=rx[ref] != 0) * 1e6, np.setdiff1d(np.arange(rx.size), ref), np.setdiff1d(np.arange(cx.size), used), ry[ref], matched_candidate)


def spectral_point_metrics(reference_mz: np.ndarray, reference_intensity: np.ndarray, candidate_mz: np.ndarray, candidate_intensity: np.ndarray, match: SpectralPointMatch) -> dict[str, float]:
    """Calculate independent localization, coverage, intensity, TIC, and size metrics."""
    ry, cy = np.asarray(reference_intensity, dtype=float), np.asarray(candidate_intensity, dtype=float)
    absolute_da, absolute_ppm = np.abs(match.mz_errors_da), np.abs(match.mz_errors_ppm)
    result: dict[str, float] = {}
    for unit, values in (("da", absolute_da), ("ppm", absolute_ppm)):
        result.update({f"localization_mae_{unit}": float(np.mean(values)) if values.size else np.nan, f"localization_rmse_{unit}": float(np.sqrt(np.mean(values ** 2))) if values.size else np.nan, f"localization_median_{unit}": float(np.median(values)) if values.size else np.nan, **{f"localization_q{int(quantile * 100)}_{unit}": float(np.quantile(values, quantile)) if values.size else np.nan for quantile in (.9, .95, .99)}, f"localization_max_{unit}": float(np.max(values)) if values.size else np.nan})
    ref_total, cand_total = float(np.sum(ry)), float(np.sum(cy)); epsilon = np.finfo(float).eps
    used_candidates = np.unique(np.concatenate(match.candidate_groups)).size if match.candidate_groups else 0
    unmatched_reference = ry[match.unmatched_reference_indices]
    unmatched_candidate = cy[match.unmatched_candidate_indices]
    aligned_reference = np.concatenate((match.matched_reference_intensity, unmatched_reference, np.zeros(unmatched_candidate.size)))
    aligned_candidate = np.concatenate((match.matched_candidate_intensity, np.zeros(unmatched_reference.size), unmatched_candidate))
    denominator = float(np.linalg.norm(aligned_reference) * np.linalg.norm(aligned_candidate))
    cosine = float(np.dot(aligned_reference, aligned_candidate) / denominator) if denominator else 0.0
    cosine = float(np.clip(cosine, -1.0, 1.0))
    result.update({"peak_recall": match.matched_reference_indices.size / ry.size if ry.size else 1.0, "peak_precision": used_candidates / cy.size if cy.size else 1.0, "matched_intensity_fraction": float(np.sum(match.matched_reference_intensity)) / (ref_total + epsilon), "local_intensity_relative_l1": float(np.sum(np.abs(match.matched_reference_intensity - match.matched_candidate_intensity))) / (float(np.sum(match.matched_reference_intensity)) + epsilon), "cosine_similarity": cosine, "spectral_angle": float(np.arccos(cosine)), "tic_relative_error": abs(ref_total - cand_total) / (abs(ref_total) + epsilon), "size_ratio": cy.size / ry.size if ry.size else 0.0, "size_reduction": 1.0 - cy.size / ry.size if ry.size else 1.0})
    positive_reference = np.clip(ry, 0, None); positive_candidate = np.clip(cy, 0, None)
    result["wasserstein"] = float(wasserstein_distance(reference_mz, candidate_mz, u_weights=positive_reference, v_weights=positive_candidate)) if np.sum(positive_reference) > 0 and np.sum(positive_candidate) > 0 else np.nan
    return result
