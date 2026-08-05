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
    r"""Match two sorted sparse spectra by m/z proximity.

    Every reference point independently searches for admissible candidate points 
    inside a tolerance window centered on its own m/z, and how that window's candidates are resolved into pairs is controlled by ``matching_strategy``.

    :param reference_mz: Sorted ascending m/z values of the reference spectrum.
    :type reference_mz: numpy.ndarray
    :param reference_intensity: Intensities aligned with ``reference_mz``.
    :type reference_intensity: numpy.ndarray
    :param candidate_mz: Sorted ascending m/z values of the candidate spectrum.
    :type candidate_mz: numpy.ndarray
    :param candidate_intensity: Intensities aligned with ``candidate_mz``.
    :type candidate_intensity: numpy.ndarray
    :param tolerance: Non-negative window half-width, in the unit selected by
        ``tolerance_unit``.
    :type tolerance: float
    :param tolerance_unit: Controls only the *search radius* used to decide which
        candidates are admissible for a given reference point — it does not change
        which error columns end up populated on the result (both ``mz_errors_da`` and
        ``mz_errors_ppm`` are always computed for every match, regardless of which unit
        was used to find it). Per reference point :math:`i` with m/z :math:`m_i`, the
        radius :math:`r_i` is:

        - ``"Da"`` — absolute, constant radius: :math:`r_i = \text{tolerance}` for
          every reference point, independent of its m/z. Matches the geometry of a
          fixed-step binning grid, where the admissible offset from a bin center is
          identical everywhere on the axis.
        - ``"ppm"`` — relative radius that grows with the reference point's own m/z:
          :math:`r_i = m_i \cdot \text{tolerance} \times 10^{-6}`. Matches instrument
          mass accuracy, which is conventionally specified in ppm and therefore
          corresponds to a wider *absolute* (Da) window at higher m/z — e.g. a 10 ppm
          tolerance is 0.001 Da at m/z 100 but 0.01 Da at m/z 1000.
    :type tolerance_unit: ToleranceUnit
    :param matching_strategy: How the candidates admissible inside one reference
        point's window are turned into a matched pair:

        - ``"nearest"`` — every reference point independently keeps its single closest
          admissible candidate. A candidate may be claimed by more than one reference
          point (no exclusivity between reference points). Diagnostic only: never use
          it for a reported metric, since one candidate counted against several
          reference points inflates apparent coverage.
        - ``"one_to_one"`` — same "closest candidate wins" rule as ``"nearest"``, but
          candidates already claimed by an earlier (lower m/z) reference point are
          excluded from later windows, so every candidate is used by at most one
          reference point. This is a genuine bijective partial matching (no double
          counting) and is the strategy every quantitative metric in this module is
          computed with. It is implemented as one linear left-to-right sweep over the
          sorted axes (``candidate_start`` only ever advances) rather than a full
          assignment-problem solver — exact here because the exclusion order matches
          the order the reference axis is walked in.
        - ``"local_mass"`` — every candidate inside a reference point's window is kept,
          not just the nearest one, and their intensities are **summed** into that
          reference point's matched candidate intensity
          (``matched_candidate_intensity``); the reported *positional* error is still
          measured to the single nearest candidate only (``mz_errors_da``/
          ``mz_errors_ppm``). Models a many-candidates-to-one-reference relationship,
          e.g. one coarse forward-binned point receiving summed mass from several
          nearby raw peaks, or (with the reference/candidate arguments swapped) how
          ``peak_collision_rate`` collects every reference peak feeding one output bin.
    :type matching_strategy: MatchingStrategy
    :return: Matched/unmatched indices on both sides and per-pair m/z errors in both
        units, aligned to the matched reference points; see :class:`SpectralPointMatch`.
    :rtype: SpectralPointMatch
    :raises ValidationError: If ``tolerance`` is negative, or ``tolerance_unit``/
        ``matching_strategy`` is not one of the documented values.
    """
    if tolerance < 0 or tolerance_unit not in {"Da", "ppm"} or matching_strategy not in {"nearest", "one_to_one", "local_mass"}:
        raise_validation_error("SpectralPointMetric", "Invalid tolerance, tolerance unit, or matching strategy.")
    rx, ry = _inputs(reference_mz, reference_intensity, "Reference")
    cx, cy = _inputs(candidate_mz, candidate_intensity, "Candidate")
    # Search radius per reference point: constant in Da, or scaled by each point's own
    # m/z in ppm (see the tolerance_unit branches documented above).
    radii = np.full(rx.size, tolerance) if tolerance_unit == "Da" else rx * tolerance * 1e-6
    pairs: list[tuple[int, int]] = []; groups: list[np.ndarray] = []
    if matching_strategy == "nearest":
        ## nearest: closest admissible candidate per reference point; candidates may repeat across reference points.
        for index in range(rx.size):
            left = int(np.searchsorted(cx, rx[index] - radii[index], side="left")); right = int(np.searchsorted(cx, rx[index] + radii[index], side="right"))
            candidates = np.arange(left, right, dtype=int)
            if candidates.size:
                chosen = int(candidates[np.argmin(np.abs(cx[candidates] - rx[index]))]); pairs.append((index, chosen)); groups.append(np.asarray([chosen]))
    elif matching_strategy == "local_mass":
        ## local_mass: keep every admissible candidate (not just the nearest); their intensities are summed later, position error still comes from the nearest one.
        for index in range(rx.size):
            left = int(np.searchsorted(cx, rx[index] - radii[index], side="left")); right = int(np.searchsorted(cx, rx[index] + radii[index], side="right"))
            candidates = np.arange(left, right, dtype=int)
            if candidates.size:
                chosen = int(candidates[np.argmin(np.abs(cx[candidates] - rx[index]))]); pairs.append((index, chosen)); groups.append(candidates)
    elif rx.size and cx.size:
        ## one_to_one: closest admissible candidate per reference point, each candidate excluded from later windows once claimed.
        # Ordered greedy matching is linear-memory and deterministic. The next
        # admissible candidate is used once; local ties prefer smaller error.
        candidate_start = 0
        for index in range(rx.size):
            left = max(candidate_start, int(np.searchsorted(cx, rx[index] - radii[index], side="left")))
            right = int(np.searchsorted(cx, rx[index] + radii[index], side="right"))
            if left < right:
                candidates = np.arange(left, right, dtype=int); chosen = int(candidates[np.argmin(np.abs(cx[candidates] - rx[index]))])
                pairs.append((index, chosen)); groups.append(np.asarray([chosen])); candidate_start = chosen + 1
    # Assemble the match: signed Da/ppm error per pair, matched/unmatched indices on
    # both sides, and (local_mass-summed) intensity aligned to each matched reference point.
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
    result["unmatched_candidate_intensity_fraction"] = float(np.sum(unmatched_candidate)) / (cand_total + epsilon)
    return result


def peak_collision_rate(reference_mz: np.ndarray, reference_intensity: np.ndarray, candidate_mz: np.ndarray, candidate_intensity: np.ndarray, tolerance: float, tolerance_unit: ToleranceUnit = "Da", min_relative_height: float = 0.0) -> float:
    """Fraction of candidate points fed by >=2 reference peaks more than one tolerance width apart.

    Roles are matched with candidate as the anchor (``local_mass``, swapped): every
    candidate point collects the reference peaks that fall within its tolerance window.
    A collision is a candidate point whose contributing reference peaks span more than
    the local tolerance radius, i.e. peaks that would not have merged had the candidate
    resolved them individually. This captures loss of resolution that a small
    localization error can otherwise hide (two source peaks collapsed into one output
    point, but the output point still sits close to at least one of them).

    :param min_relative_height: Drop contributors below this fraction of the tallest
        contributor before checking for a collision, to ignore negligible/noise peaks.
    """
    rx = np.asarray(reference_mz, dtype=np.float64); ry = np.asarray(reference_intensity, dtype=np.float64)
    cx = np.asarray(candidate_mz, dtype=np.float64)
    match = match_spectral_points(cx, np.asarray(candidate_intensity), rx, ry, tolerance, tolerance_unit, "local_mass")
    radii = np.full(cx.size, tolerance) if tolerance_unit == "Da" else cx * tolerance * 1e-6
    contributing = [group for group in match.candidate_groups if group.size]
    if not contributing:
        return 0.0
    collisions = 0
    for candidate_index, group in zip(match.matched_reference_indices, match.candidate_groups):
        if group.size < 2:
            continue
        heights = ry[group]
        if min_relative_height > 0.0 and heights.size:
            group = group[heights >= min_relative_height * np.max(heights)]
        if group.size < 2:
            continue
        span = float(np.max(rx[group]) - np.min(rx[group]))
        if span > radii[candidate_index]:
            collisions += 1
    return collisions / len(contributing)


def signal_retention_by_quantile(reference_intensity: np.ndarray, match: SpectralPointMatch, quantiles: tuple[float, ...] = (0.5, 0.9, 0.95, 0.99)) -> dict[str, dict[str, float]]:
    """Report, per signal quantile, how many top-intensity reference points it takes to
    reach that quantile and what fraction of those points survived matching.

    A single unmatched-intensity ratio cannot distinguish "we lost a few dominant peaks"
    from "we lost a long tail of noise." Low recall at a low quantile (few points, most
    of the signal) means real peaks are missing; recall dropping only near q=0.99 means
    the loss is concentrated in low-intensity, likely-noise points.
    """
    ry = np.asarray(reference_intensity, dtype=np.float64)
    matched_mask = np.zeros(ry.size, dtype=bool)
    if match.matched_reference_indices.size:
        matched_mask[match.matched_reference_indices] = True
    total = float(np.sum(ry))
    if ry.size == 0 or total <= 0:
        return {f"q{int(quantile * 100)}": {"k_peaks": 0, "recall_at_quantile": np.nan} for quantile in quantiles}
    order = np.argsort(ry)[::-1]
    cumulative = np.cumsum(ry[order])
    result: dict[str, dict[str, float]] = {}
    for quantile in quantiles:
        k = min(ry.size, int(np.searchsorted(cumulative, quantile * total, side="left") + 1))
        top_indices = order[:k]
        result[f"q{int(quantile * 100)}"] = {"k_peaks": int(k), "recall_at_quantile": float(np.mean(matched_mask[top_indices]))}
    return result
