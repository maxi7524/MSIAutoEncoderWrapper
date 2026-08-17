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
    x, y = np.asarray(axis, dtype=np.float32), np.asarray(intensity, dtype=np.float32)
    if x.ndim != 1 or y.ndim != 1 or x.size != y.size or np.any(np.diff(x) < 0):
        raise_validation_error("SpectralPointMetric", f"{name} axis and intensity must be equal, one-dimensional, and sorted.")
    valid = np.isfinite(x) & np.isfinite(y)
    return x[valid], y[valid]


def _one_to_one_matches(rx: np.ndarray, radii: np.ndarray, cx: np.ndarray) -> list[tuple[int, int]]:
    """Match every reference point to at most one candidate, most-constrained
    reference points first, resolved in vectorized rounds.

    Each round: for every reference point that still has at least one *available*
    (not yet claimed by an earlier round) candidate in its tolerance window, compute
    that window against the current candidate pool in bulk (``searchsorted`` on the
    whole remaining array at once, not per point). Among those with candidates, only
    the ones whose window currently has the *fewest* available candidates are resolved
    this round — ties (several equally-constrained points wanting the same candidate)
    go to whichever is closer; the loser stays unresolved and is re-evaluated next
    round, when the contested candidate is already gone and its window has changed.
    Reference points with zero available candidates are permanently dropped.

    This "fewest options first" order is what makes the result different from (and
    better than) a fixed left-to-right sweep by m/z: a reference point with several
    candidates to choose from can no longer starve a point that only had one option, by
    being processed first purely because it happens to sit at a lower m/z.

    Termination is guaranteed: each round either drops at least one reference point
    (assigned or permanently unmatched) or the loop exits, so it runs at most
    ``rx.size`` rounds — in practice far fewer, since real spectra rarely have more than
    a handful of reference points genuinely contesting the same candidate.

    :return: ``(reference_index, candidate_index)`` pairs, sorted by reference index.
    """
    remaining = np.arange(rx.size)
    used = np.zeros(cx.size, dtype=bool)
    pairs: list[tuple[int, int]] = []
    while remaining.size:
        available = np.flatnonzero(~used)
        if available.size == 0:
            break
        cx_available = cx[available]
        window_left = np.searchsorted(cx_available, rx[remaining] - radii[remaining], side="left")
        window_right = np.searchsorted(cx_available, rx[remaining] + radii[remaining], side="right")
        window_size = window_right - window_left

        has_candidate = window_size > 0
        if not np.any(has_candidate):
            break  # nothing left can ever be matched
        most_constrained = int(window_size[has_candidate].min())
        active = np.flatnonzero(has_candidate & (window_size == most_constrained))

        ## Resolve this round's most-constrained tier: nearest available candidate per
        ## active point, contested candidates going to whichever point is closer.
        best_distance_by_candidate: dict[int, tuple[float, int]] = {}
        for local_index in active:
            window = available[window_left[local_index]:window_right[local_index]]
            nearest_in_window = int(window[np.argmin(np.abs(cx[window] - rx[remaining[local_index]]))])
            distance = float(abs(cx[nearest_in_window] - rx[remaining[local_index]]))
            current_best = best_distance_by_candidate.get(nearest_in_window)
            if current_best is None or distance < current_best[0]:
                best_distance_by_candidate[nearest_in_window] = (distance, int(local_index))

        resolved_local = {local_index for _, local_index in best_distance_by_candidate.values()}
        for candidate_index, (_, local_index) in best_distance_by_candidate.items():
            pairs.append((int(remaining[local_index]), candidate_index))
            used[candidate_index] = True

        drop_local = resolved_local | set(np.flatnonzero(~has_candidate).tolist())
        keep_mask = np.ones(remaining.size, dtype=bool)
        keep_mask[list(drop_local)] = False
        remaining = remaining[keep_mask]
    pairs.sort(key=lambda pair: pair[0])
    return pairs


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
          reference points inflates apparent coverage. Fully vectorized (see
          implementation) — no per-point Python loop.
        - ``"one_to_one"`` — every candidate is claimed by at most one reference point (a
          genuine bijective partial matching, no double counting) — the strategy every
          quantitative metric in this module is computed with. Resolved in rounds (see
          :func:`_one_to_one_matches`): each round, among reference points that still
          have at least one *available* (not yet claimed) candidate, the ones with the
          *fewest* available candidates go first, with ties (several equally-constrained
          points contesting the same candidate) broken by proximity; winning candidates
          are removed from the pool and the round repeats for whoever is left. This
          "most-constrained-first" order — rather than a fixed left-to-right sweep by
          m/z — avoids a real failure mode of naive greedy matching: a reference point
          with several options gets processed first and happens to take the one
          candidate that was another reference point's *only* option, leaving that other
          point unmatched even though a different assignment would have matched both.
          Each round is vectorized (bulk ``searchsorted`` over the current candidate
          pool); only the number of rounds is a Python-level loop, and it collapses to
          very few rounds whenever admissible-candidate windows rarely overlap (the
          common case for sparse centroided spectra matched at a tight tolerance).
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
        ## nearest: fully vectorized nearest-candidate lookup (no per-point Python loop);
        ## candidates may repeat across reference points. Sorted-array nearest-neighbor
        ## trick: the nearest value to rx[i] in a sorted cx is always immediately before
        ## or immediately after searchsorted's insertion point.
        if rx.size and cx.size:
            insert = np.searchsorted(cx, rx)
            left_index = np.clip(insert - 1, 0, cx.size - 1)
            right_index = np.clip(insert, 0, cx.size - 1)
            left_distance = np.abs(cx[left_index] - rx)
            right_distance = np.abs(cx[right_index] - rx)
            prefer_left = left_distance <= right_distance
            nearest_index = np.where(prefer_left, left_index, right_index)
            nearest_distance = np.where(prefer_left, left_distance, right_distance)
            within_tolerance = nearest_distance <= radii
            for ref_index, cand_index in zip(np.flatnonzero(within_tolerance).tolist(), nearest_index[within_tolerance].tolist()):
                pairs.append((ref_index, cand_index)); groups.append(np.asarray([cand_index]))
    elif matching_strategy == "local_mass":
        ## local_mass: window boundaries computed in bulk (vectorized); the group itself
        ## has a variable size per reference point, so building each group still needs a
        ## per-point loop, just without the per-point searchsorted calls.
        if rx.size and cx.size:
            left = np.searchsorted(cx, rx - radii, side="left")
            right = np.searchsorted(cx, rx + radii, side="right")
            for index in range(rx.size):
                if right[index] > left[index]:
                    candidates = np.arange(left[index], right[index], dtype=int)
                    chosen = int(candidates[np.argmin(np.abs(cx[candidates] - rx[index]))])
                    pairs.append((index, chosen)); groups.append(candidates)
    elif rx.size and cx.size:
        ## one_to_one: see _one_to_one_matches for the most-constrained-first, round-based algorithm.
        pairs = _one_to_one_matches(rx, radii, cx)
        groups = [np.asarray([candidate_index]) for _, candidate_index in pairs]
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
    rx = np.asarray(reference_mz, dtype=np.float32); ry = np.asarray(reference_intensity, dtype=np.float32)
    cx = np.asarray(candidate_mz, dtype=np.float32)
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
    ry = np.asarray(reference_intensity, dtype=np.float32)
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
