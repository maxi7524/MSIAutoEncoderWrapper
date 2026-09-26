"""Ion images on a binned axis and their agreement with a reference image."""

from __future__ import annotations

import numpy as np
from scipy.stats import pearsonr, rankdata, spearmanr

from ....utils.exceptions import raise_validation_error


def ion_bin_indices(mz_values: np.ndarray, bin_edges: np.ndarray, radius: int = 1) -> list[np.ndarray]:
    """Return the bins summed into the ion image of each m/z value.

    The ion image of an m/z value is the summed intensity of the bin containing it and
    of ``radius`` neighbouring bins on either side, clipped to the axis. The same radius
    defines annotation evidence (``SignalEvidencePolicy.bin_radius``), so ion images and
    evidence states refer to the same spectral support.

    :param mz_values: Ion m/z values, shape ``(I,)``.
    :type mz_values: numpy.ndarray
    :param bin_edges: Increasing bin edges of the model axis, shape ``(M + 1,)``.
    :type bin_edges: numpy.ndarray
    :param radius: Neighbouring bins on either side.
    :type radius: int
    :return: One integer index array per ion; empty when the m/z lies outside the axis.
    :rtype: list[numpy.ndarray]
    :raises ValidationError: If the radius is negative.
    """
    if radius < 0:
        raise_validation_error("IonImages", "radius must be nonnegative.")
    edges = np.asarray(bin_edges, dtype=np.float64)
    count = edges.size - 1
    result = []
    for mz in np.asarray(mz_values, dtype=np.float64):
        if not edges[0] <= mz <= edges[-1]:
            result.append(np.empty(0, dtype=np.int64))
            continue
        centre = min(int(np.searchsorted(edges, mz, side="right")) - 1, count - 1)
        result.append(np.arange(max(0, centre - radius), min(count, centre + radius + 1), dtype=np.int64))
    return result


def ion_intensities(spectra: np.ndarray, bins: list[np.ndarray]) -> np.ndarray:
    """Sum spectra over the support of every ion.

    :param spectra: Spectra, shape ``(N, M)``.
    :type spectra: numpy.ndarray
    :param bins: Bin supports from :func:`ion_bin_indices`.
    :type bins: list[numpy.ndarray]
    :return: Ion intensities, shape ``(N, I)``; ``nan`` for ions outside the axis.
    :rtype: numpy.ndarray
    """
    values = np.asarray(spectra)
    result = np.full((values.shape[0], len(bins)), np.nan, dtype=np.float32)  # (N, I)
    for position, support in enumerate(bins):
        if support.size:
            result[:, position] = values[:, support].sum(axis=1)
    return result


def spatial_agreement(reference: np.ndarray, candidate: np.ndarray) -> dict[str, float]:
    """Rank and linear correlation between two images over their common pixels.

    :param reference: Reference pixel values, shape ``(N,)``.
    :type reference: numpy.ndarray
    :param candidate: Compared pixel values, shape ``(N,)``.
    :type candidate: numpy.ndarray
    :return: ``spearman``, ``pearson`` and the number of finite pixels. Correlations
        are ``nan`` when either image is constant.
    :rtype: dict[str, float]
    """
    reference, candidate = np.asarray(reference, dtype=np.float64), np.asarray(candidate, dtype=np.float64)
    finite = np.isfinite(reference) & np.isfinite(candidate)
    if finite.sum() < 3 or np.ptp(reference[finite]) == 0 or np.ptp(candidate[finite]) == 0:
        return {"spearman": np.nan, "pearson": np.nan, "pixels": float(finite.sum())}
    return {"spearman": float(spearmanr(reference[finite], candidate[finite]).statistic),
            "pearson": float(pearsonr(reference[finite], candidate[finite]).statistic),
            "pixels": float(finite.sum())}


def presence_auc(scores: np.ndarray, present: np.ndarray) -> float:
    """ROC AUC of ``scores`` separating pixels where an ion is present from the rest.

    Computed from the Mann-Whitney rank sum with average ranks for ties.

    :param scores: Pixel scores, shape ``(N,)``.
    :type scores: numpy.ndarray
    :param present: Boolean presence of the reference ion, shape ``(N,)``.
    :type present: numpy.ndarray
    :return: AUC in ``[0, 1]``; ``nan`` when either class is empty.
    :rtype: float
    """
    scores, present = np.asarray(scores, dtype=np.float64), np.asarray(present, dtype=bool)
    finite = np.isfinite(scores)
    scores, present = scores[finite], present[finite]
    positives, negatives = int(present.sum()), int((~present).sum())
    if positives == 0 or negatives == 0:
        return float("nan")
    ranks = rankdata(scores)
    return float((ranks[present].sum() - positives * (positives + 1) / 2) / (positives * negatives))
