"""Unsupervised segmentation of pixel representations and segment marker bins."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment
from sklearn.metrics import adjusted_rand_score
from sklearn.mixture import GaussianMixture

from ....utils.exceptions import raise_validation_error
from ....utils.logger import get_custom_logger

logger = get_custom_logger(__name__)


@dataclass(frozen=True)
class SegmentationFit:
    """Gaussian-mixture segmentation fitted on one representation.

    :param k: Number of mixture components (segments).
    :type k: int
    :param model: Fitted mixture.
    :type model: sklearn.mixture.GaussianMixture
    :param bic: Bayesian information criterion on the fitting sample.
    :type bic: float
    :param fit_rows: Rows of the fitting representation used for estimation.
    :type fit_rows: numpy.ndarray
    """

    k: int
    model: GaussianMixture
    bic: float
    fit_rows: np.ndarray

    def predict(self, representation: np.ndarray) -> np.ndarray:
        """Assign every row to its most probable segment.

        :param representation: Rows in the fitted space, shape ``(N, D)``.
        :type representation: numpy.ndarray
        :return: Segment labels, shape ``(N,)``.
        :rtype: numpy.ndarray
        """
        return self.model.predict(np.asarray(representation, dtype=np.float64)).astype(np.int64)


def fit_segmentation(representation: np.ndarray, k: int, *, seed: int, sample_size: int,
                     n_init: int = 3, reg_covar: float = 1e-6) -> SegmentationFit:
    """Fit a full-covariance Gaussian mixture on a seeded subsample.

    :param representation: Training representation, shape ``(N, D)``.
    :type representation: numpy.ndarray
    :param k: Number of segments; at least two.
    :type k: int
    :param seed: Seed of the subsample and of the mixture initialization.
    :type seed: int
    :param sample_size: Maximum number of fitting rows.
    :type sample_size: int
    :param n_init: Independent initializations; the best likelihood is kept.
    :type n_init: int
    :param reg_covar: Diagonal covariance regularization.
    :type reg_covar: float
    :return: Fitted segmentation with its BIC.
    :rtype: SegmentationFit
    :raises ValidationError: If ``k`` or the sample is invalid.
    """
    values = np.asarray(representation, dtype=np.float64)
    if k < 2 or values.ndim != 2 or len(values) < k:
        raise_validation_error("Segmentation", "Need at least k rows of a two-dimensional representation and k >= 2.")
    generator = np.random.default_rng(seed)
    rows = np.sort(generator.choice(len(values), size=min(sample_size, len(values)), replace=False))  # (S,)
    model = GaussianMixture(n_components=k, covariance_type="full", random_state=seed, n_init=n_init,
                            reg_covar=reg_covar)
    model.fit(values[rows])
    bic = float(model.bic(values[rows]))
    logger.debug("Segmentation k=%s fitted on %s rows (BIC=%.3f).", k, rows.size, bic)
    return SegmentationFit(k=k, model=model, bic=bic, fit_rows=rows)


def align_labels(reference: np.ndarray, labels: np.ndarray, k: int) -> np.ndarray:
    """Relabel ``labels`` to maximize agreement with ``reference`` (Hungarian matching).

    Mixture component indices are arbitrary; aligning them makes segment colours
    comparable across repetitions and axes. The partition itself is unchanged.

    :param reference: Reference labels, shape ``(N,)``.
    :type reference: numpy.ndarray
    :param labels: Labels to relabel, shape ``(N,)``.
    :type labels: numpy.ndarray
    :param k: Number of possible labels in both partitions.
    :type k: int
    :return: Relabelled partition, shape ``(N,)``.
    :rtype: numpy.ndarray
    """
    contingency = np.zeros((k, k), dtype=np.int64)
    np.add.at(contingency, (np.asarray(labels), np.asarray(reference)), 1)
    source, target = linear_sum_assignment(-contingency)
    mapping = np.arange(k)
    mapping[source] = target
    return mapping[np.asarray(labels)]


def adjusted_rand_between(left: np.ndarray, right: np.ndarray) -> float:
    """Adjusted Rand index of two partitions of the same pixels.

    :param left: First partition, shape ``(N,)``.
    :type left: numpy.ndarray
    :param right: Second partition, shape ``(N,)``.
    :type right: numpy.ndarray
    :return: ARI; 1 for identical partitions, about 0 for independent ones.
    :rtype: float
    """
    return float(adjusted_rand_score(np.asarray(left), np.asarray(right)))


def segment_contingency(reference: np.ndarray, other: np.ndarray, k: int) -> pd.DataFrame:
    """Pixel overlap of two segmentations of the same pixels.

    After :func:`align_labels`, matching segments share an index, so the mass on the
    diagonal is the agreement and the off-diagonal cells show which segments split or
    merge between the two partitions.

    :param reference: Reference labels in ``[0, k)``, shape ``(N,)``.
    :type reference: numpy.ndarray
    :param other: Compared labels in ``[0, k)``, shape ``(N,)``.
    :type other: numpy.ndarray
    :param k: Number of possible labels in both partitions.
    :type k: int
    :return: One row per label pair with ``pixels``, ``reference_fraction`` (share of the
        reference segment) and ``other_fraction`` (share of the compared segment); empty
        segments give ``nan`` fractions.
    :rtype: pandas.DataFrame
    :raises ValidationError: If the partitions differ in length or a label is outside ``[0, k)``.
    """
    reference, other = np.asarray(reference, dtype=np.int64), np.asarray(other, dtype=np.int64)
    if reference.shape != other.shape:
        raise_validation_error("segment_contingency", "Partitions must label the same pixels.")
    if reference.size and (min(reference.min(), other.min()) < 0 or max(reference.max(), other.max()) >= k):
        raise_validation_error("segment_contingency", f"Labels must lie in [0, {k}).")
    counts = np.zeros((k, k), dtype=np.int64)  # (k_reference, k_other)
    np.add.at(counts, (reference, other), 1)
    with np.errstate(invalid="ignore", divide="ignore"):
        by_reference = counts / counts.sum(axis=1, keepdims=True)  # (k, k)
        by_other = counts / counts.sum(axis=0, keepdims=True)  # (k, k)
    left, right = np.meshgrid(np.arange(k), np.arange(k), indexing="ij")
    return pd.DataFrame({"reference_segment": left.ravel(), "other_segment": right.ravel(), "pixels": counts.ravel(),
                         "reference_fraction": by_reference.ravel(), "other_fraction": by_other.ravel()})


def segment_marker_table(spectra: np.ndarray, labels: np.ndarray, mass_axis: np.ndarray, *,
                         top: int, minimum_mean: float) -> pd.DataFrame:
    """Rank the bins most enriched in every segment.

    Enrichment of bin :math:`j` in segment :math:`s` is
    :math:`\\log_2 (\\bar x_{s,j} + \\epsilon) - \\log_2 (\\bar x_{\\neg s,j} + \\epsilon)`,
    the log ratio of the mean TIC-normalized intensity inside and outside the segment.
    Bins whose overall mean intensity is below ``minimum_mean`` are ignored so that the
    ratio is not dominated by noise-level bins; the same value is used as
    :math:`\\epsilon`.

    :param spectra: TIC-normalized spectra, shape ``(N, M)``.
    :type spectra: numpy.ndarray
    :param labels: Segment labels, shape ``(N,)``.
    :type labels: numpy.ndarray
    :param mass_axis: Bin centres, shape ``(M,)``.
    :type mass_axis: numpy.ndarray
    :param top: Number of marker bins retained per segment.
    :type top: int
    :param minimum_mean: Minimum overall mean intensity of an eligible bin.
    :type minimum_mean: float
    :return: ``segment``, ``rank``, ``mz``, ``bin``, ``mean_inside``, ``mean_outside``,
        ``log2_enrichment`` and ``segment_fraction``.
    :rtype: pandas.DataFrame
    """
    values = np.asarray(spectra, dtype=np.float32)
    labels = np.asarray(labels)
    segments = np.unique(labels)
    ## REMARK: segment sums come from one (K, N) x (N, M) product instead of boolean
    ## row copies; a held-out image matrix is too large to copy once per segment.
    indicator = (labels[None, :] == segments[:, None]).astype(np.float32)  # (K, N)
    sums = (indicator @ values).astype(np.float64)  # (K, M)
    counts = indicator.sum(axis=1).astype(np.float64)  # (K,)
    total = sums.sum(axis=0)  # (M,)
    overall = total / len(labels)  # (M,)
    eligible = np.flatnonzero(overall >= minimum_mean)  # (E,)
    rows = []
    for position, segment in enumerate(segments):
        inside = labels == segment
        if inside.all():
            continue
        mean_inside = sums[position, eligible] / counts[position]  # (E,)
        mean_outside = (total[eligible] - sums[position, eligible]) / (len(labels) - counts[position])  # (E,)
        enrichment = np.log2(mean_inside + minimum_mean) - np.log2(mean_outside + minimum_mean)  # (E,)
        order = np.argsort(enrichment)[::-1][:top]
        for rank, position in enumerate(order):
            rows.append({"segment": int(segment), "rank": rank, "mz": float(mass_axis[eligible[position]]),
                         "bin": int(eligible[position]), "mean_inside": float(mean_inside[position]),
                         "mean_outside": float(mean_outside[position]),
                         "log2_enrichment": float(enrichment[position]), "segment_fraction": float(inside.mean())})
    return pd.DataFrame(rows)
