"""Geometry diagnostics for predictive heads on a shared held-out pixel sample."""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.spatial.distance import cdist

from ....utils.logger import get_custom_logger
from .sphere_geometry import (
    dimension_usage, knn_overlap, linear_cka, procrustes_distance, rsa_spearman,
    structure_test, trustworthiness_continuity, two_nn_intrinsic_dimension,
)

logger = get_custom_logger(__name__)


def geometry_tables(latent: np.ndarray, targets: np.ndarray, indices: np.ndarray, *, k: int = 10) -> dict[str, pd.DataFrame]:
    """Describe variance, distances and annotation neighbourhoods without a 2D proxy.

    :param latent: Finite codes ``(N, D)`` in one explicitly named space.
    :param targets: Available-positive indicator matrix ``(N, C)``.
    :param indices: Shared sampled row positions ``(S,)``; pairwise work is O(S²).
    :param k: Number of neighbours; capped at S-1.
    :return: Geometry summary, eigenvalues, per-pixel diagnostics and pair histograms.
    :rtype: dict[str, pandas.DataFrame]
    :raises ValueError: If inputs are invalid or fewer than three rows are sampled.
    """
    z = np.asarray(latent, dtype=np.float64)
    ids = np.asarray(indices, dtype=int)
    if z.ndim != 2 or z.shape[1] < 2 or not np.isfinite(z).all() or len(z) != len(targets):
        raise ValueError("Geometry needs aligned finite (N, D>=2) codes and labels.")
    if len(ids) < 3 or len(np.unique(ids)) != len(ids) or np.any(ids < 0) or np.any(ids >= len(z)) or k < 1:
        raise ValueError("Geometry requires >=3 distinct valid sample indices and k>=1.")
    # Global covariance and radial distribution, without discarding a collapsed model
    centered = z - z.mean(axis=0, keepdims=True)  # (N, D)
    collapsed = not np.any(centered)
    usage = ({"eigenvalues": np.zeros(z.shape[1]), "trace": 0.0, "effective_rank": 0.0,
              "participation_ratio": 0.0} if collapsed else dimension_usage(z))
    norms = np.linalg.norm(z, axis=1)  # (N,)
    rows = [{"metric": key, "value": value} for key, value in usage.items() if key != "eigenvalues"]
    rows.extend([{"metric": "collapsed", "value": float(collapsed)},
                 {"metric": "norm_mean", "value": norms.mean()},
                 {"metric": "norm_std", "value": norms.std()},
                 {"metric": "centroid_norm", "value": np.linalg.norm(z.mean(axis=0))},
                 {"metric": "row_sum_abs_max", "value": np.abs(z.sum(axis=1)).max()}])
    # Local metric geometry on identical rows for every model
    sample = z[ids]  # (S, D)
    distances = cdist(sample, sample, metric="euclidean")  # (S, S)
    sample_norms = np.linalg.norm(sample, axis=1)  # (S,)
    denominator = sample_norms[:, None] * sample_norms[None, :]  # (S, S)
    cosines = np.divide(sample @ sample.T, denominator, out=np.full_like(distances, np.nan), where=denominator > 0)  # (S, S)
    i, j = np.triu_indices(len(ids), 1)  # (Q,), (Q,)
    # All pairs enter medians and histograms; sampled codes retain enough information
    # to reconstruct exact distances without repeating millions of identity strings.
    pairs = pd.DataFrame({"left_row": ids[i], "right_row": ids[j],
                          "euclidean": distances[i, j], "cosine": cosines[i, j]})
    np.fill_diagonal(distances, np.inf)
    neighbours = np.argsort(distances, axis=1, kind="stable")[:, :min(k, len(ids) - 1)]  # (S, K)
    labels = np.asarray(targets[ids], dtype=bool)  # (S, C)
    local_jaccard = []
    for row, near in enumerate(neighbours):
        intersection = (labels[row] & labels[near]).sum(axis=1)  # (K,)
        union = (labels[row] | labels[near]).sum(axis=1)  # (K,)
        local_jaccard.append(np.divide(intersection, union, out=np.zeros(len(near)), where=union > 0).mean())
    rows.extend([{"metric": "knn_annotation_jaccard", "value": np.mean(local_jaccard)},
                 {"metric": "pair_distance_median", "value": pairs.euclidean.median()},
                 {"metric": "pair_cosine_median", "value": pairs.cosine.median()},
                 {"metric": "duplicate_pair_fraction", "value": float((pairs.euclidean == 0).mean())}])
    histograms = []
    for metric in ("euclidean", "cosine"):
        values = pairs[metric].to_numpy()
        finite = values[np.isfinite(values)]
        counts, edges = np.histogram(finite, bins=128)
        histograms.extend({"metric": metric, "left_edge": edges[bin_index], "right_edge": edges[bin_index + 1],
                           "count": count, "total_pairs": len(values), "finite_pairs": len(finite)}
                          for bin_index, count in enumerate(counts))
    # Deterministic SVD projection is descriptive; signs have no scientific meaning.
    sample_centered = sample - sample.mean(axis=0, keepdims=True)  # (S, D)
    _, _, axes = np.linalg.svd(sample_centered, full_matrices=False)
    coordinates = sample_centered @ axes[:2].T  # (S, 2)
    pixels = pd.DataFrame({"row_position": ids, "norm": norms[ids],
                           "nearest_distance": distances.min(axis=1),
                           "annotation_jaccard": local_jaccard,
                           "annotation_count": labels.sum(axis=1),
                           "pc1": coordinates[:, 0], "pc2": coordinates[:, 1]})
    logger.info("Geometry: %s full rows, %s shared neighbourhood rows.", len(z), len(ids))
    return {"geometry": pd.DataFrame(rows), "eigenvalues": pd.DataFrame({"component": np.arange(z.shape[1]) + 1, "value": usage["eigenvalues"]}),
            "geometry_pixels": pixels, "geometry_pairs": pd.DataFrame(histograms)}


def representation_similarity(left: np.ndarray, right: np.ndarray) -> float:
    """Return linear CKA on aligned rows; undefined for constant representations.

    :param left: First code matrix, ``(N, D_a)``.
    :param right: Second code matrix, ``(N, D_b)``.
    :return: CKA similarity, or NaN for a collapsed input.
    :rtype: float
    """
    if not np.any(left - left.mean(axis=0)) or not np.any(right - right.mean(axis=0)):
        return float("nan")
    return linear_cka(left, right)


def ridge_probe(train_latent: np.ndarray, train_targets: np.ndarray, evaluated_latent: np.ndarray, *, penalty: float = 0.01) -> np.ndarray:
    """Fit the same linear annotation probe to every frozen representation.

    :param train_latent: Training codes ``(N_train, D)``.
    :param train_targets: Training annotation indicators ``(N_train, C)``.
    :param evaluated_latent: Codes to score ``(N_eval, D)``; never used for fitting.
    :param penalty: Positive coefficient in mean squared error + penalty * ||W||².
    :return: Unbounded ranking scores ``(N_eval, C)``; not calibrated probabilities.
    :rtype: numpy.ndarray
    :raises ValueError: If the penalty is not finite and positive.
    """
    if not np.isfinite(penalty) or penalty <= 0:
        raise ValueError("The fixed probe penalty must be finite and positive.")
    x = np.asarray(train_latent, dtype=np.float64)
    mean, scale = x.mean(axis=0), x.std(axis=0)  # (D,), (D,)
    scale = np.where(scale > 1e-12, scale, 1.0)  # (D,)
    standardized = (x - mean) / scale  # (N_train, D)
    y = np.asarray(train_targets, dtype=np.float64)
    intercept = y.mean(axis=0)  # (C,)
    gram = standardized.T @ standardized / len(x)  # (D, D)
    rhs = standardized.T @ (y - intercept) / len(x)  # (D, C)
    weights = np.linalg.solve(gram + penalty * np.eye(x.shape[1]), rhs)  # (D, C)
    return ((evaluated_latent - mean) / scale) @ weights + intercept  # (N_eval, C)


def structure_summary(u: np.ndarray, indices: np.ndarray, rng: np.random.Generator, *, pair_count: int = 5000) -> dict[str, float]:
    """Test whether one model's sampled representation has any structure beyond sphere noise.

    Wraps :func:`sphere_geometry.structure_test` on the same shared sampled rows used by
    :func:`geometry_tables`, so this and the pair-distance histograms describe the same
    points. A model whose observed cos-theta spread matches the closed-form uniform-sphere
    null (``observed_sd_cos_theta`` close to ``uniform_baseline_sd_cos_theta``,
    ``observed_mean_cos_theta`` close to 0) has a representation indistinguishable from
    unstructured noise on the canonicalized sphere; a narrower spread indicates clustering,
    and a wider or skewed one (not visible in this summary — plot the raw samples for that)
    indicates polarized or multi-cluster structure rather than uniform noise.

    :param u: Canonicalized codes for the full split, ``(N, D)``.
    :type u: numpy.ndarray
    :param indices: Shared sampled row positions, ``(S,)`` — the same indices passed to
        :func:`geometry_tables` for this model and split.
    :type indices: numpy.ndarray
    :param rng: Seeded generator for the random-pair sample.
    :type rng: numpy.random.Generator
    :param pair_count: Number of random pairs drawn from the sampled rows.
    :type pair_count: int
    :return: ``observed_mean_cos_theta``, ``observed_sd_cos_theta``,
        ``uniform_baseline_sd_cos_theta``, ``effective_dimension``.
    :rtype: dict[str, float]
    """
    result = structure_test(u[np.asarray(indices, dtype=int)], rng, pair_count=pair_count)
    result.pop("cos_theta_samples", None)
    return result


def intrinsic_dimension_estimate(u: np.ndarray, indices: np.ndarray) -> float:
    """Return the TwoNN intrinsic-dimension estimate on the shared sampled rows.

    Complementary to :func:`geometry_tables`'s covariance-based ``effective_rank`` and
    ``participation_ratio``: a non-linear, local estimate of how many dimensions the point
    cloud actually occupies, rather than a global linear one. Uses the same sampled rows as
    :func:`geometry_tables` so both numbers describe the same points.

    REMARK: ``O(S^2)`` in the sample size, like the pair-distance histograms it is computed
    alongside; do not call it on the full unsampled split.

    :param u: Canonicalized codes for the full split, ``(N, D)``.
    :type u: numpy.ndarray
    :param indices: Shared sampled row positions, ``(S,)``.
    :type indices: numpy.ndarray
    :return: Estimated intrinsic dimension.
    :rtype: float
    """
    return two_nn_intrinsic_dimension(u[np.asarray(indices, dtype=int)])


def pairwise_geometry_battery(u_left: np.ndarray, u_right: np.ndarray, indices: np.ndarray, *, k: int = 10) -> dict[str, float]:
    """Compare two models' representations on the same shared, row-aligned pixels.

    A global shape match (e.g. equal ``effective_rank``) does not imply the same local
    neighbourhood structure, and vice versa: ``procrustes_distance`` and
    :func:`representation_similarity` (linear CKA) test global geometric alignment, while
    ``knn_overlap`` and ``trustworthiness_continuity`` test local neighbourhood
    preservation. All four are computed on the identical sampled rows of both inputs, in the
    same order — the caller must guarantee the row alignment (same pixels, same order),
    exactly as :func:`geometry_tables` requires for its own sample.

    :param u_left: Canonicalized codes of the first model, full split, ``(N, D)``.
    :type u_left: numpy.ndarray
    :param u_right: Canonicalized codes of the second model, full split, ``(N, D)``,
        row-aligned with ``u_left``.
    :type u_right: numpy.ndarray
    :param indices: Shared sampled row positions, ``(S,)``.
    :type indices: numpy.ndarray
    :param k: Neighbourhood size for ``knn_overlap``/``trustworthiness_continuity``; capped
        at ``S - 1``.
    :type k: int
    :return: ``procrustes_distance`` (a proper metric in ``[0, 1]``), ``linear_cka``,
        ``knn_overlap``, ``trustworthiness``, ``continuity``.
    :rtype: dict[str, float]
    """
    ids = np.asarray(indices, dtype=int)
    left, right = u_left[ids], u_right[ids]
    neighbours = min(k, len(ids) - 1)
    trust_cont = trustworthiness_continuity(left, right, n_neighbors=neighbours)
    return {"procrustes_distance": procrustes_distance(left, right),
            "linear_cka": representation_similarity(left, right),
            "knn_overlap": knn_overlap(left, right, k=neighbours),
            **trust_cont}


def label_structure_correlation(u: np.ndarray, indices: np.ndarray, targets: np.ndarray,
                                rng: np.random.Generator, *, pair_count: int = 5000) -> dict[str, float]:
    """Correlate latent angular distance with annotation-label dissimilarity.

    Wraps :func:`sphere_geometry.rsa_spearman` on the shared sampled rows. A
    representation can have rich geometric structure (:func:`structure_summary`,
    ``effective_rank``) that is nonetheless unrelated to the annotated molecules; a
    Spearman correlation near 0 between latent angle and label-Jaccard distance means the
    geometry and the annotations are unrelated, while a strong correlation means pixels with
    similar latent codes also carry similar annotations. This does not distinguish a
    representation that never saw the labels from one trained against them — comparing the
    value across conditions is what does that.

    :param u: Canonicalized codes for the full split, ``(N, D)``.
    :type u: numpy.ndarray
    :param indices: Shared sampled row positions, ``(S,)``.
    :type indices: numpy.ndarray
    :param targets: Available-positive indicator matrix for the full split, ``(N, C)``,
        row-aligned with ``u``.
    :type targets: numpy.ndarray
    :param rng: Seeded generator for the random-pair sample.
    :type rng: numpy.random.Generator
    :param pair_count: Number of random pairs drawn from the sampled rows.
    :type pair_count: int
    :return: ``spearman_r``, ``p_value``, ``pairs`` (actual sampled pair count after
        dropping identical-index draws).
    :rtype: dict[str, float]
    """
    ids = np.asarray(indices, dtype=int)
    return rsa_spearman(u[ids], targets[ids], rng, pair_count=pair_count)
