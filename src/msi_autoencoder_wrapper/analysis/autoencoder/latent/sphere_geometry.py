"""Canonicalized-latent (`LayerNorm`-sphere) geometry metrics.

See `23_08_26_architecture_predictive/report/geometria_latentu.md` for the derivation
and justification behind every function here (why the bottleneck's codes live on
`S^{D-2}(sqrt D)`, why comparisons must use canonicalized `u = (z-beta)/gamma` rather
than raw `z`, and why `angle_degrees` rather than `||J||_F` is the sensitivity measure
to trust) — this module implements only that document's §4-§7 (canonicalization, the
angle metric, and the metrics catalogue), not the theory itself.

Every function operates on already-canonicalized `u` unless noted otherwise. None of
them subsample internally: several (`two_nn_intrinsic_dimension`, `procrustes_distance`,
`linear_cka`, `knn_overlap`, `trustworthiness_continuity`) are `O(N^2)` in the number
of rows via a dense `(N, N)` cosine matrix — callers on a large split (thousands of
rows) should pass a random subsample, not the full split.
"""

from __future__ import annotations

from typing import Any, Callable, Dict, Optional, Sequence

import numpy as np
from scipy.spatial import procrustes as _scipy_procrustes
from scipy.stats import spearmanr
from sklearn.manifold import trustworthiness as _sklearn_trustworthiness

from ....utils.exceptions import raise_validation_error


# --------------------------------------------------
# Section: Canonicalization
# --------------------------------------------------

def canonicalize(z: np.ndarray, gamma: np.ndarray, beta: np.ndarray) -> np.ndarray:
    """Undo `LayerNorm`'s learned affine transform: $u = (z-\\beta)/\\gamma$.

    :param z: Post-`LayerNorm` latent codes (the model's own ``latent_space``
        output), shape ``(N, D)``.
    :type z: numpy.ndarray
    :param gamma: `LayerNorm` ``weight``, shape ``(D,)``.
    :type gamma: numpy.ndarray
    :param beta: `LayerNorm` ``bias``, shape ``(D,)``.
    :type beta: numpy.ndarray
    :return: Canonicalized codes on $S^{D-2}(\\sqrt D)$ (`geometria_latentu.md` §2).
    :rtype: numpy.ndarray
    :raises ValidationError: If any `gamma` component is zero.
    """
    z_array = np.asarray(z, dtype=np.float64)
    gamma_array = np.asarray(gamma, dtype=np.float64).reshape(1, -1)
    beta_array = np.asarray(beta, dtype=np.float64).reshape(1, -1)
    if np.any(gamma_array == 0):
        raise_validation_error("SphereGeometry", "gamma must be nonzero everywhere.")
    return (z_array - beta_array) / gamma_array


def normalize_to_constant_norm(x: np.ndarray, target_norm: Optional[float] = None) -> np.ndarray:
    """Row-normalize an embedding to a constant norm, e.g. a contrastive projection.

    Not a substitute for `canonicalize`: embeddings such as the model's own
    `projection` output (`LinearProjector`: `Linear -> LayerNorm -> ReLU -> Linear`)
    have no bottleneck `LayerNorm` as their *final* op, so there is no `(gamma, beta)`
    affine transform to undo, and their norm varies per sample. `InfoNCELoss` itself
    discards that norm via `F.normalize(..., dim=1)` before computing similarities
    (`training.criterions.autoencoder.contrastive.infoNCE_loss`) — the loss's actual
    geometry lives on a sphere of *some* constant radius, not in the raw output space.
    This function reproduces that same normalization so the rest of this module
    (`pairwise_cosine` and everything built on it — `structure_test`, `knn_overlap`,
    `two_nn_intrinsic_dimension`, `dimension_usage`, `cloud_asymmetry`) can be reused
    unmodified: every function here that assumes a fixed-norm point cloud only needs
    that norm to be constant across rows, not that it equal $\\sqrt D$ specifically —
    defaulting `target_norm` to $\\sqrt D$ simply keeps `pairwise_cosine`'s `/D`
    convention numerically meaningful as an actual cosine.

    :param x: Raw embedding, shape ``(N, D)``.
    :type x: numpy.ndarray
    :param target_norm: Norm every output row should have; defaults to $\\sqrt D$.
    :type target_norm: float | None
    :return: Row-normalized embedding, shape ``(N, D)``.
    :rtype: numpy.ndarray
    :raises ValidationError: If any row has zero norm.
    """
    array = np.asarray(x, dtype=np.float64)
    norms = np.linalg.norm(array, axis=1, keepdims=True)
    if np.any(norms == 0):
        raise_validation_error("SphereGeometry", "normalize_to_constant_norm: a zero-norm row cannot be rescaled.")
    scale = np.sqrt(array.shape[1]) if target_norm is None else float(target_norm)
    return (array / norms) * scale


def encoder_layer_norm_parameters(model: Any) -> tuple[np.ndarray, np.ndarray]:
    """Return the bottleneck `LayerNorm`'s ``(gamma, beta)`` as numpy arrays.

    Assumes a `CNNEncoder`-shaped encoder (`model.encoder.bottleneck_layer[-1]` is
    the bottleneck `nn.LayerNorm` — verified directly against a loaded
    `conv1d-ae-32-16-8-latent-10` model; every ablation compared under this
    methodology shares that architecture, `report/methodology.md` §3.1/§5).

    :param model: Loaded autoencoder (``wrapper.active_model``).
    :type model: Any
    :return: ``(gamma, beta)``, each shape ``(D,)``.
    :rtype: tuple[numpy.ndarray, numpy.ndarray]
    """
    layer_norm = model.encoder.bottleneck_layer[-1]
    return (
        layer_norm.weight.detach().cpu().numpy(),
        layer_norm.bias.detach().cpu().numpy(),
    )


def verify_canonicalization(u: np.ndarray) -> Dict[str, float]:
    """Check the two defining properties of $u$: row sums near 0, row norms near $\\sqrt D$.

    A pipeline-correctness check (`geometria_latentu.md` §2.1), not a metric to
    report as a scientific result — large deviations mean `canonicalize` was applied
    incorrectly (wrong `gamma`/`beta`, or `z` was not this encoder's own output).

    :param u: Canonicalized codes, shape ``(N, D)``.
    :type u: numpy.ndarray
    :return: ``max_abs_row_sum``, ``expected_norm``, ``mean_row_norm``,
        ``max_abs_norm_deviation``.
    :rtype: Dict[str, float]
    """
    dimension = u.shape[1]
    row_sums = u.sum(axis=1)
    row_norms = np.linalg.norm(u, axis=1)
    expected_norm = float(np.sqrt(dimension))
    return {
        "max_abs_row_sum": float(np.max(np.abs(row_sums))),
        "expected_norm": expected_norm,
        "mean_row_norm": float(np.mean(row_norms)),
        "max_abs_norm_deviation": float(np.max(np.abs(row_norms - expected_norm))),
    }


# --------------------------------------------------
# Section: Primary metric — angle theta
# --------------------------------------------------

def pairwise_cosine(u: np.ndarray, u_other: Optional[np.ndarray] = None) -> np.ndarray:
    """$\\cos\\theta_{ij} = \\langle u_i, u_j' \\rangle / D$ for every pair.

    :param u: Canonicalized codes, shape ``(N, D)``.
    :type u: numpy.ndarray
    :param u_other: Second set, shape ``(M, D)``; defaults to `u` itself.
    :type u_other: numpy.ndarray | None
    :return: Cosine matrix, shape ``(N, M)``.
    :rtype: numpy.ndarray
    """
    dimension = u.shape[1]
    other = u if u_other is None else u_other
    return (u @ other.T) / dimension


def angle_degrees(cos_theta: np.ndarray) -> np.ndarray:
    """$\\theta = \\arccos(\\operatorname{clip}(\\cos\\theta, -1, 1))$, in degrees."""
    return np.degrees(np.arccos(np.clip(cos_theta, -1.0, 1.0)))


def paired_angle_degrees(u_a: np.ndarray, u_b: np.ndarray) -> np.ndarray:
    """Row-aligned angle between two code sets — e.g. $u(x)$ vs. $u(x+\\varepsilon)$.

    :param u_a: Canonicalized codes, shape ``(N, D)``.
    :type u_a: numpy.ndarray
    :param u_b: Canonicalized codes for the *same* $N$ samples (row-aligned),
        shape ``(N, D)``.
    :type u_b: numpy.ndarray
    :return: Angle in degrees, one value per row, shape ``(N,)``.
    :rtype: numpy.ndarray
    :raises ValidationError: If the two inputs' shapes do not match.
    """
    if u_a.shape != u_b.shape:
        raise_validation_error(
            "SphereGeometry", "paired_angle_degrees requires matching, row-aligned shapes."
        )
    dimension = u_a.shape[1]
    cos_theta = np.sum(u_a * u_b, axis=1) / dimension
    return angle_degrees(cos_theta)


# --------------------------------------------------
# Section: Is there any structure at all
# --------------------------------------------------

def structure_test(
    u: np.ndarray,
    rng: np.random.Generator,
    pair_count: int = 5000,
    return_samples: bool = False,
) -> Dict[str, float]:
    """Compare observed $\\cos\\theta$ spread against the closed-form uniform-sphere null.

    `geometria_latentu.md` §3.1: for two independent points uniform on $S^d$,
    $\\operatorname{Var}[\\cos\\theta] = 1/(d+1)$, with $d = D-2$ (§2.2) — a free,
    comparable-across-ablations zero point. An observed
    ``observed_sd_cos_theta`` near ``uniform_baseline_sd_cos_theta`` (with
    ``observed_mean_cos_theta`` near 0) means this ablation's latent is
    indistinguishable from unstructured noise on the sphere.

    :param u: Canonicalized codes, shape ``(N, D)``.
    :type u: numpy.ndarray
    :param rng: Seeded generator for the random pair sample.
    :type rng: numpy.random.Generator
    :param pair_count: Number of random (non-identical) pairs to sample.
    :type pair_count: int
    :param return_samples: If ``True``, also return the raw per-pair ``cos_theta``
        values under ``cos_theta_samples`` — the summary mean/sd otherwise hides
        the shape of this distribution (e.g. bimodality, heavy tails), which a
        violin/histogram plot of the raw samples makes visible.
    :type return_samples: bool
    :return: ``observed_mean_cos_theta``, ``observed_sd_cos_theta``,
        ``uniform_baseline_sd_cos_theta``, ``effective_dimension``, and
        (if ``return_samples``) ``cos_theta_samples``.
    :rtype: Dict[str, float]
    :raises ValidationError: If the canonicalized dimension is not greater than 2.
    """
    n_samples, dimension = u.shape
    effective_dimension = dimension - 2
    if effective_dimension <= 0:
        raise_validation_error(
            "SphereGeometry", "Canonicalized dimension must exceed 2 for a defined null."
        )
    left = rng.integers(0, n_samples, size=pair_count)
    right = rng.integers(0, n_samples, size=pair_count)
    distinct = left != right
    cos_theta = np.sum(u[left[distinct]] * u[right[distinct]], axis=1) / dimension
    samples = {"cos_theta_samples": cos_theta} if return_samples else {}
    return {
        **samples,
        "observed_mean_cos_theta": float(np.mean(cos_theta)),
        "observed_sd_cos_theta": float(np.std(cos_theta)),
        "uniform_baseline_sd_cos_theta": float(np.sqrt(1.0 / (effective_dimension + 1))),
        "effective_dimension": float(effective_dimension),
    }


def null_direction_check(u: np.ndarray, gamma: np.ndarray) -> Dict[str, float]:
    """Pipeline-correctness check: is $\\operatorname{Cov}(u)$'s null direction the
    constant direction, not $1/\\gamma$?

    If canonicalization incorrectly omits the division by $\\gamma$ (e.g. computing
    $\\tilde u = z-\\beta = \\gamma \\odot u$ instead of $u = (z-\\beta)/\\gamma$): since
    $\\operatorname{Cov}(u)\\,\\mathbf{1} = 0$ exactly (every row satisfies
    $\\mathbf{1}^\\top u = 0$), substituting
    $\\operatorname{Cov}(\\tilde u) = \\operatorname{diag}(\\gamma)\\operatorname{Cov}(u)
    \\operatorname{diag}(\\gamma)$ gives $\\operatorname{Cov}(\\tilde u)\\,(1/\\gamma) =
    \\operatorname{diag}(\\gamma)\\operatorname{Cov}(u)\\,\\mathbf{1} = 0$ — the
    direction that should have exactly zero variance reappears along $1/\\gamma$
    instead of the constant direction. Report, not itself a scientific finding: for
    correct canonicalization, ``cosine_with_constant_direction`` should be near 1 and
    ``cosine_with_inverse_gamma`` should be small.

    :param u: Canonicalized codes, shape ``(N, D)``.
    :type u: numpy.ndarray
    :param gamma: The same encoder's `LayerNorm` ``weight``, shape ``(D,)``.
    :type gamma: numpy.ndarray
    :return: ``smallest_eigenvalue``, ``cosine_with_constant_direction``,
        ``cosine_with_inverse_gamma`` (all non-negative; sign of an eigenvector is
        arbitrary).
    :rtype: Dict[str, float]
    """
    covariance = np.cov(u, rowvar=False, ddof=0)
    eigenvalues, eigenvectors = np.linalg.eigh(covariance)
    null_eigenvector = eigenvectors[:, 0]  # eigh: ascending order, so index 0 is smallest.
    dimension = u.shape[1]
    constant_direction = np.ones(dimension) / np.sqrt(dimension)
    inverse_gamma = 1.0 / np.asarray(gamma, dtype=np.float64)
    inverse_gamma_direction = inverse_gamma / np.linalg.norm(inverse_gamma)
    return {
        "smallest_eigenvalue": float(eigenvalues[0]),
        "cosine_with_constant_direction": float(abs(np.dot(null_eigenvector, constant_direction))),
        "cosine_with_inverse_gamma": float(abs(np.dot(null_eigenvector, inverse_gamma_direction))),
    }


def cloud_asymmetry(u: np.ndarray) -> float:
    """$\\lVert \\bar u \\rVert^2 / D$ — how far the sample mean sits from the sphere's center.

    `geometria_latentu.md` Część V, item 6. Complements `dimension_usage`: with
    population covariance (`ddof=0`), $\\operatorname{tr}\\operatorname{Cov}(u) = D -
    \\lVert \\bar u \\rVert^2$ exactly, so a nonzero mean pulls variance out of the
    fixed trace budget rather than the budget shrinking on its own.

    :param u: Canonicalized codes, shape ``(N, D)``.
    :type u: numpy.ndarray
    :return: Squared mean-vector norm, normalized by ``D``.
    :rtype: float
    """
    dimension = u.shape[1]
    mean_u = u.mean(axis=0)
    return float(np.dot(mean_u, mean_u) / dimension)


# --------------------------------------------------
# Section: Dimension usage and intrinsic dimension
# --------------------------------------------------

def dimension_usage(u: np.ndarray) -> Dict[str, Any]:
    """Eigenvalue spectrum of $\\operatorname{Cov}(u)$, effective rank, participation ratio.

    `geometria_latentu.md` §3.2: $\\operatorname{tr}\\operatorname{Cov}(u) \\le D$ is a
    fixed budget, so the spectrum is directly comparable between ablations without
    extra normalization. One eigenvalue is expected to be numerically zero — the
    $\\mathbf{1}$-direction canonicalization already removed (§2.2) — and is a
    pipeline check, not itself a finding.

    :param u: Canonicalized codes, shape ``(N, D)``.
    :type u: numpy.ndarray
    :return: ``eigenvalues`` (descending, length ``D``), ``trace``,
        ``effective_rank`` (exponential Shannon entropy of the normalized spectrum),
        ``participation_ratio``.
    :rtype: Dict[str, Any]
    :raises ValidationError: If the total variance is zero (degenerate input).
    """
    # ddof=0 (population, not Bessel-corrected) covariance so trace(Cov(u)) <= D
    # holds as the hard bound geometria_latentu.md §3.2 states, not just approximately.
    covariance = np.cov(u, rowvar=False, ddof=0)
    eigenvalues = np.clip(np.linalg.eigvalsh(covariance)[::-1], 0.0, None)
    total = float(eigenvalues.sum())
    if total <= 0:
        raise_validation_error("SphereGeometry", "Degenerate covariance: zero total variance.")
    proportions = eigenvalues[eigenvalues > 0] / total
    effective_rank = float(np.exp(-np.sum(proportions * np.log(proportions))))
    participation_ratio = float(total**2 / np.sum(eigenvalues**2))
    return {
        "eigenvalues": eigenvalues,
        "trace": total,
        "effective_rank": effective_rank,
        "participation_ratio": participation_ratio,
    }


def two_nn_intrinsic_dimension(u: np.ndarray) -> float:
    """TwoNN intrinsic-dimension estimator (Facco et al., 2017), on angular distance.

    Uses $\\theta$ rather than chordal distance so the estimate does not depend on
    which of `geometria_latentu.md` §3.1's three rank-equivalent sphere distances is
    nominally chosen — they agree to leading order for the small, local angles TwoNN
    actually uses (nearest and second-nearest neighbor only). Expected value near
    $D-2$ (`geometria_latentu.md` §2.2); a large deviation flags either
    representational collapse onto a lower-dimensional subset or a pipeline error.
    `O(N^2)` — subsample large splits before calling.

    :param u: Canonicalized codes, shape ``(N, D)``.
    :type u: numpy.ndarray
    :return: Estimated intrinsic dimension.
    :rtype: float
    :raises ValidationError: If fewer than 3 samples are given, or every pair is
        coincident (all pairwise angles zero).
    """
    if u.shape[0] < 3:
        raise_validation_error("SphereGeometry", "two_nn_intrinsic_dimension needs at least 3 samples.")
    theta = angle_degrees(pairwise_cosine(u))
    np.fill_diagonal(theta, np.inf)
    sorted_theta = np.sort(theta, axis=1)
    nearest, second_nearest = sorted_theta[:, 0], sorted_theta[:, 1]
    valid = nearest > 0
    if not np.any(valid):
        raise_validation_error(
            "SphereGeometry", "two_nn_intrinsic_dimension: every nearest-neighbor distance is zero."
        )
    log_ratio = np.log(second_nearest[valid] / nearest[valid])
    return float(np.sum(valid) / np.sum(log_ratio))


# --------------------------------------------------
# Section: Geometry similarity between two ablations
# --------------------------------------------------

def procrustes_distance(u_a: np.ndarray, u_b: np.ndarray) -> float:
    """Normalized orthogonal-Procrustes distance between two paired point sets.

    A proper metric (satisfies the triangle inequality, unlike raw CKA) — safe to
    assemble into an $M \\times M$ distance matrix over $M$ ablations and feed to
    hierarchical clustering / MDS. Requires the two inputs to be row-aligned (same
    sample, same order) — `geometria_latentu.md` §7. `O(N^2)` via SVD on the full
    matrices; subsample large splits before calling.

    :param u_a: Canonicalized codes for one ablation, shape ``(N, D)``.
    :type u_a: numpy.ndarray
    :param u_b: Canonicalized codes for another ablation, same rows in the same
        order, shape ``(N, D)``.
    :type u_b: numpy.ndarray
    :return: $\\sqrt{\\text{disparity}} \\in [0, 1]$.
    :rtype: float
    :raises ValidationError: If the two inputs' shapes do not match.
    """
    if u_a.shape != u_b.shape:
        raise_validation_error(
            "SphereGeometry", "procrustes_distance requires paired, equal-shape inputs."
        )
    _, _, disparity = _scipy_procrustes(u_a, u_b)
    return float(np.sqrt(disparity))


def linear_cka(u_a: np.ndarray, u_b: np.ndarray) -> float:
    """Linear CKA (Kornblith et al., 2019) between two paired point sets.

    Value-dependent (`geometria_latentu.md` §7): a cheap cross-check for
    `procrustes_distance`, never trusted alone — known to sit near 1 even for
    visibly different representations, and is not itself a metric (no triangle
    inequality).

    :param u_a: Canonicalized codes for one ablation, shape ``(N, D)``.
    :type u_a: numpy.ndarray
    :param u_b: Canonicalized codes for another ablation, same rows in the same
        order, shape ``(N, D)``.
    :type u_b: numpy.ndarray
    :return: CKA similarity, nominally in ``[0, 1]``.
    :rtype: float
    :raises ValidationError: If the two inputs do not share the same row count.
    """
    if u_a.shape[0] != u_b.shape[0]:
        raise_validation_error(
            "SphereGeometry", "linear_cka requires the same number of paired rows."
        )
    centered_a = u_a - u_a.mean(axis=0, keepdims=True)
    centered_b = u_b - u_b.mean(axis=0, keepdims=True)
    cross = centered_a.T @ centered_b
    numerator = float(np.sum(cross**2))
    denominator = float(
        np.sqrt(
            np.sum((centered_a.T @ centered_a) ** 2)
            * np.sum((centered_b.T @ centered_b) ** 2)
        )
    )
    if denominator == 0:
        raise_validation_error("SphereGeometry", "linear_cka: degenerate (zero-variance) input.")
    return numerator / denominator


# --------------------------------------------------
# Section: Neighborhood similarity
# --------------------------------------------------

def knn_overlap(u_a: np.ndarray, u_b: np.ndarray, k: int = 10) -> float:
    """Mean Jaccard overlap of each point's $k$ nearest neighbors, across two spaces.

    Rank-based (`geometria_latentu.md` §7): indifferent to which of chordal/
    geodesic/cosine-based distance is nominally used. `O(N^2)`; subsample large
    splits before calling.

    :param u_a: Canonicalized codes for one ablation, shape ``(N, D)``.
    :type u_a: numpy.ndarray
    :param u_b: Canonicalized codes for another ablation, same rows in the same
        order, shape ``(N, D)``.
    :type u_b: numpy.ndarray
    :param k: Neighborhood size.
    :type k: int
    :return: Mean Jaccard overlap in ``[0, 1]``, ``1.0`` for identical neighbor sets.
    :rtype: float
    :raises ValidationError: If the two inputs do not share the same row count, or
        ``k`` leaves too few points to form a neighborhood.
    """
    if u_a.shape[0] != u_b.shape[0]:
        raise_validation_error("SphereGeometry", "knn_overlap requires the same number of paired rows.")
    if u_a.shape[0] <= k:
        raise_validation_error("SphereGeometry", "knn_overlap: k must be smaller than the sample count.")
    neighbors_a = _knn_indices(u_a, k)
    neighbors_b = _knn_indices(u_b, k)
    overlaps = [
        len(set(row_a) & set(row_b)) / k for row_a, row_b in zip(neighbors_a, neighbors_b)
    ]
    return float(np.mean(overlaps))


def _knn_indices(u: np.ndarray, k: int) -> np.ndarray:
    """Return each row's `k` nearest-neighbor indices by cosine similarity."""
    cos_theta = pairwise_cosine(u)
    np.fill_diagonal(cos_theta, -np.inf)
    return np.argpartition(-cos_theta, kth=k, axis=1)[:, :k]


def trustworthiness_continuity(
    u_reference: np.ndarray,
    u_comparison: np.ndarray,
    n_neighbors: int = 10,
) -> Dict[str, float]:
    """Trustworthiness of `u_comparison` against `u_reference`, and its reverse (continuity).

    Reuses `sklearn.manifold.trustworthiness` directly: Euclidean nearest-neighbor
    *ranking* on equal-radius canonicalized codes is rank-equivalent to angular
    distance (`geometria_latentu.md` §3.1), so no custom implementation is needed.
    Rank-based.

    :param u_reference: Canonicalized codes treated as ground truth, shape
        ``(N, D)``.
    :type u_reference: numpy.ndarray
    :param u_comparison: Canonicalized codes for the space being checked against
        it, same rows in the same order, shape ``(N, D)``.
    :type u_comparison: numpy.ndarray
    :param n_neighbors: Neighborhood size.
    :type n_neighbors: int
    :return: ``trustworthiness`` (are `u_comparison`'s neighbors also close in
        `u_reference`), ``continuity`` (the reverse direction), both in ``[0, 1]``.
    :rtype: Dict[str, float]
    """
    return {
        "trustworthiness": float(
            _sklearn_trustworthiness(u_reference, u_comparison, n_neighbors=n_neighbors)
        ),
        "continuity": float(
            _sklearn_trustworthiness(u_comparison, u_reference, n_neighbors=n_neighbors)
        ),
    }


# --------------------------------------------------
# Section: Structure vs. labels
# --------------------------------------------------

def rsa_spearman(
    u: np.ndarray,
    labels: np.ndarray,
    rng: np.random.Generator,
    pair_count: int = 5000,
) -> Dict[str, float]:
    """Spearman correlation between latent angle and multi-label Jaccard distance.

    `geometria_latentu.md` §7: directly tests `grid_0005`'s label-Jaccard-weighted
    negative-pair rationale (`report/methodology.md` §4.3). Rank-based — robust to
    $\\theta$'s nonlinearity by construction.

    :param u: Canonicalized codes, shape ``(N, D)``.
    :type u: numpy.ndarray
    :param labels: Binary multi-label target matrix, row-aligned with `u`, shape
        ``(N, C)``.
    :type labels: numpy.ndarray
    :param rng: Seeded generator for the random pair sample.
    :type rng: numpy.random.Generator
    :param pair_count: Number of random (non-identical) pairs to sample.
    :type pair_count: int
    :return: ``spearman_r``, ``p_value``, ``pairs`` (actual sampled pair count after
        dropping identical-index draws).
    :rtype: Dict[str, float]
    """
    n_samples, dimension = u.shape
    left = rng.integers(0, n_samples, size=pair_count)
    right = rng.integers(0, n_samples, size=pair_count)
    distinct = left != right
    left, right = left[distinct], right[distinct]
    cos_theta = np.sum(u[left] * u[right], axis=1) / dimension
    theta = angle_degrees(cos_theta)
    labels_bool = labels.astype(bool)
    intersection = np.sum(labels_bool[left] & labels_bool[right], axis=1)
    union = np.sum(labels_bool[left] | labels_bool[right], axis=1)
    jaccard = np.divide(
        intersection, union, out=np.zeros(len(left), dtype=np.float64), where=union > 0
    )
    correlation, p_value = spearmanr(theta, 1.0 - jaccard)
    return {"spearman_r": float(correlation), "p_value": float(p_value), "pairs": int(len(left))}


# --------------------------------------------------
# Section: Encoder sensitivity
# --------------------------------------------------

def angular_sensitivity_curve(
    encode_fn: Callable[[np.ndarray], np.ndarray],
    inputs: np.ndarray,
    epsilons: Sequence[float],
    rng: np.random.Generator,
) -> Dict[str, np.ndarray]:
    """Empirical angular sensitivity: mean angle moved per input perturbation scale.

    $\\varepsilon \\mapsto \\mathbb{E}_{x,\\delta}\\big[\\angle(u(x), u(x+\\varepsilon\\lVert x
    \\rVert\\delta))\\big]$, $\\delta$ a random unit direction — the $\\gamma$-invariant
    replacement for $\\lVert J \\rVert_F$ (`geometria_latentu.md` §6.2-§6.3). Perturbing
    by a *random* direction gives the general encoder-sensitivity curve; the
    contrastive-selectivity variant (`geometria_latentu.md` §7's last row) instead
    compares `paired_angle_degrees` directly between the original and a peak-permuted
    view — this function only owns the generic random-direction sweep.

    :param encode_fn: Raw spectrum batch (shape ``(N, M)``) to canonicalized latent
        (shape ``(N, D)``) — must already apply `canonicalize` internally.
    :type encode_fn: Callable[[numpy.ndarray], numpy.ndarray]
    :param inputs: Input spectra to perturb, shape ``(N, M)``.
    :type inputs: numpy.ndarray
    :param epsilons: Relative perturbation scales to sweep.
    :type epsilons: Sequence[float]
    :param rng: Seeded generator for the random perturbation directions.
    :type rng: numpy.random.Generator
    :return: ``epsilon`` and ``mean_angle_degrees``, both shape ``(len(epsilons),)``.
    :rtype: Dict[str, numpy.ndarray]
    """
    baseline_u = encode_fn(inputs)
    mean_angles = []
    norms = np.linalg.norm(inputs, axis=1, keepdims=True)
    for epsilon in epsilons:
        directions = rng.standard_normal(inputs.shape)
        directions /= np.linalg.norm(directions, axis=1, keepdims=True)
        perturbed = inputs + epsilon * norms * directions
        perturbed_u = encode_fn(perturbed)
        mean_angles.append(float(np.mean(paired_angle_degrees(baseline_u, perturbed_u))))
    return {
        "epsilon": np.asarray(epsilons, dtype=np.float64),
        "mean_angle_degrees": np.asarray(mean_angles, dtype=np.float64),
    }
