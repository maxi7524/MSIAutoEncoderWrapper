"""Agreement between chemical label-graph structure and spatial ion co-localization."""

from __future__ import annotations

from typing import Iterable

import numpy as np
import pandas as pd
from scipy.stats import rankdata

from .ion_images import IonImages
from .label_graph import LabelGraph


def colocalization(
    images: Iterable[IonImages],
    ion_ids: tuple[str, ...],
    *,
    device: str | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Pool per-image Pearson ion-image correlations over a collection.

    Within image :math:`i`, the correlation :math:`r_{ab}^{(i)}` of two ion images is
    computed over pixels with annotated signal. Correlations are pooled by the
    Fisher transform weighted with :math:`n_i - 3` (the inverse variance of
    :math:`\\operatorname{artanh} r`):

    .. math::

        \\bar r_{ab} = \\tanh\\Big(\\frac{\\sum_i (n_i-3) \\operatorname{artanh} r^{(i)}_{ab}}
                                   {\\sum_i (n_i-3)}\\Big).

    :param images: Ion images of all datasets of the collection.
    :type images: collections.abc.Iterable[IonImages]
    :param ion_ids: Collection ion order.
    :type ion_ids: tuple[str, ...]
    :param device: Torch device for the per-image correlation matrices; ``None``
        uses NumPy.
    :type device: str | None
    :return: Pooled correlation ``(A, A)`` (``nan`` for never co-annotated pairs)
        and the number of images contributing to each pair ``(A, A)``.
    :rtype: tuple[numpy.ndarray, numpy.ndarray]
    """
    index = {ion: position for position, ion in enumerate(ion_ids)}
    size = len(ion_ids)
    weighted = np.zeros((size, size))
    weights = np.zeros((size, size))
    counts = np.zeros((size, size), dtype=np.int64)
    for image in images:
        normalized, _ = image.tic_normalized()  # (A_i, P_+)
        keep = np.array([ion in index and normalized[i].std() > 0 for i, ion in enumerate(image.ion_ids)])
        if keep.sum() < 2 or normalized.shape[1] < 4:
            continue
        rows = np.array([index[ion] for ion in image.ion_ids[keep]])
        correlation = np.clip(_correlation(normalized[keep], device), -0.999999, 0.999999)  # (A_k, A_k)
        weight = normalized.shape[1] - 3
        block = np.ix_(rows, rows)
        weighted[block] += weight * np.arctanh(correlation)
        weights[block] += weight
        counts[block] += 1
    with np.errstate(invalid="ignore", divide="ignore"):
        pooled = np.tanh(weighted / weights)
    pooled[weights == 0] = np.nan
    return pooled, counts


def _correlation(matrix: np.ndarray, device: str | None) -> np.ndarray:
    """Return the Pearson correlation matrix of the rows of ``(A, P)``."""
    if device is None:
        return np.corrcoef(matrix)
    import torch

    values = torch.as_tensor(matrix, dtype=torch.float32, device=device)  # (A, P)
    centered = values - values.mean(dim=1, keepdim=True)
    unit = centered / centered.norm(dim=1, keepdim=True)
    return (unit @ unit.T).double().cpu().numpy()  # (A, A)


def tanimoto_similarity(graph: LabelGraph) -> np.ndarray:
    """Return continuous Tanimoto similarity of ion class-membership vectors.

    :math:`T(a,b) = w_a^\\top w_b / (\\lVert w_a \\rVert^2 + \\lVert w_b \\rVert^2 - w_a^\\top w_b)`
    equals the Jaccard index of class sets for binary memberships and generalizes it
    to fractional (``weighted``) memberships.

    :param graph: Label graph.
    :type graph: LabelGraph
    :return: Matrix ``(A, A)``; ``nan`` where either ion is unresolved.
    :rtype: numpy.ndarray
    """
    matrix = graph.membership.toarray().astype(np.float64)  # (A, C)
    inner = matrix @ matrix.T  # (A, A)
    squared = np.diag(inner)
    with np.errstate(invalid="ignore", divide="ignore"):
        similarity = inner / (squared[:, None] + squared[None, :] - inner)
    unresolved = ~graph.resolved
    similarity[unresolved, :] = np.nan
    similarity[:, unresolved] = np.nan
    return similarity


def mantel_test(
    similarity: np.ndarray,
    reference: np.ndarray,
    *,
    n_permutations: int,
    seed: int,
    device: str | None = None,
    batch_size: int = 32,
) -> dict[str, float]:
    """Rank Mantel test between graph similarity and spatial co-localization.

    Over the ions with at least one finite value in both matrices, all off-diagonal
    entries of ``similarity`` are rank-transformed once, and the entries of
    ``reference`` are rank-transformed over its finite pairs. The statistic is the
    Pearson correlation of the two rank vectors over pairs finite in both. Under the
    null hypothesis of exchangeable ion labels, ions of ``similarity`` are permuted
    jointly in rows and columns; because the rank of every entry is a property of the
    entry, permuting the rank matrix equals ranking the permuted matrix. The
    one-sided p-value is :math:`(1 + \\#\\{\\rho_\\pi \\ge \\rho\\})/(1 + B)`.

    :param similarity: Graph similarity ``(A, A)``; ``nan`` rows mark unresolved ions.
    :type similarity: numpy.ndarray
    :param reference: Spatial co-localization ``(A, A)``.
    :type reference: numpy.ndarray
    :param n_permutations: Number of permutations :math:`B`.
    :type n_permutations: int
    :param seed: Permutation seed; permutations are drawn with NumPy so that they do
        not depend on the device.
    :type seed: int
    :param device: Torch device evaluating permutations in batches; ``None`` uses
        NumPy one permutation at a time.
    :type device: str | None
    :param batch_size: Permutations evaluated together on ``device``.
    :type batch_size: int
    :return: ``rho``, ``p_value``, ``null_mean``, ``null_sd``, ``pairs``, ``ions``.
    :rtype: dict[str, float]

    REMARK: Ranks of ``similarity`` are taken over all pairs of usable ions, not only
    over pairs that are also finite in ``reference``. This keeps the permutation
    exact without re-ranking, at the cost of a statistic that differs slightly from
    Spearman's coefficient restricted to co-observed pairs.
    """
    usable = np.isfinite(np.diag(similarity)) & np.any(np.isfinite(reference) & ~np.eye(len(reference), dtype=bool), axis=1)
    sub_similarity = similarity[np.ix_(usable, usable)]
    sub_reference = reference[np.ix_(usable, usable)]
    size = sub_similarity.shape[0]
    empty = {"rho": float("nan"), "p_value": float("nan"), "null_mean": float("nan"), "null_sd": float("nan"),
             "pairs": 0, "ions": int(size)}
    if size < 3:
        return empty
    upper = np.triu_indices(size, k=1)
    ## Rank matrices
    similarity_ranks = np.full((size, size), np.nan)
    similarity_ranks[upper] = rankdata(sub_similarity[upper])
    similarity_ranks = np.fmax(similarity_ranks, similarity_ranks.T)  # symmetric (n, n)
    valid = np.isfinite(sub_reference[upper])
    reference_ranks = rankdata(sub_reference[upper][valid])  # (M,)
    rows, columns = upper[0][valid], upper[1][valid]

    def rho(order: np.ndarray) -> float:
        values = similarity_ranks[order[rows], order[columns]]
        return float(np.corrcoef(values, reference_ranks)[0, 1]) if np.ptp(values) > 0 else float("nan")

    identity = np.arange(size)
    observed = rho(identity) if valid.sum() >= 3 else float("nan")
    if not np.isfinite(observed):
        return {**empty, "pairs": int(valid.sum())}
    generator = np.random.default_rng(seed)
    orders = np.stack([generator.permutation(size) for _ in range(n_permutations)])  # (B, n)
    if device is None:
        null = np.array([rho(order) for order in orders])
    else:
        null = _batched_rank_correlation(similarity_ranks, reference_ranks, rows, columns, orders, device, batch_size)
    null = null[np.isfinite(null)]
    return {
        "rho": observed,
        "p_value": float((1 + np.sum(null >= observed)) / (1 + null.size)),
        "null_mean": float(null.mean()),
        "null_sd": float(null.std()),
        "pairs": int(valid.sum()),
        "ions": int(size),
    }


def _batched_rank_correlation(
    similarity_ranks: np.ndarray,
    reference_ranks: np.ndarray,
    rows: np.ndarray,
    columns: np.ndarray,
    orders: np.ndarray,
    device: str,
    batch_size: int,
) -> np.ndarray:
    """Evaluate the permutation statistic of :func:`mantel_test` in batches on a device."""
    import torch

    ranks = torch.as_tensor(similarity_ranks, dtype=torch.float32, device=device)  # (n, n)
    target = torch.as_tensor(reference_ranks, dtype=torch.float32, device=device)  # (M,)
    target = target - target.mean()
    target = target / target.norm()  # centered unit vector (M,)
    row_index = torch.as_tensor(rows, device=device)
    column_index = torch.as_tensor(columns, device=device)
    result = []
    for start in range(0, len(orders), batch_size):
        order = torch.as_tensor(orders[start:start + batch_size], device=device)  # (b, n)
        values = ranks[order[:, row_index], order[:, column_index]]  # (b, M)
        centered = values - values.mean(dim=1, keepdim=True)
        norms = centered.norm(dim=1)
        statistic = (centered @ target) / norms  # (b,)
        result.append(torch.where(norms > 0, statistic, torch.full_like(statistic, float("nan"))).cpu().numpy())
    return np.concatenate(result).astype(np.float64)


def class_spatial_coherence(graph: LabelGraph, reference: np.ndarray, *, min_ions: int = 3) -> pd.DataFrame:
    """Compare co-localization within each class with the collection average.

    :param graph: Label graph.
    :type graph: LabelGraph
    :param reference: Pooled co-localization ``(A, A)``.
    :type reference: numpy.ndarray
    :param min_ions: Minimum number of member ions with at least one finite pair.
    :type min_ions: int
    :return: One row per class with ``within_mean`` (mean co-localization over
        member pairs), ``overall_mean`` (over all finite pairs), ``excess``
        (difference), ``pairs`` and ``ions``.
    :rtype: pandas.DataFrame
    """
    upper = np.triu(np.isfinite(reference), k=1)
    overall = float(reference[upper].mean()) if upper.any() else float("nan")
    binary = (graph.membership > 0).tocsc()
    rows = []
    for position, record in enumerate(graph.classes.itertuples(index=False)):
        members = binary[:, position].indices
        if members.size < min_ions:
            continue
        block = reference[np.ix_(members, members)]
        values = block[np.triu_indices(members.size, k=1)]
        values = values[np.isfinite(values)]
        if values.size == 0:
            continue
        rows.append({"class_id": record.class_id, "taxonomy": record.taxonomy, "name": record.name,
                     "depth": record.depth, "ions": int(members.size), "pairs": int(values.size),
                     "within_mean": float(values.mean()), "overall_mean": overall,
                     "excess": float(values.mean() - overall)})
    return pd.DataFrame(rows)


def linear_cka(kernel: np.ndarray, reference: np.ndarray) -> float:
    """Return centered kernel alignment of two ``(n, n)`` kernels.

    :math:`\\mathrm{CKA}(K, L) = \\langle HKH, HLH\\rangle_F /
    (\\lVert HKH \\rVert_F \\lVert HLH \\rVert_F)` with centering
    :math:`H = I - \\mathbf{1}\\mathbf{1}^\\top / n`; it is invariant to isotropic
    scaling and lies in ``[0, 1]`` for positive semi-definite kernels.

    :param kernel: First kernel.
    :type kernel: numpy.ndarray
    :param reference: Second kernel.
    :type reference: numpy.ndarray
    :return: Alignment value; ``nan`` if either centered kernel vanishes.
    :rtype: float
    """
    def center(matrix: np.ndarray) -> np.ndarray:
        return matrix - matrix.mean(axis=0, keepdims=True) - matrix.mean(axis=1, keepdims=True) + matrix.mean()

    left, right = center(kernel), center(reference)
    denominator = np.linalg.norm(left) * np.linalg.norm(right)
    return float(np.sum(left * right) / denominator) if denominator > 0 else float("nan")


def propagated_kernels(
    graph: LabelGraph,
    steps: tuple[int, ...],
    ions: np.ndarray,
    *,
    device: str | None = None,
) -> dict[int, np.ndarray]:
    """Return cosine kernels of propagated ion embeddings for selected ions.

    :param graph: Label graph.
    :type graph: LabelGraph
    :param steps: Propagation depths ``k``.
    :type steps: tuple[int, ...]
    :param ions: Boolean or integer selection of graph ion rows.
    :type ions: numpy.ndarray
    :param device: Torch device for propagation and kernel products; ``None`` uses
        SciPy/NumPy.
    :type device: str | None
    :return: Mapping ``k -> (n, n)`` kernel.
    :rtype: dict[int, numpy.ndarray]
    """
    kernels = {}
    selection = np.flatnonzero(ions) if np.asarray(ions).dtype == bool else np.asarray(ions)
    sequence = graph.propagation_sequence(tuple(steps), device=device, as_tensor=device is not None)
    for k, embeddings in sequence.items():
        selected = embeddings[selection]  # (n, N)
        if device is None:
            norms = np.linalg.norm(selected, axis=1, keepdims=True)
            unit = np.divide(selected, norms, out=np.zeros_like(selected), where=norms > 0)
            kernels[k] = unit @ unit.T  # (n, n)
        else:
            norms = selected.norm(dim=1, keepdim=True)
            unit = selected / norms.clamp_min(1e-30)
            kernels[k] = (unit @ unit.T).double().cpu().numpy()  # (n, n)
    return kernels
