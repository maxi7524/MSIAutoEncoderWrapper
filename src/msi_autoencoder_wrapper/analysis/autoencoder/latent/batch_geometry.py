"""Device-accelerated forms of the latent statistics that are quadratic in sample count.

Most of :mod:`.sphere_geometry` is linear in the number of pixels and costs nothing at
sweep scale. Two of its statistics are not: ``two_nn_intrinsic_dimension`` and
``knn_overlap`` both materialize the full pairwise similarity matrix, and the latter
then loops over rows in Python to intersect neighbor sets. On a full evaluation split
across tens of models that is the difference between an analysis that runs and one that
does not, which is why the reference module tells callers to subsample before calling
it.

This module removes that constraint rather than the data: the same quantities are
computed as tensor operations on one device, so a full split can be analysed instead of
a sample of it. It is a deliberate second implementation of quantities
:mod:`.sphere_geometry` already defines, kept only for speed, and
``tests/analysis/test_batch_geometry.py`` asserts agreement with that reference.

Both functions inherit the reference's conventions exactly: similarity is
``(u @ u.T) / D``, the dimension-normalized inner product that leaves canonicalized
codes on a comparable scale, and neighborhoods exclude the point itself.
"""

from __future__ import annotations

from typing import Any, Optional

import numpy as np
import torch

from ....utils.exceptions import raise_validation_error
from ....utils.logger import get_custom_logger

logger = get_custom_logger(__name__)


def _resolve_device(device: Optional[Any]) -> torch.device:
    """Pick the evaluation device, preferring CUDA when the caller gave none."""
    if device is not None:
        return torch.device(device)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _as_tensor(u: Any, device: torch.device) -> torch.Tensor:
    """Move canonicalized codes onto the evaluation device in double precision.

    REMARK: double precision is not optional here. Both statistics depend on the
    ordering of nearly equal pairwise similarities, and a float32 similarity matrix
    reorders near-ties differently from the float64 reference, which changes neighbor
    sets and therefore the result.
    """
    return torch.as_tensor(np.asarray(u), dtype=torch.float64, device=device)


def _pairwise_cosine(u: torch.Tensor) -> torch.Tensor:
    """Dimension-normalized inner product, matching ``sphere_geometry.pairwise_cosine``."""
    return (u @ u.T) / u.shape[1]  # (N, N)


def two_nn_intrinsic_dimension_batch(u: Any, device: Optional[Any] = None) -> float:
    """TwoNN intrinsic dimension, computed on one device.

    Equivalent to :func:`.sphere_geometry.two_nn_intrinsic_dimension`: the estimator
    is $\\hat d = n / \\sum_i \\log(\\theta_{i,2}/\\theta_{i,1})$ over the angular
    distances to each point's nearest and second-nearest neighbor.

    :param u: Canonicalized codes, shape ``(N, D)``.
    :type u: numpy.ndarray | torch.Tensor
    :param device: Evaluation device; defaults to CUDA when available.
    :type device: str | torch.device | None
    :return: Estimated intrinsic dimension.
    :rtype: float
    :raises ValidationError: If fewer than three samples are given, or every nearest
        neighbor is coincident.
    """
    tensor = _as_tensor(u, _resolve_device(device))
    if tensor.shape[0] < 3:
        raise_validation_error(
            "BatchGeometry", "two_nn_intrinsic_dimension needs at least 3 samples."
        )

    ## Angular distance, self-pairs excluded
    cosine = _pairwise_cosine(tensor).clamp(-1.0, 1.0)
    theta = torch.rad2deg(torch.arccos(cosine))  # (N, N)
    theta.fill_diagonal_(float("inf"))

    ## Two nearest neighbors per point
    nearest_two = torch.topk(theta, k=2, dim=1, largest=False).values  # (N, 2)
    nearest, second_nearest = nearest_two[:, 0], nearest_two[:, 1]

    valid = nearest > 0
    if not bool(valid.any()):
        raise_validation_error(
            "BatchGeometry",
            "two_nn_intrinsic_dimension: every nearest-neighbor distance is zero.",
        )
    log_ratio = torch.log(second_nearest[valid] / nearest[valid])
    return float(valid.sum() / log_ratio.sum())


def knn_overlap_batch(
    u_a: Any, u_b: Any, k: int = 10, device: Optional[Any] = None
) -> float:
    """Mean neighborhood agreement between two representations, computed on one device.

    Equivalent to :func:`.sphere_geometry.knn_overlap`: the mean over points of the
    fraction of each point's ``k`` nearest neighbors that both representations share.

    :param u_a: First representation's canonicalized codes, shape ``(N, D)``.
    :type u_a: numpy.ndarray | torch.Tensor
    :param u_b: Second representation's codes, row-aligned with ``u_a``, shape ``(N, D)``.
    :type u_b: numpy.ndarray | torch.Tensor
    :param k: Neighborhood size.
    :type k: int
    :param device: Evaluation device; defaults to CUDA when available.
    :type device: str | torch.device | None
    :return: Mean overlap in ``[0, 1]``, ``1.0`` for identical neighbor sets.
    :rtype: float
    :raises ValidationError: If the inputs have different row counts, or ``k`` leaves
        too few points to form a neighborhood.
    """
    resolved = _resolve_device(device)
    first, second = _as_tensor(u_a, resolved), _as_tensor(u_b, resolved)
    if first.shape[0] != second.shape[0]:
        raise_validation_error(
            "BatchGeometry", "knn_overlap requires the same number of paired rows."
        )
    if first.shape[0] <= k:
        raise_validation_error(
            "BatchGeometry", "knn_overlap: k must be smaller than the sample count."
        )

    neighbors_first = _neighbor_indices(first, k)  # (N, k)
    neighbors_second = _neighbor_indices(second, k)  # (N, k)

    ## Set intersection per row without a Python loop: k is small, so comparing every
    ## pair of the two index lists is cheaper than building membership structures.
    matches = neighbors_first.unsqueeze(2) == neighbors_second.unsqueeze(1)  # (N, k, k)
    shared = matches.any(dim=2).sum(dim=1)  # (N,)
    return float(shared.to(torch.float64).mean() / k)


def _neighbor_indices(u: torch.Tensor, k: int) -> torch.Tensor:
    """Indices of each row's ``k`` most similar other rows."""
    cosine = _pairwise_cosine(u)
    cosine.fill_diagonal_(float("-inf"))
    return torch.topk(cosine, k=k, dim=1, largest=True).indices  # (N, k)
