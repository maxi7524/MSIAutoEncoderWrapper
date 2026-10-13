"""CUDA partition methods for small diagnostic spectral images."""

from __future__ import annotations

from typing import Any

import numpy as np
import torch

from ....utils.logger import get_custom_logger

logger = get_custom_logger(__name__)


def _squared_distance(x: torch.Tensor, centres: torch.Tensor) -> torch.Tensor:
    """Compute squared Euclidean distance without cancellation-prone Gram subtraction."""
    distance = torch.cdist(x, centres, compute_mode="donot_use_mm_for_euclid_dist").square()  # (N, K)
    if not torch.isfinite(distance).all():
        raise ValueError("CUDA squared distances overflowed float32.")
    return distance


def _kmeans(x: torch.Tensor, k: int, seed: int, max_iter: int, n_init: int) -> tuple[torch.Tensor, torch.Tensor, int]:
    """Run seeded k-means++ and Lloyd updates using only CUDA tensor calculations."""
    best_labels = best_centres = None
    best_cost = float("inf")
    best_iterations = 0
    # Evaluate independent seeded centroid initializations
    for init in range(n_init):
        ## Initialize centroids with k-means++
        generator = torch.Generator(device=x.device).manual_seed(seed + init)
        centres = [x[torch.randint(len(x), (1,), device=x.device, generator=generator)[0]]]
        closest = _squared_distance(x, centres[0][None]).squeeze(1)  # (N,)
        for _ in range(1, k):
            total = closest.sum()  # ()
            chosen = torch.multinomial(closest / total, 1, generator=generator)[0] if total > 0 else torch.randint(len(x), (1,), device=x.device, generator=generator)[0]
            centres.append(x[chosen])
            closest = torch.minimum(closest, _squared_distance(x, centres[-1][None]).squeeze(1))  # (N,)
        centre = torch.stack(centres)  # (K, D)
        ## Alternate assignments and centroid updates
        previous = None
        for iteration in range(1, max_iter + 1):
            distances = _squared_distance(x, centre)  # (N, K)
            labels = distances.argmin(dim=1)  # (N,)
            if previous is not None and torch.equal(labels, previous):
                break
            new = []
            for j in range(k):
                selected = x[labels == j]  # (N_j, D)
                new.append(selected.mean(dim=0) if len(selected) else x[distances.amin(dim=1).argmax()])
            centre = torch.stack(new)  # (K, D)
            previous = labels.clone()
        distances = _squared_distance(x, centre)  # (N, K)
        labels = distances.argmin(dim=1)  # (N,)
        ## Retain the candidate with minimum within-cluster squared distance
        cost = float(distances.amin(dim=1).sum().item())
        if cost < best_cost:
            best_cost, best_labels, best_centres, best_iterations = cost, labels.clone(), centre.clone(), iteration
    assert best_labels is not None and best_centres is not None
    return best_labels, best_centres, best_iterations


def _gmm_diag(x: torch.Tensor, k: int, seed: int, ridge: float, max_iter: int) -> tuple[torch.Tensor, dict[str, Any]]:
    """Compute diagonal Gaussian EM with log-sum-exp responsibilities on CUDA."""
    # Initialize the diagonal mixture from CUDA k-means centroids
    _, means, _ = _kmeans(x, k, seed, max_iter=64, n_init=3)  # (K, D)
    variances = x.var(dim=0, correction=0).clamp_min(ridge).expand(k, -1).clone()  # (K, D)
    weights = torch.full((k,), 1 / k, dtype=x.dtype, device=x.device)  # (K,)
    previous = None
    converged = False
    # Optimize mixture likelihood
    for iteration in range(1, max_iter + 1):
        ## Evaluate log-domain responsibilities
        residual = x[:, None, :] - means[None, :, :]  # (N, K, D)
        log_prob = -.5 * (torch.log(2 * torch.pi * variances)[None, :, :] + residual.square() / variances[None, :, :]).sum(dim=2)  # (N, K)
        log_joint = log_prob + weights.log()[None, :]  # (N, K)
        log_density = torch.logsumexp(log_joint, dim=1)  # (N,)
        likelihood = float(log_density.mean().item())
        if previous is not None and abs(likelihood - previous) <= 1e-4:
            converged = True
            break
        responsibilities = (log_joint - log_density[:, None]).exp()  # (N, K)
        ## Update mixture weights, means and regularized variances
        mass = responsibilities.sum(dim=0).clamp_min(1e-8)  # (K,)
        weights = mass / mass.sum()  # (K,)
        means = (responsibilities.T @ x) / mass[:, None]  # (K, D)
        residual = x[:, None, :] - means[None, :, :]  # (N, K, D)
        variances = (responsibilities[:, :, None] * residual.square()).sum(dim=0) / mass[:, None] + ridge  # (K, D)
        previous = likelihood
    # Evaluate assignments from the final parameter state, including the last M-step
    residual = x[:, None, :] - means[None, :, :]  # (N, K, D)
    log_prob = -.5 * (torch.log(2 * torch.pi * variances)[None, :, :] + residual.square() / variances[None, :, :]).sum(dim=2)  # (N, K)
    log_joint = log_prob + weights.log()[None, :]  # (N, K)
    if not torch.isfinite(log_joint).all():
        raise ValueError("Nonfinite CUDA Gaussian-mixture likelihood.")
    return log_joint.argmax(dim=1), {"converged":converged,"iterations":iteration,"covariance_type":"diag","gmm_regularization":ridge}


def _dbscan(x: torch.Tensor, min_samples: int, radius_quantile: float) -> tuple[torch.Tensor, dict[str, Any]]:
    """Compute density components and border assignment entirely with CUDA tensors.

    Radius is a prespecified quantile of min_samples-neighbour distances including
    self, never selected using reference labels. Ambiguous border points choose the
    smallest core-component ID. Small diagnostic images only: dense O(N^2) storage;
    label propagation is O(T*N^2), with T <= N.
    """
    # Construct the radius graph and identify core observations
    distance = torch.cdist(x, x, compute_mode="donot_use_mm_for_euclid_dist")  # (N, N)
    if not torch.isfinite(distance).all():
        raise ValueError("CUDA density distances overflowed float32.")
    radius = torch.quantile(distance.kthvalue(min_samples, dim=1).values, radius_quantile)  # ()
    adjacent = distance <= radius  # (N, N)
    core = adjacent.sum(dim=1) >= min_samples  # (N,)
    core_edges = adjacent & core[:, None] & core[None, :]  # (N, N)
    # Resolve connected components among core observations
    sentinel = len(x)
    components = torch.where(core, torch.arange(len(x), device=x.device), sentinel)  # (N,)
    for iteration in range(1, len(x) + 1):
        candidates = torch.where(core_edges, components[None, :], sentinel).amin(dim=1)  # (N,)
        updated = torch.minimum(components, candidates)  # (N,)
        if torch.equal(updated, components):
            break
        components = updated
    # Assign core and border observations; retain isolated points as noise
    roots = torch.unique(components[core])  # (K,)
    nearest_core = torch.where(adjacent & core[None, :], components[None, :], sentinel).amin(dim=1)  # (N,)
    labels = torch.full((len(x),), -1, dtype=torch.int64, device=x.device)  # (N,)
    if len(roots):
        available = nearest_core < sentinel  # (N,)
        labels[available] = torch.searchsorted(roots, nearest_core[available])
    return labels, {"converged":True,"iterations":iteration,"density_radius":float(radius.item()),
                    "radius_quantile":radius_quantile,"min_samples":min_samples,"covariance_type":None}


def cluster_on_cuda(values: np.ndarray, *, method: str, classes: int, seed: int,
                    min_samples: int, radius_quantile: float, gmm_regularization: float,
                    max_iter: int = 100) -> tuple[np.ndarray, dict[str, Any]]:
    """Cluster raw/learned observations with an explicitly enforced CUDA backend.

    K-means++ uses 5 initializations. Gaussian EM uses diagonal covariances. DBSCAN
    is a distinct density pipeline from archived CPU HDBSCAN. No CPU reference
    estimator is evaluated. K-means cost O(I*N*K*D); GMM O(I*N*K*D), auxiliary
    storage O(N*K*D). DBSCAN uses O(N^2*D + T*N^2), memory O(N^2 + N*D).

    :param values: Finite observations, shape ``(N, D)``.
    :param method: kmeans, gmm_diag or dbscan.
    :param classes: Prespecified K for K-means/GMM.
    :param seed: CUDA generator seed.
    :param min_samples: DBSCAN density count including self.
    :param radius_quantile: Frozen rule for the per-image density radius.
    :param gmm_regularization: Positive covariance ridge.
    :param max_iter: Positive optimization limit.
    :return: Assignments and explicit backend/estimator diagnostics.
    :raises RuntimeError: If CUDA is unavailable.
    :raises ValueError: For invalid input/settings or nonfinite results.
    """
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA partition methods refuse CPU fallback.")
    x = torch.as_tensor(values, dtype=torch.float32, device="cuda")  # (N, D)
    if x.ndim != 2 or not x.numel() or not torch.isfinite(x).all() or not 2 <= classes <= len(x) or not 2 <= min_samples <= len(x) or not 0 <= radius_quantile <= 1 or not np.isfinite(gmm_regularization) or gmm_regularization <= 0 or max_iter < 1:
        raise ValueError("Invalid CUDA clustering data or settings.")
    with torch.inference_mode():
        if method == "kmeans":
            labels, _, iterations = _kmeans(x,classes,seed,max_iter,n_init=5)
            details = {"converged":iterations<max_iter,"iterations":iterations,"covariance_type":None}
        elif method == "gmm_diag":
            labels, details = _gmm_diag(x,classes,seed,gmm_regularization,max_iter)
        elif method == "dbscan":
            labels, details = _dbscan(x,min_samples,radius_quantile)
        else:
            raise ValueError("Supported CUDA methods: kmeans, gmm_diag, dbscan.")
    logger.debug("CUDA partition completed: method=%s rows=%s dimensions=%s iterations=%s.", method, len(x), x.shape[1], details["iterations"])
    details.update(backend="torch_cuda",compute_dtype="float32")
    return labels.cpu().numpy(), details


def latent_rank_on_cuda(values: np.ndarray) -> float:
    """Compute effective covariance rank of a small latent on CUDA.

    :param values: Finite encoder latent matrix, shape ``(N, D)``, N >= 2.
    :return: Effective covariance rank; zero for a constant representation.
    """
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA latent diagnostics refuse CPU fallback.")
    x = torch.as_tensor(values,dtype=torch.float32,device="cuda")  # (N, D)
    if x.ndim != 2 or len(x) < 2 or not x.shape[1] or not torch.isfinite(x).all():
        raise ValueError("Latent rank requires a finite matrix with at least two rows.")
    # Scale before covariance construction to prevent overflow for finite latents
    scale = x.abs().amax()  # ()
    if scale > 0:
        x = x / scale  # (N, D)
    # Compute the covariance spectrum and its entropy-based effective rank
    centered = x - x.mean(dim=0,keepdim=True)  # (N, D)
    eigen = torch.linalg.eigvalsh(centered.T @ centered).clamp_min(0)  # (D,)
    total = eigen.sum()  # ()
    if total == 0:
        return 0.
    proportions = eigen / total  # (D,)
    positive = proportions > 0
    return float(torch.exp(-(proportions[positive]*proportions[positive].log()).sum()).item())
