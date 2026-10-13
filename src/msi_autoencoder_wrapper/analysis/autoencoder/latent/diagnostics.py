"""Collapse and gradient diagnostics without fitting a model or modifying weights."""

from __future__ import annotations

from collections.abc import Mapping

import numpy as np
import pandas as pd
import torch

from ....utils.logger import get_custom_logger

logger = get_custom_logger(__name__)


def representation_diagnostics(latents: np.ndarray) -> tuple[dict[str, float | int], pd.DataFrame]:
    """Estimate covariance dimension using a scaled float64 singular spectrum.

    Effective rank is exp(entropy of normalized covariance eigenvalues);
    participation ratio is (sum eigenvalues)^2 / sum squared eigenvalues.
    Constant representations explicitly return zero. Complexity O(N*D*min(N,D)),
    auxiliary storage O(N*D). No target intrinsic dimension is assumed.

    :param latents: Finite nonempty latent matrix, shape ``(N, D)``, N >= 2.
    :return: Scalar collapse diagnostics and normalized eigenvalue table.
    :raises ValueError: For invalid or nonfinite input.
    """
    z = np.asarray(latents, dtype=np.float64)  # (N, D)
    if z.ndim != 2 or len(z) < 2 or z.shape[1] == 0 or not np.isfinite(z).all():
        raise ValueError("Latents require at least two finite rows and one dimension.")
    # Scale before centering to avoid overflow without changing spectral ratios
    scale = float(np.max(np.abs(z)))
    scaled = z / scale if scale else z.copy()  # (N, D)
    centered = scaled - scaled.mean(axis=0, keepdims=True)  # (N, D)
    singular = np.linalg.svd(centered, full_matrices=False, compute_uv=False)  # (min(N,D),)
    eigen = np.zeros(z.shape[1])  # (D,)
    eigen[:len(singular)] = np.square(singular) / (len(z) - 1)
    total = float(eigen.sum())
    proportions = eigen / total if total else eigen  # (D,)
    positive = proportions > 0  # (D,)
    effective = float(np.exp(-np.sum(proportions[positive] * np.log(proportions[positive])))) if total else 0.0
    rank = int(np.count_nonzero(singular > singular[0] * max(centered.shape) * np.finfo(float).eps)) if singular[0] else 0
    summary = {"samples": len(z), "dimensions": z.shape[1], "numerical_rank": rank, "effective_rank": effective,
               "participation_ratio": float(1 / np.square(proportions).sum()) if total else 0.0,
               "constant_dimensions": int(np.sum(np.ptp(scaled, axis=0) == 0)),
               "latent_max_absolute": scale, "scaled_total_variance": total}
    logger.debug("Latent diagnostics: rank %s, effective rank %s.", rank, effective)
    return summary, pd.DataFrame({"component": np.arange(len(eigen)), "scaled_eigenvalue": eigen, "variance_fraction": proportions})


def gradient_diagnostics(gradients: Mapping[str, torch.Tensor | None]) -> pd.DataFrame:
    """Report missing, zero and nonfinite gradients without masking failures.

    :param gradients: Named gradient snapshots; None indicates no gradient route.
    :return: One record per parameter/component.
    """
    rows = []
    for name, gradient in gradients.items():
        if gradient is None:
            rows.append({"name": name, "present": False, "finite": None, "elements": 0, "norm": None, "zero_fraction": None})
            continue
        values = gradient.detach().to(dtype=torch.float64)  # (...)
        finite = bool(torch.isfinite(values).all().item())
        norm = None
        if finite:
            scale = float(values.abs().max().item()) if values.numel() else 0.0
            norm = scale * float(torch.linalg.vector_norm(values / scale).item()) if scale else 0.0
            if not np.isfinite(norm):
                norm = None
        rows.append({"name": name, "present": True, "finite": finite, "elements": values.numel(),
                     "norm": norm,
                     "zero_fraction": float((values == 0).double().mean().item()) if values.numel() else None})
    return pd.DataFrame(rows)
