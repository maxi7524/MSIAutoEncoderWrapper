"""Pure latent-space calculations."""

from __future__ import annotations

from typing import Any, Dict

import numpy as np
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

from ....utils.exceptions import raise_validation_error


def latent_statistics(latents: np.ndarray) -> Dict[str, np.ndarray]:
    """Calculate per-component distribution statistics.

    :param latents: Latent matrix of shape ``(samples, dimensions)``.
    :type latents: numpy.ndarray
    :return: Moments, extrema, norms, and component correlation.
    :rtype: Dict[str, numpy.ndarray]
    """
    values = np.asarray(latents)
    return {
        "mean": np.mean(values, axis=0),
        "std": np.std(values, axis=0),
        "min": np.min(values, axis=0),
        "max": np.max(values, axis=0),
        "correlation": np.corrcoef(values, rowvar=False),
        "norm": np.linalg.norm(values, axis=1),
    }


def project_latents(
    latents: np.ndarray,
    method: str = "pca",
    components: int = 2,
    random_seed: int = 0,
    **kwargs: Any,
) -> np.ndarray:
    """Project latent vectors using PCA or t-SNE.

    :param latents: Latent matrix.
    :type latents: numpy.ndarray
    :param method: ``pca`` or ``tsne``.
    :type method: str
    :param components: Output dimension count.
    :type components: int
    :param random_seed: Reproducible estimator seed.
    :type random_seed: int
    :param kwargs: Additional estimator arguments.
    :return: Projected latent matrix.
    :rtype: numpy.ndarray
    """
    if method == "pca":
        return PCA(
            n_components=components,
            random_state=random_seed,
            **kwargs,
        ).fit_transform(latents)
    if method == "tsne":
        return TSNE(
            n_components=components,
            random_state=random_seed,
            **kwargs,
        ).fit_transform(latents)
    raise_validation_error(
        "LatentAnalysis", "Projection method must be 'pca' or 'tsne'."
    )
