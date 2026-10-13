"""Fixed unsupervised clustering and chemical annotation overlap diagnostics."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.cluster import HDBSCAN, KMeans
from sklearn.metrics import adjusted_rand_score, homogeneity_completeness_v_measure

from .segmentation import fit_segmentation
from ....utils.logger import get_custom_logger

logger = get_custom_logger(__name__)


def cluster_latents(latents: np.ndarray, *, method: str, seed: int, n_clusters: int,
                    min_cluster_size: int = 10) -> np.ndarray:
    """Cluster encoder latents with parameters fixed before reference scoring.

    :param latents: Finite matrix, shape ``(N, D)``.
    :param method: kmeans, hdbscan or gmm; UMAP is not silently applied.
    :param seed: Reproducible initialization for stochastic estimators.
    :param n_clusters: Prespecified component count for K-means/GMM.
    :param min_cluster_size: Prespecified HDBSCAN density threshold.
    :return: Integer assignments, shape ``(N,)``; -1 denotes HDBSCAN noise.
    :raises ValueError: For invalid data or method.
    """
    z = np.asarray(latents, dtype=np.float64)  # (N, D)
    if z.ndim != 2 or not z.size or not np.isfinite(z).all():
        raise ValueError("Clustering requires a nonempty finite latent matrix.")
    logger.info("Clustering %s rows using %s, seed %s.", len(z), method, seed)
    if method == "kmeans":
        return KMeans(n_clusters=n_clusters, n_init=10, random_state=seed).fit_predict(z).astype(np.int64)  # (N,)
    if method == "hdbscan":
        # REMARK: sklearn HDBSCAN is deterministic for the fixed data/parameters;
        # it has no random_state parameter. Model seeds remain separate metadata.
        return HDBSCAN(min_cluster_size=min_cluster_size, copy=True).fit_predict(z).astype(np.int64)  # (N,)
    if method == "gmm":
        return fit_segmentation(z, n_clusters, seed=seed, sample_size=len(z)).predict(z)  # (N,)
    raise ValueError("Supported fixed pipelines: kmeans, hdbscan, gmm.")


def chemical_cluster_profiles(clusters: np.ndarray, targets: np.ndarray, availability: np.ndarray,
                              *, annotations_complete: bool) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Report chemical enrichment and partition agreement per independent class.

    With incomplete PU annotations, rates are observed-positive fractions;
    supervised ARI/homogeneity are unavailable. A chemical multilabel vector is
    never forced into one anatomical class. Complexity O(N*C*K), memory O(C*K).

    :param clusters: Integer partition assignments, shape ``(N,)``.
    :param targets: Binary chemical annotations, shape ``(N, C)``.
    :param availability: Explicit observed-entry mask, shape ``(N, C)``.
    :param annotations_complete: Whether zeros are confirmed absences.
    :return: Cluster/class rate table and optional per-class partition scores.
    :raises ValueError: For inconsistent shapes or invalid annotations.
    """
    labels, y, mask = np.asarray(clusters), np.asarray(targets), np.asarray(availability)
    if y.ndim != 2 or labels.shape != (len(y),) or mask.shape != y.shape or mask.dtype != bool or not np.isin(y, [0, 1]).all():
        raise ValueError("Chemical profiles require binary N by C targets and an explicit boolean mask.")
    rates, scores = [], []
    for c in range(y.shape[1]):
        available = mask[:, c]  # (N,)
        for cluster in np.unique(labels):
            selected = available & (labels == cluster)  # (N,)
            count = int(selected.sum())
            rates.append({"cluster": int(cluster), "class_index": c, "available_entries": count,
                          "observed_positive_fraction": float(y[selected, c].mean()) if count else None})
        valid = available & (labels != -1)  # (N,)
        record = {"class_index": c, "scored_pixels": int(valid.sum()), "annotations_complete": annotations_complete,
                  "ari": None, "homogeneity": None, "completeness": None}
        if annotations_complete and valid.sum() >= 2 and len(np.unique(y[valid, c])) > 1:
            homogeneity, completeness, _ = homogeneity_completeness_v_measure(y[valid, c], labels[valid])
            record.update(ari=float(adjusted_rand_score(y[valid, c], labels[valid])),
                          homogeneity=float(homogeneity), completeness=float(completeness))
        scores.append(record)
    return pd.DataFrame(rates), pd.DataFrame(scores)
