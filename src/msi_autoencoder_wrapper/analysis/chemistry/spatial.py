"""Projection of METASPACE ion images onto the classes of a label graph."""

from __future__ import annotations

import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score

from .ion_images import IonImages
from .label_graph import LabelGraph


def _graph_rows(images: IonImages, graph: LabelGraph) -> tuple[np.ndarray, np.ndarray]:
    """Return image-ion positions present in the graph and their graph rows."""
    index = {ion: position for position, ion in enumerate(graph.ion_ids)}
    positions = np.array([i for i, ion in enumerate(images.ion_ids) if ion in index], dtype=np.int64)  # (A_g,)
    rows = np.array([index[images.ion_ids[i]] for i in positions], dtype=np.int64)  # (A_g,)
    return positions, rows


def class_abundance(images: IonImages, graph: LabelGraph) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return the share of annotated signal attributed to every graph class.

    With TIC-normalized ion intensities :math:`\\tilde{x}_{p a}` (sum over annotated
    ions of pixel :math:`p` equals one) and closure membership weights
    :math:`w_{a c}`, the class abundance is

    .. math::

        s_{p c} = \\sum_a \\tilde{x}_{p a} w_{a c}.

    Classes overlap (an ion counts towards every ancestor), so abundances of
    nested classes are not additive across the hierarchy.

    :param images: Ion images of one dataset.
    :type images: IonImages
    :param graph: Label graph defining classes and weights.
    :type graph: LabelGraph
    :return: Abundances ``(P_+, C)``, pixel mask ``(P,)`` of pixels with annotated
        signal, and the resolved signal share ``(P_+,)`` — the fraction of annotated
        signal carried by ions with at least one class.
    :rtype: tuple[numpy.ndarray, numpy.ndarray, numpy.ndarray]
    """
    normalized, mask = images.tic_normalized()  # (A, P_+), (P,)
    positions, rows = _graph_rows(images, graph)
    weights = graph.membership[rows].toarray()  # (A_g, C)
    abundance = normalized[positions].T @ weights  # (P_+, C)
    resolved = normalized[positions[graph.resolved[rows]]].sum(axis=0)  # (P_+,)
    return abundance.astype(np.float32), mask, resolved.astype(np.float32)


def effective_class_count(abundance: np.ndarray) -> np.ndarray:
    """Return :math:`\\exp(H)` of the per-pixel normalized class distribution.

    :param abundance: Class abundances ``(P, C)``.
    :type abundance: numpy.ndarray
    :return: Effective number of classes per pixel ``(P,)``; ``nan`` for pixels
        without class signal.
    :rtype: numpy.ndarray
    """
    total = abundance.sum(axis=1, keepdims=True)  # (P, 1)
    with np.errstate(divide="ignore", invalid="ignore"):
        share = np.where(total > 0, abundance / total, 0.0)  # (P, C)
        entropy = -np.sum(np.where(share > 0, share * np.log(share), 0.0), axis=1)  # (P,)
    return np.where(total[:, 0] > 0, np.exp(entropy), np.nan)


def pixel_representations(
    images: IonImages,
    graph: LabelGraph,
    steps: tuple[int, ...],
    *,
    device: str | None = None,
) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    """Return pixel representations in ion space and in label space.

    ``ions`` is the TIC-normalized ion composition :math:`\\tilde{x}_p`.
    ``classes`` is the class abundance :math:`s_p`. ``graph_k`` is
    :math:`\\tilde{x}_p^\\top E^{(k)}`, the pixel signature after ``k`` steps of
    symmetric graph propagation of one-hot node features (:math:`E^{(k)}` are the
    ion rows of :math:`\\hat{A}^k`). Ions absent from the graph are excluded from all
    representations so that the comparison is restricted to the same signal.

    :param images: Ion images of one dataset.
    :type images: IonImages
    :param graph: Label graph.
    :type graph: LabelGraph
    :param steps: Propagation depths ``k``.
    :type steps: tuple[int, ...]
    :param device: Torch device for graph propagation; ``None`` uses SciPy.
    :type device: str | None
    :return: Mapping from representation name to ``(matrix (P_+, D), mask (P,))``;
        label-space matrices are distance-preserving factorizations (see
        :func:`_gram_factor`).
    :rtype: dict[str, tuple[numpy.ndarray, numpy.ndarray]]
    """
    normalized, mask = images.tic_normalized()  # (A, P_+)
    positions, rows = _graph_rows(images, graph)
    ## REMARK: Only ions with at least one class enter every representation. An
    ## unresolved ion would vanish from the class representation but keep its one-hot
    ## identity in graph_k, which would inflate the agreement of graph_k with ion space.
    resolved = graph.resolved[rows]
    positions, rows = positions[resolved], rows[resolved]
    composition = normalized[positions].T  # (P_+, A_r)
    result = {"ions": (composition, mask),
              "classes": (composition @ _gram_factor(graph.membership[rows].toarray()), mask)}
    for k, embedding in graph.propagation_sequence(tuple(int(k) for k in steps), device=device).items():
        result[f"graph_{k}"] = (composition @ _gram_factor(embedding[rows]), mask)  # (P_+, A_r)
    return result


def _gram_factor(embedding: np.ndarray) -> np.ndarray:
    """Return :math:`L` with :math:`L L^\top = E E^\top` for ion embeddings ``E``.

    :param embedding: Ion embeddings ``(A_g, D)``.
    :type embedding: numpy.ndarray
    :return: Factor ``(A_g, A_g)``.
    :rtype: numpy.ndarray

    REMARK: Pixel representations :math:`\tilde{x}_p^\top E` live in a
    ``D``-dimensional space (``D`` up to thousands of graph nodes), which would need
    gigabytes per image. Euclidean distances between them depend only on the Gram
    matrix :math:`E E^\top`, so :math:`\tilde{x}_p^\top L` gives identical k-means
    geometry in ``A_g`` dimensions.
    """
    gram = embedding.astype(np.float64) @ embedding.T.astype(np.float64)  # (A_g, A_g)
    values, vectors = np.linalg.eigh(gram)
    return (vectors * np.sqrt(np.clip(values, 0.0, None))).astype(np.float32)  # (A_g, A_g)


def spatial_coherence(images: IonImages, mask: np.ndarray, labels: np.ndarray) -> float:
    """Return the fraction of 4-neighbour pixel pairs carrying the same label.

    :param images: Ion images providing the pixel grid.
    :type images: IonImages
    :param mask: Pixel mask ``(P,)`` selecting labelled pixels.
    :type mask: numpy.ndarray
    :param labels: Integer labels of masked pixels ``(P_+,)``.
    :type labels: numpy.ndarray
    :return: Agreement fraction; higher values mean spatially smoother labels.
    :rtype: float
    """
    grid = images.grid(labels.astype(np.float32), mask)  # (H, W)
    agree, total = 0, 0
    for left, right in ((grid[:, :-1], grid[:, 1:]), (grid[:-1, :], grid[1:, :])):
        valid = ~np.isnan(left) & ~np.isnan(right)
        agree += int((left[valid] == right[valid]).sum())
        total += int(valid.sum())
    return agree / total if total else float("nan")


def clustering_agreement(
    images: IonImages,
    graph: LabelGraph,
    *,
    steps: tuple[int, ...],
    n_clusters: int,
    sample_size: int,
    seed: int,
    device: str | None = None,
) -> list[dict[str, float]]:
    """Compare k-means segmentations in ion space and in label spaces.

    Each representation is clustered by k-means fitted on the same seeded pixel
    sample and applied to all pixels. Agreement with the ion-space segmentation is
    the adjusted Rand index (ARI); spatial smoothness is
    :func:`spatial_coherence`.

    :param images: Ion images of one dataset.
    :type images: IonImages
    :param graph: Label graph.
    :type graph: LabelGraph
    :param steps: Propagation depths for ``graph_k`` representations.
    :type steps: tuple[int, ...]
    :param n_clusters: Number of k-means clusters.
    :type n_clusters: int
    :param sample_size: Maximum number of pixels used to fit k-means.
    :type sample_size: int
    :param seed: Seed of the pixel sample and k-means initialization.
    :type seed: int
    :param device: Torch device for graph propagation; ``None`` uses SciPy.
    :type device: str | None
    :return: One record per representation; empty if the image has fewer pixels
        than clusters or fewer than two resolved ions in the variant.
    :rtype: list[dict[str, float]]
    """
    representations = pixel_representations(images, graph, steps, device=device)
    reference_matrix, mask = representations["ions"]
    count = reference_matrix.shape[0]
    if count < n_clusters or reference_matrix.shape[1] < 2:
        return []
    sample = np.random.default_rng(seed).choice(count, size=min(sample_size, count), replace=False)  # (S,)
    labels = {}
    for name, (matrix, _) in representations.items():
        if not np.isfinite(matrix).all() or np.allclose(matrix, 0):
            continue
        model = KMeans(n_clusters=n_clusters, n_init=4, random_state=seed).fit(matrix[sample])
        labels[name] = model.predict(matrix)  # (P_+,)
    return [{"representation": name, "ari_vs_ions": float(adjusted_rand_score(labels["ions"], values)),
             "spatial_coherence": spatial_coherence(images, mask, values), "pixels": int(count),
             "resolved_ions": int(reference_matrix.shape[1])}
            for name, values in labels.items()]
