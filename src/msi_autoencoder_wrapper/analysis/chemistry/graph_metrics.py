"""Structural statistics of taxonomies and label-graph variants."""

from __future__ import annotations

from itertools import combinations
from typing import Mapping, Sequence

import numpy as np
import pandas as pd
from scipy.sparse.csgraph import connected_components
from sklearn.metrics import adjusted_mutual_info_score

from msi_dataset_manager.annotations.chemistry import ChemicalTaxonomy

from .label_graph import LabelGraph


def mean_pairwise_cosine(rows: np.ndarray) -> float:
    """Return the mean cosine similarity over all distinct row pairs.

    Uses :math:`\\sum_{i \\ne j} u_i^\\top u_j = \\lVert \\sum_i u_i \\rVert^2 - n`
    for unit rows :math:`u_i`, so the cost is linear in the number of rows.

    :param rows: Matrix ``(n, d)`` (NumPy array or torch tensor); zero rows are
        excluded.
    :type rows: numpy.ndarray | torch.Tensor
    :return: Mean cosine similarity, ``nan`` for fewer than two non-zero rows.
    :rtype: float
    """
    if not isinstance(rows, np.ndarray):
        ## Torch tensor: same computation on its device
        norms = rows.norm(dim=1)  # (n,)
        unit = rows[norms > 0] / norms[norms > 0, None]  # (n_+, d)
    else:
        norms = np.linalg.norm(rows, axis=1)  # (n,)
        unit = rows[norms > 0] / norms[norms > 0, None]  # (n_+, d)
    count = unit.shape[0]
    if count < 2:
        return float("nan")
    total = unit.sum(axis=0)  # (d,)
    return float((total @ total - count) / (count * (count - 1)))


def structure_metrics(
    graph: LabelGraph,
    *,
    trainable_support: int = 5,
    propagation_steps: Sequence[int] = (1, 2, 3),
    device: str | None = None,
) -> dict[str, float]:
    """Summarize one realized label graph.

    :param graph: Realized label graph.
    :type graph: LabelGraph
    :param trainable_support: Minimum number of ions for a class label to count as
        trainable.
    :type trainable_support: int
    :param propagation_steps: GCN depths ``k`` for the over-smoothing statistic.
    :type propagation_steps: collections.abc.Sequence[int]
    :param device: Torch device for propagation; ``None`` uses SciPy/NumPy.
    :type device: str | None
    :return: Flat metric record; see the notebook methodology for definitions.
    :rtype: dict[str, float]
    """
    # Coverage and label support
    resolved = graph.resolved  # (A,)
    ion_count, resolved_count = len(graph.ion_ids), int(resolved.sum())
    binary = (graph.membership > 0).astype(np.float32).tocsc()  # (A, C)
    support = np.asarray(binary.sum(axis=0)).ravel()  # (C,)
    classes_per_ion = np.asarray(binary.sum(axis=1)).ravel()[resolved]  # (A_+,)
    relation_counts = graph.edges.relation.value_counts()
    direct = graph.edges[graph.edges.relation == "belongs_to"].groupby("source").size()
    record: dict[str, float] = {
        "ion_count": ion_count,
        "resolved_ions": resolved_count,
        "coverage": resolved_count / ion_count if ion_count else float("nan"),
        "class_count": len(graph.class_ids),
        "max_depth": float(graph.classes.depth.max()) if len(graph.classes) else float("nan"),
        "classes_per_ion_mean": float(classes_per_ion.mean()) if resolved_count else float("nan"),
        "direct_classes_per_ion_mean": float(direct.mean()) if len(direct) else float("nan"),
        "ions_per_class_median": float(np.median(support)) if support.size else float("nan"),
        "max_class_fraction": float(support.max() / resolved_count) if support.size and resolved_count else float("nan"),
        "trainable_labels": int((support >= trainable_support).sum()),
    }
    for relation in ("belongs_to", "is_a", "equivalent_to", "same_formula"):
        record[f"edges_{relation}"] = int(relation_counts.get(relation, 0))

    # Redundant classes: identical ion sets
    if support.size:
        columns = binary.T.tocsr()
        signatures = {tuple(columns.indices[columns.indptr[i]:columns.indptr[i + 1]]) for i in range(columns.shape[0])}
        record["redundant_class_fraction"] = 1.0 - len(signatures) / columns.shape[0]
    else:
        record["redundant_class_fraction"] = float("nan")

    # Ion pairs connected through a shared class
    shared = (binary @ binary.T).tocsr()  # (A, A)
    if resolved_count > 1:
        rows = shared[resolved][:, resolved]
        record["co_class_pair_fraction"] = float((rows.getnnz() - resolved_count) / (resolved_count * (resolved_count - 1)))
    else:
        record["co_class_pair_fraction"] = float("nan")

    # Connectivity
    adjacency = graph.adjacency(normalized=False, self_loops=False)
    _, labels = connected_components(adjacency, directed=False)
    ion_labels = labels[: ion_count][resolved]
    record["components_with_ions"] = int(np.unique(ion_labels).size) if resolved_count else 0
    record["largest_component_ion_fraction"] = (float(np.bincount(ion_labels).max() / resolved_count)
                                                if resolved_count else float("nan"))

    # Over-smoothing of propagated ion representations
    sequence = graph.propagation_sequence(tuple(int(k) for k in propagation_steps), device=device, as_tensor=device is not None)
    for steps, embeddings in sequence.items():
        record[f"ion_cosine_k{steps}"] = mean_pairwise_cosine(embeddings[np.flatnonzero(resolved)])  # (A_+, N)
    return record


def level_profile(evidence: pd.DataFrame, taxonomies: Mapping[str, ChemicalTaxonomy]) -> pd.DataFrame:
    """Quantify isomer ambiguity at every hierarchy depth.

    For an ion and depth :math:`d`, ``possible`` classes are those supported by at
    least one candidate and ``certain`` classes those supported by all candidates.
    An ion is *unambiguous at depth d* when it has exactly one possible class there,
    which is then necessarily certain.

    :param evidence: Class evidence with ``taxonomy``, ``resolution``, ``ion_id``,
        ``class_id``, ``certain``.
    :type evidence: pandas.DataFrame
    :param taxonomies: Loaded taxonomies by name.
    :type taxonomies: collections.abc.Mapping[str, ChemicalTaxonomy]
    :return: One row per taxonomy, resolution and depth.
    :rtype: pandas.DataFrame
    """
    frame = evidence.copy()
    frame["depth"] = [taxonomies[t].class_depth.get(c, -1) for t, c in zip(frame.taxonomy, frame.class_id)]
    rows = []
    for (taxonomy, resolution), group in frame.groupby(["taxonomy", "resolution"], sort=True):
        resolved = group.ion_id.nunique()
        for depth, level in group.groupby("depth", sort=True):
            per_ion = level.groupby("ion_id").agg(possible=("class_id", "size"), certain=("certain", "sum"))
            rows.append({
                "taxonomy": taxonomy, "resolution": resolution, "depth": int(depth),
                "resolved_ions": resolved,
                "ions_possible": int(len(per_ion)),
                "ions_certain": int((per_ion.certain > 0).sum()),
                "ions_unambiguous": int((per_ion.possible == 1).sum()),
                "possible_classes_per_ion_mean": float(per_ion.possible.mean()),
                "classes_possible": int(level.class_id.nunique()),
                "classes_certain": int(level.loc[level.certain.astype(bool), "class_id"].nunique()),
            })
    return pd.DataFrame(rows)


def taxonomy_agreement(
    evidence: pd.DataFrame,
    taxonomies: Mapping[str, ChemicalTaxonomy],
    *,
    resolution: str,
    max_depth: int = 6,
    min_ions: int = 10,
) -> pd.DataFrame:
    """Measure how consistently two taxonomies partition the same ions.

    At depths :math:`d_a` and :math:`d_b` each ion with exactly one certain class in
    both taxonomies receives two labels; their adjusted mutual information (AMI) is
    1 for identical partitions up to relabelling and about 0 for independent ones.

    :param evidence: Class evidence table.
    :type evidence: pandas.DataFrame
    :param taxonomies: Loaded taxonomies by name.
    :type taxonomies: collections.abc.Mapping[str, ChemicalTaxonomy]
    :param resolution: Candidate resolution used for both taxonomies.
    :type resolution: str
    :param max_depth: Deepest level compared.
    :type max_depth: int
    :param min_ions: Minimum number of jointly labelled ions for a reported value;
        level pairs where either side has a single class are skipped.
    :type min_ions: int
    :return: Rows ``taxonomy_a``, ``depth_a``, ``taxonomy_b``, ``depth_b``,
        ``ions``, ``classes_a``, ``classes_b``, ``ami``.
    :rtype: pandas.DataFrame
    """
    certain = evidence[(evidence.resolution == resolution) & evidence.certain.astype(bool)].copy()
    certain["depth"] = [taxonomies[t].class_depth.get(c, -1) for t, c in zip(certain.taxonomy, certain.class_id)]
    single = certain.groupby(["taxonomy", "depth", "ion_id"]).class_id.transform("size") == 1
    unique = certain[single].set_index(["taxonomy", "depth", "ion_id"]).class_id.sort_index()
    levels = set(unique.index.droplevel(2))
    rows = []
    names = sorted(certain.taxonomy.unique())
    for left, right in combinations(names, 2):
        for depth_a in range(0, max_depth + 1):
            for depth_b in range(0, max_depth + 1):
                if (left, depth_a) not in levels or (right, depth_b) not in levels:
                    continue
                labels_a, labels_b = unique.loc[(left, depth_a)], unique.loc[(right, depth_b)]
                common = labels_a.index.intersection(labels_b.index)
                if len(common) < min_ions:
                    continue
                ### REMARK: AMI of a constant labelling is 1 by convention and carries no
                ### information about agreement; such level pairs are skipped.
                if labels_a[common].nunique() < 2 or labels_b[common].nunique() < 2:
                    continue
                rows.append({"taxonomy_a": left, "depth_a": depth_a, "taxonomy_b": right, "depth_b": depth_b,
                             "ions": len(common), "classes_a": labels_a[common].nunique(),
                             "classes_b": labels_b[common].nunique(),
                             "ami": adjusted_mutual_info_score(labels_a[common], labels_b[common])})
    return pd.DataFrame(rows)


def class_contingency(
    evidence: pd.DataFrame,
    taxonomies: Mapping[str, ChemicalTaxonomy],
    *,
    resolution: str,
    left: tuple[str, int],
    right: tuple[str, int],
) -> pd.DataFrame:
    """Return ion counts per pair of unique certain classes at two levels.

    :param evidence: Class evidence table.
    :type evidence: pandas.DataFrame
    :param taxonomies: Loaded taxonomies by name.
    :type taxonomies: collections.abc.Mapping[str, ChemicalTaxonomy]
    :param resolution: Candidate resolution.
    :type resolution: str
    :param left: ``(taxonomy, depth)`` defining rows.
    :type left: tuple[str, int]
    :param right: ``(taxonomy, depth)`` defining columns.
    :type right: tuple[str, int]
    :return: Contingency table indexed by class names.
    :rtype: pandas.DataFrame
    """
    labels = []
    for taxonomy, depth in (left, right):
        frame = evidence[(evidence.resolution == resolution) & (evidence.taxonomy == taxonomy) & evidence.certain.astype(bool)]
        frame = frame[[taxonomies[taxonomy].class_depth.get(c, -1) == depth for c in frame.class_id]]
        counts = frame.groupby("ion_id").class_id.agg(list)
        names = taxonomies[taxonomy].classes.set_index("class_id").name
        labels.append(counts[counts.map(len) == 1].map(lambda values: names[values[0]]))
    joined = pd.concat(labels, axis=1, join="inner", keys=["left", "right"])
    return pd.crosstab(joined.left, joined.right)

