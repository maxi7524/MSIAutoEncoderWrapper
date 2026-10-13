"""Variants of the chemical label graph over annotated ions and chemical classes.

The label graph :math:`G = (V_M \\cup V_C, E, R)` has one node per annotated ion
(``formula|adduct``) and one node per chemical class used by at least one ion.
Relation types :math:`R` are:

``belongs_to``
    ion -> class, weighted by the class assignment rule;
``is_a``
    class -> nearest retained ancestor class of the same taxonomy;
``equivalent_to``
    class <-> class of another taxonomy declared ``EXACT`` (or, for the
    ``declared`` alignment, also ``BROAD``/``NARROW``) by ChemOnt synonyms;
``same_formula``
    ion <-> ion with the same neutral formula and a different adduct.

A :class:`GraphVariant` fixes every structural decision, so that variants can be
enumerated and compared without changing any other part of the analysis.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from itertools import combinations
from typing import Mapping, Sequence

import numpy as np
import pandas as pd
from scipy import sparse

from msi_dataset_manager.annotations.chemistry import RESOLUTION_MODES, ChemicalTaxonomy

from ...utils.logger import get_custom_logger

logger = get_custom_logger(__name__)

TAXONOMY_ABBREVIATIONS = {"LIPID_MAPS": "LM", "ClassyFire": "CF", "ChEBI": "ChEBI", "SwissLipids": "SL"}
ASSIGNMENT_RULES = ("certain", "possible", "weighted")
ATTACHMENTS = ("most_specific", "all_ancestors")
ALIGNMENTS = ("none", "exact", "declared")
ALIGNMENT_PREDICATES = {"none": (), "exact": ("EXACT",), "declared": ("EXACT", "BROAD", "NARROW")}
PRUNING_RULES = ("none", "informative")
RELATIONS = ("belongs_to", "is_a", "equivalent_to", "same_formula")
EVIDENCE_COLUMNS = ("taxonomy", "resolution", "ion_id", "class_id", "support_fraction", "certain", "candidate_count")


def class_taxonomy(class_id: str) -> str:
    """Return the taxonomy namespace encoded in a class identifier."""
    return class_id.split(":", maxsplit=1)[0]


@dataclass(frozen=True)
class GraphVariant:
    """Complete structural specification of one label-graph variant.

    :param taxonomies: Non-empty tuple of taxonomy names contributing classes.
    :type taxonomies: tuple[str, ...]
    :param resolution: Candidate resolution, one of ``metaspace_ids`` or ``formula``.
    :type resolution: str
    :param assignment: ``certain`` (class shared by every candidate), ``possible``
        (class of any candidate), or ``weighted`` (candidate support fraction).
    :type assignment: str
    :param attachment: ``most_specific`` attaches an ion only to its most specific
        classes and leaves ancestors to ``is_a`` propagation; ``all_ancestors``
        attaches it to its complete class closure.
    :type attachment: str
    :param alignment: ``exact`` adds ``equivalent_to`` edges for ChemOnt ``EXACT``
        synonyms; ``declared`` also for ``BROAD`` and ``NARROW`` synonyms.
    :type alignment: str
    :param adduct_links: Whether ions sharing a neutral formula are linked.
    :type adduct_links: bool
    :param pruning: ``informative`` removes classes with too little support to be
        learned and classes covering most resolved ions of their taxonomy.
    :type pruning: str
    :raises ValueError: If an option is unknown.
    """

    taxonomies: tuple[str, ...]
    resolution: str
    assignment: str
    attachment: str = "all_ancestors"
    alignment: str = "none"
    adduct_links: bool = False
    pruning: str = "none"

    def __post_init__(self) -> None:
        if not self.taxonomies:
            raise ValueError("A graph variant needs at least one taxonomy.")
        options = ((self.resolution, RESOLUTION_MODES), (self.assignment, ASSIGNMENT_RULES),
                   (self.attachment, ATTACHMENTS), (self.alignment, ALIGNMENTS), (self.pruning, PRUNING_RULES))
        for value, allowed in options:
            if value not in allowed:
                raise ValueError(f"Unknown graph option {value!r}; expected one of {allowed}.")

    @property
    def taxonomy_label(self) -> str:
        """Return the compact taxonomy combination label, e.g. ``LM+CF``."""
        return "+".join(TAXONOMY_ABBREVIATIONS.get(name, name) for name in self.taxonomies)

    @property
    def variant_id(self) -> str:
        """Return a stable human-readable identifier of the complete variant."""
        return (f"{self.taxonomy_label}/{self.resolution}/{self.assignment}/{self.attachment}/"
                f"align={self.alignment}/adduct={int(self.adduct_links)}/prune={self.pruning}")

    def to_record(self) -> dict[str, object]:
        """Return a flat record for result tables."""
        record = asdict(self)
        record["taxonomies"] = "+".join(self.taxonomies)
        record.update(variant_id=self.variant_id, taxonomy_label=self.taxonomy_label,
                      taxonomy_count=len(self.taxonomies))
        return record


def variant_grid(taxonomies: Sequence[str], axes: Mapping[str, Sequence[object]]) -> list[GraphVariant]:
    """Enumerate every taxonomy combination crossed with the configured axes.

    Alignments other than ``none`` are only enumerated for combinations of at least two
    taxonomies including ClassyFire, because ChemOnt is the only source declaring
    cross-taxonomy class correspondences; elsewhere it would duplicate ``none``.

    :param taxonomies: Available taxonomy names in display order.
    :type taxonomies: collections.abc.Sequence[str]
    :param axes: Lists for ``resolution``, ``assignment``, ``attachment``,
        ``alignment``, ``adduct_links`` and ``pruning``.
    :type axes: collections.abc.Mapping[str, collections.abc.Sequence[object]]
    :return: Variants in deterministic order.
    :rtype: list[GraphVariant]
    """
    subsets = [subset for size in range(1, len(taxonomies) + 1) for subset in combinations(taxonomies, size)]
    variants = []
    for subset in subsets:
        alignments = [value for value in axes["alignment"]
                      if value == "none" or (len(subset) > 1 and "ClassyFire" in subset)]
        for resolution in axes["resolution"]:
            for assignment in axes["assignment"]:
                for attachment in axes["attachment"]:
                    for alignment in alignments:
                        for adduct_links in axes["adduct_links"]:
                            for pruning in axes["pruning"]:
                                variants.append(GraphVariant(tuple(subset), str(resolution), str(assignment),
                                                             str(attachment), str(alignment), bool(adduct_links),
                                                             str(pruning)))
    return variants


@dataclass(frozen=True, eq=False)
class LabelGraph:
    """One realized label graph.

    :param variant: Structural specification.
    :type variant: GraphVariant
    :param ion_ids: All ions of the collection; ions without classes are isolated
        nodes, so every variant has the same ion node set.
    :type ion_ids: tuple[str, ...]
    :param classes: Retained classes with ``class_id``, ``taxonomy``, ``name``,
        ``depth`` and ``support`` (number of ions).
    :type classes: pandas.DataFrame
    :param membership: Ion-to-class closure weights after pruning, shape ``(A, C)``.
    :type membership: scipy.sparse.csr_matrix
    :param edges: ``source``, ``target``, ``relation``, ``weight`` over node ids.
    :type edges: pandas.DataFrame
    """

    variant: GraphVariant
    ion_ids: tuple[str, ...]
    classes: pd.DataFrame
    membership: sparse.csr_matrix
    edges: pd.DataFrame

    @property
    def class_ids(self) -> tuple[str, ...]:
        """Return class node identifiers in membership column order."""
        return tuple(self.classes.class_id)

    @property
    def node_ids(self) -> tuple[str, ...]:
        """Return ion nodes followed by class nodes."""
        return self.ion_ids + self.class_ids

    @property
    def resolved(self) -> np.ndarray:
        """Return the mask of ions with at least one retained class, shape ``(A,)``."""
        return np.asarray(self.membership.getnnz(axis=1) > 0)

    def adjacency(self, *, normalized: bool = True, self_loops: bool = True) -> sparse.csr_matrix:
        """Return the symmetric weighted adjacency over :attr:`node_ids`.

        :param normalized: Apply :math:`D^{-1/2} (A + I) D^{-1/2}` as in GCN.
        :type normalized: bool
        :param self_loops: Add the identity before normalization.
        :type self_loops: bool
        :return: Sparse matrix of shape ``(N, N)`` with ``N = A + C``.
        :rtype: scipy.sparse.csr_matrix

        REMARK: Relations are symmetrized. A relational GCN would keep one weight
        matrix per relation and direction; the symmetric single-relation operator
        is used here only to characterize how far information can propagate.
        """
        index = {node: position for position, node in enumerate(self.node_ids)}
        size = len(index)
        rows = self.edges.source.map(index).to_numpy()
        columns = self.edges.target.map(index).to_numpy()
        weights = self.edges.weight.to_numpy(dtype=np.float64)
        matrix = sparse.coo_matrix((weights, (rows, columns)), shape=(size, size)).tocsr()
        matrix = matrix + matrix.T  # (N, N)
        if self_loops:
            matrix = matrix + sparse.identity(size, format="csr")
        if not normalized:
            return matrix.tocsr()
        degree = np.asarray(matrix.sum(axis=1)).ravel()  # (N,)
        scale = sparse.diags(np.where(degree > 0, degree ** -0.5, 0.0))
        return (scale @ matrix @ scale).tocsr()

    def propagated_ion_embeddings(self, steps: int) -> np.ndarray:
        """Return ion rows of :math:`\\hat{A}^k` for one-hot initial node features.

        With :math:`H^{(0)} = I` and no learned weights, :math:`H^{(k)} = \\hat{A}^k`
        is the receptive field of a ``k``-layer linear GCN (SGC). Its ion rows show
        which nodes can influence each ion's classifier vector.

        :param steps: Number of propagation steps ``k >= 0``.
        :type steps: int
        :return: Dense matrix of shape ``(A, N)``.
        :rtype: numpy.ndarray
        """
        return self.propagation_sequence((steps,))[steps]

    def propagation_sequence(
        self,
        steps: tuple[int, ...],
        *,
        device: str | None = None,
        as_tensor: bool = False,
    ) -> dict[int, np.ndarray]:
        """Return :meth:`propagated_ion_embeddings` for several depths in one pass.

        :param steps: Requested depths ``k``.
        :type steps: tuple[int, ...]
        :param device: Torch device (``"cuda"``/``"cpu"``) for sparse-dense
            propagation; ``None`` uses SciPy on the CPU.
        :type device: str | None
        :param as_tensor: Return torch tensors on ``device`` instead of arrays.
        :type as_tensor: bool
        :return: Mapping ``k -> (A, N)`` matrix.
        :rtype: dict[int, numpy.ndarray]
        """
        operator = self.adjacency()
        count = len(self.ion_ids)
        result = {}
        if device is None:
            state = sparse.identity(operator.shape[0], format="csr")[:, :count].toarray().astype(np.float32)  # (N, A)
            for k in range(max(steps) + 1):
                if k in steps:
                    result[k] = np.ascontiguousarray(state.T)  # (A, N)
                state = operator @ state  # (N, A)
            return result
        import torch

        coo = operator.tocoo()
        matrix = torch.sparse_coo_tensor(np.vstack([coo.row, coo.col]), coo.data.astype(np.float32),
                                         coo.shape, device=device).coalesce()  # (N, N)
        state = torch.eye(operator.shape[0], count, device=device)  # (N, A)
        for k in range(max(steps) + 1):
            if k in steps:
                result[k] = state.T.contiguous() if as_tensor else state.T.cpu().numpy()  # (A, N)
            state = torch.sparse.mm(matrix, state)  # (N, A)
        return result


def _minimal(classes: set[str], taxonomy: ChemicalTaxonomy) -> set[str]:
    """Return members of ``classes`` that are not a proper ancestor of another member."""
    proper = {ancestor for class_id in classes for ancestor in taxonomy.ancestors(class_id) if ancestor != class_id}
    return classes - proper


def build_label_graph(
    variant: GraphVariant,
    ions: pd.DataFrame,
    evidence: pd.DataFrame,
    taxonomies: Mapping[str, ChemicalTaxonomy],
    *,
    min_class_support: int = 2,
    max_class_fraction: float = 0.5,
) -> LabelGraph:
    """Realize one label-graph variant for an ion collection.

    :param variant: Structural specification.
    :type variant: GraphVariant
    :param ions: Ion table with ``ion_id`` and ``formula``.
    :type ions: pandas.DataFrame
    :param evidence: Class evidence with :data:`EVIDENCE_COLUMNS`, produced by
        :func:`msi_dataset_manager.annotations.chemistry.class_assignments`.
    :type evidence: pandas.DataFrame
    :param taxonomies: Loaded taxonomies by name.
    :type taxonomies: collections.abc.Mapping[str, ChemicalTaxonomy]
    :param min_class_support: Minimum ion count of a class under ``informative``
        pruning.
    :type min_class_support: int
    :param max_class_fraction: Maximum fraction of the taxonomy's resolved ions a
        class may cover under ``informative`` pruning.
    :type max_class_fraction: float
    :return: Realized graph.
    :rtype: LabelGraph
    :raises KeyError: If the variant needs a taxonomy that is not loaded.
    """
    ion_ids = tuple(ions.ion_id)
    ion_index = {ion: position for position, ion in enumerate(ion_ids)}

    # Class evidence under the assignment rule
    selected = evidence[evidence.taxonomy.isin(variant.taxonomies)
                        & (evidence.resolution == variant.resolution)
                        & evidence.ion_id.isin(ion_index)]
    if variant.assignment == "certain":
        selected = selected[selected.certain.astype(bool)].assign(weight=1.0)
    elif variant.assignment == "possible":
        selected = selected.assign(weight=1.0)
    else:
        selected = selected.assign(weight=selected.support_fraction.astype(float))

    # Class pruning
    support = selected.groupby("class_id").ion_id.nunique()
    keep = support[support >= 1]
    if variant.pruning == "informative":
        resolved_per_taxonomy = selected.groupby("taxonomy").ion_id.nunique()
        limit = keep.index.map(lambda c: max_class_fraction * resolved_per_taxonomy[class_taxonomy(c)])
        keep = keep[(keep >= min_class_support) & (keep.to_numpy() <= np.asarray(limit))]
    selected = selected[selected.class_id.isin(keep.index)]
    kept = set(keep.index)

    # Class records
    class_rows = []
    for class_id in sorted(kept):
        taxonomy = taxonomies[class_taxonomy(class_id)]
        class_rows.append((class_id, class_taxonomy(class_id), int(taxonomy.class_depth.get(class_id, -1)),
                           int(keep[class_id])))
    classes = pd.DataFrame(class_rows, columns=["class_id", "taxonomy", "depth", "support"])
    names = pd.concat([taxonomies[name].classes[["class_id", "name"]] for name in variant.taxonomies])
    classes = classes.merge(names.drop_duplicates("class_id"), on="class_id", how="left")
    class_index = {class_id: position for position, class_id in enumerate(classes.class_id)}

    # Ion-class closure membership
    membership = sparse.coo_matrix(
        (selected.weight.to_numpy(dtype=np.float32),
         (selected.ion_id.map(ion_index).to_numpy(), selected.class_id.map(class_index).to_numpy())),
        shape=(len(ion_ids), len(class_index)),
    ).tocsr()  # (A, C)
    membership.sum_duplicates()

    # Edges
    edges: list[tuple[str, str, str, float]] = []
    ## is_a: nearest retained ancestors
    for class_id in classes.class_id:
        taxonomy = taxonomies[class_taxonomy(class_id)]
        ancestors = (set(taxonomy.ancestors(class_id)) & kept) - {class_id}
        edges.extend((class_id, parent, "is_a", 1.0) for parent in sorted(_minimal(ancestors, taxonomy)))
    ## belongs_to: closure or most specific classes per taxonomy
    weights = {(ion, class_id): weight for ion, class_id, weight in selected[["ion_id", "class_id", "weight"]].itertuples(index=False)}
    for (ion, name), group in selected.groupby(["ion_id", "taxonomy"], sort=True):
        targets = set(group.class_id)
        if variant.attachment == "most_specific":
            targets = _minimal(targets, taxonomies[name])
        edges.extend((ion, class_id, "belongs_to", float(weights[(ion, class_id)])) for class_id in sorted(targets))
    ## equivalent_to: declared cross-taxonomy correspondences
    predicates = ALIGNMENT_PREDICATES[variant.alignment]
    for name in variant.taxonomies if predicates else ():
        alignments = taxonomies[name].alignments
        declared = alignments[alignments.predicate.isin(predicates) & alignments.class_id.isin(kept) & alignments.target_id.isin(kept)]
        edges.extend((source, target, "equivalent_to", 1.0) for source, target in declared[["class_id", "target_id"]].itertuples(index=False))
    ## same_formula: adducts of one neutral formula
    if variant.adduct_links:
        for _, group in ions.groupby("formula"):
            members = sorted(group.ion_id)
            edges.extend((left, right, "same_formula", 1.0) for left, right in combinations(members, 2))
    edge_frame = pd.DataFrame(edges, columns=["source", "target", "relation", "weight"]).drop_duplicates(["source", "target", "relation"])
    return LabelGraph(variant, ion_ids, classes, membership, edge_frame.reset_index(drop=True))
