"""Chemical label graphs, graph statistics, spatial projection and m/z ambiguity."""

import numpy as np
import pandas as pd
import pytest

from msi_dataset_manager.annotations.chemistry import ChemicalTaxonomy, class_assignments, resolve_candidates

from msi_autoencoder_wrapper.analysis.chemistry import consistency, graph_metrics, mz_distribution, spatial
from msi_autoencoder_wrapper.analysis.chemistry.ion_images import IonImages
from msi_autoencoder_wrapper.analysis.chemistry.label_graph import GraphVariant, build_label_graph, variant_grid


def _taxonomy(name, classes, edges, compounds, memberships, alignments=()):
    return ChemicalTaxonomy(name, "1", {
        "classes": pd.DataFrame({"class_id": classes, "name": [c.split(":")[-1] for c in classes], "rank": ""}),
        "class_edges": pd.DataFrame(edges, columns=["child_id", "parent_id"]),
        "compounds": pd.DataFrame(compounds, columns=["compound_id", "name", "formula"]),
        "memberships": pd.DataFrame(memberships, columns=["compound_id", "class_id"]),
        "aliases": pd.DataFrame(columns=["alias", "compound_id"]),
        "alignments": pd.DataFrame(list(alignments), columns=["class_id", "target_id", "predicate"]),
    })


@pytest.fixture
def collection():
    """Two taxonomies over five ions; C2H4 is ambiguous between PC and PE in A."""
    a = _taxonomy("A", ["A:root", "A:GP", "A:PC", "A:PE", "A:SP"],
                  [("A:GP", "A:root"), ("A:PC", "A:GP"), ("A:PE", "A:GP"), ("A:SP", "A:root")],
                  [("a1", "", "C2H4"), ("a2", "", "C2H4"), ("a3", "", "C3H6"), ("a4", "", "C4H8"), ("a5", "", "C5H10")],
                  [("a1", "A:PC"), ("a2", "A:PE"), ("a3", "A:PC"), ("a4", "A:PE"), ("a5", "A:SP")],
                  alignments=[("A:GP", "B:lipid", "EXACT")])
    b = _taxonomy("B", ["B:lipid", "B:other"], [],
                  [("b1", "", "C2H4"), ("b2", "", "C3H6"), ("b3", "", "C6H6")],
                  [("b1", "B:lipid"), ("b2", "B:lipid"), ("b3", "B:other")])
    ions = pd.DataFrame({"ion_id": ["C2H4|-H", "C3H6|-H", "C4H8|-H", "C5H10|-H", "C2H4|+Cl", "C9H9|-H"],
                         "formula": ["C2H4", "C3H6", "C4H8", "C5H10", "C2H4", "C9H9"],
                         "mz": [27.0, 41.0, 55.0, 69.0, 63.0, 117.0], "dataset_count": [3, 3, 3, 1, 3, 3]})
    taxonomies = {"A": a, "B": b}
    evidence = pd.concat([class_assignments(resolve_candidates(ions.assign(identifiers=[[]] * len(ions)), t, "formula"), t)
                          .assign(taxonomy=n, resolution="formula") for n, t in taxonomies.items()], ignore_index=True)
    return ions, evidence, taxonomies


def test_variant_grid_enumerates_alignment_only_for_classyfire_combinations():
    axes = {"resolution": ["formula"], "assignment": ["certain"], "attachment": ["all_ancestors"],
            "alignment": ["none", "exact"], "adduct_links": [False], "pruning": ["none"]}
    variants = variant_grid(["LIPID_MAPS", "ClassyFire", "ChEBI"], axes)
    ## 7 subsets; exact alignment only for LM+CF, CF+ChEBI, LM+CF+ChEBI
    assert len(variants) == 7 + 3
    assert {v.taxonomy_label for v in variants if v.alignment == "exact"} == {"LM+CF", "CF+ChEBI", "LM+CF+ChEBI"}


def test_certain_and_possible_assignment_and_attachment(collection):
    ions, evidence, taxonomies = collection
    certain = build_label_graph(GraphVariant(("A",), "formula", "certain", "most_specific"), ions, evidence, taxonomies)
    ## C2H4 candidates are PC and PE -> only GP and root are certain; most specific = GP
    belongs = certain.edges[(certain.edges.relation == "belongs_to") & (certain.edges.source == "C2H4|-H")]
    assert belongs.target.tolist() == ["A:GP"]
    assert certain.resolved.tolist() == [True, True, True, True, True, False]
    possible = build_label_graph(GraphVariant(("A",), "formula", "possible", "all_ancestors"), ions, evidence, taxonomies)
    targets = set(possible.edges[(possible.edges.relation == "belongs_to") & (possible.edges.source == "C2H4|-H")].target)
    assert targets == {"A:PC", "A:PE", "A:GP", "A:root"}
    weighted = build_label_graph(GraphVariant(("A",), "formula", "weighted"), ions, evidence, taxonomies)
    column = list(weighted.class_ids).index("A:PC")
    assert weighted.membership[0, column] == pytest.approx(0.5)
    ## is_a edges reproduce the hierarchy among used classes
    is_a = set(map(tuple, certain.edges[certain.edges.relation == "is_a"][["source", "target"]].to_numpy()))
    assert ("A:PC", "A:GP") in is_a and ("A:GP", "A:root") in is_a


def test_pruning_reconnects_hierarchy_and_alignment_links_taxonomies(collection):
    ions, evidence, taxonomies = collection
    variant = GraphVariant(("A", "B"), "formula", "certain", "all_ancestors", "exact", True, "informative")
    graph = build_label_graph(variant, ions, evidence, taxonomies, min_class_support=2, max_class_fraction=0.7)
    ## root covers 5/5 resolved A-ions (> 0.7) and SP has one ion: both removed
    assert "A:root" not in graph.class_ids and "A:SP" not in graph.class_ids
    ## GP lost its parent root, so it becomes a root itself
    assert not ((graph.edges.relation == "is_a") & (graph.edges.source == "A:GP")).any()
    ## B:lipid covers all resolved B-ions and is pruned too; alignment needs both ends
    assert "B:lipid" not in graph.class_ids and "equivalent_to" not in set(graph.edges.relation)
    unpruned = build_label_graph(GraphVariant(("A", "B"), "formula", "certain", "all_ancestors", "exact", True),
                                 ions, evidence, taxonomies)
    relations = unpruned.edges.groupby("relation").size().to_dict()
    assert relations["equivalent_to"] == 1
    assert relations["same_formula"] == 1  # C2H4|-H <-> C2H4|+Cl


def test_structure_metrics_and_oversmoothing(collection):
    ions, evidence, taxonomies = collection
    graph = build_label_graph(GraphVariant(("A",), "formula", "certain"), ions, evidence, taxonomies)
    metrics = graph_metrics.structure_metrics(graph, trainable_support=2, propagation_steps=(1, 2))
    assert metrics["coverage"] == pytest.approx(5 / 6)
    assert metrics["class_count"] == 5
    assert metrics["max_class_fraction"] == pytest.approx(1.0)  # root holds every resolved ion
    assert 0.0 < metrics["ion_cosine_k1"] < metrics["ion_cosine_k2"] <= 1.0
    assert graph_metrics.mean_pairwise_cosine(np.eye(3)) == pytest.approx(0.0)
    assert graph_metrics.mean_pairwise_cosine(np.ones((3, 2))) == pytest.approx(1.0)


def test_level_profile_counts_ambiguity(collection):
    _, evidence, taxonomies = collection
    profile = graph_metrics.level_profile(evidence, taxonomies).set_index(["taxonomy", "depth"])
    ## depth 2 of A: C2H4 ions have two possible classes (PC, PE), others one
    row = profile.loc[("A", 2)]
    assert row.ions_possible == 4 and row.ions_unambiguous == 2 and row.ions_certain == 2


def test_tanimoto_mantel_and_cka():
    similarity = np.array([[1.0, 0.8, 0.1], [0.8, 1.0, 0.2], [0.1, 0.2, 1.0]])
    result = consistency.mantel_test(similarity, similarity, n_permutations=20, seed=0)
    assert result["rho"] == pytest.approx(1.0) and result["pairs"] == 3
    kernel = np.random.default_rng(0).normal(size=(6, 3))
    kernel = kernel @ kernel.T
    assert consistency.linear_cka(kernel, 3 * kernel) == pytest.approx(1.0)


def test_colocalization_pools_fisher_z_by_pixel_count():
    images = [IonImages("d1", np.array(["a", "b"]), np.array([[0, 0], [1, 0], [2, 0], [3, 0], [4, 0]]),
                        np.array([[1, 2, 3, 4, 5], [5, 4, 3, 2, 1]], dtype=np.float32))]
    pooled, counts = consistency.colocalization(images, ("a", "b", "c"))
    assert pooled[0, 1] == pytest.approx(-1.0, abs=1e-5)
    assert np.isnan(pooled[0, 2]) and counts[0, 1] == 1


def test_class_abundance_is_share_of_annotated_signal(collection):
    ions, evidence, taxonomies = collection
    graph = build_label_graph(GraphVariant(("A",), "formula", "certain"), ions, evidence, taxonomies)
    images = IonImages("d", np.array(["C3H6|-H", "C9H9|-H"]), np.array([[0, 0], [1, 0], [0, 1]]),
                       np.array([[1.0, 3.0, 0.0], [1.0, 1.0, 0.0]], dtype=np.float32))
    abundance, mask, resolved = spatial.class_abundance(images, graph)
    assert mask.tolist() == [True, True, False]
    pc = list(graph.class_ids).index("A:PC")
    assert abundance[:, pc] == pytest.approx([0.5, 0.75])
    assert resolved == pytest.approx([0.5, 0.75])
    assert spatial.effective_class_count(np.array([[1.0, 1.0], [2.0, 0.0]])) == pytest.approx([2.0, 1.0])


def test_gram_factor_preserves_pixel_distances():
    embedding = np.random.default_rng(1).normal(size=(4, 9))
    composition = np.random.default_rng(2).random((5, 4))
    direct, factored = composition @ embedding, composition @ spatial._gram_factor(embedding)
    distance = lambda x: np.linalg.norm(x[:, None] - x[None], axis=-1)
    assert distance(factored) == pytest.approx(distance(direct), abs=1e-4)


def test_bin_ambiguity_metrics_on_analytic_example():
    ## Bin 0: classes P (3) and Q (1); bin 1: P only (2); bin 2: Q only (4)
    counts = mz_distribution.bin_class_counts(np.array([0.1, 0.2, 0.3, 0.4, 1.1, 1.2, 2.1, 2.2, 2.3, 2.4]),
                                              np.array(list("PPPQPPQQQQ")), 1.0, 0.0)
    summary, per_class = mz_distribution.ambiguity_metrics(counts, purity_threshold=0.7)
    assert summary["bayes_accuracy"] == pytest.approx(9 / 10)
    assert summary["unambiguous_bin_fraction"] == pytest.approx(2 / 3)
    assert summary["exclusive_member_fraction"] == pytest.approx(6 / 10)
    per_class = per_class.set_index("class_id")
    assert per_class.loc["P", "exclusive_fraction"] == pytest.approx(2 / 5)
    assert per_class.loc["P", "pure_fraction"] == pytest.approx(1.0)  # 3/4 >= 0.7
    assert per_class.loc["Q", "pure_fraction"] == pytest.approx(4 / 5)


cuda = pytest.mark.skipif(not __import__("torch").cuda.is_available(), reason="CUDA unavailable")


@cuda
def test_device_paths_match_cpu_reference(collection):
    ## Tolerances: float32 on the GPU versus float64/float32 SciPy on the CPU.
    ions, evidence, taxonomies = collection
    graph = build_label_graph(GraphVariant(("A", "B"), "formula", "weighted", "most_specific", "exact", True),
                              ions, evidence, taxonomies)
    cpu = graph.propagation_sequence((0, 2))
    gpu = graph.propagation_sequence((0, 2), device="cuda")
    assert gpu[2] == pytest.approx(cpu[2], abs=1e-5)
    assert graph_metrics.structure_metrics(graph, device="cuda")["ion_cosine_k2"] == pytest.approx(
        graph_metrics.structure_metrics(graph)["ion_cosine_k2"], abs=1e-5)
    selection = np.array([True, True, True, False, True, True])
    for k in (1, 2):
        assert consistency.propagated_kernels(graph, (k,), selection, device="cuda")[k] == pytest.approx(
            consistency.propagated_kernels(graph, (k,), selection)[k], abs=1e-5)


@cuda
def test_batched_mantel_and_correlation_match_cpu():
    generator = np.random.default_rng(3)
    points = generator.normal(size=(12, 2))
    reference = np.corrcoef(generator.normal(size=(12, 30)))
    similarity = np.exp(-np.linalg.norm(points[:, None] - points[None], axis=-1))
    reference[0, 5] = reference[5, 0] = np.nan
    cpu = consistency.mantel_test(similarity, reference, n_permutations=50, seed=4)
    gpu = consistency.mantel_test(similarity, reference, n_permutations=50, seed=4, device="cuda", batch_size=7)
    assert gpu["rho"] == pytest.approx(cpu["rho"]) and gpu["p_value"] == pytest.approx(cpu["p_value"])
    assert gpu["null_mean"] == pytest.approx(cpu["null_mean"], abs=1e-5)
    matrix = generator.random((5, 40))
    assert consistency._correlation(matrix, "cuda") == pytest.approx(np.corrcoef(matrix), abs=1e-5)


def test_taxonomy_agreement_reports_identical_partitions(collection):
    _, evidence, taxonomies = collection
    ## Compare taxonomy A with a renamed copy of itself: AMI must be 1 at equal depths.
    renamed = evidence[evidence.taxonomy == "A"].assign(taxonomy="A2", class_id=lambda f: "A2" + f.class_id.str[1:])
    copy = taxonomies["A"]
    tables = {key: frame.replace(regex={r"^A:": "A2:"}) for key, frame in copy.tables.items()}
    extended = {**taxonomies, "A2": ChemicalTaxonomy("A2", "1", tables)}
    agreement = graph_metrics.taxonomy_agreement(pd.concat([evidence, renamed]), extended, resolution="formula", min_ions=2)
    same = agreement[(agreement.taxonomy_a == "A") & (agreement.taxonomy_b == "A2") & (agreement.depth_a == agreement.depth_b)]
    assert not same.empty and (same.ami > 0.999).all()


def test_pixel_representations_use_only_resolved_ions(collection):
    ions, evidence, taxonomies = collection
    graph = build_label_graph(GraphVariant(("A",), "formula", "certain"), ions, evidence, taxonomies)
    images = IonImages("d", np.array(["C3H6|-H", "C9H9|-H", "C4H8|-H"]), np.array([[0, 0], [1, 0]]),
                       np.array([[1.0, 3.0], [1.0, 1.0], [2.0, 0.0]], dtype=np.float32))
    representations = spatial.pixel_representations(images, graph, (1,))
    ## C9H9|-H is unresolved and must not appear in any representation
    assert {name: matrix.shape[1] for name, (matrix, _) in representations.items()} == {"ions": 2, "classes": 2, "graph_1": 2}
