"""Taxonomy normalization, closure, candidate resolution and source readers."""

import gzip

import pandas as pd
import pytest

from msi_dataset_manager.annotations.chemistry import (
    ChemicalTaxonomy,
    canonical_formula,
    class_assignments,
    load_chebi_taxonomy,
    load_classyfire_taxonomy,
    load_lipidmaps_taxonomy,
    load_swisslipids_taxonomy,
    normalize_identifier,
    resolve_candidates,
)


def toy_taxonomy() -> ChemicalTaxonomy:
    """Two isomers of C2H4 in sibling classes PC/PE plus a multiply-inherited class."""
    return ChemicalTaxonomy("T", "1", {
        "classes": pd.DataFrame({"class_id": ["T:root", "T:GP", "T:PC", "T:PE", "T:X"],
                                 "name": ["root", "GP", "PC", "PE", "X"], "rank": [""] * 5}),
        "class_edges": pd.DataFrame({"child_id": ["T:GP", "T:PC", "T:PE", "T:X", "T:X"],
                                     "parent_id": ["T:root", "T:GP", "T:GP", "T:PC", "T:root"]}),
        "compounds": pd.DataFrame({"compound_id": ["c1", "c2", "c3", "c4"], "name": ["a", "b", "c", "d"],
                                   "formula": ["C2H4", "C2H4", "C3H6", "C4H8"]}),
        "memberships": pd.DataFrame({"compound_id": ["c1", "c2", "c3"], "class_id": ["T:PC", "T:PE", "T:X"]}),
        "aliases": pd.DataFrame({"alias": ["LIPIDMAPS:LMGP00000001", "HMDB:HMDB0000002"], "compound_id": ["c1", "c2"]}),
        "alignments": pd.DataFrame(columns=["class_id", "target_id", "predicate"]),
    })


def test_identifier_normalization_unifies_versions_and_prefixes():
    assert normalize_identifier("HMDB00220") == normalize_identifier("hmdb:HMDB0000220") == "HMDB:HMDB0000220"
    assert normalize_identifier("kegg.compound:C00249") == normalize_identifier("C00249") == "KEGG:C00249"
    assert normalize_identifier("SLM:000056508") == "SWISSLIPIDS:SLM:000056508"
    assert normalize_identifier("CHEBI:0015756") == "CHEBI:15756"
    assert normalize_identifier("12345") is None


def test_canonical_formula_hill_order_and_neutralization():
    assert canonical_formula("O2C16H32") == "C16H32O2"
    assert canonical_formula("C16H31O2", charge=-1) == "C16H32O2"
    assert canonical_formula("C5H14NO", charge=1) == "C5H13NO"
    assert canonical_formula("ClNa") == "ClNa"
    assert canonical_formula("C2H4R") is None


def test_dag_closure_and_shortest_depth():
    taxonomy = toy_taxonomy()
    assert taxonomy.ancestors("T:X") == {"T:X", "T:PC", "T:GP", "T:root"}
    assert taxonomy.class_depth == {"T:root": 0, "T:GP": 1, "T:PC": 2, "T:PE": 2, "T:X": 1}
    assert taxonomy.compound_closure("c4") == frozenset()


def test_resolution_modes_and_ambiguity_aggregation():
    taxonomy = toy_taxonomy()
    ions = pd.DataFrame({"ion_id": ["C2H4|-H", "C3H6|-H", "C9H2|-H"], "formula": ["C2H4", "C3H6", "C9H2"],
                         "identifiers": [["LMGP00000001"], [], ["HMDB0000002"]]})
    by_formula = resolve_candidates(ions, taxonomy, "formula")
    assert set(map(tuple, by_formula.to_numpy())) == {("C2H4|-H", "c1"), ("C2H4|-H", "c2"), ("C3H6|-H", "c3")}
    by_ids = resolve_candidates(ions, taxonomy, "metaspace_ids")
    ## The HMDB alias belongs to C2H4 and must not explain the C9H2 ion.
    assert set(map(tuple, by_ids.to_numpy())) == {("C2H4|-H", "c1")}
    evidence = class_assignments(by_formula, taxonomy).set_index(["ion_id", "class_id"])
    assert evidence.loc[("C2H4|-H", "T:PC"), "support_fraction"] == pytest.approx(0.5)
    assert not evidence.loc[("C2H4|-H", "T:PC"), "certain"]
    assert evidence.loc[("C2H4|-H", "T:GP"), "certain"]
    assert set(evidence.loc["C3H6|-H"].index) == {"T:X", "T:PC", "T:GP", "T:root"}


def test_taxonomy_roundtrip(tmp_path):
    taxonomy = toy_taxonomy()
    loaded = ChemicalTaxonomy.load(taxonomy.save(tmp_path / "t"))
    assert loaded.ancestors("T:X") == taxonomy.ancestors("T:X")
    assert loaded.formula_index == taxonomy.formula_index


def test_lipidmaps_reader_uses_codes_and_most_specific_membership(tmp_path):
    path = tmp_path / "lm.tsv"
    path.write_text("09/12/26\nlm_id\tname\tcore\tmain_class\tsub_class\tclass_level4\tformula\thmdb_id\tchebi_id\tkegg_id\n"
                    "LMGP01010001\tPC a\tGlycerophospholipids [GP]\tGlycerophosphocholines [GP01]\tDiacylglycerophosphocholines [GP0101]\t\tC8H18NO6P\tHMDB00564\t1234\t\n")
    taxonomy = load_lipidmaps_taxonomy(path, "fixture")
    assert taxonomy.compound_closure("LIPIDMAPS:LMGP01010001") == {"LIPID_MAPS:GP", "LIPID_MAPS:GP01", "LIPID_MAPS:GP0101"}
    assert taxonomy.alias_index["HMDB:HMDB0000564"] == ("LIPIDMAPS:LMGP01010001",)
    assert taxonomy.alias_index["CHEBI:1234"] == ("LIPIDMAPS:LMGP01010001",)


def test_chebi_reader_keeps_neutral_compounds_and_maps_conjugates(tmp_path):
    path = tmp_path / "chebi.obo.gz"
    text = """format-version: 1.2

[Term]
id: CHEBI:1
name: chemical entity

[Term]
id: CHEBI:2
name: fatty acid
is_a: CHEBI:1 ! chemical entity

[Term]
id: CHEBI:10
name: palmitic acid
is_a: CHEBI:2 ! fatty acid
xref: hmdb:HMDB0000220 {source="hmdb"}
property_value: http://purl.obolibrary.org/obo/chemrof:generalized_empirical_formula "C16H32O2" xsd:string
property_value: http://purl.obolibrary.org/obo/chemrof:charge "0" xsd:integer

[Term]
id: CHEBI:11
name: palmitate
is_a: CHEBI:1 ! chemical entity
relationship: RO:0018033 CHEBI:10 ! is conjugate base of palmitic acid
property_value: http://purl.obolibrary.org/obo/chemrof:generalized_empirical_formula "C16H31O2" xsd:string
property_value: http://purl.obolibrary.org/obo/chemrof:charge "-1" xsd:integer
"""
    with gzip.open(path, "wt") as stream:
        stream.write(text)
    taxonomy = load_chebi_taxonomy(path, "fixture")
    assert list(taxonomy.compounds.compound_id) == ["CHEBI:10"]
    assert taxonomy.compound_closure("CHEBI:10") == {"ChEBI:CHEBI:2", "ChEBI:CHEBI:1"}
    assert taxonomy.alias_index["CHEBI:11"] == ("CHEBI:10",)
    assert taxonomy.alias_index["HMDB:HMDB0000220"] == ("CHEBI:10",)


def test_swisslipids_reader_collapses_subspecies_and_neutralizes(tmp_path):
    path = tmp_path / "sl.tsv.gz"
    columns = ["Lipid ID", "Level", "Name", "Lipid class*", "Parent", "Formula (pH7.3)", "Charge (pH7.3)",
               "CHEBI", "LIPID MAPS", "HMDB"]
    rows = [["SLM:1", "Category", "Lipid", "", "", "", "", "", "", ""],
            ["SLM:2", "Class", "Fatty acids", "SLM:1", "", "C1HO2R", "-1", "", "", ""],
            ["SLM:3", "Species", "FA(16:0)", "SLM:2", "", "C16H31O2", "-1", "", "", ""],
            ["SLM:4", "Isomeric subspecies", "palmitate", "", "SLM:3", "C16H31O2", "-1", "15756", "LMFA01010001", ""]]
    pd.DataFrame(rows, columns=columns).to_csv(path, sep="\t", index=False, compression="gzip")
    taxonomy = load_swisslipids_taxonomy(path, "fixture")
    assert taxonomy.compound_formula == {"SWISSLIPIDS:SLM:3": "C16H32O2"}
    assert taxonomy.compound_closure("SWISSLIPIDS:SLM:3") == {"SwissLipids:SLM:2", "SwissLipids:SLM:1"}
    assert taxonomy.alias_index["LIPIDMAPS:LMFA01010001"] == ("SWISSLIPIDS:SLM:3",)
    assert taxonomy.alias_index["CHEBI:15756"] == ("SWISSLIPIDS:SLM:3",)


def test_classyfire_reader_combines_chemont_and_hmdb(tmp_path):
    ontology = tmp_path / "chemont.obo"
    ontology.write_text("""[Term]
id: CHEMONTID:9999999
name: Chemical entities

[Term]
id: CHEMONTID:0000000
name: Organic compounds
is_a: CHEMONTID:9999999 ! Chemical entities

[Term]
id: CHEMONTID:0000262
name: Fatty acids and conjugates
synonym: "Fatty Acids and Conjugates" EXACT LIPIDMAPS_TERM [LIPIDMAPS:FA01]
is_a: CHEMONTID:0000000 ! Organic compounds
""")
    hmdb = tmp_path / "hmdb.xml"
    hmdb.write_text("""<?xml version="1.0"?><hmdb xmlns="http://www.hmdb.ca"><metabolite>
<accession>HMDB0000220</accession><secondary_accessions><accession>HMDB00220</accession></secondary_accessions>
<name>Palmitic acid</name><chemical_formula>C16H32O2</chemical_formula><kegg_id>C00249</kegg_id><chebi_id>15756</chebi_id>
<taxonomy><direct_parent>Fatty acids and conjugates</direct_parent><alternative_parents><alternative_parent>Organic compounds</alternative_parent><alternative_parent>Unknown class</alternative_parent></alternative_parents></taxonomy>
</metabolite></hmdb>""")
    taxonomy = load_classyfire_taxonomy(ontology, hmdb, "fixture")
    compound = "HMDB:HMDB0000220"
    assert taxonomy.compound_closure(compound) == {"ClassyFire:CHEMONTID:0000262", "ClassyFire:CHEMONTID:0000000",
                                                   "ClassyFire:CHEMONTID:9999999"}
    assert taxonomy.alias_index["KEGG:C00249"] == (compound,)
    assert taxonomy.alignments.to_dict("records") == [
        {"class_id": "ClassyFire:CHEMONTID:0000262", "target_id": "LIPID_MAPS:FA01", "predicate": "EXACT"}]
    ranks = dict(zip(taxonomy.classes.class_id, taxonomy.classes["rank"]))
    assert ranks["ClassyFire:CHEMONTID:0000000"] == "kingdom"
    assert taxonomy.direct_classes[compound] == {"ClassyFire:CHEMONTID:0000262"}
    with_alternatives = load_classyfire_taxonomy(ontology, hmdb, "fixture", alternative_parents=True)
    assert with_alternatives.direct_classes[compound] == {"ClassyFire:CHEMONTID:0000262", "ClassyFire:CHEMONTID:0000000"}
