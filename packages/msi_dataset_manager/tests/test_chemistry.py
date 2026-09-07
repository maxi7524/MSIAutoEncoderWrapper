"""Versioned chemical enrichment, ambiguity, and source-preservation tests."""

import json
import sqlite3

import pytest

from msi_dataset_manager.annotations.chemistry import (
    ChemicalCandidate, SnapshotProvider, enrich_annotations, parse_formula, read_chemistry,
)


@pytest.fixture
def chemical_snapshot(tmp_path):
    path = tmp_path / "chemistry.json"
    path.write_text(json.dumps({"provider": "fixture", "version": "1", "candidates": [
        {"identifier": "a", "formula": "C2H4", "classes": ["lipid", "PC"]},
        {"identifier": "b", "formula": "C2H4", "classes": ["lipid", "PE"]},
        {"identifier": "c", "formula": "C3H6", "classes": ["lipid", "PC"]},
    ]}))
    return SnapshotProvider(path)


def test_formula_counts_and_strict_validation():
    assert parse_formula("C6H12O6") == {"C": 6, "H": 12, "O": 6}
    assert parse_formula("CH3COOH") == {"C": 2, "H": 4, "O": 2}
    for formula in ("", "C0H2", "Xx2", "C6H12O6+Na", "[13C]6H12O6", "Ca(OH)2"):
        with pytest.raises(ValueError):
            parse_formula(formula)


def test_enrichment_preserves_sql_and_intersects_ambiguous_classes(tmp_path, chemical_snapshot):
    db = tmp_path / "annotations.sqlite"
    with sqlite3.connect(db) as connection:
        connection.execute("CREATE TABLE original (value TEXT)")
        connection.execute("INSERT INTO original VALUES ('untouched')")
    records = [{"formula": "C2H4", "adduct": "+H"}, {"formula": "C3H6", "adduct": "+H"}]
    assert enrich_annotations(records, db, chemical_snapshot) == 2
    assert enrich_annotations(records, db, chemical_snapshot) == 0
    data = read_chemistry(db, provider="fixture", version="1")
    assert data["C2H4|+H"]["certain_classes"] == ["lipid"]
    assert data["C2H4|+H"]["possible_classes"] == ["PC", "PE", "lipid"]
    assert data["C2H4|+H"]["status"] == "ambiguous"
    with sqlite3.connect(db) as connection:
        assert connection.execute("SELECT value FROM original").fetchone()[0] == "untouched"


def test_identifier_lookup_preserves_the_source_assignment(tmp_path, chemical_snapshot):
    db = tmp_path / "annotations.sqlite"
    enrich_annotations([{"formula": "C2H4", "adduct": "+H", "molecule_ids": ["a"]}], db, chemical_snapshot)
    record = read_chemistry(db, provider="fixture", version="1")["C2H4|+H"]
    assert record["status"] == "unique"
    assert record["certain_classes"] == ["PC", "lipid"]


def test_failed_provider_rolls_back_descriptors(tmp_path):
    class InvalidProvider:
        name, version = "bad", "1"
        def lookup(self, formula, identifiers):
            return (ChemicalCandidate("wrong", "C3H6"),)
    db = tmp_path / "annotations.sqlite"
    with pytest.raises(ValueError, match="conflicting"):
        enrich_annotations([{"formula": "C2H4", "adduct": "+H"}], db, InvalidProvider())
    with sqlite3.connect(db) as connection:
        assert connection.execute("SELECT COUNT(*) FROM chemistry_annotations").fetchone()[0] == 0


def test_lipidmaps_adapter_preserves_hierarchy_and_snapshot(monkeypatch):
    from io import BytesIO
    from msi_dataset_manager.annotations.chemistry import providers

    urls = []

    def response(url, timeout):
        urls.append(url)
        assert timeout == 5
        record = ({"category": "Glycerophospholipids", "main_class": "PC"}
                  if "/classification/" in url else
                  {"lm_id": "LMGP01010001", "formula": "C2H4", "smiles": "C=C"})
        return BytesIO(json.dumps(record).encode())

    monkeypatch.setattr(providers, "urlopen", response)
    provider = providers.LipidMapsProvider("fixture-release", timeout=5)
    candidates = provider.lookup("C2H4", ("LMGP01010001",))
    assert candidates[0].classes == ("LIPID_MAPS:Glycerophospholipids", "LIPID_MAPS:PC")
    assert candidates[0].smiles == "C=C"
    assert provider.version == "fixture-release"
    assert len(urls) == 2 and all("/lm_id/LMGP01010001/" in url for url in urls)
