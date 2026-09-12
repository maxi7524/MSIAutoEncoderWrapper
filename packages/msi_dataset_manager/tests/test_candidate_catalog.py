"""External candidate-provider, cache, mass, and metadata regression tests."""

from __future__ import annotations

from io import BytesIO
import gzip
import json
import zipfile

import pytest

from msi_dataset_manager.annotations.candidates import (
    CandidateCatalogReader,
    CandidateClass,
    CandidateCompound,
    ChEBICandidateProvider,
    HMDBCandidateProvider,
    LIPIDMAPSCandidateProvider,
    build_candidate_catalog,
    calculate_theoretical_mz,
    materialize_candidate_sources,
)
from msi_dataset_manager.cli import build_parser
from msi_dataset_manager.metadata import (
    read_candidate_metadata,
    write_dataset_metadata,
)


class _Response(BytesIO):
    """Bytes response exposing the subset of ``urlopen`` used by providers."""


def test_metaspace_formula_adduct_mass_matches_known_glucose_annotation() -> None:
    """Neutral formula plus +H reproduces METASPACE's glucose m/z convention."""
    mz, charge = calculate_theoretical_mz("C6H12O6", "+H")

    assert charge == 1
    assert mz == pytest.approx(181.07066457, abs=2e-8)


def test_lipidmaps_provider_normalizes_public_tsv(monkeypatch: pytest.MonkeyPatch) -> None:
    """LIPID MAPS fields and hierarchy become common compound records."""
    payload = (
        "09/12/26\n"
        "LM_ID\tNAME\tFORMULA\tEXACT_MASS\tCATEGORY\tMAIN_CLASS\tSUB_CLASS\tSMILES\n"
        "LMFA0001\tTest lipid\tC2H4\t28.0313\tFatty Acyls\tFatty acids\tAcyclic\tC=C\n"
    ).encode()
    monkeypatch.setattr(
        "msi_dataset_manager.annotations.candidates.providers.urlopen",
        lambda request, timeout: _Response(payload),
    )

    record = LIPIDMAPSCandidateProvider(version="fixture").fetch_candidates()[0]

    assert record.provider == "LIPID_MAPS"
    assert record.identifier == "LMFA0001"
    assert record.formula == "C2H4"
    assert [item.name for item in record.classes] == [
        "Fatty Acyls",
        "Fatty acids",
        "Acyclic",
    ]


def test_hmdb_provider_normalizes_public_xml_archive(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path,
) -> None:
    """HMDB XML is parsed without exposing provider-specific XML to consumers."""
    xml = b"""<?xml version='1.0' encoding='UTF-8'?>
<hmdb xmlns='http://www.hmdb.ca'>
  <metabolite>
    <accession>HMDB0000001</accession><name>Fixture metabolite</name>
    <chemical_formula>C2H4</chemical_formula>
    <monisotopic_molecular_weight>28.0313001</monisotopic_molecular_weight>
    <smiles>C=C</smiles><inchikey>TEST</inchikey>
    <taxonomy><kingdom>Organic compounds</kingdom><class>Hydrocarbons</class></taxonomy>
  </metabolite>
</hmdb>"""
    archive = BytesIO()
    with zipfile.ZipFile(archive, "w") as stream:
        stream.writestr("hmdb_metabolites.xml", xml)
    monkeypatch.setattr(
        "msi_dataset_manager.annotations.candidates.providers.urlopen",
        lambda request, timeout: _Response(archive.getvalue()),
    )

    record = HMDBCandidateProvider(version="fixture").fetch_candidates()[0]

    assert record.provider == "HMDB"
    assert record.identifier == "HMDB0000001"
    assert record.monoisotopic_mass == pytest.approx(28.0313001)
    assert [item.name for item in record.classes] == [
        "Organic compounds",
        "Hydrocarbons",
    ]
    xml_path = tmp_path / "hmdb_metabolites.xml"
    xml_path.write_bytes(xml)
    cached_record = HMDBCandidateProvider(version="fixture").fetch_candidates(xml_path)[0]
    assert cached_record.identifier == "HMDB0000001"


def test_chebi_provider_normalizes_ontology_and_direct_classes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """ChEBI OBO chemistry and hierarchy become common compound records."""
    payload = gzip.compress(
        b"""format-version: 1.2

[Term]
id: CHEBI:1
name: Fixture molecule
property_value: http://purl.obolibrary.org/obo/chebi/formula \"C2H4\" xsd:string
property_value: http://purl.obolibrary.org/obo/chebi/monoisotopicmass \"28.0313001\" xsd:float
property_value: http://purl.obolibrary.org/obo/chebi/smiles \"C=C\" xsd:string
is_a: CHEBI:2 ! alkene

"""
    )
    monkeypatch.setattr(
        "msi_dataset_manager.annotations.candidates.providers.urlopen",
        lambda request, timeout: _Response(payload),
    )

    record = ChEBICandidateProvider(version="fixture").fetch_candidates()[0]

    assert record.provider == "ChEBI"
    assert record.identifier == "CHEBI:1"
    assert record.formula == "C2H4"
    assert record.monoisotopic_mass == pytest.approx(28.0313001)
    assert [(item.identifier, item.name) for item in record.classes] == [
        ("CHEBI:2", "alkene")
    ]


def test_provider_source_snapshot_is_reused_and_manifested(
    tmp_path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Raw provider exports are retained separately from candidate SQLite files."""
    payload = (
        "LM_ID\tNAME\tFORMULA\n"
        "LMFA0001\tTest lipid\tC2H4\n"
    ).encode()
    monkeypatch.setattr(
        "msi_dataset_manager.annotations.candidates.providers.urlopen",
        lambda request, timeout: _Response(payload),
    )
    provider = LIPIDMAPSCandidateProvider(version="fixture")

    snapshots = materialize_candidate_sources(
        cache_dir=tmp_path / "external_databases",
        providers=(provider,),
    )
    assert snapshots[0].read_bytes() == payload
    manifest = json.loads(
        (tmp_path / "external_databases" / "sources_manifest.json").read_text()
    )
    assert manifest["sources"][0]["name"] == "LIPID_MAPS"
    assert manifest["sources"][0]["local_path"] == str(snapshots[0])

    monkeypatch.setattr(
        "msi_dataset_manager.annotations.candidates.providers.urlopen",
        lambda request, timeout: pytest.fail("the existing source cache must be used"),
    )
    assert provider.materialize_snapshot(tmp_path / "external_databases") == snapshots[0]
    assert provider.fetch_candidates(snapshots[0])[0].identifier == "LMFA0001"
    monkeypatch.setattr(
        "msi_dataset_manager.annotations.candidates.providers.urlopen",
        lambda request, timeout: _Response(payload),
    )
    materialize_candidate_sources(
        cache_dir=tmp_path / "external_databases",
        providers=(LIPIDMAPSCandidateProvider(version="fixture-two"),),
    )
    merged_manifest = json.loads(
        (tmp_path / "external_databases" / "sources_manifest.json").read_text()
    )
    assert {source["version"] for source in merged_manifest["sources"]} == {
        "fixture",
        "fixture-two",
    }


def test_catalog_can_be_built_from_a_durable_source_snapshot(
    tmp_path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Dataset-specific filtering consumes local raw source exports without I/O."""
    write_dataset_metadata(
        workspace_path=tmp_path,
        source="metaspace",
        dataset_id="liver",
        name="Liver",
        metadata={"polarity": "Positive", "mz_min": 20, "mz_max": 40},
    )
    payload = (
        "LM_ID\tNAME\tFORMULA\n"
        "LMFA0001\tTest lipid\tC2H4\n"
    ).encode()
    monkeypatch.setattr(
        "msi_dataset_manager.annotations.candidates.providers.urlopen",
        lambda request, timeout: _Response(payload),
    )
    provider = LIPIDMAPSCandidateProvider(version="fixture")
    build_candidate_catalog(
        workspace_path=tmp_path,
        dataset_id="liver",
        providers=(provider,),
        source_cache_dir=tmp_path / "external_databases",
    )

    monkeypatch.setattr(
        "msi_dataset_manager.annotations.candidates.providers.urlopen",
        lambda request, timeout: pytest.fail("the existing source cache must be used"),
    )
    path = build_candidate_catalog(
        workspace_path=tmp_path,
        dataset_id="liver",
        providers=(provider,),
        refresh_cache=True,
        source_cache_dir=tmp_path / "external_databases",
    )
    assert CandidateCatalogReader(path).get_candidate_ions()[0]["formula"] == "C2H4"


def test_candidate_sources_cli_exposes_explicit_snapshot_versions() -> None:
    """The CLI makes source retrieval and version labels auditable."""
    arguments = build_parser().parse_args(
        [
            "candidate-sources",
            "--cache-dir",
            "data/external_databases",
            "--chebi-version",
            "255",
        ]
    )

    assert arguments.command == "candidate-sources"
    assert arguments.chebi_version == "255"


def test_catalog_uses_local_cache_and_refreshes_only_when_requested(tmp_path) -> None:
    """A per-dataset candidate SQLite prevents repeated provider access."""
    write_dataset_metadata(
        workspace_path=tmp_path,
        source="metaspace",
        dataset_id="liver",
        name="Liver",
        metadata={"polarity": "Positive", "mz_min": 20, "mz_max": 40},
    )

    class Provider:
        name, version = "fixture", "1"

        def __init__(self) -> None:
            self.calls = 0

        def fetch_candidates(self):
            self.calls += 1
            return (
                CandidateCompound(
                    provider=self.name,
                    provider_version=self.version,
                    identifier="one",
                    name="One",
                    formula="C2H4",
                    classes=(CandidateClass("fixture", "class", "Class", 0),),
                ),
                CandidateCompound(
                    provider=self.name,
                    provider_version=self.version,
                    identifier="outside",
                    name="Outside",
                    formula="C100H2",
                ),
                CandidateCompound(
                    provider=self.name,
                    provider_version=self.version,
                    identifier="unsupported",
                    name="Unsupported",
                    formula="SiH4",
                ),
            )

    provider = Provider()
    path = build_candidate_catalog(
        workspace_path=tmp_path,
        dataset_id="liver",
        providers=[provider],
    )
    assert provider.calls == 1
    assert build_candidate_catalog(
        workspace_path=tmp_path,
        dataset_id="liver",
        providers=[provider],
    ) == path
    assert provider.calls == 1
    build_candidate_catalog(
        workspace_path=tmp_path,
        dataset_id="liver",
        providers=[provider],
        refresh_cache=True,
    )
    assert provider.calls == 2

    reader = CandidateCatalogReader(path)
    assert reader.get_manifest()["filters"]["polarity"] == "Positive"
    assert [item["adduct"] for item in reader.get_candidate_ions()] == ["+H"]
    assert reader.get_candidates()[0]["classes"][0]["name"] == "Class"


def test_candidate_metadata_requires_a_normalized_local_artifact(tmp_path) -> None:
    """Candidate creation cannot silently proceed without dataset provenance."""
    with pytest.raises(ValueError, match="No normalized metadata"):
        read_candidate_metadata(workspace_path=tmp_path, dataset_id="missing")
