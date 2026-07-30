"""Offline contract tests for the PRIDE source adapter."""

from __future__ import annotations

import hashlib
import uuid
from pathlib import Path
from typing import Any, Dict, List

import pytest

from msi_autoencoder_wrapper.dataset_management.sources.strategies.pride import (
    PrideDatasetSource,
    _read_annotation_table,
)
from msi_autoencoder_wrapper.utils.exceptions import ProjectConfigError


PROJECT = {
    "accession": "PXD000001",
    "title": "Curated mouse bladder MSI",
    "organisms": [{"accession": "NEWT:10090", "name": "Mus musculus"}],
    "organismParts": [{"accession": "BTO:0001418", "name": "Urinary bladder"}],
    "diseases": [{"name": "bladder carcinoma"}],
}


def _file(name: str, size: int) -> Dict[str, Any]:
    return {"fileName": name, "fileSizeBytes": size, "publicFileLocations": []}


FILES = [
    _file("bladder.imzML", 100),
    _file("bladder.ibd", 900),
    _file("bladder_annotations.csv", 50),
    _file("orphan.imzML", 20),
]


class FakeProjectClient:
    """Return one stable project and capture native search parameters."""

    def __init__(self) -> None:
        self.search_calls: List[Dict[str, Any]] = []

    def search_by_keywords_and_filters(self, **kwargs: Any) -> List[Dict[str, Any]]:
        self.search_calls.append(kwargs)
        return [PROJECT] if kwargs["page"] == 0 else []

    def get_by_accession(self, accession: str) -> Dict[str, Any]:
        assert accession == PROJECT["accession"]
        return PROJECT


class FakeFileProvider:
    """Return project files without network access."""

    def list_files(self, accession: str) -> List[Dict[str, Any]]:
        assert accession == PROJECT["accession"]
        return FILES


class FakeDownloadClient:
    """Materialize requested records and capture checksum options."""

    def __init__(self) -> None:
        self.calls: List[Dict[str, Any]] = []

    def download_files_by_list(self, **kwargs: Any) -> None:
        self.calls.append(kwargs)
        target = Path(kwargs["output_folder"])
        pair_uuid = uuid.UUID("12345678-1234-5678-1234-567812345678")
        (target / "bladder.imzML").write_text(
            '<imzML><fileContent><cvParam accession="IMS:1000080" '
            f'value="{pair_uuid}"/></fileContent></imzML>',
            encoding="utf-8",
        )
        (target / "bladder.ibd").write_bytes(pair_uuid.bytes + b"spectra")
        (target / "bladder_annotations.csv").write_text(
            "formula,adduct,spectrum_id\nC6H12O6,+H,0\nC6H12O6,+H,2\n",
            encoding="utf-8",
        )


def _checksums(_: str) -> Dict[str, str]:
    return {
        record["fileName"]: hashlib.md5(record["fileName"].encode()).hexdigest()
        for record in FILES
    }


def test_pride_search_expands_projects_to_complete_annotated_pairs() -> None:
    """Discovery returns one record per complete pair with remote sizes."""
    projects = FakeProjectClient()
    source = PrideDatasetSource(
        project_client=projects,
        file_provider=FakeFileProvider(),
        checksum_provider=_checksums,
    )

    records = source.search_datasets(
        {
            "organisms": ["Mus musculus (mouse)"],
            "required_metadata_fields": ["organisms", "organismParts"],
            "single_value_metadata_fields": ["organisms", "organismParts", "diseases"],
        }
    )

    assert len(records) == 1
    assert records[0]["dataset_id"].startswith("PXD000001__bladder-")
    assert records[0]["metadata"]["total_size_bytes"] == 1000
    assert records[0]["metadata"]["download_size_bytes"] == 1050
    assert records[0]["metadata"]["annotation_status"] == "supported"
    assert "organisms==Mus%20musculus%20%28mouse%29" in projects.search_calls[0]["query_filter"]


def test_pride_download_uses_official_client_checksum_and_imports_annotations(
    tmp_path: Path,
) -> None:
    """A selected pair is downloaded with MD5 validation and canonical names."""
    downloader = FakeDownloadClient()
    source = PrideDatasetSource(
        project_client=FakeProjectClient(),
        download_client=downloader,
        file_provider=FakeFileProvider(),
        checksum_provider=_checksums,
    )
    record = source.search_datasets({})[0]
    dataset_id = record["dataset_id"]

    destination = source.download_dataset(dataset_id, tmp_path / dataset_id)
    annotations = source.get_annotations(dataset_id)

    assert downloader.calls[0]["checksum_check"] is True
    assert downloader.calls[0]["file_names"] == [
        "bladder.imzML",
        "bladder.ibd",
        "bladder_annotations.csv",
    ]
    assert (destination / f"{dataset_id}.imzML").is_file()
    assert (destination / f"{dataset_id}.ibd").is_file()
    assert annotations[0]["formula"] == "C6H12O6"
    assert annotations[0]["spectrum_ids"] == [0, 2]


def test_unsupported_annotation_schema_is_rejected(tmp_path: Path) -> None:
    """PRIDE tables without explicit molecule-to-pixel links are not inferred."""
    path = tmp_path / "image_annotations.tsv"
    path.write_text("formula\tmz\nC6H12O6\t181.07\n", encoding="utf-8")

    with pytest.raises(ProjectConfigError, match="expected a spectrum_id column"):
        _read_annotation_table(path)


def test_ambiguous_biological_metadata_rejects_whole_project() -> None:
    """Project-level values are accepted only when configured as unambiguous."""
    ambiguous = {**PROJECT, "diseases": [{"name": "healthy"}, {"name": "cancer"}]}

    class AmbiguousProjectClient(FakeProjectClient):
        def search_by_keywords_and_filters(self, **kwargs: Any) -> List[Dict[str, Any]]:
            return [ambiguous]

        def get_by_accession(self, accession: str) -> Dict[str, Any]:
            return ambiguous

    source = PrideDatasetSource(
        project_client=AmbiguousProjectClient(),
        file_provider=FakeFileProvider(),
        checksum_provider=_checksums,
    )

    assert source.search_datasets({"single_value_metadata_fields": ["diseases"]}) == []
    assert source.get_search_diagnostics()[0]["reason"].startswith(
        "ambiguous metadata field"
    )
    assert source.get_search_diagnostics()[0]["project_url"].endswith("PXD000001")


def test_pride_source_documents_explorer_filters() -> None:
    """The notebook explorer can display provider filter capabilities."""
    filters = PrideDatasetSource().get_available_filters()

    assert filters["organism_parts"]["api_field"] == "organismsPart"
    assert filters["require_annotation_source"]["required"] is True
