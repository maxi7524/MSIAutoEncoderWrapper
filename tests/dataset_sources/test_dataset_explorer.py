"""Tests for notebook-oriented dataset exploration."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional

from msi_autoencoder_wrapper.dataset_management.exploration import DatasetExplorer
from msi_autoencoder_wrapper.dataset_management.sources.base import DatasetSource
from msi_autoencoder_wrapper.dataset_management.sources.strategies.metaspace import (
    MetaspaceDatasetSource,
)


class FakeExplorationSource(DatasetSource):
    """Return stable accepted records and rejection diagnostics."""

    source_name = "fake"

    def __init__(self) -> None:
        self.seen_filters: Dict[str, Any] = {}
        self._config = {}
        self._accepted: List[Dict[str, Any]] = []

    def get_available_filters(self) -> Dict[str, Any]:
        return {"organisms": {"type": "list"}}

    def filter(self, filters: Mapping[str, Any]) -> List[Dict[str, Any]]:
        self.seen_filters = dict(filters)
        self._accepted = [
            {
                "dataset_id": "one",
                "name": "One",
                "metadata": {
                    "project_accession": "PXD000001",
                    "project_url": "https://example.test/PXD000001",
                    "project": {
                        "organisms": [{"name": "Mus musculus"}],
                        "organismParts": [{"name": "Urinary bladder"}],
                    },
                    "total_size_bytes": 1000,
                    "annotation_status": "supported",
                },
            },
            {
                "dataset_id": "two",
                "name": "Two",
                "metadata": {"total_size_bytes": 2000},
            },
        ]
        return self.get_accepted_records()

    def get_accepted_records(self) -> List[Dict[str, Any]]:
        return [dict(record) for record in self._accepted]

    def get_rejected_records(self) -> List[Dict[str, Any]]:
        return [
            {
                "project_accession": "PXD000002",
                "reason": "unsupported annotation format",
                "project_url": "https://example.test/PXD000002",
            }
        ]

    def get_dataset_metadata(self, dataset_id: str) -> Dict[str, Any]:
        raise NotImplementedError

    def get_annotations(
        self,
        dataset_id: str,
        options: Optional[Mapping[str, Any]] = None,
    ) -> List[Dict[str, Any]]:
        raise NotImplementedError

    def download_dataset(self, dataset_id: str, destination: Path | str) -> Path:
        raise NotImplementedError


def test_explorer_search_exclude_and_export_use_query_configuration(
    tmp_path: Path,
) -> None:
    """Reviewed exclusions are exported in the same mapping passed to query."""
    source = FakeExplorationSource()
    explorer = DatasetExplorer(source)
    filters = {
        "organisms": ["Mus musculus"],
        "exclude_dataset_ids": ["already-excluded"],
    }

    results = explorer.filter(filters)
    explorer.exclude("two")
    output = explorer.export_config(tmp_path / "filters.json")

    assert source.seen_filters == {"organisms": ["Mus musculus"]}
    assert explorer.accepted()["dataset_id"].tolist() == ["one"]
    assert results.loc[0, "organisms"] == "Mus musculus"
    assert results.loc[0, "organism_parts"] == "Urinary bladder"
    assert explorer.results()["dataset_id"].tolist() == ["one"]
    assert explorer.results(include_excluded=True).loc[1, "excluded"]
    assert json.loads(output.read_text(encoding="utf-8"))["exclude_dataset_ids"] == [
        "already-excluded",
        "two",
    ]


def test_explorer_exposes_filter_help_and_rejection_links() -> None:
    """Notebook users can inspect provider filters and rejected records."""
    explorer = DatasetExplorer(FakeExplorationSource())
    explorer.search({})

    assert explorer.get_available_filters() == {"organisms": {"type": "list"}}
    assert explorer.rejected().loc[0, "reason"] == "unsupported annotation format"
    assert explorer.rejected().loc[0, "project_url"].endswith("PXD000002")


def test_explorer_supports_metaspace_filters_and_metadata() -> None:
    """The same explorer reports and searches native METASPACE fields."""

    class Dataset:
        id = "metaspace-one"
        name = "Mouse bladder"
        metadata = {
            "Sample_Information": {
                "Organism": "Mouse",
                "Organism_Part": "Urinary bladder",
            }
        }
        polarity = "Positive"
        status = "FINISHED"
        image_size = {"x": 2, "y": 2}
        database_details = [
            type("Database", (), {"name": "HMDB", "version": "v4", "id": 1})()
        ]

    class Client:
        def __init__(self) -> None:
            self.filters: Dict[str, Any] = {}

        def datasets(self, **filters: Any) -> List[Dataset]:
            self.filters = filters
            return [Dataset()]

    client = Client()
    explorer = DatasetExplorer(MetaspaceDatasetSource(client=client))

    results = explorer.search({"organism": "Mus musculus", "polarity": "Positive"})

    assert explorer.available_filters()["polarity"]["type"] == "Positive | Negative"
    assert client.filters == {
        "organism": "Mus musculus",
        "polarity": "Positive",
        "status": "FINISHED",
    }
    assert results.loc[0, "source"] == "metaspace"
    assert results.loc[0, "organisms"] == "Mouse"
    assert results.loc[0, "organism_parts"] == "Urinary bladder"
    assert results.loc[0, "polarity"] == "Positive"
    assert results.loc[0, "processing_status"] == "FINISHED"
    assert results.loc[0, "image_size"] == "x=2, y=2"
    assert results.loc[0, "databases"] == "HMDB v4"
    assert results.loc[0, "project_url"].endswith("/metaspace-one")
