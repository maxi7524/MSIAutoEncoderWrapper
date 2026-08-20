"""Tests for notebook-oriented dataset exploration."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional

from msi_dataset_manager.exploration import DatasetExplorer
from msi_dataset_manager.sources.base import DatasetSource
from msi_dataset_manager.sources.strategies.metaspace import (
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
        return {
            "organisms": {"type": "list"},
            "verified": {"type": "boolean", "choices": [True, False]},
        }

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
                    "download_size_bytes": 1000,
                    "mz_min": 100,
                    "mz_max": 1000,
                    "annotation_status": "supported",
                },
            },
            {
                "dataset_id": "two",
                "name": "Two",
                "metadata": {
                    "total_size_bytes": 2000,
                    "download_size_bytes": 2000,
                    "mz_min": 250,
                    "mz_max": 850,
                },
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

    assert explorer.get_available_filters() == {
        "organisms": {"type": "list"},
        "verified": {"type": "boolean", "choices": [True, False]},
    }
    assert explorer.get_available_values("verified")["value"].tolist() == [True, False]
    assert explorer.rejected().loc[0, "reason"] == "unsupported annotation format"
    assert explorer.rejected().loc[0, "project_url"].endswith("PXD000002")


def test_explorer_previews_and_filters_mz_ranges(tmp_path: Path) -> None:
    """Range preview is non-mutating and query filters retain the selection."""
    explorer = DatasetExplorer(FakeExplorationSource())
    explorer.search({})

    coverage = explorer.count_mz_range_coverage([100, 200], [800, 900])
    selected = explorer.select_mz_range(200, 900)
    assert explorer.filters == {}
    assert explorer.accepted()["dataset_id"].tolist() == ["one", "two"]

    explorer.filter({"mz_min": 200, "mz_max": 900})
    exported = explorer.export_selection(
        tmp_path / "kidney",
        sort_by="download_size_bytes",
        ascending=False,
    )

    assert coverage["dataset_count"].tolist() == [1, 1, 1, 1]
    assert selected["dataset_id"].tolist() == ["one"]
    payload = json.loads(exported["selection"].read_text(encoding="utf-8"))
    assert payload["filters"]["mz_min"] == 200
    assert payload["filters"]["mz_max"] == 900
    # The fake source does not implement provider-side filtering; this test
    # verifies explorer state and export semantics, while source filtering is
    # covered by the METASPACE source tests.
    assert payload["datasets"][0]["metadata"]["mz_min"] == 250
    assert payload["dataset_ids"] == ["two", "one"]
    assert "rejected" not in payload
    assert payload["exported_at"]


def test_explorer_previews_and_applies_current_filters_without_requery() -> None:
    """Current-result filters reuse records from the preceding broad query."""
    source = FakeExplorationSource()
    explorer = DatasetExplorer(source)
    explorer.filter({})

    preview = explorer.filter_current({"mz_min": 200, "mz_max": 900})
    accepted = explorer.apply_current_filters({"mz_min": 200, "mz_max": 900})

    assert preview["dataset_id"].tolist() == ["one"]
    assert accepted["dataset_id"].tolist() == ["one"]
    assert source.seen_filters == {}
    assert explorer.results(include_excluded=True).loc[1, "excluded"]


def test_explorer_reviews_current_records_and_applies_selected_rules() -> None:
    """Review flags are visible before explicit exclusion recommendations apply."""

    class ReviewSource(FakeExplorationSource):
        def filter(self, filters: Mapping[str, Any]) -> List[Dict[str, Any]]:
            self.seen_filters = dict(filters)
            self._accepted = [
                {
                    "dataset_id": "technical-copy",
                    "name": "2024-01-01_mouse_brain_5ppm",
                    "metadata": {"pixel_count": 1000, "mz_min": 100, "mz_max": 900},
                },
                {
                    "dataset_id": "keeper",
                    "name": "mouse_brain",
                    "metadata": {"pixel_count": 1000, "mz_min": 100, "mz_max": 900},
                },
                {
                    "dataset_id": "qc",
                    "name": "mouse_brain_null_mz_shift",
                    "metadata": {"pixel_count": 1000, "mz_min": 100, "mz_max": 910},
                },
                {
                    "dataset_id": "fragment",
                    "name": "granular_layer_mouse_brain",
                    "metadata": {"pixel_count": 2000, "mz_min": 100, "mz_max": 900},
                },
            ]
            return self.get_accepted_records()

    explorer = DatasetExplorer(ReviewSource())
    explorer.filter({})
    review = explorer.review_current(profile="brain")

    assert review.exclusion_ids(["high_confidence_duplicates"]) == [
        "technical-copy"
    ]
    assert review.exclusion_ids(["mz_shift_qc_variants"]) == ["qc"]
    assert review.exclusion_ids(["explicit_regional_fragments"]) == ["fragment"]
    assert explorer.accepted()["dataset_id"].tolist() == [
        "technical-copy",
        "keeper",
        "qc",
        "fragment",
    ]

    accepted = explorer.apply_review(
        review,
        rules=["high_confidence_duplicates", "mz_shift_qc_variants"],
    )

    assert accepted["dataset_id"].tolist() == ["keeper", "fragment"]


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

    results = explorer.search({"organism": "mouse", "polarity": "Positive"})

    assert explorer.available_filters()["polarity"]["type"] == "Positive | Negative"
    # "organism" is unstructured METASPACE free text, so it is matched
    # locally and case-insensitively rather than forwarded to the API.
    assert client.filters == {
        "idMask": ["metaspace-one"],
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
