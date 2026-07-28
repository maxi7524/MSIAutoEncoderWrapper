"""Offline tests for discovery and materialization pipeline stages."""

from __future__ import annotations

import shutil
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional

from msi_autoencoder_wrapper.dataset_management.sources.base import DatasetSource
from msi_autoencoder_wrapper.dataset_management.operations import (
    materialize_and_merge_selection,
    materialize_selection,
    query_to_selection,
)
from msi_autoencoder_wrapper.readers.strategies.pyimzml_reader import PyImzMLReader
from msi_autoencoder_wrapper.dataset_management.catalog import DatasetCatalog


class FakeDatasetSource(DatasetSource):
    """Small provider adapter used without network access."""

    source_name = "fake"

    def __init__(self, fixture_path: Path) -> None:
        self.fixture_path = fixture_path
        self._config = {}

    def search_datasets(self, filters: Mapping[str, Any]) -> List[Dict[str, Any]]:
        return [{"dataset_id": "one", "name": "One", "metadata": dict(filters)}]

    def get_dataset_metadata(self, dataset_id: str) -> Dict[str, Any]:
        return {"dataset_id": dataset_id, "name": "One", "metadata": {"condition": "healthy"}}

    def get_annotations(
        self,
        dataset_id: str,
        options: Optional[Mapping[str, Any]] = None,
    ) -> List[Dict[str, Any]]:
        return [{"id": "annotation-1", "sumFormula": "C6H12O6", "fdr": 0.5}]

    def download_dataset(self, dataset_id: str, destination: Path | str) -> Path:
        target = Path(destination)
        target.mkdir(parents=True, exist_ok=True)
        shutil.copy2(self.fixture_path, target / f"{dataset_id}.imzML")
        shutil.copy2(self.fixture_path.with_suffix(".ibd"), target / f"{dataset_id}.ibd")
        return target


def test_discovery_and_download_are_separate_stages(
    tmp_path: Path,
    msi_fixture_path: Path,
) -> None:
    """Catalog discovery performs no download; materialization imports all labels."""
    source = FakeDatasetSource(msi_fixture_path)
    datasets_dir = tmp_path / "datasets"
    catalog = DatasetCatalog(datasets_dir / "catalog.sqlite")
    selection = datasets_dir / "selections" / "candidate.json"

    query_to_selection(
        source=source,
        filters={"organism": "mouse"},
        catalog=catalog,
        selection_path=selection,
    )
    assert selection.is_file()
    assert not (datasets_dir / "sources" / "fake" / "one").exists()

    materialized = materialize_selection(
        source=source,
        selection_path=selection,
        datasets_dir=datasets_dir,
        catalog=catalog,
    )
    assert materialized == [datasets_dir / "sources" / "fake" / "one"]
    assert (materialized[0] / "one.imzML").is_file()
    assert len(catalog.get_annotations(source="fake", dataset_id="one")) == 1


def test_low_disk_mode_downloads_merges_and_releases_staging_data(
    tmp_path: Path,
    msi_fixture_path: Path,
) -> None:
    """One-pass materialization retains catalog data but removes source files."""
    source = FakeDatasetSource(msi_fixture_path)
    datasets_dir = tmp_path / "datasets"
    catalog = DatasetCatalog(datasets_dir / "catalog.sqlite")
    selection = datasets_dir / "selections" / "candidate.json"
    query_to_selection(
        source=source,
        filters={"organism": "mouse"},
        catalog=catalog,
        selection_path=selection,
    )
    output = datasets_dir / "merged" / "pilot" / "dataset.imzML"

    result = materialize_and_merge_selection(
        source=source,
        selection_path=selection,
        datasets_dir=datasets_dir,
        catalog=catalog,
        output_path=output,
        merged_dataset_id="pilot",
        row_width=3,
    )

    assert PyImzMLReader(result).GetNumberOfSpectra() == 6
    assert not (datasets_dir / ".staging" / "fake" / "one").exists()
    assert catalog.get_dataset("fake", "one")["status"] == "merged"
    assert catalog.get_source_index(
        merged_dataset_id="pilot",
        merged_spectrum_index=5,
    ) == {
        "source": "fake",
        "source_dataset_id": "one",
        "source_spectrum_id": 5,
    }
