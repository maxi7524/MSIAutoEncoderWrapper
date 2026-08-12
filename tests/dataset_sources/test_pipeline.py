"""Offline tests for discovery and materialization pipeline stages."""

from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional

import pytest

from msi_autoencoder_wrapper.annotations import SQLiteAnnotationReader
from msi_dataset_manager.sources.base import DatasetSource
from msi_dataset_manager.operations import (
    materialize_and_merge_selection,
    materialize_selection,
    query_to_selection,
)
from msi_autoencoder_wrapper.readers.strategies.pyimzml_reader import PyImzMLReader
from msi_dataset_manager.catalog import DatasetCatalog
from msi_dataset_manager.utils.exceptions import ValidationError
from msi_dataset_manager.utils.exceptions import DownloadLimitError


class FakeDatasetSource(DatasetSource):
    """Small provider adapter used without network access."""

    source_name = "fake"

    def __init__(self, fixture_path: Path) -> None:
        self.fixture_path = fixture_path
        self.seen_filters: Dict[str, Any] = {}
        self._accepted: List[Dict[str, Any]] = []
        self.annotation_options: Optional[Mapping[str, Any]] = None
        self.download_calls: List[tuple[str, Path]] = []
        self.annotation_calls: List[str] = []
        self.events: List[str] = []
        self._config = {}

    def get_available_filters(self) -> Dict[str, Any]:
        return {"organism": {"type": "string"}}

    def filter(self, filters: Mapping[str, Any]) -> List[Dict[str, Any]]:
        self.seen_filters = dict(filters)
        self._accepted = [
            {"dataset_id": "one", "name": "One", "metadata": dict(filters)}
        ]
        return self.get_accepted_records()

    def get_accepted_records(self) -> List[Dict[str, Any]]:
        return [dict(record) for record in self._accepted]

    def get_rejected_records(self) -> List[Dict[str, Any]]:
        return []

    def get_dataset_metadata(self, dataset_id: str) -> Dict[str, Any]:
        return {"dataset_id": dataset_id, "name": "One", "metadata": {"condition": "healthy"}}

    def get_annotations(
        self,
        dataset_id: str,
        options: Optional[Mapping[str, Any]] = None,
    ) -> List[Dict[str, Any]]:
        self.annotation_calls.append(dataset_id)
        self.events.append(f"annotations:{dataset_id}")
        self.annotation_options = options
        return [
            {
                "id": "annotation-1",
                "sumFormula": "C6H12O6",
                "mz": 181.0707,
                "fdr": 0.5,
                "spectrum_ids": list(range(6)),
            }
        ]

    def download_dataset(self, dataset_id: str, destination: Path | str) -> Path:
        target = Path(destination)
        self.download_calls.append((dataset_id, target))
        self.events.append(f"download:{dataset_id}")
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
        filters={"organism": "mouse", "annotation_fdr": 0.1},
        catalog=catalog,
        selection_path=selection,
    )
    assert selection.is_file()
    assert not (datasets_dir / "one").exists()

    materialized = materialize_selection(
        source=source,
        selection_path=selection,
        datasets_dir=datasets_dir,
        catalog=catalog,
    )
    assert materialized == [datasets_dir / "one"]
    assert (materialized[0] / "one.imzML").is_file()
    assert len(catalog.get_annotations(source="fake", dataset_id="one")) == 1
    assert source.annotation_options == {"annotation_fdr": 0.1}


def test_materialization_reuses_workspace_source_before_provider_request(
    tmp_path: Path,
    msi_fixture_path: Path,
) -> None:
    """A canonical source pair bypasses the provider download operation."""
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
    retained = datasets_dir / "one"
    retained.mkdir(parents=True)
    shutil.copy2(msi_fixture_path, retained / "one.imzML")
    shutil.copy2(msi_fixture_path.with_suffix(".ibd"), retained / "one.ibd")

    materialize_selection(
        source=source,
        selection_path=selection,
        datasets_dir=datasets_dir,
        catalog=catalog,
    )

    assert source.download_calls == []


def test_materialization_downloads_all_pairs_before_annotations_and_reuses_both(
    tmp_path: Path,
    msi_fixture_path: Path,
) -> None:
    """Files form phase one and completed annotation imports form phase two."""
    source = FakeDatasetSource(msi_fixture_path)
    datasets_dir = tmp_path / "datasets"
    catalog = DatasetCatalog(datasets_dir / "catalog.sqlite")
    selection = datasets_dir / "selection.json"
    selection.write_text(
        '{"source":"fake","datasets":['
        '{"dataset_id":"one","name":"One"},'
        '{"dataset_id":"two","name":"Two"}]}'
    )

    materialize_selection(
        source=source,
        selection_path=selection,
        datasets_dir=datasets_dir,
        catalog=catalog,
        annotation_options={"annotation_fdr": 0.1},
    )

    assert source.events == [
        "download:one",
        "download:two",
        "annotations:one",
        "annotations:two",
    ]
    source.events.clear()
    source.download_calls.clear()
    source.annotation_calls.clear()

    materialize_selection(
        source=source,
        selection_path=selection,
        datasets_dir=datasets_dir,
        catalog=catalog,
        annotation_options={"annotation_fdr": 0.1},
    )

    assert source.download_calls == []
    assert source.annotation_calls == []
    manifest = selection.parent / "materialization.json"
    assert manifest.is_file()


def test_download_limit_continues_with_annotations_for_later_local_pairs(
    tmp_path: Path,
    msi_fixture_path: Path,
) -> None:
    """File quota ends phase one but does not hide later cached datasets."""
    class LimitedSource(FakeDatasetSource):
        def download_dataset(self, dataset_id: str, destination: Path | str) -> Path:
            if dataset_id == "missing":
                Path(destination).mkdir(parents=True, exist_ok=True)
                raise DownloadLimitError("provider refused download links")
            return super().download_dataset(dataset_id, destination)

    source = LimitedSource(msi_fixture_path)
    datasets_dir = tmp_path / "datasets"
    catalog = DatasetCatalog(datasets_dir / "catalog.sqlite")
    selection = datasets_dir / "selection.json"
    selection.write_text(
        '{"source":"fake","datasets":['
        '{"dataset_id":"missing","name":"Missing"},'
        '{"dataset_id":"local","name":"Local"}]}'
    )
    local = datasets_dir / "local"
    local.mkdir(parents=True)
    shutil.copy2(msi_fixture_path, local / "local.imzML")
    shutil.copy2(msi_fixture_path.with_suffix(".ibd"), local / "local.ibd")

    materialized = materialize_selection(
        source=source,
        selection_path=selection,
        datasets_dir=datasets_dir,
        catalog=catalog,
    )

    assert materialized == [local]
    assert source.annotation_calls == ["local"]
    assert (local / "annotations.csv").is_file()
    assert (local / "pixel_intensities.csv").is_file()
    manifest = json.loads(
        (selection.parent / "materialization.json").read_text()
    )
    assert manifest["file_limit_reached"] is True
    assert manifest["file_statuses"]["missing"] == "download_limit"
    assert manifest["annotation_statuses"]["local"] == "downloaded_and_imported"


def test_query_applies_generic_dataset_exclusions_outside_provider(
    tmp_path: Path,
    msi_fixture_path: Path,
) -> None:
    """Local exclusion IDs are not forwarded as native provider filters."""
    source = FakeDatasetSource(msi_fixture_path)
    catalog = DatasetCatalog(tmp_path / "datasets" / "catalog.sqlite")

    selected = query_to_selection(
        source=source,
        filters={"organism": "mouse", "exclude_dataset_ids": ["one"]},
        catalog=catalog,
        selection_path=tmp_path / "selection.json",
    )

    assert selected == []
    assert source.seen_filters == {"organism": "mouse"}


def test_download_rejects_annotation_fdr_different_from_selection(
    tmp_path: Path,
    msi_fixture_path: Path,
) -> None:
    """Discovery and materialization cannot use different annotation thresholds."""
    source = FakeDatasetSource(msi_fixture_path)
    datasets_dir = tmp_path / "datasets"
    catalog = DatasetCatalog(datasets_dir / "catalog.sqlite")
    selection = datasets_dir / "selections" / "candidate.json"
    query_to_selection(
        source=source,
        filters={"annotation_fdr": 0.1},
        catalog=catalog,
        selection_path=selection,
    )

    with pytest.raises(ValidationError, match="must match"):
        materialize_selection(
            source=source,
            selection_path=selection,
            datasets_dir=datasets_dir,
            catalog=catalog,
            annotation_options={"annotation_fdr": 0.2},
        )


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
    annotation_reader = SQLiteAnnotationReader(
        catalog.path,
        merged_dataset_id="pilot",
    )
    merged_annotations = annotation_reader.get_spectrum_annotations(5)
    assert merged_annotations[0]["sumFormula"] == "C6H12O6"
    assert merged_annotations[0]["source_spectrum_id"] == 5


def test_download_merge_reuses_workspace_source_before_provider_request(
    tmp_path: Path,
    msi_fixture_path: Path,
) -> None:
    """A canonical source pair bypasses provider download during merge."""
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
    retained = datasets_dir / "one"
    retained.mkdir(parents=True)
    shutil.copy2(msi_fixture_path, retained / "one.imzML")
    shutil.copy2(msi_fixture_path.with_suffix(".ibd"), retained / "one.ibd")

    materialize_and_merge_selection(
        source=source,
        selection_path=selection,
        datasets_dir=datasets_dir,
        catalog=catalog,
        output_path=datasets_dir / "merged" / "pilot" / "dataset.imzML",
        merged_dataset_id="pilot",
        row_width=3,
    )

    assert source.download_calls == []
    assert (retained / "one.imzML").is_file()
    assert (retained / "one.ibd").is_file()


def test_download_merge_completes_partial_workspace_source_in_place(
    tmp_path: Path,
    msi_fixture_path: Path,
) -> None:
    """An incomplete canonical pair is passed alone to the provider adapter."""
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
    retained = datasets_dir / "one"
    retained.mkdir(parents=True)
    shutil.copy2(msi_fixture_path, retained / "one.imzML")

    materialize_and_merge_selection(
        source=source,
        selection_path=selection,
        datasets_dir=datasets_dir,
        catalog=catalog,
        output_path=datasets_dir / "merged" / "pilot" / "dataset.imzML",
        merged_dataset_id="pilot",
        row_width=3,
    )

    assert source.download_calls == [("one", retained)]
    assert (retained / "one.imzML").is_file()
    assert (retained / "one.ibd").is_file()
