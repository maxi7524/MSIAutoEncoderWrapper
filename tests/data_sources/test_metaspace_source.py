"""Offline contract tests for the METASPACE source adapter."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, List

from msi_autoencoder_wrapper.data_sources.strategies.metaspace_source import (
    MetaspaceDatasetSource,
)


class FakeMetaspaceDataset:
    """Minimal object matching the official client's dataset surface."""

    id = "dataset-a"
    name = "Dataset A"
    metadata = {"organism": "mouse", "condition": "healthy"}
    polarity = "Positive"
    status = "FINISHED"
    image_size = {"x": 2, "y": 2}
    database_details = [SimpleNamespace(id=1, name="HMDB", version="v4")]

    def __init__(self) -> None:
        self.result_calls: List[Dict[str, Any]] = []

    def results(self, **kwargs: Any) -> List[Dict[str, Any]]:
        self.result_calls.append(kwargs)
        return [{"id": "one", "sumFormula": "C6H12O6", "fdr": 0.5}]

    def download_to_dir(self, path: Path, base_name: str) -> None:
        path.mkdir(parents=True, exist_ok=True)
        (path / f"{base_name}.imzML").write_text("imzml", encoding="utf-8")
        (path / f"{base_name}.ibd").write_bytes(b"ibd")


class FakeMetaspaceClient:
    """Capture discovery filters and return a stable fake dataset."""

    def __init__(self) -> None:
        self.value = FakeMetaspaceDataset()
        self.filters: Dict[str, Any] = {}

    def datasets(self, **filters: Any) -> List[FakeMetaspaceDataset]:
        self.filters = filters
        return [self.value]

    def dataset(self, id: str) -> FakeMetaspaceDataset:
        assert id == self.value.id
        return self.value


def test_metaspace_adapter_preserves_metadata_and_imports_broad_annotations(
    tmp_path: Path,
) -> None:
    """The adapter passes native filters and adds database provenance."""
    client = FakeMetaspaceClient()
    source = MetaspaceDatasetSource(client=client)

    records = source.search_datasets({"organism": "Mus musculus"})
    annotations = source.get_annotations("dataset-a")
    destination = source.download_dataset("dataset-a", tmp_path / "dataset-a")

    assert client.filters == {"organism": "Mus musculus"}
    assert records[0]["metadata"]["condition"] == "healthy"
    assert records[0]["metadata"]["databases"] == [
        {"name": "HMDB", "version": "v4", "id": 1}
    ]
    assert annotations[0]["database_name"] == "HMDB"
    assert annotations[0]["database_version"] == "v4"
    assert client.value.result_calls == [{"database": ("HMDB", "v4"), "fdr": 0.5}]
    assert (destination / "dataset-a.imzML").is_file()
