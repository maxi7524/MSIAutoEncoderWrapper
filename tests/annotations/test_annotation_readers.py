"""Tests for standalone and active-context annotation readers."""

from __future__ import annotations

from pathlib import Path

from msi_autoencoder_wrapper.annotations.annotations_manager import AnnotationReaderManager
from msi_autoencoder_wrapper.annotations.sqlite_annotation_reader import SQLiteAnnotationReader
from msi_autoencoder_wrapper.core.wrapper import MSIAutoEncoderWrapper
from msi_autoencoder_wrapper.dataset_management.catalog import DatasetCatalog
from tests.mocks.components import MockMSIReader


def test_sqlite_annotation_reader_defaults_to_all_annotations(tmp_path: Path) -> None:
    """Filters are opt-in and do not modify imported catalog records."""
    catalog_path = tmp_path / "catalog.sqlite"
    catalog = DatasetCatalog(catalog_path)
    catalog.upsert_dataset(
        source="metaspace",
        dataset_id="dataset-a",
        name="Dataset A",
        metadata={"condition": "healthy"},
    )
    catalog.replace_annotations(
        source="metaspace",
        dataset_id="dataset-a",
        annotations=[
            {"id": "one", "fdr": 0.05},
            {"id": "two", "fdr": 0.5},
        ],
    )
    reader = SQLiteAnnotationReader(catalog_path, "metaspace", "dataset-a")

    assert len(reader.get_annotations()) == 2
    assert [item["id"] for item in reader.get_annotations({"max_fdr": 0.1})] == ["one"]


def test_annotation_reader_is_available_from_active_context(
    tmp_path: Path,
    msi_fixture_path: Path,
) -> None:
    """The annotation reader is configured independently from the imzML reader."""
    wrapper = MSIAutoEncoderWrapper(project_path=str(tmp_path))
    data_reader = MockMSIReader(msi_fixture_path)
    annotation_reader = SQLiteAnnotationReader(
        tmp_path / "datasets" / "catalog.sqlite",
        "metaspace",
        "dataset-a",
    )

    wrapper.context_manager.set_reader(data_reader, str(msi_fixture_path))
    wrapper.context_manager.set_annotation_reader(
        annotation_reader,
        str(msi_fixture_path),
    )
    wrapper.workspace.set_active_image(str(msi_fixture_path))

    assert wrapper.active_context.annotation_reader is annotation_reader
    AnnotationReaderManager.load_builtin_reader()
    assert "SQLiteAnnotationReader" in AnnotationReaderManager.REGISTRY


def test_sqlite_annotation_reader_uses_one_canonical_spectrum_schema(tmp_path: Path) -> None:
    """The canonical reader returns all molecules and indexed spectrum labels."""
    catalog_path = tmp_path / "catalog.sqlite"
    catalog = DatasetCatalog(catalog_path)
    catalog.upsert_dataset(
        source="metaspace",
        dataset_id="dataset-a",
        name="Dataset A",
        metadata={"condition": "disease"},
    )
    catalog.replace_annotations(
        source="metaspace",
        dataset_id="dataset-a",
        annotations=[
            {"id": "one", "formula": "A", "spectrum_ids": [0]},
            {"id": "two", "formula": "B", "spectrum_ids": [1]},
        ],
    )
    reader = SQLiteAnnotationReader(catalog_path, "metaspace", "dataset-a")

    assert len(reader.get_annotations()) == 2
    assert [item["formula"] for item in reader.get_spectrum_annotations(1)] == ["B"]
    assert reader.get_spectrum_metadata(1)["metadata"]["condition"] == "disease"
