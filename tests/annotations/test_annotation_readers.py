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


def test_data_reader_automatically_loads_workspace_annotations(
    tmp_path: Path,
    msi_fixture_path: Path,
) -> None:
    """A registered image automatically receives the workspace annotation reader."""
    catalog = DatasetCatalog(tmp_path / "datasets" / "catalog.sqlite")
    catalog.upsert_dataset(
        source="metaspace",
        dataset_id="dataset-a",
        name="Dataset A",
        metadata={"condition": "disease"},
        local_path=msi_fixture_path.parent,
        status="materialized",
    )
    catalog.replace_annotations(
        source="metaspace",
        dataset_id="dataset-a",
        annotations=[{"id": "one", "formula": "A", "spectrum_ids": [0]}],
    )
    wrapper = MSIAutoEncoderWrapper(project_path=str(tmp_path))

    wrapper.context_manager.set_reader(
        MockMSIReader(msi_fixture_path),
        str(msi_fixture_path),
    )
    wrapper.workspace.set_active_image(str(msi_fixture_path))

    assert wrapper.active_context.annotation_reader is not None
    assert (
        wrapper.active_context.annotation_reader.get_dataset_metadata()["name"]
        == "Dataset A"
    )
    assert (
        wrapper.active_context.annotation_reader.get_spectrum_annotations(0)[0]["formula"]
        == "A"
    )


def test_data_reader_accepts_explicit_annotation_catalog(
    tmp_path: Path,
    msi_fixture_path: Path,
) -> None:
    """A catalog outside the workspace can be selected explicitly."""
    catalog_path = tmp_path / "external" / "annotations.sqlite"
    catalog = DatasetCatalog(catalog_path)
    catalog.upsert_dataset(
        source="metaspace",
        dataset_id="dataset-a",
        name="External Dataset",
        metadata={},
        local_path=msi_fixture_path,
        status="materialized",
    )
    wrapper = MSIAutoEncoderWrapper(project_path=str(tmp_path / "workspace"))

    wrapper.context_manager.set_reader(
        MockMSIReader(msi_fixture_path),
        str(msi_fixture_path),
        annotation_catalog_path=str(catalog_path),
    )
    wrapper.workspace.set_active_image(str(msi_fixture_path))

    assert wrapper.active_context.annotation_reader is not None
    assert (
        wrapper.active_context.annotation_reader.get_dataset_metadata()["name"]
        == "External Dataset"
    )


def test_merged_annotations_survive_removal_of_source_files(tmp_path: Path) -> None:
    """Merged metadata and annotations remain in SQLite after source cleanup."""
    source_directory = tmp_path / "source-dataset"
    source_directory.mkdir()
    source_imzml = source_directory / "source.imzML"
    source_csv = source_directory / "annotations.csv"
    source_imzml.touch()
    source_csv.touch()
    merged_imzml = tmp_path / "merged-dataset" / "merged.imzML"
    merged_imzml.parent.mkdir()
    merged_imzml.touch()

    catalog_path = tmp_path / "catalog.sqlite"
    catalog = DatasetCatalog(catalog_path)
    catalog.upsert_dataset(
        source="metaspace",
        dataset_id="dataset-a",
        name="Dataset A",
        metadata={"condition": "disease"},
        local_path=source_directory,
        status="merged",
    )
    catalog.replace_annotations(
        source="metaspace",
        dataset_id="dataset-a",
        annotations=[{"id": "one", "formula": "A", "spectrum_ids": [0]}],
    )
    catalog.register_merged_dataset("merged-a", merged_imzml)
    catalog.replace_spectrum_mappings(
        "merged-a",
        [
            {
                "source": "metaspace",
                "source_dataset_id": "dataset-a",
                "source_spectrum_id": 0,
                "merged_spectrum_index": 0,
            }
        ],
    )

    source_imzml.unlink()
    source_csv.unlink()
    source_directory.rmdir()
    reader = SQLiteAnnotationReader(catalog_path, merged_dataset_id="merged-a")

    assert reader.get_spectrum_metadata(0)["metadata"]["condition"] == "disease"
    assert reader.get_spectrum_annotations(0)[0]["formula"] == "A"


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
