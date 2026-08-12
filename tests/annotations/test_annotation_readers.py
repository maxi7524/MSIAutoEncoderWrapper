"""Tests for standalone and active-context annotation readers."""

from __future__ import annotations

import shutil
from pathlib import Path

import pytest

from msi_autoencoder_wrapper.annotations.annotations_manager import AnnotationReaderManager
from msi_autoencoder_wrapper.annotations.strategies.metaspace_csv_annotation_reader import (
    MetaspaceCSVAnnotationReader,
)
from msi_autoencoder_wrapper.annotations.strategies.sqlite_annotation_reader import (
    SQLiteAnnotationReader,
)
from msi_autoencoder_wrapper.core.wrapper import MSIAutoEncoderWrapper
from msi_autoencoder_wrapper.readers.strategies.pyimzml_reader import PyImzMLReader
from msi_autoencoder_wrapper.utils.exceptions import ValidationError
from msi_dataset_manager.catalog import DatasetCatalog
from tests.mocks.components import MockMSIReader


def _copy_image_with_metaspace_csv(
    destination: Path,
    source_image: Path,
    *,
    intensity: float = 12.5,
) -> Path:
    """Create one local image with its paired METASPACE CSV exports."""
    destination.mkdir()
    image_path = destination / "example.imzML"
    shutil.copy2(source_image, image_path)
    shutil.copy2(source_image.with_suffix(".ibd"), image_path.with_suffix(".ibd"))
    x, y, _ = PyImzMLReader(image_path).GetSpectrumPosition(0)
    (destination / "metaspace_annotations.csv").write_text(
        "#METASPACE export\n"
        "group,datasetName,datasetId,formula,adduct,mz,fdr\n"
        "example,Example image,dataset-a,C6H12O6,+H,181.0707,0.05\n",
        encoding="utf-8",
    )
    (destination / "example_pixel_intensities.csv").write_text(
        "mol_formula,adduct,mz,moleculeNames,moleculeIds,"
        f"x{x - 1}_y{y - 1}\n"
        f"C6H12O6,+H,181.0707,Glucose,HMDB0000122,{intensity}\n",
        encoding="utf-8",
    )
    return image_path


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
    AnnotationReaderManager.load_builtin_readers()
    assert "SQLiteAnnotationReader" in AnnotationReaderManager.REGISTRY


def test_annotation_manager_uses_sqlite_as_default(tmp_path: Path) -> None:
    """The canonical SQLite strategy is selected when no name is supplied."""
    catalog_path = tmp_path / "catalog.sqlite"
    catalog = DatasetCatalog(catalog_path)
    catalog.upsert_dataset(
        source="metaspace",
        dataset_id="one",
        name="One",
        metadata={},
    )

    reader = AnnotationReaderManager.get_reader(
        catalog_path=catalog_path,
        source="metaspace",
        dataset_id="one",
    )

    assert isinstance(reader, SQLiteAnnotationReader)


def test_data_reader_detects_local_metaspace_csv_annotations(
    tmp_path: Path,
    msi_fixture_path: Path,
) -> None:
    """A local METASPACE CSV pair is attached when no catalog entry exists."""
    image_path = _copy_image_with_metaspace_csv(tmp_path / "image", msi_fixture_path)
    wrapper = MSIAutoEncoderWrapper(project_path=str(tmp_path / "workspace"))

    wrapper.context_manager.set_reader(MockMSIReader(image_path), str(image_path))
    wrapper.workspace.set_active_image(str(image_path))

    reader = wrapper.active_context.annotation_reader
    assert isinstance(reader, MetaspaceCSVAnnotationReader)
    assert reader.get_dataset_metadata()["dataset_id"] == "dataset-a"
    assert reader.get_spectrum_annotations(0)[0]["formula"] == "C6H12O6"


def test_annotation_reader_can_auto_detect_after_data_reader_setup(
    tmp_path: Path,
    msi_fixture_path: Path,
) -> None:
    """Explicit auto-detection uses the existing local image context."""
    image_path = _copy_image_with_metaspace_csv(tmp_path / "image", msi_fixture_path)
    wrapper = MSIAutoEncoderWrapper(project_path=str(tmp_path / "workspace"))
    wrapper.context_manager.set_reader(
        MockMSIReader(image_path),
        str(image_path),
        auto_load_annotations=False,
    )

    reader = wrapper.context_manager.set_annotation_reader(
        img_name_or_path=str(image_path)
    )

    assert isinstance(reader, MetaspaceCSVAnnotationReader)


def test_local_metaspace_csv_rejects_negative_intensity(
    tmp_path: Path,
    msi_fixture_path: Path,
) -> None:
    """Negative ion intensities are invalid annotation data."""
    image_path = _copy_image_with_metaspace_csv(
        tmp_path / "image",
        msi_fixture_path,
        intensity=-1.0,
    )

    with pytest.raises(ValidationError, match="non-negative"):
        MetaspaceCSVAnnotationReader(
            image_path=image_path,
            annotations_path=image_path.parent / "metaspace_annotations.csv",
            pixel_intensities_path=image_path.parent / "example_pixel_intensities.csv",
        )


def test_explicit_missing_annotation_catalog_is_an_error(
    tmp_path: Path,
    msi_fixture_path: Path,
) -> None:
    """An explicit catalog path is strict rather than silently ignored."""
    wrapper = MSIAutoEncoderWrapper(project_path=str(tmp_path / "workspace"))

    with pytest.raises(ValidationError, match="does not exist"):
        wrapper.context_manager.set_reader(
            MockMSIReader(msi_fixture_path),
            str(msi_fixture_path),
            annotation_catalog_path=str(tmp_path / "missing.sqlite"),
        )


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


def test_data_reader_automatically_loads_sibling_composed_catalog(
    tmp_path: Path,
    msi_fixture_path: Path,
) -> None:
    """A composed image discovers its same-named SQLite annotation store."""
    directory = tmp_path / "datasets" / "kidney"
    directory.mkdir(parents=True)
    image_path = directory / "kidney.imzML"
    shutil.copy2(msi_fixture_path, image_path)
    shutil.copy2(msi_fixture_path.with_suffix(".ibd"), image_path.with_suffix(".ibd"))
    catalog = DatasetCatalog(directory / "kidney.sqlite")
    catalog.register_merged_dataset("kidney", image_path)
    wrapper = MSIAutoEncoderWrapper(project_path=str(tmp_path))

    wrapper.context_manager.set_reader(MockMSIReader(image_path), str(image_path))
    wrapper.workspace.set_active_image(str(image_path))

    reader = wrapper.active_context.annotation_reader
    assert isinstance(reader, SQLiteAnnotationReader)
    assert reader.merged_dataset_id == "kidney"


def test_local_csv_precedes_sibling_composed_catalog(
    tmp_path: Path,
    msi_fixture_path: Path,
) -> None:
    """Portable CSV annotations take priority during automatic discovery."""
    image_path = _copy_image_with_metaspace_csv(tmp_path / "kidney", msi_fixture_path)
    catalog = DatasetCatalog(image_path.with_suffix(".sqlite"))
    catalog.register_merged_dataset("kidney", image_path)
    wrapper = MSIAutoEncoderWrapper(project_path=str(tmp_path / "workspace"))

    wrapper.context_manager.set_reader(MockMSIReader(image_path), str(image_path))
    wrapper.workspace.set_active_image(str(image_path))

    assert isinstance(
        wrapper.active_context.annotation_reader,
        MetaspaceCSVAnnotationReader,
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
