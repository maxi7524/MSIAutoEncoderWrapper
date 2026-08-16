"""Integration tests for the dataset-manager annotation reader."""

from __future__ import annotations

import shutil
from pathlib import Path

import numpy as np

from msi_autoencoder_wrapper.core.wrapper import MSIAutoEncoderWrapper
from msi_dataset_manager.annotations.merge import (
    AnnotationMergeInput,
    MergedAnnotationWriter,
)
from msi_dataset_manager.imzml import PyImzMLReader
from msi_dataset_manager.sources.base import SourceAnnotationExport
from msi_dataset_manager.sources.strategies.metaspace.csv import write_annotation_csv_pair
from tests.mocks.components import MockMSIReader


def _copy_image(destination: Path, source_image: Path, name: str) -> Path:
    destination.mkdir(parents=True)
    image_path = destination / f"{name}.imzML"
    shutil.copy2(source_image, image_path)
    shutil.copy2(source_image.with_suffix(".ibd"), image_path.with_suffix(".ibd"))
    return image_path


def _write_source_annotations(image_path: Path) -> None:
    reader = PyImzMLReader(image_path)
    coordinates = [
        reader.GetSpectrumPosition(index)
        for index in range(reader.GetNumberOfSpectra())
    ]
    width = max(coordinate[0] for coordinate in coordinates)
    height = max(coordinate[1] for coordinate in coordinates)
    ion_image = np.zeros((height, width), dtype=np.float32)
    x, y, _ = coordinates[0]
    ion_image[y - 1, x - 1] = 1.0
    write_annotation_csv_pair(
        directory=image_path.parent,
        dataset_id=image_path.stem,
        dataset_name="Source image",
        annotations=[
            (
                {
                    "schema_version": 2,
                    "source": "metaspace",
                    "source_annotation_id": "db:C6H12O6+H",
                    "formula": "C6H12O6",
                    "adduct": "+H",
                    "mz": 181.0707,
                    "fdr": 0.05,
                    "database_id": "db",
                    "source_record": {"provider": "metaspace"},
                },
                ion_image,
            )
        ],
        reader=reader,
    )


def test_wrapper_auto_loads_current_source_export(
    tmp_path: Path,
    msi_fixture_path: Path,
) -> None:
    """The wrapper delegates current METASPACE CSV files to Dataset Manager."""
    image_path = _copy_image(tmp_path / "source", msi_fixture_path, "dataset-a")
    _write_source_annotations(image_path)
    wrapper = MSIAutoEncoderWrapper(project_path=str(tmp_path / "workspace"))

    wrapper.context_manager.set_reader(MockMSIReader(image_path), str(image_path))
    wrapper.workspace.set_active_image(str(image_path))

    reader = wrapper.active_context.annotation_reader
    assert type(reader).__module__.startswith("msi_dataset_manager.annotations")
    assert reader.get_dataset_metadata()["dataset_id"] == "dataset-a"
    assert reader.get_spectrum_annotations(0)[0]["formula"] == "C6H12O6"


def test_wrapper_auto_loads_sibling_merged_store(
    tmp_path: Path,
    msi_fixture_path: Path,
) -> None:
    """A composed image receives the one final SQLite reader."""
    image_path = _copy_image(tmp_path / "merged", msi_fixture_path, "cohort")
    export = SourceAnnotationExport(
        source="metaspace",
        dataset_id="source-a",
        schema_version=2,
        metadata={"condition": "disease"},
        records=[
            {
                "source_annotation_id": "db:A",
                "formula": "A",
                "adduct": "+H",
                "mz": 100.0,
                "fdr": 0.05,
                "source_record": {},
                "spectrum_ids": [0],
            }
        ],
    )
    MergedAnnotationWriter().write(
        path=image_path.with_suffix(".sqlite"),
        inputs=[
            AnnotationMergeInput(
                source="metaspace",
                dataset_id="source-a",
                name="Source A",
                imzml_path=tmp_path / "source-a.imzML",
                annotation_export=export,
                metadata={},
            )
        ],
        pixel_mappings=[
            {
                "source": "metaspace",
                "source_dataset_id": "source-a",
                "source_spectrum_id": 0,
                "merged_spectrum_index": 0,
            }
        ],
        filtering_metadata={},
    )
    wrapper = MSIAutoEncoderWrapper(project_path=str(tmp_path / "workspace"))

    wrapper.context_manager.set_reader(MockMSIReader(image_path), str(image_path))
    wrapper.workspace.set_active_image(str(image_path))

    reader = wrapper.active_context.annotation_reader
    assert type(reader).__module__.startswith("msi_dataset_manager.annotations")
    assert reader.get_annotations()[0]["formula"] == "A"
    assert reader.get_spectrum_metadata(0)["source_spectrum_id"] == 0


def test_wrapper_does_not_load_old_csv_names(
    tmp_path: Path,
    msi_fixture_path: Path,
) -> None:
    """Old METASPACE filenames are not a supported fallback."""
    image_path = _copy_image(tmp_path / "legacy", msi_fixture_path, "dataset-a")
    (image_path.parent / "metaspace_annotations.csv").write_text("old", encoding="utf-8")
    (image_path.parent / "dataset-a_pixel_intensities.csv").write_text(
        "old",
        encoding="utf-8",
    )
    wrapper = MSIAutoEncoderWrapper(project_path=str(tmp_path / "workspace"))

    wrapper.context_manager.set_reader(MockMSIReader(image_path), str(image_path))
    wrapper.workspace.set_active_image(str(image_path))

    assert wrapper.active_context.annotation_reader is None
