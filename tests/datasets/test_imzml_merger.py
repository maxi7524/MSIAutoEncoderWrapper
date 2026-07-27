"""Tests for rectangular imzML merge and source-index provenance."""

from __future__ import annotations

from pathlib import Path

from msi_autoencoder_wrapper.datasets.imzml_merger import ImzMLMergeInput, ImzMLMerger
from msi_autoencoder_wrapper.readers.strategies.pyimzml_reader import PyImzMLReader
from msi_autoencoder_wrapper.workspace.dataset_catalog import DatasetCatalog


def test_merger_writes_rectangular_coordinates_and_index_mapping(
    tmp_path: Path,
    msi_fixture_path: Path,
) -> None:
    """Selected source indices become consecutive merged spectrum indices."""
    catalog = DatasetCatalog(tmp_path / "datasets" / "catalog.sqlite")
    output = tmp_path / "datasets" / "merged" / "test-merge" / "dataset.imzML"
    merger = ImzMLMerger(catalog)

    result = merger.merge(
        inputs=[
            ImzMLMergeInput(
                source="metaspace",
                dataset_id="dataset-a",
                imzml_path=msi_fixture_path,
                spectrum_ids=[0, 2],
            ),
            ImzMLMergeInput(
                source="metaspace",
                dataset_id="dataset-b",
                imzml_path=msi_fixture_path,
                spectrum_ids=[1],
            ),
        ],
        output_path=output,
        merged_dataset_id="test-merge",
        row_width=2,
    )

    reader = PyImzMLReader(result)
    assert reader.GetNumberOfSpectra() == 3
    assert [reader.GetSpectrumPosition(index) for index in range(3)] == [
        (1, 1, 1),
        (2, 1, 1),
        (1, 2, 1),
    ]
    assert catalog.get_merged_index(
        merged_dataset_id="test-merge",
        source="metaspace",
        source_dataset_id="dataset-a",
        source_spectrum_id=2,
    ) == 1
    assert catalog.get_source_index(
        merged_dataset_id="test-merge",
        merged_spectrum_index=2,
    )["source_dataset_id"] == "dataset-b"
