"""Tests for rectangular imzML merge and source-index provenance."""

from __future__ import annotations

from pathlib import Path

from msi_dataset_manager.operations import ImzMLMergeInput, ImzMLMerger
from msi_autoencoder_wrapper.readers.strategies.pyimzml_reader import PyImzMLReader
from msi_dataset_manager.catalog import DatasetCatalog
from msi_dataset_manager.operations.spectrum_selection import (
    select_merge_spectrum_ids,
)


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


def test_merger_defaults_to_all_candidate_spectra(
    tmp_path: Path,
    msi_fixture_path: Path,
) -> None:
    """An omitted unannotated limit retains every available source spectrum."""
    catalog = DatasetCatalog(tmp_path / "datasets" / "catalog.sqlite")
    catalog.upsert_dataset(
        source="pride",
        dataset_id="bladder",
        name="Bladder",
        metadata={},
    )
    catalog.replace_annotations(
        source="pride",
        dataset_id="bladder",
        annotations=[
            {
                "annotation_id": "molecule-a",
                "formula": "C6H12O6",
                "spectrum_ids": [1, 3],
            }
        ],
    )
    output = tmp_path / "datasets" / "merged" / "annotated" / "dataset.imzML"

    result = ImzMLMerger(catalog).merge(
        inputs=[
            ImzMLMergeInput(
                source="pride",
                dataset_id="bladder",
                imzml_path=msi_fixture_path,
            )
        ],
        output_path=output,
        merged_dataset_id="annotated",
    )

    assert PyImzMLReader(result).GetNumberOfSpectra() == 6
    assert catalog.get_source_index(
        merged_dataset_id="annotated",
        merged_spectrum_index=0,
    )["source_spectrum_id"] == 1
    assert catalog.get_source_index(
        merged_dataset_id="annotated",
        merged_spectrum_index=1,
    )["source_spectrum_id"] == 3


def test_unannotated_sampling_uses_larger_limit_and_available_cap() -> None:
    """Ratio and amount combine as min(max(ratio, amount), available)."""
    selected = select_merge_spectrum_ids(
        candidate_ids=list(range(10)),
        annotated_ids=[0, 1],
        unannotated_ratio=3.0,
        unannotated_amount=4,
        random_seed=42,
        seed_namespace="pride:bladder",
    )

    assert selected[:2] == [0, 1]
    assert len(selected[2:]) == 6
    assert selected == select_merge_spectrum_ids(
        candidate_ids=list(range(10)),
        annotated_ids=[0, 1],
        unannotated_ratio=3.0,
        unannotated_amount=4,
        random_seed=42,
        seed_namespace="pride:bladder",
    )
