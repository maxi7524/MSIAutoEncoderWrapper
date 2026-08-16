"""Tests for rectangular imzML merge and source-index provenance."""

from __future__ import annotations

from pathlib import Path

from msi_dataset_manager.operations.composition.imzml_writer import (
    ImzMLMergeInput,
    ImzMLMerger,
)
from msi_autoencoder_wrapper.readers.strategies.pyimzml_reader import PyImzMLReader
from msi_dataset_manager.operations.composition.selection import (
    select_merge_spectrum_ids,
)


def test_merger_writes_rectangular_coordinates_and_index_mapping(
    tmp_path: Path,
    msi_fixture_path: Path,
) -> None:
    """Selected source indices become consecutive merged spectrum indices."""
    output = tmp_path / "datasets" / "merged" / "test-merge" / "dataset.imzML"
    merger = ImzMLMerger()

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
        row_width=2,
    )

    reader = PyImzMLReader(result.path)
    assert reader.GetNumberOfSpectra() == 3
    assert [reader.GetSpectrumPosition(index) for index in range(3)] == [
        (1, 1, 1),
        (2, 1, 1),
        (1, 2, 1),
    ]
    assert result.pixel_mappings[1]["source_spectrum_id"] == 2
    assert result.pixel_mappings[1]["merged_spectrum_index"] == 1
    assert result.pixel_mappings[2]["source_dataset_id"] == "dataset-b"


def test_merger_defaults_to_all_candidate_spectra(
    tmp_path: Path,
    msi_fixture_path: Path,
) -> None:
    """An omitted unannotated limit retains every available source spectrum."""
    output = tmp_path / "datasets" / "merged" / "annotated" / "dataset.imzML"

    result = ImzMLMerger().merge(
        inputs=[
            ImzMLMergeInput(
                source="pride",
                dataset_id="bladder",
                imzml_path=msi_fixture_path,
                annotation_records=[{"fdr": 0.05, "spectrum_ids": [1, 3]}],
            )
        ],
        output_path=output,
    )

    assert PyImzMLReader(result.path).GetNumberOfSpectra() == 6
    assert result.pixel_mappings[0]["source_spectrum_id"] == 1
    assert result.pixel_mappings[1]["source_spectrum_id"] == 3


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
