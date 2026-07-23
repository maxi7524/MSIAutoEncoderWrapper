"""Tests for dataset metadata, annotations, and minimal merge provenance."""

from __future__ import annotations

from pathlib import Path

from msi_autoencoder_wrapper.core.wrapper import MSIAutoEncoderWrapper
from msi_autoencoder_wrapper.workspace.dataset_catalog import DatasetCatalog


def test_catalog_preserves_metadata_and_filters_annotations_at_read_time(
    tmp_path: Path,
) -> None:
    """Import stores every record while explicit readers may select a subset."""
    catalog = DatasetCatalog(tmp_path / "datasets" / "catalog.sqlite")
    catalog.upsert_dataset(
        source="metaspace",
        dataset_id="dataset-a",
        name="Dataset A",
        metadata={"organism": "mouse", "condition": "disease-a"},
    )
    catalog.replace_annotations(
        source="metaspace",
        dataset_id="dataset-a",
        annotations=[
            {
                "id": "a-1",
                "sumFormula": "C6H12O6",
                "adduct": "+H",
                "fdr": 0.05,
                "database_name": "HMDB",
                "database_version": "v4",
            },
            {
                "id": "a-2",
                "sumFormula": "C5H11NO2",
                "adduct": "+Na",
                "fdr": 0.2,
                "database_name": "HMDB",
                "database_version": "v4",
            },
        ],
    )

    assert catalog.get_dataset("metaspace", "dataset-a")["metadata"] == {
        "condition": "disease-a",
        "organism": "mouse",
    }
    assert len(catalog.get_annotations(source="metaspace", dataset_id="dataset-a")) == 2
    selected = catalog.get_annotations(
        source="metaspace",
        dataset_id="dataset-a",
        filters={"max_fdr": 0.1, "adduct": "+H"},
    )
    assert [annotation["id"] for annotation in selected] == ["a-1"]


def test_catalog_maps_only_source_spatial_id_to_merged_index(tmp_path: Path) -> None:
    """Merge provenance remains independent from spatial coordinate transforms."""
    catalog = DatasetCatalog(tmp_path / "catalog.sqlite")
    catalog.register_merged_dataset("merged-a", tmp_path / "merged-a.imzML")
    catalog.replace_spectrum_mappings(
        "merged-a",
        [
            {
                "source": "metaspace",
                "source_dataset_id": "dataset-a",
                "source_spatial_id": 7,
                "merged_spectrum_index": 2,
            }
        ],
    )

    assert catalog.get_merged_index(
        merged_dataset_id="merged-a",
        source="metaspace",
        source_dataset_id="dataset-a",
        source_spatial_id=7,
    ) == 2
    assert catalog.get_source_index(
        merged_dataset_id="merged-a",
        merged_spectrum_index=2,
    ) == {
        "source": "metaspace",
        "source_dataset_id": "dataset-a",
        "source_spatial_id": 7,
    }


def test_workspace_resolves_nested_dataset_and_keeps_legacy_alias(tmp_path: Path) -> None:
    """New dataset folders work through both the new and legacy workspace APIs."""
    dataset_path = tmp_path / "datasets" / "sample" / "sample.imzML"
    dataset_path.parent.mkdir(parents=True)
    dataset_path.write_text("fixture", encoding="utf-8")
    wrapper = MSIAutoEncoderWrapper(project_path=str(tmp_path))

    wrapper.workspace.set_active_image("sample")

    assert wrapper.workspace.get_active_image_file_path() == dataset_path
    assert wrapper.workspace._resolve_and_verify_image_file("sample") == dataset_path
    assert wrapper.workspace.get_imgs_dir() == wrapper.workspace.get_datasets_dir()
