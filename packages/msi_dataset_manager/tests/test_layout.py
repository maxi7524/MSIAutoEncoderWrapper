"""Tests for independent workspace path contracts."""

from msi_dataset_manager import DatasetWorkspaceLayout


def test_layout_shares_source_images_and_names_catalogues_by_cohort(tmp_path) -> None:
    """Source files are global while SQLite and manifests are cohort-specific."""
    layout = DatasetWorkspaceLayout(tmp_path)

    assert layout.imzml_path("image-a") == (
        tmp_path / "datasets" / "image-a" / "image-a.imzML"
    )
    assert layout.catalog_path("kidney") == (
        tmp_path / "configs" / "datasets" / "kidney" / "kidney.sqlite"
    )
    assert layout.composed_catalog_path("kidney") == (
        tmp_path / "datasets" / "kidney" / "kidney.sqlite"
    )
    assert layout.materialization_path("kidney") == (
        tmp_path / "configs" / "datasets" / "kidney" / "materialization.json"
    )
    assert layout.composition_path("kidney") == (
        tmp_path / "datasets" / "kidney" / "composition.json"
    )
