"""Tests for independent workspace path contracts."""

from msi_dataset_manager import DatasetWorkspaceLayout


def test_layout_shares_sources_and_creates_sqlite_only_for_composition(tmp_path) -> None:
    """Source files are shared while only composed cohorts own SQLite."""
    layout = DatasetWorkspaceLayout(tmp_path)

    assert layout.imzml_path("image-a") == (
        tmp_path / "datasets" / "image-a" / "image-a.imzML"
    )
    assert layout.composed_catalog_path("kidney") == (
        tmp_path / "datasets" / "kidney" / "kidney.sqlite"
    )
    assert layout.composition_path("kidney") == (
        tmp_path / "datasets" / "kidney" / "composition.json"
    )
