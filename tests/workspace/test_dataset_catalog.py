"""Tests for dataset paths managed by the wrapper workspace."""

from __future__ import annotations

from pathlib import Path

from msi_autoencoder_wrapper.core.wrapper import MSIAutoEncoderWrapper


def test_workspace_resolves_nested_dataset(tmp_path: Path) -> None:
    """Dataset keys resolve through the canonical nested workspace layout."""
    dataset_path = tmp_path / "datasets" / "sample" / "sample.imzML"
    dataset_path.parent.mkdir(parents=True)
    dataset_path.write_text("fixture", encoding="utf-8")
    wrapper = MSIAutoEncoderWrapper(project_path=str(tmp_path))

    wrapper.workspace.set_active_image("sample")

    assert wrapper.workspace.get_active_image_file_path() == dataset_path
    assert wrapper.workspace._resolve_and_verify_image_file("sample") == dataset_path
