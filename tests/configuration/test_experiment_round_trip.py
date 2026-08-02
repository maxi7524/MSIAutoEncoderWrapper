"""Integration tests for schema-v2 experiment folders."""

from __future__ import annotations

from pathlib import Path

from msi_autoencoder_wrapper.binners.binners_manager import BinnerManager
from msi_autoencoder_wrapper.core.wrapper import MSIAutoEncoderWrapper
from tests.mocks.components import build_small_autoencoder


def test_saved_experiment_loads_context_and_model_from_one_directory(
    tmp_path: Path, msi_fixture_path: Path
) -> None:
    source = MSIAutoEncoderWrapper(project_path=str(tmp_path / "source"))
    binner = BinnerManager.get_binner(
        "LinearBinning", x_min=0.0, x_max=32.0, bin_step=1.0
    )
    source.context_manager.set_reader("PyImzMLReader", str(msi_fixture_path))
    source.context_manager.set_binner(binner, str(msi_fixture_path))
    source.workspace.set_active_image(str(msi_fixture_path))
    source.models_manager.attach_model(
        build_small_autoencoder(), model_name="experiment-ae", trained=True
    )
    model_dir = source.workspace.save_model()

    restored = MSIAutoEncoderWrapper(project_path=str(tmp_path / "restored"))
    config = restored.load_experiment(str(model_dir))

    assert config["schema_version"] == 2
    assert restored.active_model is not None
    assert restored.models_manager._active_model_name == "experiment-ae"
    assert restored.active_context.reader.GetNumberOfSpectra() == 6
