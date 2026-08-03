"""Tests for cohort activation and immutable member datasets."""

from __future__ import annotations

from pathlib import Path

from msi_autoencoder_wrapper.binners.binners_manager import BinnerManager
from msi_autoencoder_wrapper.core.wrapper import MSIAutoEncoderWrapper
from msi_autoencoder_wrapper.models.datasets.strategies.cohort_dataset import (
    CohortPixelDataset,
)
from tests.mocks.components import MockMSIReader
from tests.mocks.components import build_small_autoencoder


def _register(wrapper, image_key: str, fixture: Path) -> None:
    reader = MockMSIReader(fixture)
    wrapper.context_manager.config_ledger[image_key] = {
        "reader": reader,
        "annotation_reader": None,
        "binner": BinnerManager.get_binner(
            "LinearBinning", x_min=0.0, x_max=32.0, bin_step=1.0
        ),
        "inverse_binner": None,
        "normalization": None,
        "model_functionality": None,
        "tmp": {},
    }


def test_cohort_activation_keeps_local_context_and_use_image_restores_scope(
    tmp_path: Path, msi_fixture_path: Path
) -> None:
    wrapper = MSIAutoEncoderWrapper(project_path=str(tmp_path))
    _register(wrapper, "image-a", msi_fixture_path)
    _register(wrapper, "image-b", msi_fixture_path)
    wrapper.workspace.set_active_image("image-a")
    wrapper.cohorts.create("study")
    wrapper.cohorts.set_images(["image-a", "image-b"], name="study")
    wrapper.cohorts.activate("study")

    assert wrapper.workspace.execution_scope == "cohort"
    assert wrapper.workspace.active_img_name == "image-a"
    with wrapper.workspace.use_image("image-b"):
        assert wrapper.workspace.execution_scope == "local"
        assert wrapper.workspace.active_img_name == "image-b"
    assert wrapper.workspace.execution_scope == "cohort"
    assert wrapper.workspace.active_img_name == "image-a"


def test_cohort_dataset_preserves_member_identity_and_saves_definition(
    tmp_path: Path, msi_fixture_path: Path
) -> None:
    wrapper = MSIAutoEncoderWrapper(project_path=str(tmp_path))
    _register(wrapper, "image-a", msi_fixture_path)
    _register(wrapper, "image-b", msi_fixture_path)
    wrapper.cohorts.create("study")
    context = wrapper.cohorts.set_images(["image-a", "image-b"], name="study")
    dataset = CohortPixelDataset(context, normalization="none")

    assert len(dataset) == 12
    assert dataset.get_sample_id(6) == {"image_key": "image-b", "spectrum_id": 0}
    assert dataset[6][0] == ("image-b", 0)
    assert wrapper.cohorts.save("study") == tmp_path / "models" / "cohort_study" / "cohort.json"


def test_common_autoencoder_is_loaded_lazily_without_replacing_active_model(
    tmp_path: Path, msi_fixture_path: Path
) -> None:
    wrapper = MSIAutoEncoderWrapper(project_path=str(tmp_path))
    _register(wrapper, "image-a", msi_fixture_path)
    wrapper.workspace.set_active_image("image-a")
    dependency = build_small_autoencoder()
    wrapper.models_manager.attach_model(dependency, model_name="shared-ae", trained=True)
    wrapper.workspace.save_model(img_name="image-a", model_name="shared-ae")

    wrapper.cohorts.create("study")
    wrapper.cohorts.set_images(["image-a"], name="study")
    context = wrapper.cohorts.set_autoencoder(
        policy="common", model="image-a/shared-ae", name="study"
    )
    wrapper.cohorts.activate("study")
    sentinel = build_small_autoencoder()
    wrapper.models_manager.attach_model(sentinel, model_name="cohort-model")

    with wrapper.workspace.use_image("image-a"):
        interface = wrapper.active_context.autoencoder

    assert interface is not None
    assert interface.torch_object is not sentinel
    assert wrapper.active_model is sentinel
    assert context.common_autoencoder.fingerprint
