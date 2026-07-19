"""Tests for the shared component configuration contract."""

from __future__ import annotations

from pathlib import Path

import pytest

from msi_autoencoder_wrapper.binners.binners_manager import BinnerManager
from msi_autoencoder_wrapper.utils.configuration import (
    get_component_config,
    make_json_compatible,
)
from msi_autoencoder_wrapper.utils.exceptions import ProjectConfigError
from msi_autoencoder_wrapper.training.criterions.criterions_manager import CriterionsManager


def test_component_config_is_portable_and_isolated() -> None:
    """Functional modules return copied JSON-compatible parameter dictionaries."""
    BinnerManager.discover_strategies()
    binner = BinnerManager.get_binner(
        "LinearBinning",
        x_min=100.0,
        x_max=200.0,
        bin_step=0.5,
    )

    config = get_component_config(binner)
    config["parameters"]["bin_step"] = 10.0

    assert config["type"] == "LinearBinning"
    assert binner.get_config()["bin_step"] == 0.5
    assert binner.GetConfig() == binner.get_config()


def test_json_normalization_handles_paths_and_rejects_runtime_objects() -> None:
    """Portable configuration converts paths and rejects arbitrary runtime state."""
    assert make_json_compatible({"source": Path("image.imzML")}) == {
        "source": "image.imzML"
    }

    with pytest.raises(ProjectConfigError):
        make_json_compatible({"runtime": object()})


def test_json_normalization_describes_ready_functional_components() -> None:
    """Manually constructed components embedded in settings remain reproducible."""
    CriterionsManager.discover_criterions()
    criterion = CriterionsManager._REGISTRY["autoencoder"]["reconstruction"]["MSELoss"](
        reduction="sum"
    )

    normalized = make_json_compatible({"criterion": criterion})

    assert normalized["criterion"]["type"] == "MSIMSELoss"
    assert normalized["criterion"]["parameters"] == {"reduction": "sum"}
