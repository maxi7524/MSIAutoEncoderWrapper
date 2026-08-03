"""Tests for the shared component configuration contract."""

from __future__ import annotations

from pathlib import Path

import pytest

from msi_autoencoder_wrapper.binners.binners_manager import BinnerManager
from msi_autoencoder_wrapper.configuration import (
    ConfigurableComponent,
    get_component_config,
    make_json_compatible,
)
from msi_autoencoder_wrapper.utils.exceptions import ProjectConfigError
from msi_autoencoder_wrapper.training.criterions.criterions_manager import CriterionsManager
from msi_autoencoder_wrapper.metrics import BaseMetric, MetricsRegistry
from msi_autoencoder_wrapper.models.architectures import ArchitecturesManager
from msi_autoencoder_wrapper.models.architectures.types.autoencoders.decoders.base_decoder import MSIBaseDecoder
from msi_autoencoder_wrapper.models.architectures.types.autoencoders.encoders.base_encoder import MSIBaseEncoder
from msi_autoencoder_wrapper.models.architectures.types.autoencoders.heads.base_head import MSIBaseHead
from msi_autoencoder_wrapper.models.architectures.types.autoencoders.projectors.base_projector import MSIBaseProjector


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
    assert binner.export_config()["parameters"] == binner.get_config()


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


def test_architecture_registry_enforces_configurable_category_contracts() -> None:
    """Every registered architecture strategy uses the shared configuration system."""
    ArchitecturesManager.discover_architectures()
    category_bases = {
        "decoder": MSIBaseDecoder,
        "encoder": MSIBaseEncoder,
        "head": MSIBaseHead,
        "projector": MSIBaseProjector,
    }

    for categories in ArchitecturesManager._COMPONENT_REGISTRY.values():
        for category, implementations in categories.items():
            for implementation in implementations.values():
                assert issubclass(implementation, category_bases[category])
                assert issubclass(implementation, ConfigurableComponent)


def test_registered_metric_classes_use_shared_configuration_contract() -> None:
    """Stateful metric implementations participate in portable configuration."""
    for definitions in MetricsRegistry.available().values():
        for definition in definitions.values():
            implementation = definition.implementation
            if isinstance(implementation, type):
                assert issubclass(implementation, BaseMetric)
                metric = implementation()
                restored = type(metric).from_config(metric.get_config())
                assert restored.get_config() == metric.get_config()
