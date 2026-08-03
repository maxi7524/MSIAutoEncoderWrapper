"""Tests for manual and registered dataset setup targets."""

from __future__ import annotations

from types import SimpleNamespace

from msi_autoencoder_wrapper.core.mixins.models_manager.proxies.dataset_proxy import (
    DatasetProxy,
)
from msi_autoencoder_wrapper.models.datasets.dataset_manager import DatasetManager
from tests.mocks.components import MockActiveContext, MockDataset


def test_dataset_manager_accepts_class_and_ready_instance(
    mock_active_context: MockActiveContext,
) -> None:
    """Dataset classes are initialized and ready datasets pass through unchanged."""
    from_class = DatasetManager.get_dataset(
        MockDataset,
        active_context=mock_active_context,
    )
    from_instance = DatasetManager.get_dataset(from_class)

    assert isinstance(from_class, MockDataset)
    assert from_instance is from_class


def test_dataset_proxy_stores_ready_instance_as_build_target(
    mock_active_context: MockActiveContext,
) -> None:
    """The public dataset setup retains a user-created dataset object."""
    dataset = MockDataset(active_context=mock_active_context)
    proxy = DatasetProxy(wrapper_ref=SimpleNamespace())

    proxy.set_dataset(dataset)

    assert proxy._building_buffer["dataset"]["target"] is dataset
    assert proxy._building_buffer["dataset"]["strategy"] == "MockDataset"
