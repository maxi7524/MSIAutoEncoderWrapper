"""Tests for shared component discovery, creation, and presentation utilities."""

from __future__ import annotations

import importlib
import sys
from types import SimpleNamespace

import pytest

from msi_autoencoder_wrapper.core.mixins.models_manager.proxies.architecture_proxy import (
    ArchitectureProxy,
)
from msi_autoencoder_wrapper.utils.printing import present_available_components
from msi_autoencoder_wrapper.utils.exceptions import (
    IncompatibleInterfaceError,
    ProjectConfigError,
)
from msi_autoencoder_wrapper.utils.module_search import discover_modules
from msi_autoencoder_wrapper.utils.validators import resolve_component


class ComponentBase:
    """Base contract used by component factory tests."""


class ExampleComponent(ComponentBase):
    """Example implementation used by component infrastructure tests."""

    def __init__(self, required_value: int, optional_value: str = "default") -> None:
        self.required_value = required_value
        self.optional_value = optional_value


class IncompatibleComponent:
    """Implementation that intentionally violates the test contract."""


def test_discover_modules_imports_public_modules(tmp_path, monkeypatch) -> None:
    """Discovery imports public modules and ignores private implementation files."""
    package_dir = tmp_path / "sample_plugins"
    package_dir.mkdir()
    (package_dir / "__init__.py").write_text("", encoding="utf-8")
    (package_dir / "implementation.py").write_text("REGISTERED = True\n", encoding="utf-8")
    (package_dir / "_private.py").write_text("IMPORTED = True\n", encoding="utf-8")

    monkeypatch.syspath_prepend(str(tmp_path))
    importlib.invalidate_caches()

    discovered = discover_modules("sample_plugins", recursive=False)

    assert discovered == ["sample_plugins.implementation"]
    assert "sample_plugins.implementation" in sys.modules
    assert "sample_plugins._private" not in sys.modules


def test_resolve_component_supports_registry_class_and_instance_targets() -> None:
    """The shared factory supports every documented component target form."""
    registry = {"example": ExampleComponent}

    from_registry = resolve_component(
        "example",
        registry,
        "TestComponent",
        expected_type=ComponentBase,
        required_value=1,
    )
    from_class = resolve_component(
        ExampleComponent,
        registry,
        "TestComponent",
        expected_type=ComponentBase,
        required_value=2,
    )
    existing_instance = ExampleComponent(required_value=3)
    from_instance = resolve_component(
        existing_instance,
        registry,
        "TestComponent",
        expected_type=ComponentBase,
    )

    assert from_registry.required_value == 1
    assert from_class.required_value == 2
    assert from_instance is existing_instance


def test_resolve_component_uses_standardized_errors() -> None:
    """Factory lookup and interface failures use global domain exceptions."""
    registry = {"example": ExampleComponent}

    with pytest.raises(ProjectConfigError, match=r"\[TESTCOMPONENT CONFIG ERROR\]"):
        resolve_component(
            "missing",
            registry,
            "TestComponent",
            expected_type=ComponentBase,
        )

    with pytest.raises(IncompatibleInterfaceError, match=r"\[TESTCOMPONENT INTERFACE ERROR\]"):
        resolve_component(
            IncompatibleComponent(),
            registry,
            "TestComponent",
            expected_type=ComponentBase,
        )


def test_available_component_presentation_has_one_shape(capsys) -> None:
    """Printed and returned implementation information share one deterministic shape."""
    result = present_available_components(
        {"example": ExampleComponent},
        title="Available Test Components",
        key_label="Component",
        print_return=True,
        return_value=True,
    )

    output = capsys.readouterr().out
    assert "Available Test Components" in output
    assert "[Component]: 'example'" in output
    assert result == {
        "example": {
            "docstring": "Example implementation used by component infrastructure tests.",
            "parameters": {
                "required_value": "Required",
                "optional_value": "default",
            },
        }
    }


def test_architecture_availability_uses_active_model_context() -> None:
    """Architecture categories and components resolve within the selected model family."""
    proxy = ArchitectureProxy(wrapper_ref=SimpleNamespace())
    proxy.active_model_type = "autoencoder"

    categories = proxy.get_available_component_categories(
        print_return=False,
        return_value=True,
    )
    encoders = proxy.get_available_components(
        "encoder",
        print_return=False,
        return_value=True,
    )

    assert categories is not None
    assert "encoder" in categories
    assert set(categories["encoder"]) == {"docstring", "parameters"}
    assert encoders is not None
    assert "CNNEncoder" in encoders
    assert set(encoders["CNNEncoder"]) == {"docstring", "parameters"}
