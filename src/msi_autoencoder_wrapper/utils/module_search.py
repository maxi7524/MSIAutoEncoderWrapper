"""Utilities for discovering and importing implementation modules."""

from __future__ import annotations

import importlib
import pkgutil
import sys
from collections.abc import Iterable
from types import ModuleType

from .exceptions import raise_project_config_error
from .logger import get_custom_logger

logger = get_custom_logger(__name__)


def discover_modules(
    package: str | ModuleType,
    *,
    recursive: bool = True,
    excluded_parts: Iterable[str] = (),
) -> list[str]:
    """Import implementation modules contained in a package.

    :param package: Importable package name or an imported package object.
    :type package: str | types.ModuleType
    :param recursive: Whether nested packages should be scanned, defaults to True.
    :type recursive: bool
    :param excluded_parts: Exact module path segments that should not be imported.
    :type excluded_parts: collections.abc.Iterable[str]
    :return: Fully qualified names of modules considered by the discovery pass.
    :rtype: list[str]
    :raises ProjectConfigError: If the package cannot be inspected or a discovered
        module cannot be imported.

    The function is idempotent with respect to Python's module cache. Modules that
    are already present in :data:`sys.modules` are reported but not imported again.
    Importing modules can execute registration decorators as a side effect.
    """
    package_module = _resolve_package(package)
    package_paths = getattr(package_module, "__path__", None)
    if package_paths is None:
        raise_project_config_error(
            context_name="ModuleDiscovery",
            message=f"Module '{package_module.__name__}' is not a package and cannot be scanned.",
        )

    excluded = set(excluded_parts)
    prefix = f"{package_module.__name__}."
    scanner = pkgutil.walk_packages if recursive else pkgutil.iter_modules
    discovered_modules: list[str] = []

    logger.debug(
        "Discovering modules in package '%s' with recursive=%s and excluded parts=%s",
        package_module.__name__,
        recursive,
        sorted(excluded),
    )

    for module_info in scanner(package_paths, prefix):
        module_name = module_info.name
        relative_parts = module_name.removeprefix(prefix).split(".")
        if any(part.startswith("_") or part in excluded for part in relative_parts):
            logger.debug("Skipping excluded implementation module: %s", module_name)
            continue

        discovered_modules.append(module_name)
        if module_name in sys.modules:
            continue

        try:
            importlib.import_module(module_name)
        except Exception as error:
            raise_project_config_error(
                context_name="ModuleDiscovery",
                message=f"Failed to import implementation module '{module_name}': {error}",
            )

    logger.info(
        "Discovered %s implementation module(s) in package '%s'.",
        len(discovered_modules),
        package_module.__name__,
    )
    return discovered_modules


def _resolve_package(package: str | ModuleType) -> ModuleType:
    """Resolve a package reference to an imported module object.

    :param package: Importable package name or module object.
    :type package: str | types.ModuleType
    :return: Imported package module.
    :rtype: types.ModuleType
    :raises ProjectConfigError: If the reference has an unsupported type or the
        package cannot be imported.
    """
    if isinstance(package, ModuleType):
        return package

    if not isinstance(package, str) or not package.strip():
        raise_project_config_error(
            context_name="ModuleDiscovery",
            message="Package reference must be a non-empty import path or module object.",
        )

    try:
        return importlib.import_module(package)
    except Exception as error:
        raise_project_config_error(
            context_name="ModuleDiscovery",
            message=f"Failed to import package '{package}': {error}",
        )
