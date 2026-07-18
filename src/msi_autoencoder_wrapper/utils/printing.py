"""Shared presentation helpers for registered library implementations."""

from __future__ import annotations

import inspect
from typing import Any, Dict, Optional


def extract_component_signatures(registry: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
    """Extract documentation and constructor parameters from a registry.

    :param registry: Mapping of public names to implementation classes or callables.
    :type registry: Dict[str, Any]
    :return: Structured implementation documentation and parameters.
    :rtype: Dict[str, Dict[str, Any]]
    """
    result: Dict[str, Dict[str, Any]] = {}
    for name in sorted(registry):
        implementation = registry[name]
        inspected_callable = implementation if not inspect.isclass(implementation) else implementation.__init__
        parameters: Dict[str, Any] = {}
        try:
            signature = inspect.signature(inspected_callable)
        except (ValueError, TypeError):
            signature = None

        if signature is not None:
            for parameter_name, parameter in signature.parameters.items():
                if parameter_name in ("self", "args", "kwargs"):
                    continue
                default_value = (
                    "Required"
                    if parameter.default == inspect.Parameter.empty
                    else parameter.default
                )
                parameters[parameter_name] = default_value

        result[name] = {
            "docstring": inspect.getdoc(implementation),
            "parameters": parameters,
        }
    return result


def print_formatted_components(
    title: str,
    key_label: str,
    components_info: Dict[str, Dict[str, Any]],
) -> None:
    """Print structured implementation information using one consistent format.

    :param title: Header displayed above the implementation list.
    :type title: str
    :param key_label: Label displayed next to every implementation name.
    :type key_label: str
    :param components_info: Structured implementation descriptions and parameters.
    :type components_info: Dict[str, Dict[str, Any]]
    """
    print("\n" + "=" * 80)
    print(f" {title}")
    print("=" * 80)

    for name, info in components_info.items():
        documentation = info["docstring"]
        cleaned_documentation = (
            documentation.strip() if documentation else "No documentation provided."
        )
        print(f"\n[{key_label}]: '{name}'")
        print(f" Description: {cleaned_documentation}")
        print(" Parameters (kwargs):")
        if info["parameters"]:
            for parameter_name, default_value in info["parameters"].items():
                print(f"   - {parameter_name}: {default_value}")
        else:
            print("   - None")
    print("=" * 80 + "\n")


def present_available_components(
    registry: Dict[str, Any],
    *,
    title: str,
    key_label: str,
    print_return: bool = True,
    return_value: bool = False,
) -> Optional[Dict[str, Dict[str, Any]]]:
    """Inspect and optionally print implementations using one public format.

    :param registry: Mapping of public implementation names to classes or callables.
    :type registry: Dict[str, Any]
    :param title: Heading displayed above the implementation list.
    :type title: str
    :param key_label: Label displayed next to every implementation name.
    :type key_label: str
    :param print_return: Whether the formatted information should be printed.
    :type print_return: bool
    :param return_value: Whether the structured information should be returned.
    :type return_value: bool
    :return: Structured implementation information when requested, otherwise None.
    :rtype: Optional[Dict[str, Dict[str, Any]]]
    """
    return present_components_info(
        extract_component_signatures(registry),
        title=title,
        key_label=key_label,
        print_return=print_return,
        return_value=return_value,
    )


def present_components_info(
    components_info: Dict[str, Dict[str, Any]],
    *,
    title: str,
    key_label: str,
    print_return: bool = True,
    return_value: bool = False,
) -> Optional[Dict[str, Dict[str, Any]]]:
    """Optionally print or return precomputed implementation information.

    :param components_info: Structured implementation descriptions and parameters.
    :type components_info: Dict[str, Dict[str, Any]]
    :param title: Heading displayed above the implementation list.
    :type title: str
    :param key_label: Label displayed next to every implementation name.
    :type key_label: str
    :param print_return: Whether the formatted information should be printed.
    :type print_return: bool
    :param return_value: Whether the structured information should be returned.
    :type return_value: bool
    :return: ``components_info`` when requested, otherwise None.
    :rtype: Optional[Dict[str, Dict[str, Any]]]
    """
    if print_return:
        print_formatted_components(title, key_label, components_info)
    if return_value:
        return components_info
    return None
