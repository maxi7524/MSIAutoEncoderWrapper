# Heading 1 (Unified System Printing Utilities)
## Consolidated stdout formatter for library configurations, strategies, and components

import inspect
from typing import Any, Dict, Optional
from ...utils.logger import get_custom_logger

# Logger initialization
logger = get_custom_logger(__name__)


# --------------------------------------------------
# Section: Shared Strategy Inspection
# --------------------------------------------------

def extract_component_signatures(registry: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
    """
    Extracts documentation strings and constructor signatures across any strategy registry.

    :param registry: Target manager registry dictionary mapping keys to component classes.
    :type registry: Dict[str, Any]
    :return: Mapping matching strategy aliases to their parameters and docstring.
    :rtype: Dict[str, Dict[str, Any]]
    """
    # Extraction loop
    result = {}
    for name, cls in registry.items():
        init_method = getattr(cls, "__init__", None)
        params = {}
        if init_method:
            try:
                sign = inspect.signature(init_method)
                for param_name, param in sign.parameters.items():
                    if param_name in ("self", "args", "kwargs"):
                        continue
                    default_val = "Required" if param.default == inspect.Parameter.empty else param.default
                    params[param_name] = default_val
            except (ValueError, TypeError):
                pass
        
        result[name] = {
            "docstring": cls.__doc__,
            "parameters": params
        }
    return result


# --------------------------------------------------
# Section: Output Formatting Block
# --------------------------------------------------

def print_formatted_components(
    title: str, 
    key_label: str, 
    components_info: Dict[str, Dict[str, Any]]
) -> None:
    """
    Outputs strategy signatures and documentation to stdout in a unified visual style.

    :param title: Header string used during print sequence.
    :type title: str
    :param key_label: Contextual descriptor label pointing to the strategy type.
    :type key_label: str
    :param components_info: Pre-extracted dictionary with parameters and docstrings.
    :type components_info: Dict[str, Dict[str, Any]]
    """
    # Header Rendering
    print("\n" + "=" * 80)
    print(f" {title}")
    print("=" * 80)
    
    # Iterative component rendering
    for name, info in components_info.items():
        doc = info["docstring"]
        cleaned_doc = doc.strip() if doc else "No documentation provided."
        print(f"\n[{key_label}]: '{name}'")
        print(f" Description: {cleaned_doc}")
        print(" Parameters (kwargs):")
        if info["parameters"]:
            for p_name, p_default in info["parameters"].items():
                print(f"   - {p_name}: {p_default}")
        else:
            print("   - None")
    print("=" * 80 + "\n")