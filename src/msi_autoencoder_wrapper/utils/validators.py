"""
Core validation utilities for the MSI AutoEncoder Wrapper ecosystem.
Provides reflection-based constructor signature inspection.
"""

import inspect
from typing import Type, Any, Dict, List

def validate_constructor_kwargs(cls: Type[Any], name: str, kwargs: Dict[str, Any]) -> None:
    """
    Validates whether the provided kwargs satisfy the required arguments of a class constructor.
    
    Inspects the __init__ signature of the target class. If any required arguments 
    (those without default values) are missing from the kwargs dictionary, it raises 
    a precise and informative ValueError.
    
    :param cls: The target class type to inspect.
    :type cls: Type[Any]
    :param name: The registry identifier or alias string of the strategy.
    :type name: str
    :param kwargs: The dictionary of keyword arguments passed for instantiation.
    :type kwargs: Dict[str, Any]
    :raises ValueError: If one or more required constructor arguments are missing.
    """
    init_method = getattr(cls, "__init__", None)
    if not init_method:
        return

    try:
        sign = inspect.signature(init_method)
    except (ValueError, TypeError):
        return

    missing_args: List[str] = []
    
    for param_name, param in sign.parameters.items():
        if param_name in ("self", "args", "kwargs"):
            continue
            
        if param.default == inspect.Parameter.empty and param.kind in (
            inspect.Parameter.POSITIONAL_OR_KEYWORD,
            inspect.Parameter.KEYWORD_ONLY
        ):
            if param_name not in kwargs:
                missing_args.append(param_name)
                
    if missing_args:
        raise ValueError(
            f"Cannot initialize configuration for '{name}' using class '{cls.__name__}'. "
            f"Missing required keyword argument(s): {', '.join([repr(a) for a in missing_args])}. "
            f"Provided arguments: {list(kwargs.keys())}."
        )