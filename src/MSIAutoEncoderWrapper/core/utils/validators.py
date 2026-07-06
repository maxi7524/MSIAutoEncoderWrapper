"""
Validation utilities for checking the existence of runtime objects and directory permissions.
"""

import os
from pathlib import Path
from typing import List, Tuple, Any, Dict, Type
from .exceptions import ValidationError

def validate_components(items_to_validate: List[Tuple[Any, str]]) -> None:
    """
    Validates a list of components or file paths. If any are missing, accumulates
    all errors and raises a single ValidationError outlining everything that is missing.

    :param items_to_validate: A list of tuples containing the object/Path to check and its descriptive name.
    :type items_to_validate: List[Tuple[Any, str]]
    :return: None
    :rtype: None
    :raises ValidationError: If one or more items are None or a Path does not exist.
    """
    missing_items = []

    for item, name in items_to_validate:
        if item is None:
            missing_items.append(f"Instance Object '{name}' [Not Initialized]")
        elif isinstance(item, Path):
            if not item.exists():
                missing_items.append(f"File/Directory Path '{name}' -> ({item}) [Does Not Exist]")
        elif isinstance(item, str) and not item.strip():
            missing_items.append(f"Identifier '{name}' [Empty String]")

    if missing_items:
        raise ValidationError(missing_items)

def validate_dir_writable(dir_path: Path) -> None:
    """
    Checks if a directory is writable by the system process to prevent runtime crashes.

    :param dir_path: Path to the target directory to verify.
    :type dir_path: Path
    :return: None
    :rtype: None
    :raises PermissionError: If the directory structure cannot be written to.
    """
    if dir_path.exists():
        if not os.access(dir_path, os.W_OK):
            raise PermissionError(f"Directory '{dir_path}' is not writable.")
    else:
        for parent in dir_path.parents:
            if parent.exists():
                if not os.access(parent, os.W_OK):
                    raise PermissionError(f"Parent directory '{parent}' is not writable.")
                break

def resolve_component(
    target: Any, 
    registry: Dict[str, Type], 
    component_type: str,
    **kwargs: Any
) -> Any:
    """
    Resolves a component strategy either from an active instance or via registry lookup name.

    :param target: Concrete instantiated object or unique string registry identifier lookup key.
    :type target: Any
    :param registry: Reference targeting internal manager class driver mapping stores (_REGISTRY).
    :type registry: Dict[str, Type]
    :param component_type: Explanatory name of the managed pipeline component for logging.
    :type component_type: str
    :param kwargs: Arbitrary initialization key-value parameters passed onto factory constructors.
    :return: Validated instantiated strategy engine type matching structural criteria.
    :rtype: Any
    :raises ProjectConfigError: If string identifier is absent from registration registries.
    """
    from .exceptions import ProjectConfigError

    if isinstance(target, str):
        if target not in registry:
            raise ProjectConfigError(
                f"Requested {component_type} identifier '{target}' is not registered. "
                f"Available drivers: {list(registry.keys())}"
            )
        return registry[target](**kwargs)
    
    if target is None:
        raise ProjectConfigError(f"Target configuration reference for {component_type} cannot be None.")
        
    return target