"""
Validation utilities for checking the existence of runtime objects and directory permissions.
"""

import os
from pathlib import Path
from typing import List, Tuple, Any
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