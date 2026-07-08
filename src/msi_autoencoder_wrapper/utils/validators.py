"""
Global independent validation utilities for the MSI AutoEncoder Wrapper ecosystem.
Provides signature reflection and basic filesystem permission verifications.
"""

import inspect
import os
from pathlib import Path
from typing import List, Tuple, Any, Dict, Type
from .logger import get_custom_logger
from .exceptions import ValidationError, ProjectConfigError

from pathlib import Path


# Logger initialization
logger = get_custom_logger(__name__)


# =====================================================================
# Section: System & File Validators
# =====================================================================


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
    # Signature inspection block
    ## Extract the __init__ method reference from the class object
    init_method = getattr(cls, "__init__", None)
    if not init_method:
        return

    try:
        ## Parse parameter signatures using reflection
        sign = inspect.signature(init_method)
    except (ValueError, TypeError):
        logger.error("Failed to extract signature for class: %s", cls.__name__, exc_info=True)
        return

    missing_args: List[str] = []

    # Parameter evaluation loop
    ## Iterate over all formal parameters of the constructor signature
    for param_name, param in sign.parameters.items():
        if param_name in ("self", "args", "kwargs"):
            continue

        if param.default == inspect.Parameter.empty and param.kind in (
            inspect.Parameter.POSITIONAL_OR_KEYWORD,
            inspect.Parameter.KEYWORD_ONLY
        ):
            ### Check if mandatory argument is missing in provided configurations
            if param_name not in kwargs:
                missing_args.append(param_name)

    if missing_args:
        error_message = (
            f"Cannot initialize configuration for '{name}' using class '{cls.__name__}'. "
            f"Missing required keyword argument(s): {', '.join([repr(a) for a in missing_args])}. "
            f"Provided arguments: {list(kwargs.keys())}."
        )
        logger.error("Validation failed for constructor setup of: %s", name)
        raise ValueError(error_message)


def validate_dir_writable(dir_path: Path) -> None:
    """
    Checks if a directory or its closest existing parent is writable by the system process.

    :param dir_path: Path to the target directory to validate.
    :type dir_path: Path
    :raises IOError: If the resolved path location is not writable.
    """
    # Path resolution block
    ## Find the closest existing parent directory if the path does not exist yet
    current_path = dir_path
    while not current_path.exists() and current_path.parent != current_path:
        current_path = current_path.parent

    # Permission verification block
    ## Test for write permissions using OS access levels
    if not os.access(current_path, os.W_OK):
        error_msg = f"Target directory path or parent location is not writable: {dir_path}"
        logger.error("Write permission verification failed for path: %s", dir_path)
        raise IOError(error_msg)
    
# =====================================================================
# Section: Modules components Validators
# =====================================================================


def validate_components(items_to_validate: List[Tuple[Any, str]]) -> None:
    """
    Validates a list of components or file paths.

    If any are missing, accumulates all errors and raises a single ValidationError
    outlining everything that is missing.

    :param items_to_validate: A list of tuples containing the object/Path to check and its descriptive name.
    :type items_to_validate: List[Tuple[Any, str]]
    :raises ValidationError: If one or more items are None, empty, or paths do not exist.
    """
    missing_items: List[str] = []

    # Validation inspection loop
    ## Scan through each individual target item and evaluate its structural completeness
    for item, name in items_to_validate:
        if item is None:
            ### Track uninitialized object references
            missing_items.append(f"Instance Object '{name}' [Not Initialized]")
        elif isinstance(item, Path):
            ### Verify path presence in local file system container
            if not item.exists():
                missing_items.append(f"File/Directory Path '{name}' -> ({item}) [Does Not Exist]")
        elif isinstance(item, str) and not item.strip():
            ### Detect blank or empty string identifier tokens
            missing_items.append(f"Identifier '{name}' [Empty String]")

    if missing_items:
        logger.error("Core component validation failed. Missing components count: %s", len(missing_items))
        raise ValidationError(missing_items)


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
    :param registry: Reference targeting internal manager class driver mapping stores.
    :type registry: Dict[str, Type]
    :param component_type: Explanatory name of the managed pipeline component for logging.
    :type component_type: str
    :param kwargs: Arbitrary initialization key-value parameters passed onto factory constructors.
    :return: Validated instantiated strategy engine type matching structural criteria.
    :rtype: Any
    :raises ProjectConfigError: If string identifier is absent from registration registries.
    """
    # Strategic component resolution pipeline
    ## Check if the input target reference is a string key lookup
    if isinstance(target, str):
        ### Validate entry presence within structural class database cache
        if target not in registry:
            error_msg = (
                f"Requested {component_type} identifier '{target}' is not registered. "
                f"Available drivers: {list(registry.keys())}"
            )
            logger.error("Registry lookup failed for type '%s' with key: %s", component_type, target)
            raise ProjectConfigError(error_msg)
        
        logger.debug("Instantiating component driver: %s from registry", target)
        return registry[target](**kwargs)
    
    ## Validate that the direct instance target is not null
    if target is None:
        logger.error("Component resolution aborted: target blueprint for %s is None", component_type)
        raise ProjectConfigError(f"Target configuration reference for {component_type} cannot be None.")
        
    return target