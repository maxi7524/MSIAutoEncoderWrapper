"""
Global independent validation utilities for the MSI AutoEncoder Wrapper ecosystem.
Provides signature reflection and basic filesystem permission verifications.
"""

import inspect
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Type

from .logger import get_custom_logger
from .exceptions import (
    raise_incompatible_interface_error,
    raise_project_config_error,
    raise_validation_error,
)


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
    a precise and informative project configuration error.

    :param cls: The target class type to inspect.
    :type cls: Type[Any]
    :param name: The registry identifier or alias string of the strategy.
    :type name: str
    :param kwargs: The dictionary of keyword arguments passed for instantiation.
    :type kwargs: Dict[str, Any]
    :raises ProjectConfigError: If required constructor arguments are missing.
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
        raise_project_config_error(context_name=name, message=error_message)


def validate_dir_writable(dir_path: Path) -> None:
    """
    Checks if a directory or its closest existing parent is writable by the system process.

    :param dir_path: Path to the target directory to validate.
    :type dir_path: Path
    :raises ValidationError: If the resolved path location is not writable.
    """
    # Path resolution block
    ## Find the closest existing parent directory if the path does not exist yet
    current_path = dir_path
    while not current_path.exists() and current_path.parent != current_path:
        current_path = current_path.parent

    # Permission verification block
    ## Test for write permissions using OS access levels
    if not os.access(current_path, os.W_OK):
        raise_validation_error(
            context_name="Filesystem",
            message=f"Target directory path or parent location is not writable: {dir_path}",
        )
    
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
        raise_validation_error(
            context_name="Components",
            message="Missing or invalid components: " + ", ".join(missing_items),
        )


def validate_subclass(
    candidate: Type[Any],
    expected_base: Type[Any],
    component_type: str,
) -> None:
    """Validate a registered implementation class against its base contract.

    :param candidate: Class submitted for registration.
    :type candidate: Type[Any]
    :param expected_base: Required base class.
    :type expected_base: Type[Any]
    :param component_type: Human-readable component category.
    :type component_type: str
    :raises IncompatibleInterfaceError: If ``candidate`` is not a class or does
        not inherit from ``expected_base``.
    """
    if not inspect.isclass(candidate) or not issubclass(candidate, expected_base):
        candidate_name = getattr(candidate, "__name__", type(candidate).__name__)
        raise_incompatible_interface_error(
            context_name=component_type,
            message=(
                f"Implementation '{candidate_name}' must inherit from "
                f"'{expected_base.__name__}'."
            ),
        )


def resolve_component(
    target: Any,
    registry: Dict[str, Type[Any]],
    component_type: str,
    expected_type: Optional[Type[Any]] = None,
    **kwargs: Any
) -> Any:
    """
    Resolves a component strategy either from an active instance, a registered name, or a raw factory class.

    :param target: Concrete instance, class type blueprint, or unique string registry identifier lookup key.
    :type target: Any
    :param registry: Reference targeting internal manager class driver mapping stores.
    :type registry: Dict[str, Type]
    :param component_type: Explanatory name of the managed pipeline component for logging.
    :type component_type: str
    :param expected_type: Optional base class required for classes and instances.
    :type expected_type: Optional[Type[Any]]
    :param kwargs: Arbitrary initialization parameters passed onto factory constructors.
    :return: Validated instantiated strategy engine type matching structural criteria.
    :rtype: Any
    """
    validate_component_target(
        target=target,
        registry=registry,
        component_type=component_type,
        expected_type=expected_type,
    )

    if isinstance(target, str):
        component_class = registry[target]
        validate_constructor_kwargs(component_class, target, kwargs)
        logger.debug("Instantiating component driver: %s from registry", target)
        return component_class(**kwargs)

    if inspect.isclass(target):
        validate_constructor_kwargs(target, target.__name__, kwargs)
        logger.debug("Instantiating component driver via raw class reference: %s", target.__name__)
        return target(**kwargs)

    logger.debug(
        "Direct compatible component instance verified for type '%s'. Bypassing factory initialization.",
        component_type,
    )
    return target


def validate_component_target(
    target: Any,
    registry: Dict[str, Type[Any]],
    component_type: str,
    expected_type: Optional[Type[Any]] = None,
) -> None:
    """Validate a registry key, implementation class, or ready instance.

    :param target: Registry key, implementation class, or initialized instance.
    :type target: Any
    :param registry: Registry used to resolve string targets.
    :type registry: Dict[str, Type[Any]]
    :param component_type: Human-readable component category.
    :type component_type: str
    :param expected_type: Optional base contract required for the target.
    :type expected_type: Optional[Type[Any]]
    :raises ProjectConfigError: If the target is missing or a string key is unknown.
    :raises IncompatibleInterfaceError: If a class or instance violates the base contract.
    """
    if target is None:
        raise_project_config_error(
            context_name=component_type,
            message=f"Target configuration reference for {component_type} cannot be None.",
        )

    if isinstance(target, str):
        if target not in registry:
            raise_project_config_error(
                context_name=component_type,
                message=(
                    f"Requested identifier '{target}' is not registered. "
                    f"Available implementations: {sorted(registry)}"
                ),
            )
        _validate_component_class(registry[target], expected_type, component_type)
        return

    if inspect.isclass(target):
        _validate_component_class(target, expected_type, component_type)
        return

    if expected_type is not None and not isinstance(target, expected_type):
        raise_incompatible_interface_error(
            context_name=component_type,
            message=(
                f"Instance of type '{type(target).__name__}' must inherit from "
                f"'{expected_type.__name__}'."
            ),
        )


def _validate_component_class(
    component_class: Type[Any],
    expected_type: Optional[Type[Any]],
    component_type: str,
) -> None:
    """Validate a resolved component class when a base contract is provided."""
    if expected_type is not None:
        validate_subclass(component_class, expected_type, component_type)
