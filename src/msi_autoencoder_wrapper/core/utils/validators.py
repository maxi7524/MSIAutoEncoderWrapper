"""
Mixin and Core system validators for the MSI library.
"""

import os
from pathlib import Path
from typing import Any, Dict, Type, Union, TYPE_CHECKING
from ...utils.logger import get_custom_logger
from ...utils.exceptions import raise_validation_error

if TYPE_CHECKING:
    from ..mixins.workspace.workspace_manager_mixin import WorkspaceMixin

# Logger initialization
logger = get_custom_logger(__name__)


# =====================================================================
# Section: Context and State Validators
# =====================================================================


def validate_active_context(wrapper: Any) -> None:
    """
    Verifies that the wrapper has a currently selected and initialized active image context.

    :param wrapper: The master facade wrapper instance.
    :type wrapper: Any
    :raises ValidationError: If active_context or active_image_key is missing.
    """
    # State verification checks
    ## Retrieve active context proxy layer references
    active_ctx = getattr(wrapper, "active_context", None)
    if active_ctx is None:
        raise_validation_error(
            context_name="ActiveContext",
            message="ActiveContextProxy layer is not mounted on the wrapper."
        )

    ## Retrieve active image key representation from the runtime storage
    active_key = getattr(active_ctx, "_instantiated_image_key", None)
    if active_key is None:
        raise_validation_error(
            context_name="ActiveContext",
            message="No active image context has been set. Execute set_reader() first to establish context."
        )
        
    logger.debug("Active context validated successfully for image key: %s", active_key)


def validate_active_model(wrapper: Any) -> None:
    """
    Validates that a PyTorch model has been actively mounted and compiled within the wrapper lifecycle.

    :param wrapper: The master facade wrapper instance.
    :type wrapper: Any
    :raises ValidationError: If no model is actively mounted.
    """
    # Model configuration check
    ## Extract compiled model attributes
    active_model = getattr(wrapper, "active_model", None)
    if active_model is None:
        raise_validation_error(
            context_name="ModelManager",
            message="No active model is currently mounted or built. Use models_manager to build or load a model first."
        )

    logger.debug("Active model instance verified successfully.")


# =====================================================================
# Section: Workspace Path and File System Validators
# =====================================================================


def validate_workspace_path(path: Union[str, Path]) -> Path:
    """
    Validates that a workspace project root directory exists and is writable.

    :param path: Path to the proposed project root directory.
    :type path: Union[str, Path]
    :return: Resolved absolute path.
    :rtype: Path
    :raises ValidationError: If path does not exist, is not a directory, or has insufficient permissions.
    """
    # File System Resolution Check
    ## Convert input string/Path to an absolute path reference
    resolved_path = Path(path).resolve()
    
    if not resolved_path.exists():
        raise_validation_error(
            context_name="Workspace",
            message=f"Provided project path does not exist: {resolved_path}"
        )
        
    if not resolved_path.is_dir():
        raise_validation_error(
            context_name="Workspace",
            message=f"Provided project path is not a directory: {resolved_path}"
        )
        
    ## Verify basic write access privileges on the resolved directory
    if not os.access(resolved_path, os.W_OK):
        raise_validation_error(
            context_name="Workspace",
            message=f"Insufficient write permissions on directory: {resolved_path}"
        )
        
    logger.debug("Workspace project root directory validated: %s", resolved_path)
    return resolved_path