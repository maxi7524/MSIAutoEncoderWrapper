"""
Module providing foundational lifecycle and context execution decorators for the MSI library.
"""

import functools
import inspect
from typing import Any, Callable, Optional
from ...utils.logger import get_custom_logger
from ...utils.exceptions import raise_validation_error

# Logger initialization
logger = get_custom_logger(__name__)


# Context execution management
## Define an isolation decorator to synchronize and protect the active image context
def manage_image_context(func: Callable[..., Any]) -> Callable[..., Any]:
    """
    Decorator to dynamically extract image arguments and manage the active workspace context lifecycle.

    Inspects method signatures to locate the target image parameter, routes execution to the 
    underlying workspace state manager, and guarantees context clearing if a downstream execution failure occurs.

    :param func: The bound method requiring synchronization with an active image dataset.
    :type func: Callable[..., Any]
    :return: The protected execution wrapper closure.
    :rtype: Callable[..., Any]
    :raises ValueError: If no valid image context can be established by the workspace engine.
    """
    @functools.wraps(func)
    def wrapper(instance: Any, *args: Any, **kwargs: Any) -> Any:
        # Context reference extraction
        ## Extract the root facade object and locate the bound workspace proxy component
        wrapper_ref = getattr(instance, "_wrapper", instance)
        workspace = getattr(wrapper_ref, "workspace", None)
        
        if not workspace:
            raise_validation_error(
                context_name="ImageContext",
                message="The target instance does not expose a workspace proxy.",
            )

        # Signature inspection and argument extraction
        ## Bind incoming positional and keyword arguments to locate the image parameter token
        sig = inspect.signature(func)
        bound_args = sig.bind(instance, *args, **kwargs)
        bound_args.apply_defaults()
        
        img_name_or_path: Optional[Any] = bound_args.arguments.get("img_name_or_path")

        # State transition delegation
        ## Route the extracted argument directly to the specialized workspace context mechanism
        logger.debug("Delegating context transition to workspace manager with parameter: %s", img_name_or_path)
        workspace.set_active_image(str(img_name_or_path) if img_name_or_path is not None else None)

        # Context presence validation
        ## Verify if a valid context was successfully established or if it was left unassigned
        if getattr(workspace, "active_img_name", None) is None:
            ### Abort execution if both the explicit parameter and workspace defaults are missing
            raise_validation_error(
                context_name="ImageContext",
                message="No image was provided and no default image is configured.",
            )

        # Protected operational execution block
        try:
            logger.debug("Executing method '%s' under verified context: %s", func.__name__, workspace.active_img_name)
            return func(instance, *args, **kwargs)
        except Exception:
            ### Capture runtime system exceptions to initiate emergency pipeline state cleanup
            logger.error("Exception intercepted in '%s'. Triggering active reader context reset.", func.__name__, exc_info=True)
            
            ### Access active lazy-loading drivers to clear cached filesystem file handles
            active_context = getattr(wrapper_ref, "active_context", None)
            if active_context:
                if hasattr(active_context, "clear_active_context"):
                    active_context.clear_active_context()
            raise
        finally:
            ### Clean up temporary workspace structural states to prevent context leaks
            if hasattr(workspace, "clear_active_context"):
                logger.debug("Operation finalized. Discharging temporary workspace image context.")
                workspace.clear_active_context()

    return wrapper
