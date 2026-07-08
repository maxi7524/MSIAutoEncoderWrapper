"""
Module providing foundational lifecycle and context execution decorators for the MSI library.
"""

import functools
import inspect
from typing import Any, Callable, Optional
from ...utils.logger import get_custom_logger

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
            logger.error("Context synchronization aborted: Bound workspace proxy is missing from instance.")
            raise AttributeError("Execution blocked: Target instance does not possess an active workspace proxy.")

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
            logger.error("Context resolution failed: Workspace state remained unassigned after resolution attempt.")
            raise ValueError("Execution blocked: Missing required image context parameters, and no default is configured.")

        # Protected operational execution block
        try:
            logger.debug("Executing method '%s' under verified context: %s", func.__name__, workspace.active_img_name)
            return func(instance, *args, **kwargs)
        except Exception as error:
            ### Capture runtime system exceptions to initiate emergency pipeline state cleanup
            logger.error("Exception intercepted in '%s'. Triggering active reader context reset.", func.__name__, exc_info=True)
            
            ### Access active lazy-loading drivers to clear cached filesystem file handles
            active_reader = getattr(wrapper_ref, "active_reader", None)
            if active_reader:
                if hasattr(active_reader, "clear_active_context"):
                    active_reader.clear_active_context()
                elif hasattr(active_reader, "release_active_reader"):
                    active_reader.release_active_reader()
            
            ### Propagate the original error up the processing execution tree
            raise error

    return wrapper