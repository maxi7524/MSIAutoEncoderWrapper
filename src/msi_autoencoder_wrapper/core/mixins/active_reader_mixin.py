"""
Module defining the execution proxy and mixin for managing live binary pipeline sessions.
"""

from typing import Any, Optional
from ...utils.logger import get_custom_logger

# Logger initialization
logger = get_custom_logger(__name__)


# --------------------------------------------------
# Section: ActiveReaderProxy Operational Bridge
# --------------------------------------------------

class ActiveReaderProxy:
    """
    Command proxy coordinating runtime execution, lazy loading, and instance exposure for the active image.
    """

    def __init__(self, wrapper_ref: Any) -> None:
        """
        Initializes the active execution proxy bounded to the parent wrapper instance.

        :param wrapper_ref: Reference to the hosting main MSIAutoEncoderWrapper instance.
        :type wrapper_ref: Any
        """
        self._wrapper = wrapper_ref
        self._instantiated_image_key: Optional[str] = None
        
        # Operational runtime object caches
        self._cached_reader: Optional[Any] = None
        self._cached_binner: Optional[Any] = None
        self._cached_inverse_binner: Optional[Any] = None

    
# --------------------------------------------------
# Section: Dynamic Component Exposure Properties
# --------------------------------------------------

    @property
    def reader(self) -> Any:
        """
        Returns the fully initialized concrete MSIBaseReader instance for the active image context.

        Automatically triggers lazy context pipeline synchronization upon access.

        :return: Implemented data reader object instance.
        :rtype: Any
        """
        self._sync_active_pipeline()
        return self._cached_reader

    @property
    def binner(self) -> Any:
        """
        Returns the fully initialized concrete MSIBaseBinner instance for the active image context.

        Automatically triggers lazy context pipeline synchronization upon access.

        :return: Implemented forward spectrum binner object instance, or None if unconfigured.
        :rtype: Any
        """
        self._sync_active_pipeline()
        return self._cached_binner

    @property
    def inverse_binner(self) -> Any:
        """
        Returns the fully initialized concrete MSIBaseInverseBinner instance for the active image context.

        Automatically triggers lazy context pipeline synchronization upon access.

        :return: Implemented reverse reconstruction binner object instance, or None if unconfigured.
        :rtype: Any
        """
        self._sync_active_pipeline()
        return self._cached_inverse_binner

# --------------------------------------------------
# Section: Helpers
# --------------------------------------------------

    def clear_active_context(self) -> None:
        """
        Explicitly releases active binary components, closes file descriptors, and clears memory caches.
        """
        # Session release layer
        ## Evict cached reader and enforce explicit file descriptor closure if supported
        if self._cached_reader:
            logger.info("Discharging active binary reader session for image context: %s", self._instantiated_image_key)
            if hasattr(self._cached_reader, "close"):
                try:
                    self._cached_reader.close()
                except Exception:
                    logger.error("Failed to gracefully close active reader file handles during context cleanup.", exc_info=True)
        
        # Clear all state properties
        self._cached_reader = None
        self._cached_binner = None
        self._cached_inverse_binner = None
        self._instantiated_image_key = None

    def _sync_active_pipeline(self) -> None:
        """
        Synchronizes initialized binary objects against the current active image token in workspace.

        :raises ValueError: If no active image context has been set inside the workspace mixin.
        :raises KeyError: If the requested image context has no compiled reader strategy configured.
        """
        # Context tracking lookup execution
        ## Extract active state identifiers from the coupled workspace engine
        workspace = self._wrapper.workspace
        current_target = workspace.active_img_name

        if not current_target:
            logger.error("Synchronization blocked: Active image context is unassigned in workspace.")
            raise ValueError("Execution blocked: Active image context is unassigned in workspace. Execute set_active_image() first.")

        # Context change detection
        ## Evaluate if the cached memory pipeline matches the current workspace selection
        if self._instantiated_image_key != current_target:
            logger.info("Context change detected. Discharging old pipeline memory map to mount: %s", current_target)
            self.clear_active_context()

            ## Access configuration registries from the reader manager ledger container
            manager = self._wrapper.reader_manager
            if current_target not in manager.config_ledger or "reader" not in manager.config_ledger[current_target]:
                logger.error("Active reader mapping failed: No reader strategy configured for image context '%s'", current_target)
                raise KeyError(f"No reader strategy has been configured for image context '{current_target}'. Call set_reader() first.")

            # Lazy loading mounting sequence
            ## Mount raw instances directly from the synchronized config ledger data structures
            img_bucket = manager.config_ledger[current_target]
            self._cached_reader = img_bucket.get("reader")
            self._cached_binner = img_bucket.get("binner")
            self._cached_inverse_binner = img_bucket.get("inverse_binner")
            
            self._instantiated_image_key = current_target
            logger.info("Successfully synchronized active pipeline components for context: %s", current_target)



# --------------------------------------------------
# Section: ActiveReaderMixin Injection Hook
# --------------------------------------------------

class ActiveReaderMixin:
    """
    Mixin class injecting execution workspace proxy controls targeting active image dataset streaming.
    """

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        """
        Instantiates active pipeline routing interfaces attached to the execution wrapper context.
        """
        # Module instantiation hook
        ## Set active reader command bridge attribute reference proxy
        self.active_reader = ActiveReaderProxy(wrapper_ref=self)
        super().__init__(*args, **kwargs)