"""
Module defining the execution proxy and mixin for managing live binary pipeline sessions.
"""

from typing import Any, Optional

from sympy import Q
from ....utils.logger import get_custom_logger

from ....readers.base_reader import MSIBaseReader
from ....binners.base_binner import MSIBaseBinner
from ....binners.base_inverse import MSIBaseInverseBinner
from ..models.autoencoder_context_manager import AutoencoderContextInterface

# Logger initialization
logger = get_custom_logger(__name__)


# --------------------------------------------------
# Section: ActiveContextProxy Operational Bridge
# --------------------------------------------------

class ActiveContextProxy:
    """
    Stateful boundary proxy representing the currently selected image execution context.
    Provides direct, lazy-loaded access to memory-resident pipelines (Readers, Binners, Latent spaces).
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
        #TODO(future) - in future we should add here different _cached valeus as latent space etc. 

    
# --------------------------------------------------
# Section: Dynamic Component Exposure Properties
# --------------------------------------------------

    @property
    def reader(self) -> MSIBaseReader:
        """
        Returns the fully initialized concrete MSIBaseReader instance for the active image context.

        Automatically triggers lazy context pipeline synchronization upon access.

        :return: Implemented data reader object instance.
        :rtype: Any
        """
        self._sync_active_context()
        return self._cached_reader

    @property
    def binner(self) -> Optional[MSIBaseBinner]:
        """
        Returns the fully initialized concrete MSIBaseBinner instance for the active image context.

        Automatically triggers lazy context pipeline synchronization upon access.

        :return: Implemented forward spectrum binner object instance, or None if unconfigured.
        :rtype: Any
        """
        self._sync_active_context()
        return self._cached_binner

    @property
    def inverse_binner(self) -> Optional[MSIBaseInverseBinner]:
        """
        Returns the fully initialized concrete MSIBaseInverseBinner instance for the active image context.

        Automatically triggers lazy context pipeline synchronization upon access.

        :return: Implemented reverse reconstruction binner object instance, or None if unconfigured.
        :rtype: Any
        """
        self._sync_active_context()
        return self._cached_inverse_binner

    @property
    def autoencoder(self) -> Optional[AutoencoderContextInterface]:
        """
        #TODO
        """
        self._sync_active_context()
        return self.autoencoder

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

    def _sync_active_context(self) -> None:
        """
        Synchronizes memory-resident pipeline objects against the active image token defined in the workspace.
        Automatically triggers a fallback transaction to the default workspace image configuration if no active context is set.

        :raises ValueError: If both active image and default workspace configurations are completely unassigned.
        :raises KeyError: If the target image context has no compiled reader strategy configured in the ledger.
        """
        # Context tracking lookup execution
        ## Extract active state identifiers from the coupled workspace engine
        workspace = self._wrapper.workspace
        current_target = workspace.active_img_name

        # Automated default context fallback logic
        ## If active image is blank, check if we can transparently boot the default session configuration
        if not current_target:
            if getattr(workspace, "default_img_name", None):
                logger.info("Active context is unassigned. Performing lazy automatic fallback activation using default image: %s", workspace.default_img_name)
                workspace.set_active_image(None)  # Triggers default image setup and synchronization cascades
                current_target = workspace.active_img_name
            else:
                logger.error("Synchronization blocked: Active image context and default configuration are both unassigned.")
                raise ValueError("Execution blocked: Active image context is unassigned. Execute set_active_image() or set_default_path() first.")

        # Lazy synchronization block
        if self._instantiated_image_key != current_target:
            logger.info("Context transition detected. Synchronizing active memory structures to: %s", current_target)
            self.clear_active_context()

            manager = self._wrapper.context_manager
            if current_target in manager.config_ledger:
                img_bucket = manager.config_ledger[current_target]
                self._cached_reader = img_bucket.get("reader")
                self._cached_binner = img_bucket.get("binner")
                self._cached_inverse_binner = img_bucket.get("inverse_binner")
            else:
                logger.error("Active reader mapping failed: No reader configuration has been recorded for image context '%s'", current_target)
                raise KeyError(f"No strategy has been configured for image context '{current_target}'. Call set_reader() first.")
            
            self._instantiated_image_key = current_target
            logger.info("Successfully bound active context memory maps for: %s", current_target)



# --------------------------------------------------
# Section: ActiveReaderMixin Injection Hook
# --------------------------------------------------

class ActiveContextMixin:
    """
    Mixin class injecting execution workspace proxy controls targeting active image dataset streaming.
    """

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        """
        Instantiates active pipeline routing interfaces attached to the execution wrapper context.
        """
        # Module instantiation hook
        ## Set active reader command bridge attribute reference proxy
        self.active_context = ActiveContextProxy(wrapper_ref=self)
        super().__init__(*args, **kwargs)