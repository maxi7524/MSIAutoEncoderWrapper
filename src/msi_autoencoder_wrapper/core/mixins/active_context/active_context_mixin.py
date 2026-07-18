# Heading 1 (Active Context Management Implementation)
## Core component executing dynamic, lazy-loaded pipeline routing for Mass Spectrometry Imaging

from __future__ import annotations
from typing import Any, Optional, TYPE_CHECKING
import torch

# Base autoencoder operational interfaces and exceptions
from .autoencoder_context_manager import AutoencoderContextInterface
from .latent_context_mixin import LatentContextMixin
from ....utils.logger import get_custom_logger
from ....utils.exceptions import raise_validation_error

if TYPE_CHECKING:
    pass

# Logger initialization
logger = get_custom_logger(__name__)


class ActiveContextProxy(LatentContextMixin):
    """
    Stateful boundary proxy representing the currently selected image execution context.
    Provides direct, lazy-loaded access to memory-resident pipelines (Readers, Binners, Autoencoders).
    """

    def __init__(self, wrapper_ref: Any) -> None:
        """
        Initializes the active execution proxy bounded to the parent wrapper instance.

        :param wrapper_ref: Reference to the hosting main wrapper facade instance.
        :type wrapper_ref: Any
        """
        self._wrapper = wrapper_ref
        self._instantiated_image_key: Optional[str] = None
        
        # Operational runtime object caches
        self._cached_reader: Optional[Any] = None
        self._cached_binner: Optional[Any] = None
        self._cached_inverse_binner: Optional[Any] = None
        self._initialize_latent_context()

    # --------------------------------------------------
    # Section: Lazy Pipeline Resolution
    # --------------------------------------------------

    def _resolve_active_pipeline(self) -> None:
        """
        Resolves the active pipelines from the context manager config ledger.
        Loads drivers and local models dynamically based on the wrapper's current image key.
        """
        # Resolve target key
        ## Retrieve the currently selected image key from the wrapper or workspace fallback
        current_target = getattr(self._wrapper, "active_image_key", None)
        if current_target is None:
            current_target = getattr(self._wrapper.workspace, "active_img_name", None)

        # Automated workspace fallback
        ## If no active image key is found, attempt to automatically trigger the workspace helper fallback
        if current_target is None and hasattr(self._wrapper, "workspace"):
            workspace_proxy = self._wrapper.workspace
            default_img = getattr(workspace_proxy, "default_img_name", None)
            
            if default_img is not None:
                logger.info(
                    "No active image set. Automatically resolving default workspace context: %s", 
                    default_img
                )
                ### Trigger set_active_image(None) which internally falls back to the default image name
                workspace_proxy.set_active_image(None)
                current_target = getattr(workspace_proxy, "active_img_name", None)

        if current_target is None:
            logger.error("Active pipeline resolution failed: No active image key context has been set.")
            raise_validation_error(
                context_name="ActiveContext",
                message="Cannot resolve active pipelines: No image context has been set. Call set_active_image() first."
            )

        # Config ledger resolution pass
        ## Fetch registered configuration structures from the centralized ledger manager
        manager = getattr(self._wrapper, "context_manager", None)
        if manager is None:
            raise_validation_error(
                context_name="ActiveContext",
                message="ContextManager mixin is missing from the wrapper hierarchy."
            )

        if current_target in manager.config_ledger:
            ## Ledger mapping and caching
            ### Load reader, binner, and optional local model instances from ledger registry
            img_bucket = manager.config_ledger[current_target]
            self._cached_reader = img_bucket.get("reader")
            self._cached_binner = img_bucket.get("binner")
            self._cached_inverse_binner = img_bucket.get("inverse_binner")
        else:
            logger.error("Active reader mapping failed: No reader configuration has been recorded for image context '%s'", current_target)
            raise_validation_error(
                context_name="ActiveContext",
                message=f"No strategy has been configured for image context '{current_target}'. Call set_reader() first."
            )

        self._instantiated_image_key = current_target
        logger.info("Successfully bound active context memory maps for: %s", current_target)

    # --------------------------------------------------
    # Section: Properties and Getters
    # --------------------------------------------------

    @property
    def reader(self) -> Any:
        """
        Lazy-loaded property returning the active data reader instance.
        """
        if self._cached_reader is None:
            self._resolve_active_pipeline()
        return self._cached_reader

    @property
    def binner(self) -> Any:
        """
        Lazy-loaded property returning the active spectrum binner instance.
        """
        if self._cached_binner is None:
            self._resolve_active_pipeline()
        return self._cached_binner

    @property
    def inverse_binner(self) -> Any:
        """
        Lazy-loaded property returning the active spectrum inverse binner instance.
        """
        if self._cached_inverse_binner is None:
            self._resolve_active_pipeline()
        return self._cached_inverse_binner

    @property
    def autoencoder(self) -> Optional[AutoencoderContextInterface]:
        """
        Exposes direct access to high-level autoencoder transformation methods with full type hint support.

        :return: Autoencoder wrapper operational interface, or None if another model family is active.
        :rtype: Optional[AutoencoderContextInterface]
        """
        models_manager = getattr(self._wrapper, "models_manager", None)
        return models_manager.autoencoder if models_manager is not None else None

    # --------------------------------------------------
    # Section: Model Capturing and Attachment
    # --------------------------------------------------

    def attach_local_model(self, torch_model: torch.nn.Module, model_type: str) -> None:
        """
        Intercepts a compiled network module, maps its operational strategy, and deploys it onto the active session.

        :param torch_model: Initialized PyTorch graph module object instance.
        :type torch_model: torch.nn.Module
        :param model_type: Identity token family string defining the model class layout rules.
        :type model_type: str
        """
        self._wrapper.models_manager.attach_model(
            torch_model=torch_model,
            model_type=model_type,
        )

    # --------------------------------------------------
    # Section: Context Teardown and Cleanup
    # --------------------------------------------------

    def clear_active_context(self) -> None:
        """
        Resets all working context session variables and active pipeline objects back to None.
        """
        logger.debug("ActiveContextProxy: Clearing active cached reader, binner and autoencoder interfaces.")
        self._instantiated_image_key = None
        self._cached_reader = None
        self._cached_binner = None
        self._cached_inverse_binner = None
        self.unload_latent()


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
