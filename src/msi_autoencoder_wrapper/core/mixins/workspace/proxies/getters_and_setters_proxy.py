# Heading 1 (Getters and Setters Proxy Implementation)
## Core component resolving folder absolute paths and setting active context states

from __future__ import annotations
from pathlib import Path
from typing import Any, Optional, List, Union

from .base_workspace_proxy import BaseWorkspaceProxy
from .....utils.exceptions import raise_workspace_error
from .....utils.logger import get_custom_logger

# Logger initialization
logger = get_custom_logger(__name__)

# =====================================================================
# Section: Module Constants
# =====================================================================
# Central configuration token for identifying global model contexts
GLOBAL_CONTEXT_KEY = "global"


class GettersAndSettersProxy(BaseWorkspaceProxy):
    """
    Proxy component responsible for path resolution and image/model state orchestration.
    """

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        """
        Initializes the getters and setters workspace proxy.
        """
        super().__init__(*args, **kwargs)

# --------------------------------------------------
# Section: Setters
# --------------------------------------------------

    # --------------------------------------------------
    # Subsection: Single image setters
    # --------------------------------------------------

    def set_active_image(self, img_name_or_path: Optional[str] = None) -> None:
        """
        Sets the active image context, resolving raw names or direct custom paths.

        :param img_name_or_path: Name of the target image or its direct file path location.
        :type img_name_or_path: Optional[str]
        """
        # Context evaluation block
        ## Handle empty input arguments by attempting default fallback transitions
        if img_name_or_path is None:
            if getattr(self, "default_img_name", None):
                logger.info("No image name provided. Falling back to default workspace image configuration.")
                self.active_img_name = self.default_img_name
                self._active_img_custom_path = getattr(self, "_default_img_custom_path", None)
            else:
                logger.info("Active image configuration cleared.")
                self.active_img_name = None
                self._active_img_custom_path = None
            return

        target_path = Path(img_name_or_path)
        
        ## Heading 2 (Path resolution logic branch)
        ### Check if the provided argument points directly to an existing absolute/relative file or has image extension
        if target_path.suffix in [".imzML", ".ibd"] or target_path.is_absolute() or target_path.exists():
            self.active_img_name = target_path.stem
            self._active_img_custom_path = target_path.parent.resolve()
            logger.info("Active image set via direct filesystem path: %s (Location: %s)", 
                        self.active_img_name, self._active_img_custom_path)
        else:
            ### Fall back to treating the argument as a simple image key index inside workspace
            self.active_img_name = img_name_or_path
            self._active_img_custom_path = None
            logger.info("Active image context mapped by index key: %s", self.active_img_name)

    def set_default_image(self, img_name_or_path: str) -> None:
        """
        Defines the default fallback image configuration to prevent repetitive setup calls.

        :param img_name_or_path: Name of the default image or its file path.
        :type img_name_or_path: str
        """
        # Default initialization block
        target_path = Path(img_name_or_path)
        if target_path.suffix in [".imzML", ".ibd"] or target_path.is_absolute() or target_path.exists():
            self.default_img_name = target_path.stem
            self._default_img_custom_path = target_path.parent.resolve()
        else:
            self.default_img_name = img_name_or_path
            self._default_img_custom_path = None
        
        logger.info("Default fallback image anchored: %s", self.default_img_name)

    # --------------------------------------------------
    # Subsection: Multiple image setters
    # --------------------------------------------------

    def set_active_images(self, img_names: List[str]) -> None:
        """
        Sets a batch collection of active image keys representing a multi-image setup.

        :param img_names: List of target image name keys or paths.
        :type img_names: List[str]
        """
        # Batch cache allocation
        self.active_img_names = [Path(name).stem for name in img_names]
        logger.info("Active multi-image batch registry updated with %s keys.", len(self.active_img_names))

    # --------------------------------------------------
    # Subsection: Model setters
    # --------------------------------------------------

    def set_active_model(self, model_name: str) -> None:
        """
        Registers the target machine learning model to be used within the active scope.

        :param model_name: Name of the target model architecture instance.
        :type model_name: str
        """
        # Active model synchronization
        self.active_model_name = model_name
        logger.info("Workspace active model focus changed to: %s", model_name)


# --------------------------------------------------
# Section: Getters
# --------------------------------------------------

    # --------------------------------------------------
    # Subsection: Path resolving helpers
    # --------------------------------------------------

    def _build_absolute_path(self, img_name: Optional[str], custom_path: Optional[Path], suffix: str = ".imzML") -> Optional[Path]:
        """
        Internal utility compiling absolute paths for datasets, properly handling external custom 
        paths to prevent directory configuration conflicts.

        :param img_name: Resolved target image name.
        :type img_name: Optional[str]
        :param custom_path: External directory path targeting raw data, or None.
        :type custom_path: Optional[Path]
        :param suffix: Expected dataset file format extension.
        :type suffix: str
        :return: Compiled absolute path, or None if context parameters are missing.
        :rtype: Optional[Path]
        """
        if not img_name:
            return None

        # Path compilation block
        ## If a custom path is registered, we must resolve the image location externally (outside project_path)
        if custom_path:
            logger.debug("Resolving external dataset absolute path for: %s inside %s", img_name, custom_path)
            return (custom_path / img_name).with_suffix(suffix)
        
        ## Otherwise, build path targeting the internal default workspace structure
        logger.debug("Resolving internal workspace dataset path for: %s", img_name)
        return self.project_path_resolved / self._layout["imgs"] / f"{img_name}{suffix}"
    
    ## Helper 1: Resolve incoming path or identifier token
    def _resolve_incoming_path(self, img_name_or_path: Optional[Union[str, Path]]) -> Tuple[Optional[str], Optional[Path]]:
        """
        Analyzes an incoming image descriptor to extract its stem name and custom base path context.

        Discriminates between local workspace keys and absolute/external file system paths.
        If the path resides inside the workspace root, it strips the absolute context to maintain local layout integrity.

        :param img_name_or_path: Raw text key, relative path, or absolute file system path.
        :type img_name_or_path: Optional[Union[str, Path]]
        :return: A tuple containing the resolved image stem name and the custom base directory path (if external).
        :rtype: Tuple[Optional[str], Optional[Path]]
        """
        if img_name_or_path is None:
            return None, None

        resolved_path = Path(img_name_or_path)
        img_name = resolved_path.stem
        custom_path: Optional[Path] = None

        # Path discrimination logic
        ## Evaluate if the parameter represents a complex file system location
        if resolved_path.is_absolute() or len(resolved_path.parts) > 1 or resolved_path.exists():
            ### Determine layout association relative to the project root directory
            if hasattr(self, "is_path_in_workspace") and self.is_path_in_workspace(resolved_path):
                logger.debug("Resolved internal path for image token: %s", img_name)
                custom_path = None
            else:
                logger.debug("Resolved external path mapping for image token: %s", img_name)
                custom_path = resolved_path.parent / resolved_path.stem
        else:
            logger.debug("Resolved standalone workspace identifier key: %s", img_name)
            custom_path = None

        return img_name, custom_path

    # --------------------------------------------------
    # Subsection: Single image getters
    # --------------------------------------------------

    def get_active_image_path(self, suffix: str = ".imzML") -> Optional[Path]:
        """
        Calculates the absolute file path targeting the currently active image context file.

        :param suffix: Dataset file format extension. Defaults to ".imzML".
        :type suffix: str
        :return: Absolute Path object representing the dataset file location, or None.
        :rtype: Optional[Path]
        """
        # Dynamic property reflection check
        ## Access local cache first, fallback dynamically to the active wrapper context to avoid collision
        img_target = self.active_img_name or self.active_image_key
        return self._build_absolute_path(img_target, self._active_img_custom_path, suffix)

    def get_default_image_path(self, suffix: str = ".imzML") -> Optional[Path]:
        """
        Calculates the absolute file path targeting the fallback default image context file.

        :param suffix: Dataset file format extension. Defaults to ".imzML".
        :type suffix: str
        :return: Absolute Path object representing the dataset file location, or None.
        :rtype: Optional[Path]
        """
        return self._build_absolute_path(self.default_img_name, self._default_img_custom_path, suffix)

    # --------------------------------------------------
    # Subsection: Model path getters
    # --------------------------------------------------

    def get_models_root(self) -> Path:
        """
        Calculates the absolute root directory holding all model representations.

        :return: Absolute path to the main models folder.
        :rtype: Path
        """
        return self.project_path_resolved / self._layout["models"]

    def get_model_dir(self, img_name: str, model_name: str) -> Path:
        """
        Calculates the path to a specific model workspace directory, enforcing the new structure:
        models/<img_name>/<model_name> or models/global/<model_name>.

        :param img_name: Name of the image context ("global" for global models).
        :type img_name: str
        :param model_name: Name of the target model architecture instance.
        :type model_name: str
        :return: Path to the specific model directory.
        :rtype: Path
        """
        # Directory resolution switch
        ## Differentiate between local context models and centralized global models using the module constant
        if img_name.lower() == GLOBAL_CONTEXT_KEY:
            return self.get_models_root() / GLOBAL_CONTEXT_KEY / model_name
        
        return self.get_models_root() / img_name / model_name

    def get_config_dir(self, img_name: str, model_name: str) -> Path:
        """
        Calculates the path to the configuration directory associated with the target model.

        :param img_name: Name of the image context ("global" for global models).
        :type img_name: str
        :param model_name: Name of the target model architecture instance.
        :type model_name: str
        :return: Path to the nested configuration directory.
        :rtype: Path
        """
        # Path compilation with structural dictionary key
        return self.get_model_dir(img_name=img_name, model_name=model_name) / self._layout["config"]

    def get_latent_dir(self, img_name: str, model_name: str) -> Path:
        """
        Calculates the path to the latent space storage directory associated with the target model.

        :param img_name: Name of the image context ("global" for global models).
        :type img_name: str
        :param model_name: Name of the target model architecture instance.
        :type model_name: str
        :return: Path to the nested latent space representation directory.
        :rtype: Path
        """
        # Path compilation with structural dictionary key
        return self.get_model_dir(img_name=img_name, model_name=model_name) / self._layout["latent"]

    def get_active_model_dir(self) -> Path:
        """
        Calculates the directory path targeting the currently active model context.

        :return: Path to the currently active model.
        :rtype: Path
        :raises WorkspaceConfigError: If no model focus has been specified.
        """
        # Active model status verification
        ## Retrieve model name dynamically through wrapper reflection fallback
        model_target = self.active_model_name or self.active_model_name_global
        img_target = self.active_img_name or self.active_image_key

        if not model_target:
            raise_workspace_error(
                context_name="Workspace",
                message="Cannot resolve active model directory: Active model name is unassigned."
            )

        if not img_target:
            raise_workspace_error(
                context_name="Workspace",
                message="Cannot resolve active model directory: Active image context is unassigned."
            )

        return self.get_model_dir(img_name=img_target, model_name=model_target)

    def get_active_model_name(self) -> Optional[str]:
        """
        Retrieves the raw string identifier of the currently active machine learning model context.

        :return: The active model name string, or None if no model context is currently set.
        :rtype: Optional[str]
        """
        return self.active_model_name or self.active_model_name_global