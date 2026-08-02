# Heading 1 (Helpers Proxy Implementation)
## Core component executing directory creations, directory traversal, and context cleanup

from __future__ import annotations
import os
from pathlib import Path
from typing import Any, List, Optional, Dict

from .base_workspace_proxy import BaseWorkspaceProxy
from .getters_and_setters_proxy import GLOBAL_CONTEXT_KEY
from .....utils.exceptions import raise_workspace_error
from .....utils.logger import get_custom_logger

# Logger initialization
logger = get_custom_logger(__name__)


class HelpersProxy(BaseWorkspaceProxy):
    """
    Proxy component handling directory structure creation, scanning utilities, 
    and workspace visualization.
    """

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        """
        Initializes the helpers workspace proxy.
        """
        super().__init__(*args, **kwargs)


    # --------------------------------------------------
    # Section: Structural Visualization Utilities
    # --------------------------------------------------

    def __str__(self) -> str:
        """
        Defines the native string representation of the WorkspaceProxy configuration.
        Constructs the current filesystem schema block for terminal logging and print output.
        """
        model_part = self.active_model_name or "<model_name>"
        img_part = self.active_img_name or "<img_name>"
        
        lines = [
            f"{self.project_path_resolved}/",
            f"├── {self._layout['datasets_dir']}/",
            f"│   ├── catalog.sqlite",
            f"│   └── {img_part}/",
            f"│       └── {img_part}.imzML / {img_part}.ibd",
            f"└── {self._layout['models_root']}/",
            f"    ├── {GLOBAL_CONTEXT_KEY}/",
            f"    │   └── {model_part}/ (Global models layout)",
            f"    │       ├── {self._layout['model_config_subdir']}/",
            f"    │       │   ├── config.json",
            f"    │       │   ├── weights.pt",
            f"    │       │   └── history.json",
            f"    │       └── {self._layout['model_latent_subdir']}/",
            f"    └── {img_part}/ (Local models layout)",
            f"        └── {model_part}/",
            f"            ├── {self._layout['model_config_subdir']}/",
            f"            │   ├── config.json",
            f"            │   ├── weights.pt",
            f"            │   └── history.json",
            f"            └── {self._layout['model_latent_subdir']}/"
        ]
        return "\n".join(lines)

    def print_workspace_layout(self) -> None:
        """
        Dynamically prints the project workspace file layout tree structure to stdout based on the current
        active model and image context variables. Useful for verification inside interactive notebooks.
        """
        print(self)

    # --------------------------------------------------
    # Section: Directory Tree Generation
    # --------------------------------------------------

    def create_required_directories(self) -> None:
        """
        Explicitly triggers physical creation of base workspace layouts on the storage system.
        Registers parent project directories as well as basic target structural subdirectories.
        """
        try:
            ## Ensure root workspace project path exists
            if not self.project_path_resolved.exists():
                logger.info("Creating project root directory: %s", self.project_path_resolved)
                self.project_path_resolved.mkdir(parents=True, exist_ok=True)

            ## Create base dataset and model directories.
            self.get_datasets_dir()
            models_dir = self.project_path_resolved / self._layout["models_root"]

            if not models_dir.exists():
                logger.info("Creating models root directory: %s", models_dir)
                models_dir.mkdir(parents=True, exist_ok=True)

        except Exception as err:
            logger.error("Failed to create required project directories", exc_info=True)
            raise_workspace_error(
                context_name="WorkspaceHelpers",
                message=f"Could not provision filesystem: {err}"
            )

    def create_structure(self, img_name: str, model_name: str) -> None:
        """
        Guarantees and automatically provisions the directory structure for a specific model context.
        Creates both 'config' and 'latent' directories under the correct new hierarchy path.

        :param img_name: Name of the target image context (or GLOBAL_CONTEXT_KEY).
        :type img_name: str
        :param model_name: Name of the target model instance.
        :type model_name: str
        """
        # Directory resolving pass
        ## Fetch nested directory targets from the getters_and_setters definitions
        config_dir = self.get_config_dir(img_name=img_name, model_name=model_name)
        latent_dir = self.get_latent_dir(img_name=img_name, model_name=model_name)

        # File system write block
        try:
            ## Ensure config directory exists
            if not config_dir.exists():
                logger.debug("Creating configuration subfolder: %s", config_dir)
                config_dir.mkdir(parents=True, exist_ok=True)

            ## Ensure latent directory exists
            if not latent_dir.exists():
                logger.debug("Creating latent space representation subfolder: %s", latent_dir)
                latent_dir.mkdir(parents=True, exist_ok=True)

            logger.info("Successfully provisioned workspace directory structure for model '%s' under context '%s'.", 
                        model_name, img_name)

        except Exception as error:
            raise_workspace_error(
                context_name="Workspace",
                message=f"Failed to create workspace directory structure on disk: {error}"
            )

    # --------------------------------------------------
    # Section: Directory Discovery / Scanning Helpers
    # --------------------------------------------------

    def scan_available_models(self, img_name: Optional[str] = None) -> List[str]:
        """
        Scans the 'models/' root directory to discover initialized models.
        If an img_name is specified, it scans local models inside models/<img_name>/.
        If img_name == GLOBAL_CONTEXT_KEY, it scans models/global/.
        If no img_name is specified, it performs a full directory traversal and returns all model names.

        :param img_name: Optional context filter (e.g. image key or GLOBAL_CONTEXT_KEY).
        :type img_name: Optional[str]
        :return: List of discovered model names.
        :rtype: List[str]
        """
        models_root = self.get_models_root()
        if not models_root.exists() or not models_root.is_dir():
            logger.warning("Models root directory does not exist: %s", models_root)
            return []

        discovered_models: List[str] = []

        # Directory traversal branch
        if img_name is not None:
            ## Scanning a specific context folder (either 'global' or a specific image directory)
            target_sub_root = models_root / img_name
            if target_sub_root.exists() and target_sub_root.is_dir():
                for item in target_sub_root.iterdir():
                    if item.is_dir() and not item.name.startswith("."):
                        discovered_models.append(item.name)
        else:
            ## Full discovery traversal across all directories
            for context_item in models_root.iterdir():
                if context_item.is_dir() and not context_item.name.startswith("."):
                    for model_item in context_item.iterdir():
                        if model_item.is_dir() and not model_item.name.startswith("."):
                            # We collect the path representation (e.g., "img_name/model_name")
                            discovered_models.append(f"{context_item.name}/{model_item.name}")

        logger.debug("Discovered %s models on disk.", len(discovered_models))
        return sorted(discovered_models)

    # --------------------------------------------------
    # Section: Session Context Teardown
    # --------------------------------------------------

    def clear_active_context(self) -> None:
        """
        Resets all working context session variables back to None after execution is complete.
        Also signals the coupled active reader proxy to release handles and clear cache allocation.
        """
        logger.debug("Executing full workspace session context teardown.")
        
        # Cache reset execution
        self.active_model_name = None
        self.active_img_name = None
        self.active_img_names = None
        self._active_img_custom_path = None
        
        # State signaling block
        ## Access the central wrapper to notify active_context if present
        wrapper_instance = getattr(self, "_wrapper", None)
        if wrapper_instance is not None:
            active_ctx = getattr(wrapper_instance, "active_context", None)
            if active_ctx is not None and hasattr(active_ctx, "clear_active_context"):
                logger.debug("Signaling active reader proxy to clear file handles and memory allocation.")
                active_ctx.clear_active_context()
